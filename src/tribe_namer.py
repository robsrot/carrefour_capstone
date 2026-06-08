"""Tribe naming via Claude API.

Takes a tribe profile row (output of profile_tribes()) and returns a commercial
name and one-paragraph business description for that tribe.

Public API
----------
name_tribe(profile_row)     → {"name": str, "description": str, "action": str}
name_all_tribes(profiles_df) → polars DataFrame with name/description/action columns added
"""
from __future__ import annotations

import logging
from typing import Any

import polars as pl

from src.config import CLAUDE_MAX_TOKENS, CLAUDE_MODEL

_log = logging.getLogger(__name__)


def _build_prompt(row: dict[str, Any]) -> str:
    cluster_id   = row.get("cluster", "?")
    n_customers  = row.get("n_customers", 0)
    avg_basket   = row.get("avg_basket", 0.0)
    avg_visits   = row.get("avg_visits", 0.0)
    revenue_share = row.get("revenue_share", 0.0)
    avg_promo    = row.get("avg_promo_rate", 0.0)

    top_products = row.get("top_products") or []
    top_lifts    = row.get("top_lifts") or []
    top_sectors  = row.get("top_sectors") or []
    top_themes   = row.get("top_themes") or []

    product_lines = "\n".join(
        f"  - {p} (lift {l:.1f}x)"
        for p, l in zip(top_products[:8], top_lifts[:8])
    )
    sector_lines = ", ".join(
        f"{s} ({l:.1f}x)" for s, l in zip(top_sectors[:3], row.get("top_sector_lifts", [])[:3])
    )
    theme_lines = ", ".join(top_themes[:5]) if top_themes else "none detected"

    return f"""You are a retail strategist for Carrefour Spain.

Below is the behavioural profile of customer tribe {cluster_id}, discovered through
product embedding and density clustering. Name and describe this tribe commercially.

## Tribe metrics
- Customers: {n_customers:,}
- Revenue share: {revenue_share:.1f}%
- Average basket value: €{avg_basket:.2f}
- Average visits (6 months): {avg_visits:.1f}
- Average promo rate: {avg_promo:.0%}

## Distinctive sectors (lift vs. average customer)
{sector_lines}

## Strategic product themes
{theme_lines}

## Top distinctive products (by lift — how much more than average this tribe buys)
{product_lines}

## Your task
Return ONLY a JSON object with three keys:
- "name": a short (2–4 word) commercial tribe name in English, e.g. "Fresh Family Cooks"
- "description": one paragraph (3–5 sentences) describing who this tribe is, what drives their shopping, and what makes them commercially valuable to Carrefour
- "action": one concrete recommended Carrefour action for this tribe (one sentence)

Respond with valid JSON only. No markdown, no commentary."""


def name_tribe(profile_row: dict[str, Any]) -> dict[str, str]:
    """Call the Claude API to generate a name, description, and action for one tribe.

    Parameters
    ----------
    profile_row : one row from profile_tribes() converted to a dict via .to_dicts()[i]

    Returns
    -------
    dict with keys "name", "description", "action"

    Raises
    ------
    ImportError  if the anthropic package is not installed
    RuntimeError if the API call fails or returns unexpected content
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for tribe naming. "
            "Install it with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic()
    prompt = _build_prompt(profile_row)

    _log.info(
        "Naming tribe %s via %s (max_tokens=%d) ...",
        profile_row.get("cluster", "?"),
        CLAUDE_MODEL,
        CLAUDE_MAX_TOKENS,
    )

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    import json
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise RuntimeError(
                f"Claude returned non-JSON content for tribe {profile_row.get('cluster', '?')}:\n{raw}"
            ) from exc

    for key in ("name", "description", "action"):
        if key not in result:
            result[key] = ""

    _log.info("  Tribe %s → '%s'", profile_row.get("cluster", "?"), result.get("name", ""))
    return result


def name_all_tribes(profiles: pl.DataFrame) -> pl.DataFrame:
    """Name every tribe in a profiles DataFrame returned by profile_tribes().

    Adds three string columns: tribe_name, tribe_description, tribe_action.
    Processes tribes sequentially to stay within API rate limits.

    Parameters
    ----------
    profiles : DataFrame from profile_tribes()

    Returns
    -------
    profiles with tribe_name, tribe_description, and tribe_action columns appended
    """
    rows = profiles.to_dicts()
    names, descriptions, actions = [], [], []

    for row in rows:
        try:
            result = name_tribe(row)
        except Exception as exc:
            _log.warning("Failed to name tribe %s: %s", row.get("cluster", "?"), exc)
            result = {"name": "", "description": "", "action": ""}

        names.append(result.get("name", ""))
        descriptions.append(result.get("description", ""))
        actions.append(result.get("action", ""))

    return profiles.with_columns([
        pl.Series("tribe_name", names, dtype=pl.Utf8),
        pl.Series("tribe_description", descriptions, dtype=pl.Utf8),
        pl.Series("tribe_action", actions, dtype=pl.Utf8),
    ])
