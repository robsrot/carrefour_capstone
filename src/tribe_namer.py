"""Evidence-based tribe naming helpers."""

from __future__ import annotations

import os
from typing import Any

import polars as pl


def fallback_tribe_name(profile_row: dict[str, Any]) -> str:
    sectors = profile_row.get("top_sectors") or []
    products = profile_row.get("top_products") or []
    if sectors:
        return f"{str(sectors[0]).title()} Loyalists"
    if products:
        first = str(products[0]).split()[0].title()
        return f"{first} Mission Shoppers"
    return f"Tribe {profile_row.get('tribe_id')}"


def build_tribe_prompt(profile_row: dict[str, Any]) -> str:
    return (
        "Name this Carrefour customer tribe using only purchase evidence.\n"
        f"Tribe id: {profile_row.get('tribe_id')}\n"
        f"Population share: {profile_row.get('population_share')}\n"
        f"Top lifted products: {profile_row.get('top_products')}\n"
        f"Product lifts: {profile_row.get('top_product_lifts')}\n"
        f"Top lifted sectors: {profile_row.get('top_sectors')}\n"
        f"Sector lifts: {profile_row.get('top_sector_lifts')}\n"
        "Return a concise commercial tribe name and one sentence of rationale."
    )


def name_tribes(profile_path: str, use_claude: bool = False) -> pl.DataFrame:
    """Name tribes, optionally using Claude when ANTHROPIC_API_KEY is available."""

    profiles = pl.read_parquet(profile_path)
    rows = []
    client = None
    if use_claude and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
        except Exception:
            client = None

    for row in profiles.iter_rows(named=True):
        name = fallback_tribe_name(row)
        rationale = "Name generated from the highest lifted sector or product evidence."
        if client is not None:
            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=120,
                    messages=[{"role": "user", "content": build_tribe_prompt(row)}],
                )
                text = response.content[0].text.strip()
                if ":" in text:
                    name, rationale = [part.strip() for part in text.split(":", 1)]
                else:
                    name = text
                    rationale = "Claude-generated name from product and sector lift evidence."
            except Exception:
                pass
        rows.append({"tribe_id": row["tribe_id"], "tribe_name": name, "naming_rationale": rationale})
    return pl.DataFrame(rows).sort("tribe_id")
