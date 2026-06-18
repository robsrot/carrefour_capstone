"""Organic Stage 7 profiling helpers derived only from transaction evidence."""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import timedelta
from typing import Any

import polars as pl

from src.utils import collect_streaming, schema_names


def _compute_copurchase_missions(
    tribe_id: int,
    tribe_customers: pl.DataFrame,
    transactions: pl.LazyFrame,
    min_baskets: int = 5,
    top_n_pairs: int = 12,
    *,
    rest_customers: pl.DataFrame | None = None,
    candidate_product_ids: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Return product pairs that co-occur in tribe baskets more than in the rest."""

    columns = set(schema_names(transactions))
    required = {"cliente", "ticket", "idarticu"}
    if not required.issubset(columns) or tribe_customers.is_empty():
        return []

    desc_expr = (
        pl.col("desc_larga_articulo").cast(pl.Utf8)
        if "desc_larga_articulo" in columns
        else pl.col("idarticu").cast(pl.Utf8)
    )
    product_filter = (
        pl.col("idarticu").is_in(candidate_product_ids)
        if candidate_product_ids
        else pl.lit(True)
    )

    tribe_pairs, tribe_baskets = _pair_counts_for_customers(
        transactions,
        tribe_customers,
        desc_expr=desc_expr,
        product_filter=product_filter,
    )
    if tribe_pairs.is_empty() or tribe_baskets <= 0:
        return []
    tribe_pairs = tribe_pairs.filter(pl.col("cooccurrence_count") >= int(min_baskets))
    if tribe_pairs.is_empty():
        return []

    rest_pair_counts = pl.DataFrame(
        schema={
            "idarticu_a": tribe_pairs.schema["idarticu_a"],
            "idarticu_b": tribe_pairs.schema["idarticu_b"],
            "rest_cooccurrence_count": pl.Int64,
        }
    )
    rest_baskets = 0
    if rest_customers is not None and not rest_customers.is_empty():
        rest_pairs, rest_baskets = _pair_counts_for_customers(
            transactions,
            rest_customers,
            desc_expr=desc_expr,
            product_filter=product_filter,
            rest=True,
        )
        if not rest_pairs.is_empty():
            rest_pair_counts = rest_pairs.select(
                ["idarticu_a", "idarticu_b", "rest_cooccurrence_count"]
            )

    joined = (
        tribe_pairs.join(rest_pair_counts, on=["idarticu_a", "idarticu_b"], how="left")
        .with_columns(
            [
                (pl.col("cooccurrence_count") / max(tribe_baskets, 1)).alias("cooccurrence_rate"),
                (pl.col("rest_cooccurrence_count").fill_null(0) / max(rest_baskets, 1)).alias(
                    "rest_cooccurrence_rate"
                ),
            ]
        )
        .with_columns(
            pl.when(pl.col("rest_cooccurrence_rate") > 0)
            .then(pl.col("cooccurrence_rate") / pl.col("rest_cooccurrence_rate"))
            .otherwise(None)
            .alias("lift_vs_rest")
        )
        .with_columns(
            pl.when(pl.col("lift_vs_rest").is_not_null())
            .then(pl.col("lift_vs_rest"))
            .otherwise(pl.col("cooccurrence_rate") * 100.0)
            .alias("_sort_score")
        )
        .sort(["_sort_score", "cooccurrence_count"], descending=[True, True])
        .head(top_n_pairs)
    )

    rows: list[dict[str, Any]] = []
    for row in joined.iter_rows(named=True):
        desc_a = _repair_text(row.get("desc_a") or row.get("idarticu_a"))
        desc_b = _repair_text(row.get("desc_b") or row.get("idarticu_b"))
        rows.append(
            {
                "product_a": desc_a,
                "product_b": desc_b,
                "idarticu_a": row.get("idarticu_a"),
                "idarticu_b": row.get("idarticu_b"),
                "cooccurrence_count": int(row.get("cooccurrence_count") or 0),
                "cooccurrence_rate": _round_float(row.get("cooccurrence_rate"), 6),
                "rest_cooccurrence_rate": _round_float(row.get("rest_cooccurrence_rate"), 6),
                "lift_vs_rest": _round_float(row.get("lift_vs_rest"), 3),
                "mission_label": f"{_mission_words(desc_a)} + {_mission_words(desc_b)}",
            }
        )
    return rows


def _pair_counts_for_customers(
    transactions: pl.LazyFrame,
    customers: pl.DataFrame,
    *,
    desc_expr: pl.Expr,
    product_filter: pl.Expr,
    rest: bool = False,
) -> tuple[pl.DataFrame, int]:
    basket_items = (
        transactions.join(customers.lazy().select("cliente"), on="cliente", how="inner")
        .filter(pl.col("ticket").is_not_null() & pl.col("idarticu").is_not_null())
        .filter(product_filter)
        .select(
            [
                pl.col("cliente").cast(pl.Utf8),
                pl.col("ticket").cast(pl.Utf8),
                pl.col("idarticu"),
                desc_expr.alias("desc"),
            ]
        )
        .with_columns(
            pl.concat_str([pl.col("cliente"), pl.lit("::"), pl.col("ticket")]).alias("_basket_id")
        )
        .unique(subset=["_basket_id", "idarticu"])
    )
    basket_count = int(
        collect_streaming(basket_items.select(pl.col("_basket_id").n_unique().alias("n_baskets")))[
            0,
            "n_baskets",
        ]
        or 0
    )
    if basket_count <= 0:
        return pl.DataFrame(), 0

    left = basket_items.select(["_basket_id", "idarticu", "desc"]).rename(
        {"idarticu": "idarticu_a", "desc": "desc_a"}
    )
    right = basket_items.select(["_basket_id", "idarticu", "desc"]).rename(
        {"idarticu": "idarticu_b", "desc": "desc_b"}
    )
    count_name = "rest_cooccurrence_count" if rest else "cooccurrence_count"
    pairs = collect_streaming(
        left.join(right, on="_basket_id", how="inner")
        .filter(pl.col("idarticu_a") < pl.col("idarticu_b"))
        .group_by(["idarticu_a", "desc_a", "idarticu_b", "desc_b"])
        .agg(pl.col("_basket_id").n_unique().alias(count_name))
    )
    return pairs, basket_count


def _compute_temporal_patterns(
    tribe_id: int,
    tribe_customers: pl.DataFrame,
    transactions: pl.LazyFrame,
    total_assigned_customers: int,
    *,
    rest_customers: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute day and hour shopping over-indexes for a tribe."""

    del tribe_id, total_assigned_customers
    columns = set(schema_names(transactions))
    if "fecha" not in columns or "ticket" not in columns or tribe_customers.is_empty():
        return {}

    tribe = _basket_time_counts(transactions, tribe_customers)
    rest = _basket_time_counts(transactions, rest_customers) if rest_customers is not None else {}
    if not tribe.get("dow_counts"):
        return {}

    weekday_overindex = _share_overindex(tribe["dow_counts"], rest.get("dow_counts", {}), _DOW_NAMES)
    dominant_day = max(weekday_overindex.items(), key=lambda item: item[1])[0] if weekday_overindex else None
    temporal: dict[str, Any] = {
        "weekday_overindex": weekday_overindex,
        "dominant_shopping_day": dominant_day,
        "weekend_vs_weekday_ratio": _weekend_ratio(
            tribe["dow_counts"],
            rest.get("dow_counts", {}),
        ),
    }

    if "hora" in columns:
        hour_overindex = _share_overindex(tribe.get("hour_counts", {}), rest.get("hour_counts", {}), _TIME_SLOTS)
        temporal["hour_overindex"] = hour_overindex
        dominant_time = _dominant_time_label(max(hour_overindex.items(), key=lambda item: item[1])[0]) if hour_overindex else None
        temporal["dominant_shopping_time"] = dominant_time
    else:
        temporal["hour_overindex"] = {}
        temporal["dominant_shopping_time"] = None
    return temporal


def _basket_time_counts(transactions: pl.LazyFrame, customers: pl.DataFrame | None) -> dict[str, Any]:
    if customers is None or customers.is_empty():
        return {"dow_counts": {}, "hour_counts": {}}
    columns = set(schema_names(transactions))
    exprs = [
        pl.col("cliente").cast(pl.Utf8),
        pl.col("ticket").cast(pl.Utf8),
        pl.col("fecha"),
        (((pl.col("fecha").dt.strftime("%w").cast(pl.Int8) + 6) % 7).alias("_dow")),
    ]
    if "hora" in columns:
        exprs.append(_time_slot_expr().alias("_time_slot"))
    baskets = (
        transactions.join(customers.lazy().select("cliente"), on="cliente", how="inner")
        .filter(pl.col("ticket").is_not_null() & pl.col("fecha").is_not_null())
        .select(exprs)
        .with_columns(
            pl.concat_str([pl.col("cliente"), pl.lit("::"), pl.col("ticket")]).alias("_basket_id")
        )
        .unique(subset=["_basket_id"])
    )
    dow_counts = {
        int(row["_dow"]): int(row["baskets"])
        for row in collect_streaming(
            baskets.group_by("_dow").agg(pl.len().alias("baskets"))
        ).iter_rows(named=True)
    }
    hour_counts: dict[str, int] = {}
    if "hora" in columns:
        hour_counts = {
            str(row["_time_slot"]): int(row["baskets"])
            for row in collect_streaming(
                baskets.group_by("_time_slot").agg(pl.len().alias("baskets"))
            ).iter_rows(named=True)
        }
    return {"dow_counts": dow_counts, "hour_counts": hour_counts}


def _time_slot_expr() -> pl.Expr:
    hour = pl.col("hora").cast(pl.Int16)
    return (
        pl.when((hour >= 6) & (hour < 12))
        .then(pl.lit("morning_6_12"))
        .when((hour >= 12) & (hour < 18))
        .then(pl.lit("afternoon_12_18"))
        .when((hour >= 18) & (hour < 24))
        .then(pl.lit("evening_18_24"))
        .otherwise(pl.lit("night_0_6"))
    )


def _compute_loyalty_profile(
    tribe_id: int,
    tribe_customers: pl.DataFrame,
    transactions: pl.LazyFrame,
    reference_date: Any,
    *,
    rest_customers: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute tenure, recency, and visit-trend profile for a tribe."""

    del tribe_id
    columns = set(schema_names(transactions))
    if "fecha" not in columns or tribe_customers.is_empty() or reference_date is None:
        return {}

    tribe_dates = _customer_date_profile(transactions, tribe_customers, reference_date)
    rest_dates = _customer_date_profile(transactions, rest_customers, reference_date) if rest_customers is not None else pl.DataFrame()
    if tribe_dates.is_empty():
        return {}

    tribe_stats = _date_stats(tribe_dates)
    rest_stats = _date_stats(rest_dates)
    visit_trend = _visit_trend(transactions, tribe_customers, reference_date)
    mean_tenure = tribe_stats.get("mean_tenure_days")
    loyalty_profile = {
        **tribe_stats,
        "new_customer_share_pct": _share_pct(tribe_dates, pl.col("tenure_days") < 30),
        "loyal_customer_share_pct": _share_pct(tribe_dates, pl.col("tenure_days") > 90),
        "tenure_vs_rest_ratio": _safe_ratio(mean_tenure, rest_stats.get("mean_tenure_days")),
        "recency_vs_rest_ratio": _safe_ratio(tribe_stats.get("mean_recency_days"), rest_stats.get("mean_recency_days")),
        "visit_trend": visit_trend,
    }
    return loyalty_profile


def _customer_date_profile(
    transactions: pl.LazyFrame,
    customers: pl.DataFrame | None,
    reference_date: Any,
) -> pl.DataFrame:
    if customers is None or customers.is_empty():
        return pl.DataFrame()
    return collect_streaming(
        transactions.join(customers.lazy().select("cliente"), on="cliente", how="inner")
        .filter(pl.col("fecha").is_not_null())
        .group_by("cliente")
        .agg(
            [
                pl.col("fecha").min().alias("first_purchase"),
                pl.col("fecha").max().alias("last_purchase"),
                pl.col("ticket").n_unique().alias("total_baskets") if "ticket" in schema_names(transactions) else pl.len().alias("total_baskets"),
            ]
        )
        .with_columns(
            [
                (pl.lit(reference_date) - pl.col("first_purchase")).dt.total_days().alias("tenure_days"),
                (pl.lit(reference_date) - pl.col("last_purchase")).dt.total_days().alias("recency_days"),
            ]
        )
    )


def _date_stats(frame: pl.DataFrame) -> dict[str, float | None]:
    if frame.is_empty():
        return {
            "mean_tenure_days": None,
            "median_tenure_days": None,
            "mean_recency_days": None,
            "median_recency_days": None,
        }
    row = frame.select(
        [
            pl.col("tenure_days").mean().alias("mean_tenure_days"),
            pl.col("tenure_days").median().alias("median_tenure_days"),
            pl.col("recency_days").mean().alias("mean_recency_days"),
            pl.col("recency_days").median().alias("median_recency_days"),
        ]
    ).row(0, named=True)
    return {key: _round_float(value, 3) for key, value in row.items()}


def _visit_trend(transactions: pl.LazyFrame, customers: pl.DataFrame, reference_date: Any) -> str:
    columns = set(schema_names(transactions))
    if "ticket" not in columns:
        return "unknown"
    recent_start = reference_date - timedelta(days=30)
    prior_start = reference_date - timedelta(days=60)
    counts = collect_streaming(
        transactions.join(customers.lazy().select("cliente"), on="cliente", how="inner")
        .filter(pl.col("fecha").is_not_null())
        .with_columns(
            [
                (pl.col("fecha") > pl.lit(recent_start)).alias("_recent"),
                ((pl.col("fecha") > pl.lit(prior_start)) & (pl.col("fecha") <= pl.lit(recent_start))).alias("_prior"),
            ]
        )
        .select(
            [
                pl.col("ticket").filter(pl.col("_recent")).n_unique().alias("recent_baskets"),
                pl.col("ticket").filter(pl.col("_prior")).n_unique().alias("prior_baskets"),
            ]
        )
    ).row(0, named=True)
    recent = float(counts.get("recent_baskets") or 0.0)
    prior = float(counts.get("prior_baskets") or 0.0)
    if prior <= 0 and recent > 0:
        return "growing"
    if prior <= 0:
        return "stable"
    ratio = recent / prior
    if ratio > 1.10:
        return "growing"
    if ratio < 0.90:
        return "declining"
    return "stable"


def synthesize_tribe_with_claude(
    tribe_id: int,
    n_customers: int,
    population_share_pct: float,
    top_products: list[dict[str, Any]],
    top_sectors: list[dict[str, Any]],
    behavior_ratios: dict[str, float],
    copurchase_pairs: list[dict[str, Any]],
    temporal_pattern: dict[str, Any],
    loyalty_profile: dict[str, Any],
    api_key: str,
    model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """Use Claude to synthesize an evidence-only working tribe read."""

    prompt = _claude_prompt(
        tribe_id,
        n_customers,
        population_share_pct,
        top_products,
        top_sectors,
        behavior_ratios,
        copurchase_pairs,
        temporal_pattern,
        loyalty_profile,
    )
    raw = _call_claude(prompt, api_key=api_key, model=model)
    parsed = _parse_json_response(raw)
    parsed["label_source"] = "claude_synthesis"
    return parsed


def synthesize_tribe_with_gemini(
    tribe_id: int,
    n_customers: int,
    population_share_pct: float,
    top_products: list[dict[str, Any]],
    top_sectors: list[dict[str, Any]],
    behavior_ratios: dict[str, float],
    copurchase_pairs: list[dict[str, Any]],
    temporal_pattern: dict[str, Any],
    loyalty_profile: dict[str, Any],
    api_key: str,
    model: str = "gemini-3.5-flash",
) -> dict[str, Any]:
    """Use Gemini to synthesize an evidence-only working tribe read."""

    prompt = _claude_prompt(
        tribe_id,
        n_customers,
        population_share_pct,
        top_products,
        top_sectors,
        behavior_ratios,
        copurchase_pairs,
        temporal_pattern,
        loyalty_profile,
    )
    raw = _call_gemini(prompt, api_key=api_key, model=model)
    parsed = _parse_json_response(raw)
    parsed["label_source"] = "gemini_synthesis"
    return parsed


def format_products_for_prompt(top_products: list[dict[str, Any]], max_products: int = 12) -> str:
    lines = []
    for i, p in enumerate(top_products[:max_products], 1):
        lines.append(
            f"  {i}. {p['desc']} -- {float(p['lift_vs_rest']):.1f}x lift, "
            f"{float(p['reach_pct']):.1f}% of tribe bought this (q={float(p['q_value']):.3f}, n={int(p['customers'])})"
        )
    return "\n".join(lines) if lines else "  No product evidence available."


def format_sectors_for_prompt(top_sectors: list[dict[str, Any]]) -> str:
    lines = []
    for s in top_sectors:
        direction = "over" if float(s.get("lift") or 0.0) > 1 else "under"
        lines.append(f"  {s['desc_sector']}: {float(s['lift']):.2f}x vs population ({direction}-indexed)")
    return "\n".join(lines) if lines else "  No sector evidence available."


def format_behavior_for_prompt(behavior_ratios: dict[str, float]) -> str:
    lines = []
    metric_labels = {
        "avg_ticket_count": "Total baskets",
        "avg_basket_value": "Avg basket value (EUR)",
        "avg_promo_share": "Promo sensitivity",
        "avg_unique_products": "Product variety",
        "avg_recency_days": "Recency (lower = more recent)",
        "avg_frequency_per_30d": "Visit frequency per 30d",
    }
    for key, ratio in sorted(behavior_ratios.items(), key=lambda item: abs(float(item[1]) - 1), reverse=True):
        label = metric_labels.get(key, key)
        direction = "higher" if float(ratio) > 1 else "lower"
        lines.append(f"  {label}: {float(ratio):.2f}x vs rest ({direction})")
    return "\n".join(lines) if lines else "  No behavior contrast available."


def format_copurchase_for_prompt(pairs: list[dict[str, Any]], max_pairs: int = 6) -> str:
    if not pairs:
        return "  No strong co-purchase patterns detected above threshold."
    lines = []
    for p in pairs[:max_pairs]:
        lines.append(
            f"  {p['product_a']} + {p['product_b']}: "
            f"bought together in {float(p['cooccurrence_rate'] or 0.0) * 100:.1f}% of tribe baskets "
            f"({float(p['lift_vs_rest'] or 0.0):.1f}x more than other tribes)"
        )
    return "\n".join(lines)


def format_temporal_for_prompt(temporal: dict[str, Any]) -> str:
    if not temporal:
        return "  Temporal data not available."
    lines = [
        f"  Dominant shopping day: {temporal.get('dominant_shopping_day', 'n/a')}",
        f"  Weekend vs weekday ratio vs rest: {float(temporal.get('weekend_vs_weekday_ratio') or 1.0):.2f}x",
    ]
    if temporal.get("dominant_shopping_time"):
        lines.append(f"  Dominant shopping time: {temporal['dominant_shopping_time']}")
    return "\n".join(lines)


def format_loyalty_for_prompt(loyalty: dict[str, Any]) -> str:
    if not loyalty:
        return "  Loyalty data not available."
    return "\n".join(
        [
            f"  Avg customer tenure: {float(loyalty.get('mean_tenure_days') or 0):.0f} days "
            f"({float(loyalty.get('tenure_vs_rest_ratio') or 1.0):.2f}x vs rest)",
            f"  Avg recency (days since last visit): {float(loyalty.get('mean_recency_days') or 0):.0f} days "
            f"({float(loyalty.get('recency_vs_rest_ratio') or 1.0):.2f}x vs rest)",
            f"  Loyal customers (>90d tenure): {float(loyalty.get('loyal_customer_share_pct') or 0):.1f}%",
            f"  Visit trend: {loyalty.get('visit_trend', 'unknown')}",
        ]
    )


def _claude_prompt(
    tribe_id: int,
    n_customers: int,
    population_share_pct: float,
    top_products: list[dict[str, Any]],
    top_sectors: list[dict[str, Any]],
    behavior_ratios: dict[str, float],
    copurchase_pairs: list[dict[str, Any]],
    temporal_pattern: dict[str, Any],
    loyalty_profile: dict[str, Any],
) -> str:
    del tribe_id
    return f"""You are a grocery retail data scientist interpreting an organic customer cluster for Carrefour Spain.
The cluster was discovered by unsupervised machine learning on product purchase patterns — no category taxonomy was used to form it.
Your job is to read the raw evidence and describe who these customers are purely from what they buy, when, and how.

STRICT RULES:
- Base every claim directly on the evidence provided. Do not infer from category names.
- Do not use demographic vocabulary (age, gender, family size, income, nationality, religion).
- Do not reference any predefined segment taxonomy (e.g. do not say "organic buyers" unless the word organic appears in lifted product descriptions).
- If evidence is mixed or unclear, say so plainly.
- All output must be in English.

---

TRIBE EVIDENCE
==============

Size: {n_customers} customers ({population_share_pct:.1f}% of assigned population)

LAYER 1 — PRODUCTS THIS TRIBE BUYS MORE THAN ALL OTHER ASSIGNED CUSTOMERS
(sorted by lift × coverage; lift_vs_rest = how many times more likely tribe members are to buy this product compared to customers in different tribes)

{format_products_for_prompt(top_products)}

LAYER 2 — STORE DEPARTMENTS (SECTOR) OVER/UNDER-INDEXED
(lift = tribe share of transaction lines in this sector / all-customers share)

{format_sectors_for_prompt(top_sectors)}

LAYER 3 — BEHAVIOURAL PROFILE VS REST OF ASSIGNED CUSTOMERS
(ratio = tribe mean / rest mean; > 1.0 means tribe is higher)

{format_behavior_for_prompt(behavior_ratios)}

LAYER 4 — CO-PURCHASE MISSIONS (products frequently bought in the same basket)
(lift_vs_rest = how much more often this pair appears together in tribe baskets vs other tribes)

{format_copurchase_for_prompt(copurchase_pairs)}

LAYER 5 — TEMPORAL SHOPPING PATTERN
{format_temporal_for_prompt(temporal_pattern)}

LAYER 6 — LOYALTY AND RECENCY PROFILE
{format_loyalty_for_prompt(loyalty_profile)}

---

Based on the above evidence only, produce the following JSON response:

{{
  "working_label": "<3-5 word label describing what these customers do, not who they are>",
  "shopping_mission": "<2-3 sentences: what is the primary shopping mission that unites this cluster? Reference specific products or pairs from the evidence.>",
  "customer_description": "<2-3 sentences: what can we infer about these customers' relationship with Carrefour and their shopping behaviour? Reference behavioral ratios and loyalty data.>",
  "commercial_opportunities": [
    "<specific action 1: name a product, category, or mechanic Carrefour could use>",
    "<specific action 2>",
    "<specific action 3>"
  ],
  "confidence_note": "<1 sentence: what is the main caveat about this profile? e.g. low coverage of top products, mixed mission signals, etc.>"
}}

Return ONLY the JSON object, no preamble, no markdown fences.
"""


def _call_claude(prompt: str, api_key: str, model: str = "claude-sonnet-4-6") -> str:
    import httpx

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


def _call_gemini(prompt: str, api_key: str, model: str = "gemini-3.5-flash") -> str:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 1000,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    max_retries = 4
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60.0) as response:
                response_json = json.loads(response.read().decode("utf-8"))
                return _extract_gemini_text(response_json)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < max_retries:
                time.sleep(_gemini_retry_sleep_seconds(exc, attempt))
                continue
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {body[:500]}") from exc
    raise RuntimeError("Gemini API request failed after retries.")


def _gemini_retry_sleep_seconds(exc: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            return min(max(float(retry_after), 1.0), 120.0)
        except ValueError:
            pass
    return min(10.0 * (2 ** attempt), 90.0)


def _extract_gemini_text(response_json: dict[str, Any]) -> str:
    candidates = response_json.get("candidates") or []
    for candidate in candidates:
        parts = ((candidate.get("content") or {}).get("parts") or [])
        text = "".join(str(part.get("text") or "") for part in parts)
        if text.strip():
            return text
    raise RuntimeError("Gemini response did not contain text.")


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


_DOW_NAMES = {
    0: "monday",
    1: "tuesday",
    2: "wednesday",
    3: "thursday",
    4: "friday",
    5: "saturday",
    6: "sunday",
}
_TIME_SLOTS = {
    "morning_6_12": "morning_6_12",
    "afternoon_12_18": "afternoon_12_18",
    "evening_18_24": "evening_18_24",
    "night_0_6": "night_0_6",
}
_MISSION_STOPWORDS = {
    "carrefour",
    "crf",
    "merca",
    "marca",
    "producto",
    "pack",
    "caja",
    "bolsa",
    "botella",
    "lata",
    "sobre",
    "tarrina",
    "und",
    "uds",
    "unidad",
    "unidades",
    "kg",
    "gr",
    "g",
    "ml",
    "l",
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "con",
    "sin",
    "para",
    "fresco",
    "fresca",
    "frescos",
    "natural",
    "extra",
}


def _share_overindex(counts: dict[Any, int], rest_counts: dict[Any, int], labels: dict[Any, str]) -> dict[str, float]:
    total = sum(counts.values())
    rest_total = sum(rest_counts.values())
    result: dict[str, float] = {}
    for key, label in labels.items():
        share = counts.get(key, 0) / total if total > 0 else 0.0
        rest_share = rest_counts.get(key, 0) / rest_total if rest_total > 0 else 0.0
        result[label] = round(share / rest_share, 3) if rest_share > 0 else (0.0 if share == 0 else 999.0)
    return result


def _weekend_ratio(tribe_counts: dict[int, int], rest_counts: dict[int, int]) -> float | None:
    def ratio(counts: dict[int, int]) -> float | None:
        weekend = counts.get(5, 0) + counts.get(6, 0)
        weekday = sum(counts.get(day, 0) for day in range(5))
        return weekend / weekday if weekday > 0 else None

    tribe_ratio = ratio(tribe_counts)
    rest_ratio = ratio(rest_counts)
    return _round_float(_safe_ratio(tribe_ratio, rest_ratio), 3)


def _dominant_time_label(slot: str) -> str:
    return {
        "morning_6_12": "morning",
        "afternoon_12_18": "afternoon",
        "evening_18_24": "evening",
        "night_0_6": "night",
    }.get(str(slot), str(slot))


def _share_pct(frame: pl.DataFrame, predicate: pl.Expr) -> float:
    if frame.is_empty():
        return 0.0
    return round(frame.filter(predicate).height / max(frame.height, 1) * 100.0, 3)


def _safe_ratio(value: Any, baseline: Any) -> float | None:
    try:
        numeric = float(value)
        base = float(baseline)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not math.isfinite(base) or base <= 0:
        return None
    return round(numeric / base, 3)


def _round_float(value: Any, digits: int = 3) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def _mission_words(description: str) -> str:
    text = unicodedata.normalize("NFKD", str(description).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text):
        if token in _MISSION_STOPWORDS:
            continue
        if len(token) < 3 or any(char.isdigit() for char in token):
            continue
        tokens.append(token)
        if len(tokens) >= 2:
            break
    return " ".join(tokens) if tokens else "product"


def _repair_text(value: Any) -> str:
    text = "" if value is None else str(value)
    try:
        text = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        pass
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
