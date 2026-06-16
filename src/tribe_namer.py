"""Evidence-based tribe naming helpers."""

from __future__ import annotations

import os
import math
import re
from typing import Any

import polars as pl


THEME_LABELS = {
    "alcohol": "Beer & Alcohol Buyers",
    "alcohol_free": "Alcohol-Free Beer Buyers",
    "baby": "Baby Product Signal",
    "halal": "Halal-Labelled Product Signal",
    "pet": "Pet Care Buyers",
    "pet_dog": "Dog Product Buyers",
    "pet_cat": "Cat Product Buyers",
    "kids_general": "Kids/Toys Product Signal",
    "kids_girls": "Doll/Girls-Labelled Product Signal",
    "kids_boys": "Action/Vehicle Toy Product Signal",
    "world_foods_asian": "Asian-Style World Food Buyers",
    "world_foods_mexican": "Mexican-Style World Food Buyers",
    "world_foods_middle_eastern": "Middle Eastern-Style World Food Buyers",
    "world_foods_latin": "Latin World Food Buyers",
    "apparel_textile": "Apparel & Textile Buyers",
    "books_toys": "Kids, Books & Toys Buyers",
    "organic_bio": "Organic/Bio Buyers",
    "plant_based": "Plant-Based Buyers",
    "protein_fitness": "Protein & Fitness Buyers",
    "gluten_free": "Gluten-Free Buyers",
    "lactose_free": "Lactose-Free Buyers",
    "health_wellness": "Health & Wellness Buyers",
    "ready_meals": "Ready-Meal Buyers",
    "premium_indulgence": "Premium & Indulgence Buyers",
    "fresh_produce": "Fresh Produce Buyers",
    "meat_charcuterie": "Fresh Meat Buyers",
    "seafood": "Seafood Buyers",
    "dairy_eggs": "Dairy & Eggs Buyers",
    "bakery_breakfast": "Bakery & Breakfast Buyers",
    "pantry_staples": "Pantry Staples Buyers",
    "snacks_sweets": "Snacks & Sweets Buyers",
    "beverages_soft": "Soft Drinks Buyers",
    "coffee_tea": "Coffee & Tea Buyers",
    "frozen_ice_cream": "Frozen & Ice Cream Buyers",
    "home_cleaning": "Home Cleaning Buyers",
    "personal_care_beauty": "Personal Care Buyers",
    "home_kitchen": "Home & Kitchen Buyers",
    "electronics_appliances": "Electronics & Appliances Buyers",
    "diy_auto_garden": "DIY, Auto & Garden Buyers",
    "fuel_convenience": "Fuel & Convenience Buyers",
    "seasonal_celebration": "Seasonal Buyers",
}

def fallback_tribe_name(profile_row: dict[str, Any]) -> str:
    """Return the deterministic core label used in profile tables."""

    return tribe_name_evidence(profile_row)["working_tribe_name"]


def tribe_name_evidence(profile_row: dict[str, Any]) -> dict[str, str]:
    """Build a strict evidence-led label and describe where it came from.

    Core tribe names should not be created from client/persona assumptions. The
    order is intentionally conservative: data-driven product terms first,
    individual lifted product descriptions second, and the raw lifted sector
    label third. Rule-based strategic themes remain evidence overlays, not
    primary core names.
    """

    term_candidates = _rank_term_candidates(profile_row)
    if term_candidates:
        selected = _select_name_parts(term_candidates)
        return _name_payload(
            f"{' + '.join(item['label'] for item in selected)} Purchase Cluster",
            "data_driven_product_terms",
            _evidence_summary(selected),
        )

    product_candidates = _rank_product_candidates(profile_row)
    if product_candidates:
        selected = _select_name_parts(product_candidates, max_parts=1)
        return _name_payload(
            f"{selected[0]['label']} Purchase Cluster",
            "lifted_product_description",
            _evidence_summary(selected),
        )

    sector_name = _best_sector_name(profile_row)
    if sector_name:
        return _name_payload(
            f"{sector_name} Purchase Cluster",
            "lifted_sector",
            sector_name,
        )

    return _name_payload(
        f"Tribe {profile_row.get('tribe_id')} Product Evidence Cluster",
        "insufficient_distinctive_evidence",
        "No strong data-derived term, lifted product, or raw sector signal passed naming thresholds.",
    )


def _name_payload(name: str, source: str, evidence: str) -> dict[str, str]:
    return {
        "working_tribe_name": name,
        "working_tribe_name_source": source,
        "working_tribe_name_evidence": evidence,
    }


def _select_name_parts(candidates: list[dict[str, Any]], max_parts: int = 2) -> list[dict[str, Any]]:
    if not candidates:
        return []
    selected = [candidates[0]]
    for candidate in candidates[1:max_parts]:
        if candidate["score"] >= candidates[0]["score"] * 0.65:
            selected.append(candidate)
    return selected


def _evidence_summary(candidates: list[dict[str, Any]]) -> str:
    parts = []
    for item in candidates:
        q_text = "" if item.get("q_value") is None else f", q={float(item['q_value']):.3g}"
        parts.append(
            f"{item['raw_label']} (lift={float(item['lift']):.2f}, coverage={float(item['coverage']):.1%}{q_text})"
        )
    return "; ".join(parts)


def _rank_term_candidates(profile_row: dict[str, Any]) -> list[dict[str, Any]]:
    terms = profile_row.get("top_product_terms") or []
    lifts = profile_row.get("top_product_term_lifts") or []
    counts = profile_row.get("top_product_term_customer_counts") or []
    q_values = profile_row.get("top_product_term_q_values") or []
    n_customers = max(int(profile_row.get("n_customers") or 0), 1)
    candidates = []
    for idx, term in enumerate(terms):
        lift = (_safe_float(lifts[idx]) if idx < len(lifts) else None) or 0.0
        count = _safe_int(counts[idx]) if idx < len(counts) else 0
        q_value = _safe_float(q_values[idx]) if idx < len(q_values) else None
        coverage = count / n_customers
        if lift < 1.25:
            continue
        if coverage < 0.01 and count < 50:
            continue
        raw_label = str(term).strip()
        label = _term_label(raw_label)
        if not label:
            continue
        candidates.append(
            {
                "label": label,
                "raw_label": raw_label,
                "lift": lift,
                "coverage": coverage,
                "q_value": q_value,
                "score": _evidence_score(lift, coverage, q_value),
            }
        )
    return sorted(candidates, key=lambda item: (-item["score"], -item["lift"], item["raw_label"]))


def _rank_product_candidates(profile_row: dict[str, Any]) -> list[dict[str, Any]]:
    products = profile_row.get("top_products") or []
    lifts = profile_row.get("top_product_lifts") or []
    counts = profile_row.get("top_product_customer_counts") or []
    q_values = profile_row.get("top_product_q_values") or []
    n_customers = max(int(profile_row.get("n_customers") or 0), 1)
    candidates = []
    for idx, product in enumerate(products):
        lift = (_safe_float(lifts[idx]) if idx < len(lifts) else None) or 0.0
        count = _safe_int(counts[idx]) if idx < len(counts) else 0
        q_value = _safe_float(q_values[idx]) if idx < len(q_values) else None
        coverage = count / n_customers
        if lift < 1.50:
            continue
        if coverage < 0.005 and count < 25:
            continue
        raw_label = str(product).strip()
        label = _product_name_hint(raw_label)
        if not label:
            continue
        candidates.append(
            {
                "label": label,
                "raw_label": raw_label,
                "lift": lift,
                "coverage": coverage,
                "q_value": q_value,
                "score": _evidence_score(lift, coverage, q_value),
            }
        )
    return sorted(candidates, key=lambda item: (-item["score"], -item["lift"], item["raw_label"]))


def _evidence_score(lift: float, coverage: float, q_value: float | None) -> float:
    significance_bonus = 1.25 if q_value is not None and q_value <= 0.05 else 1.0
    return lift * math.sqrt(max(coverage, 0.0)) * significance_bonus


def _term_label(term: str) -> str:
    words = [part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", term) if len(part) >= 3]
    if not words:
        return ""
    return " ".join(words[:3])


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _best_sector_name(profile_row: dict[str, Any]) -> str | None:
    sectors = profile_row.get("top_sectors") or []
    lifts = profile_row.get("top_sector_lifts") or []
    for idx, sector in enumerate(sectors):
        lift = float(lifts[idx]) if idx < len(lifts) and lifts[idx] is not None else 0.0
        if lift >= 1.20:
            return _sector_label(str(sector))
    return None


def _sector_label(sector: str) -> str:
    text = _repair_display_text(sector).strip()
    if not text:
        return ""
    return text.title()


def _repair_display_text(text: str) -> str:
    repaired = text
    for _ in range(3):
        if "Ãƒ" not in repaired and "Ã‚" not in repaired:
            break
        try:
            next_text = repaired.encode("latin1").decode("utf-8")
        except UnicodeError:
            break
        if next_text == repaired:
            break
        repaired = next_text
    return repaired


def _product_name_hint(product: str) -> str | None:
    words = [part.title() for part in product.replace("/", " ").split() if len(part) >= 4]
    if not words:
        return None
    return " ".join(words[:2])


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
