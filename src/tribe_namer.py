"""Evidence-based tribe naming helpers."""

from __future__ import annotations

import os
import math
from typing import Any

import polars as pl

from src.product_themes import detect_product_themes


THEME_LABELS = {
    "alcohol": "Beer & Alcohol Buyers",
    "alcohol_free": "Alcohol-Free Beer Buyers",
    "baby": "Baby Care Buyers",
    "halal": "Halal Product Buyers",
    "pet": "Pet Care Buyers",
    "pet_dog": "Dog Product Buyers",
    "pet_cat": "Cat Product Buyers",
    "kids_general": "Kids & Family Buyers",
    "kids_girls": "Girls Apparel & Toy Buyers",
    "kids_boys": "Boys Apparel & Toy Buyers",
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

THEME_COMBINATION_LABELS = {
    frozenset({"plant_based", "organic_bio"}): "Plant-Based & Bio Buyers",
    frozenset({"dairy_eggs", "organic_bio"}): "Organic Dairy Buyers",
    frozenset({"plant_based", "protein_fitness"}): "Plant-Based & Protein Buyers",
    frozenset({"organic_bio", "protein_fitness"}): "Organic Protein Buyers",
    frozenset({"gluten_free", "lactose_free"}): "Special Diet Buyers",
    frozenset({"baby", "apparel_textile"}): "Baby & Family Buyers",
    frozenset({"baby", "kids_general"}): "Young Family Buyers",
    frozenset({"baby", "books_toys"}): "Young Family Buyers",
    frozenset({"kids_general", "books_toys"}): "Kids, Books & Toys Buyers",
    frozenset({"kids_girls", "apparel_textile"}): "Girls Apparel Buyers",
    frozenset({"kids_boys", "apparel_textile"}): "Boys Apparel Buyers",
    frozenset({"pet_dog", "pet_cat"}): "Multi-Pet Buyers",
    frozenset({"pet", "pet_dog"}): "Dog Product Buyers",
    frozenset({"pet", "pet_cat"}): "Cat Product Buyers",
    frozenset({"world_foods_mexican", "world_foods_asian"}): "World Food Buyers",
    frozenset({"world_foods_middle_eastern", "world_foods_latin"}): "World Food Buyers",
    frozenset({"alcohol", "beverages_soft"}): "Drinks Buyers",
    frozenset({"meat_charcuterie", "fresh_produce"}): "Fresh Food Buyers",
    frozenset({"seafood", "premium_indulgence"}): "Premium Seafood Buyers",
    frozenset({"apparel_textile", "books_toys"}): "Family Non-Food Buyers",
    frozenset({"home_cleaning", "personal_care_beauty"}): "Household Essentials Buyers",
}

SECTOR_LABELS = {
    "TEXTIL": "Textile & Apparel Buyers",
    "ELECTROFOTO": "Electronics & Appliance Buyers",
    "BAZAR": "Home & General Merchandise Buyers",
    "PROD. FRESCOS TRADIC": "Fresh Food Buyers",
    "P.G.C.": "Packaged Grocery Buyers",
}


def fallback_tribe_name(profile_row: dict[str, Any]) -> str:
    product_theme_name = _product_theme_name(profile_row)
    if _is_low_distinction(profile_row):
        return product_theme_name or "Mixed Basket Generalists"

    theme_candidates = _rank_theme_candidates(profile_row)
    if theme_candidates:
        top_theme = theme_candidates[0]
        if len(theme_candidates) > 1:
            second_theme = theme_candidates[1]
            combo = THEME_COMBINATION_LABELS.get(frozenset({top_theme["theme"], second_theme["theme"]}))
            if combo and second_theme["score"] >= top_theme["score"] * 0.65:
                return combo
        return THEME_LABELS.get(top_theme["theme"], f"{str(top_theme['theme']).replace('_', ' ').title()} Buyers")

    if product_theme_name:
        return product_theme_name

    sector_name = _best_sector_name(profile_row)
    if sector_name:
        return sector_name

    products = profile_row.get("top_products") or []
    if products:
        product_hint = _product_name_hint(str(products[0]))
        if product_hint:
            return f"{product_hint} Buyers"
    return f"Tribe {profile_row.get('tribe_id')}"


def _rank_theme_candidates(profile_row: dict[str, Any]) -> list[dict[str, Any]]:
    themes = profile_row.get("top_themes") or []
    lifts = profile_row.get("top_theme_lifts") or []
    counts = profile_row.get("top_theme_customer_counts") or []
    n_customers = max(int(profile_row.get("n_customers") or 0), 1)
    candidates = []
    for idx, theme in enumerate(themes):
        lift = float(lifts[idx]) if idx < len(lifts) and lifts[idx] is not None else 0.0
        count = int(counts[idx]) if idx < len(counts) and counts[idx] is not None else 0
        coverage = count / n_customers
        if lift < 1.15 and not (lift >= 1.08 and coverage >= 0.60):
            continue
        if coverage < 0.05 and count < 75:
            continue
        score = lift * math.sqrt(max(coverage, 0.0))
        candidates.append({"theme": str(theme), "lift": lift, "coverage": coverage, "score": score})
    return sorted(candidates, key=lambda item: (-item["score"], -item["lift"], item["theme"]))


def _best_sector_name(profile_row: dict[str, Any]) -> str | None:
    sectors = profile_row.get("top_sectors") or []
    lifts = profile_row.get("top_sector_lifts") or []
    for idx, sector in enumerate(sectors):
        lift = float(lifts[idx]) if idx < len(lifts) and lifts[idx] is not None else 0.0
        sector_text = str(sector).upper()
        if lift >= 1.20:
            return SECTOR_LABELS.get(sector_text, f"{sector_text.title()} Buyers")
    return None


def _product_name_hint(product: str) -> str | None:
    words = [part.title() for part in product.replace("/", " ").split() if len(part) >= 4]
    if not words:
        return None
    return " ".join(words[:2])


def _product_theme_name(profile_row: dict[str, Any]) -> str | None:
    products = profile_row.get("top_products") or []
    theme_counts: dict[str, int] = {}
    for product in products[:8]:
        for theme in detect_product_themes(str(product)):
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
    if not theme_counts:
        return None
    theme, count = sorted(theme_counts.items(), key=lambda item: (-item[1], item[0]))[0]
    if count < 2:
        return None
    return THEME_LABELS.get(theme, f"{theme.replace('_', ' ').title()} Buyers")


def _is_low_distinction(profile_row: dict[str, Any]) -> bool:
    theme_lifts = [float(value) for value in (profile_row.get("top_theme_lifts") or []) if value is not None]
    sector_lifts = [float(value) for value in (profile_row.get("top_sector_lifts") or []) if value is not None]
    strong_themes = sum(1 for value in theme_lifts if value >= 1.20)
    strong_sectors = sum(1 for value in sector_lifts if value >= 1.20)
    return strong_themes == 0 and strong_sectors == 0


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
