"""Lightweight product-theme patterns for tribe narrative overlays."""

from __future__ import annotations

import re


STRATEGIC_THEME_PATTERNS: dict[str, list[str]] = {
    "baby": [r"\bbebe\b", r"\bbaby\b", r"panal"],
    "pet": [r"\bperro\b", r"\bgato\b", r"mascota"],
    "organic_bio": [r"\bbio\b", r"ecolog", r"organ"],
    "gluten_free": [r"sin gluten", r"\bsg\b"],
    "lactose_free": [r"sin lactosa"],
    "protein_fitness": [r"protein", r"fitness", r"\bzero\b"],
    "health_wellness": [r"integral", r"light", r"salud"],
    "plant_based": [r"vegano", r"vegetal", r"tofu"],
    "ready_meals": [r"preparad", r"microondas", r"listo"],
    "premium_indulgence": [r"premium", r"gourmet", r"delicat"],
}


def detect_product_themes(description: str | None) -> list[str]:
    text = (description or "").lower()
    themes = []
    for theme, patterns in STRATEGIC_THEME_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            themes.append(theme)
    return themes
