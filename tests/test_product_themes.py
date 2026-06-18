from src.product_themes import detect_product_themes


def test_baby_food_tags_suppress_generic_ingredient_themes():
    themes = detect_product_themes("TARRITO HERO RECETAS CASERAS COCIDO TERNERA 2 X 190 GR")

    assert "baby" in themes
    assert "baby_food" in themes
    assert "meat_charcuterie" not in themes


def test_yogolino_is_baby_product_not_generic_dairy_theme():
    themes = detect_product_themes("YOGOLINO MELOCOTON PLATANO S/AZUCAR ANADIDO 4X100G")

    assert "baby" in themes
    assert "baby_food" in themes
    assert "dairy_eggs" not in themes


def test_baby_girls_clothing_keeps_granular_and_broad_product_tags():
    themes = detect_product_themes("VESTIDO TEX BABY NINA ALGODON ECOLOGICO")

    assert "baby" in themes
    assert "baby_clothing" in themes
    assert "baby_girls_clothing" in themes
    assert "apparel_textile" in themes
    assert "organic_bio" in themes


def test_product_tags_are_multi_label_for_overlapping_claims():
    themes = detect_product_themes("YOGUR BIO SIN LACTOSA NATURAL")

    assert "organic_bio" in themes
    assert "lactose_free" in themes
    assert "dairy_eggs" in themes
