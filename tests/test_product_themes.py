from src.product_themes import detect_product_themes, normalize_product_text


def test_normalize_product_text_handles_accents_and_mojibake():
    assert normalize_product_text("CAF\u00c9 MOLIDO") == "cafe molido"
    assert normalize_product_text("CAF\u00c3\u0089 MOLIDO") == "cafe molido"


def test_detect_product_themes_covers_grocery_missions():
    themes = set(detect_product_themes("ARROZ CRUJIENTE CACAO BIO SIN GLUTEN SIN LACTOSA 330 G"))

    assert {"organic_bio", "gluten_free", "lactose_free", "pantry_staples"} <= themes


def test_detect_product_themes_covers_ready_meals_and_dairy():
    themes = set(detect_product_themes("PIZZA 4 QUESOS CARREFOUR 580G"))

    assert {"ready_meals", "dairy_eggs"} <= themes


def test_detect_product_themes_covers_non_food_departments():
    apparel = set(detect_product_themes("PIJAMA MANGA LARGA DISNEY PANTALON MICROPRINT BEBE TEX BABY"))
    appliance = set(detect_product_themes("LAVAVAJILLAS AEG F56312W0 60CM LIBRE INSTALACION A++"))
    detergent = set(detect_product_themes("LAVAVAJILLAS PASTILLAS FINISH QUANTUM ESSENTIAL 50D"))

    assert {"baby", "apparel_textile"} <= apparel
    assert "electronics_appliances" in appliance
    assert "home_cleaning" in detergent
    assert "electronics_appliances" not in detergent


def test_detect_product_themes_reduces_pet_false_positives():
    book = set(detect_product_themes("EL GATO CON BOTAS - PUSS IN BOOTS (VVAA) (SUSAETA)"))
    pet_food = set(detect_product_themes("POUCH YOGURT SABOR NATURAL PARA GATO YOW UP 3X85GR"))

    assert "books_toys" in book
    assert "pet" not in book
    assert "coffee_tea" not in book
    assert "pet" in pet_food


def test_detect_product_themes_reduces_department_word_false_positives():
    rain_boots = set(detect_product_themes("BOTA DE AGUA GOMA TEX"))
    computer_desk = set(detect_product_themes("MESA ORDENADOR 75X130X60 KALMAR COLOR BLANCO"))
    coffee = set(detect_product_themes("CAFE MOLIDO MEZCLA FORTALEZA 250 GRS"))
    vitamins = set(detect_product_themes("OMEGA 3 HEALTH 4U 30 CAPSULAS BLANDAS 42,5G"))

    assert "beverages_soft" not in rain_boots
    assert "apparel_textile" in rain_boots
    assert "electronics_appliances" not in computer_desk
    assert "home_kitchen" in computer_desk
    assert "coffee_tea" in coffee
    assert "health_wellness" in vitamins
    assert "coffee_tea" not in vitamins
