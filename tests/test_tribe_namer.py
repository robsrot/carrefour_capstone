from src.tribe_namer import fallback_tribe_name, tribe_name_evidence


def test_data_driven_terms_take_priority_over_theme_combo_label():
    row = {
        "tribe_id": 5,
        "n_customers": 1000,
        "top_themes": ["plant_based", "organic_bio"],
        "top_theme_lifts": [1.6, 1.5],
        "top_theme_customer_counts": [400, 350],
        "top_product_terms": ["bebida soja", "tofu"],
        "top_product_term_lifts": [2.4, 2.1],
        "top_product_term_customer_counts": [220, 180],
        "top_product_term_q_values": [0.001, 0.003],
    }

    evidence = tribe_name_evidence(row)

    assert evidence["working_tribe_name"] == "Bebida Soja + Tofu Purchase Cluster"
    assert evidence["working_tribe_name_source"] == "data_driven_product_terms"
    assert fallback_tribe_name(row) == evidence["working_tribe_name"]
    assert "Plant-Based & Bio Buyers" not in evidence["working_tribe_name"]


def test_strategic_themes_are_not_used_as_primary_core_names_without_data_terms():
    row = {
        "tribe_id": 8,
        "n_customers": 1000,
        "top_themes": ["halal", "plant_based", "organic_bio"],
        "top_theme_lifts": [4.0, 3.0, 2.5],
        "top_theme_customer_counts": [120, 300, 280],
        "top_theme_q_values": [0.001, 0.001, 0.001],
        "top_product_terms": [],
        "top_product_term_lifts": [],
        "top_product_term_customer_counts": [],
    }

    evidence = tribe_name_evidence(row)

    assert evidence["working_tribe_name"] == "Tribe 8 Product Evidence Cluster"
    assert evidence["working_tribe_name_source"] == "insufficient_distinctive_evidence"


def test_raw_lifted_sector_can_be_used_without_manual_sector_mapping():
    row = {
        "tribe_id": 2,
        "n_customers": 1000,
        "top_sectors": ["P.G.C."],
        "top_sector_lifts": [1.4],
        "top_product_terms": [],
        "top_product_term_lifts": [],
        "top_product_term_customer_counts": [],
        "top_products": [],
        "top_product_lifts": [],
        "top_product_customer_counts": [],
    }

    evidence = tribe_name_evidence(row)

    assert evidence["working_tribe_name"] == "P.G.C. Purchase Cluster"
    assert evidence["working_tribe_name_source"] == "lifted_sector"
