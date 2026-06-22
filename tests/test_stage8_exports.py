import json

import polars as pl

from src.config import PipelineConfig
from src.stage8 import STAGE8_SCHEMA_VERSION, stage8_artifact_paths, write_stage8_dashboard_pack


def _cfg(tmp_path):
    return PipelineConfig(
        values={
            "run": {"random_seed": 42},
            "paths": {
                "outputs": "outputs",
                "processed": "data/processed",
                "dev": "data/dev",
                "raw_parquet": "data/raw/parquet",
                "raw_csv": "data/raw/csv",
            },
            "data": {
                "prepared_transactions": "df_combined.parquet",
                "customer_kpis": "customer_kpis.parquet",
                "product_master": "maestra_articulos.parquet",
            },
            "word2vec": {"embeddings_output": "product_embeddings.parquet", "vector_size": 128},
            "customer_embeddings": {
                "output": "customer_embeddings.parquet",
                "weight_strategy": "quantity_idf",
                "quantity_transform": "log1p",
                "normalize_vectors": True,
            },
            "behavioral_features": {"output": "customer_behavior_features.parquet"},
            "official_model_suite": {
                "umap_hdbscan": {
                    "umap": {"output": "feature_set_umap_pca64_u20_n75.parquet", "n_components": 20},
                    "hdbscan": {"min_cluster_size": 50},
                },
                "two_stage_hdbscan": {"second_stage_hdbscan": {"min_cluster_size": 25}},
                "three_stage_hdbscan": {
                    "model_name": "model_e_three_stage_hdbscan_lift_core",
                    "variant_prefix": "test",
                    "third_stage_hdbscan": {"min_cluster_size": 10},
                    "lift_filter": {"min_strong_product_lifts": 2},
                },
            },
            "modeling": {"feature_set_for_selection": "embeddings_only"},
            "profiling": {
                "soft_audience_min_affinity": 0.70,
                "soft_audience_min_margin": 0.10,
                "soft_audience_medium_affinity": 0.58,
                "soft_audience_medium_margin": 0.04,
                "remaining_bridge_min_affinity": 0.58,
                "remaining_bridge_max_margin": 0.06,
                "remaining_near_tribe_min_affinity": 0.70,
                "remaining_near_tribe_min_margin": 0.10,
            },
        },
        mode="dev",
        root=tmp_path,
    )


def _write_fixture(cfg):
    stage6 = cfg.artifacts / "stage6"
    evidence = stage6 / "stage6_8_evidence"
    stage7 = cfg.artifacts / "stage7"
    handoff_support = stage7 / "final_handoff" / "supporting_tables"
    feature_dir = cfg.outputs / "features"
    model_dir = cfg.model_selection_cache
    for path in [evidence, stage7, handoff_support, feature_dir, model_dir]:
        path.mkdir(parents=True, exist_ok=True)

    (evidence / f"stage68_manifest_{cfg.mode}.json").write_text(
        json.dumps(
            {
                "row_counts": {"product_lifts": 2, "sector_lifts": 2},
                "outputs": {"tribe_evidence_path": "tribe_evidence.parquet"},
            }
        ),
        encoding="utf-8",
    )
    (stage7 / "final_handoff").mkdir(parents=True, exist_ok=True)
    (stage7 / "final_handoff" / f"stage7_final_manifest_{cfg.mode}.json").write_text(
        json.dumps({"promotion_policy": {"readiness_source": "stage6_profile_readiness"}}),
        encoding="utf-8",
    )

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "tribe_id": [0, 0, 1, -1, -1, -1],
            "assignment_probability": [0.95, 0.90, 0.80, None, None, None],
            "assignment_confidence_score": [0.95, 0.90, 0.80, None, None, None],
            "assignment_confidence_type": ["hard", "hard", "hard", "", "", ""],
            "assignment_source": ["hdbscan", "hdbscan", "hdbscan", "noise", "noise", "noise"],
        }
    ).write_parquet(model_dir / "cluster_assignments_model_e_three_stage_hdbscan_lift_core.parquet")
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "u0": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
            "u1": [0.0, 0.2, 1.0, 1.2, 2.0, 2.2],
            "u2": [0.0, 0.3, 1.0, 1.3, 2.0, 2.3],
        }
    ).write_parquet(feature_dir / "feature_set_umap_pca64_u20_n75.parquet")
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "pc0": [0.0, 0.2, 1.0, 1.2, 2.0, 2.2],
            "pc1": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
            "pc2": [0.0, 0.4, 1.0, 1.4, 2.0, 2.4],
        }
    ).write_parquet(feature_dir / "feature_set_pca_for_umap.parquet")
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "ticket_count": [8, 7, 4, 9, 1, 3],
            "total_spend": [100.0, 120.0, 60.0, 200.0, 5.0, 30.0],
            "avg_basket_value": [20.0, 22.0, 15.0, 40.0, 5.0, 10.0],
            "promo_share": [0.1, 0.2, 0.3, 0.1, 0.0, 0.4],
            "unique_products": [12, 14, 9, 30, 1, 4],
            "unique_sectors": [4, 5, 3, 8, 1, 2],
            "recency_days": [5, 6, 10, 8, 100, 30],
            "frequency_per_30d": [2.0, 2.2, 1.0, 3.0, 0.1, 0.8],
        }
    ).write_parquet(feature_dir / "customer_behavior_features.parquet")
    pl.DataFrame(
        {
            "cliente": ["c4", "c5", "c6"],
            "top_tribe_id": [0, 1, 0],
            "top_affinity_score": [0.8, 0.5, 0.6],
            "second_tribe_id": [1, 0, 1],
            "second_affinity_score": [0.6, 0.45, 0.58],
            "affinity_margin": [0.2, 0.05, 0.02],
            "affinity_confidence_band": ["high", "low", "medium"],
            "recommended_use": ["soft audience opportunity", "diagnostic only", "diagnostic only"],
        }
    ).write_parquet(evidence / f"stage6_7_remaining_customer_affinity_{cfg.mode}.parquet")

    pl.DataFrame(
        {
            "tribe_id": [0, 1],
            "tribe_name": ["Fresh Mission Buyers", "Review Snack Buyers"],
            "business_name": ["Fresh Mission Buyers", "Review Snack Buyers"],
            "technical_name": ["Fresh Mission Buyers", "Review Snack Buyers"],
            "promotion_decision": ["promoted", "review"],
            "tribe_status": ["final_strong", "potential_review"],
            "tribe_status_label": ["Final - strong", "Potential / Review"],
            "stage6_profile_readiness": ["strong", "review"],
            "validation_tier": ["promoted_strong", "review"],
            "business_confidence": ["high", "review_only"],
            "validation_blockers": ["pass", "stage6_profile_readiness=review"],
            "coverage_group": ["core_promoted_tribe", "review_tribe"],
            "membership_policy": ["official hard-assigned promoted tribe", "hard-assigned tribe held outside core set"],
            "recommended_use": ["Ready for campaign planning.", "Review only."],
            "customers": [2, 1],
            "population_share_pct": [33.3, 16.7],
            "mean_assignment_confidence": [0.925, 0.8],
            "p10_assignment_confidence": [0.9, 0.8],
            "readiness_caveat": ["Strong.", "Review."],
        }
    ).write_csv(stage7 / f"stage7_all_tribe_profiles_{cfg.mode}.csv")
    pl.DataFrame(
        {
            "segment_id": ["tribe_00", "tribe_01", "near_tribe_fringe_customers", "sparse_low_signal_shoppers"],
            "segment_name": ["Fresh Mission Buyers", "Review Snack Buyers", "Near-tribe fringe customers", "Sparse or low-signal shoppers"],
            "coverage_group": ["core_promoted_tribes", "review_tribes", "near_tribe_customers", "sparse_customers"],
            "tribe_id": [0, 1, None, None],
            "promotion_decision": ["promoted", "review", "not_applicable", "not_applicable"],
            "validation_tier": ["promoted_strong", "review", "remaining_customer_segment", "remaining_customer_segment"],
            "customers": [2, 1, 1, 2],
            "share_of_total_pct": [33.3, 16.7, 16.7, 33.3],
            "counts_toward_population_total": ["true", "true", "true", "true"],
            "membership_policy": ["official", "review", "descriptive", "descriptive"],
            "recommended_use": ["Campaign.", "Review.", "Test expansion.", "Onboard."],
        }
    ).write_csv(stage7 / f"stage7_4_customer_coverage_report_{cfg.mode}.csv")
    pl.DataFrame(
        {
            "segment_id": ["near_tribe_fringe_customers", "sparse_low_signal_shoppers"],
            "segment_name": ["Near-tribe fringe customers", "Sparse or low-signal shoppers"],
            "customer_count": [1, 2],
            "share_of_remaining_pct": [33.3, 66.7],
            "share_of_total_pct": [16.7, 33.3],
            "recommended_action": ["Test expansion.", "Onboard."],
            "targetability": ["high", "low"],
        }
    ).write_csv(stage7 / f"stage7_remaining_customer_analysis_{cfg.mode}.csv")
    pl.DataFrame(
        {
            "segment_id": ["tribe_00", "near_tribe_fringe_customers"],
            "segment_name": ["Fresh Mission Buyers", "Near-tribe fringe customers"],
            "coverage_group": ["core_promoted_tribe", "near_tribe_customers"],
            "tribe_id": [0, None],
            "promotion_decision": ["promoted", "not_applicable"],
            "validation_tier": ["promoted_strong", "remaining_customer_segment"],
            "business_confidence": ["high", "descriptive"],
            "recommended_use": ["Campaign.", "Test expansion."],
            "marketing_actions": ["Offer fresh bundle.", "Test closest tribe."],
            "merchandising_actions": ["Bundle products.", "Adjacent products."],
            "cross_sell_opportunities": ["Fresh add-ons.", "Nearest tribe mission."],
            "retention_opportunities": ["Grow baskets.", "Measured expansion."],
            "exclusions": ["Suppress controls.", "No hard membership."],
            "primary_kpi": ["Incremental revenue.", "Conversion."],
        }
    ).write_csv(stage7 / f"stage7_5_segment_action_playbook_{cfg.mode}.csv")
    pl.DataFrame(
        {
            "tribe_a_id": [0],
            "tribe_a_name": ["Fresh Mission Buyers"],
            "tribe_b_id": [1],
            "tribe_b_name": ["Review Snack Buyers"],
            "relationship_type": ["different_product_mission"],
            "relationship_score": [0.3],
            "campaign_guidance": ["Suppress overlap."],
        }
    ).write_csv(stage7 / f"stage7_all_tribe_relationship_atlas_{cfg.mode}.csv")
    pl.DataFrame(
        {
            "tribe_id": [0, 1],
            "rank": [1, 1],
            "product_id": ["p1", "p2"],
            "product_description": ["Fresh Salad", "Snack Bar"],
            "category": ["Fresh", "Grocery"],
            "lift_vs_rest": [2.5, 1.8],
            "lift_vs_population": [2.0, 1.6],
            "reach_pct": [50.0, 100.0],
            "q_value": [0.01, 0.02],
            "customers": [1, 1],
            "product_rank_score": [1.0, 1.0],
            "statistical_result": ["strong_significant_overindex", "strong_significant_overindex"],
        }
    ).write_csv(handoff_support / f"stage7_all_tribe_product_summary_long_{cfg.mode}.csv")
    pl.DataFrame(
        {
            "tribe_id": [0, 1],
            "desc_sector": ["Fresh", "Grocery"],
            "lift_vs_rest": [2.0, 1.5],
            "customers": [2, 1],
            "q_value": [0.01, 0.02],
        }
    ).write_parquet(evidence / f"sector_lifts_{cfg.mode}.parquet")

    for path in [
        stage6 / "stage6_1_pca_for_umap_summary.csv",
        stage6 / "stage6_1_umap_checks.csv",
        stage6 / "stage6_5_merged_three_stage_hdbscan_checks.csv",
        stage6 / "stage6_6_representation_cluster_quality.csv",
        stage6 / "stage6_6_cluster_stability_readiness_clusters_summary.csv",
        cfg.artifacts / "stage3" / "embedding_validation.csv",
        stage7 / f"stage7_stakeholder_readiness_{cfg.mode}.csv",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"metric": ["status"], "value": ["pass"], "status": ["pass"]}).write_csv(path)


def test_stage8_pack_writes_manifest_and_reconciles_customer_coverage(tmp_path):
    cfg = _cfg(tmp_path)
    _write_fixture(cfg)

    outputs = write_stage8_dashboard_pack(cfg=cfg, embedding_sample_size=20)
    paths = stage8_artifact_paths(cfg=cfg)
    manifest = json.loads(paths["dashboard_manifest"].read_text(encoding="utf-8"))
    readiness = pl.read_csv(paths["readiness_csv"])

    assert outputs["dashboard_manifest"] == paths["dashboard_manifest"]
    assert manifest["schema_version"] == STAGE8_SCHEMA_VERSION
    assert manifest["readiness_summary"]["status"] == "pass"
    assert readiness.filter(pl.col("status") == "fail").is_empty()
    assert paths["readme"].exists()
    assert pl.read_parquet(paths["artifact_manifest"]).height > 0
    assert paths["relational_manifest"].exists()
    assert paths["relational_schema"].exists()
    assert pl.read_parquet(paths["relational_integrity_report"]).filter(pl.col("status") == "fail").is_empty()
    assert pl.read_parquet(paths["rel_dim_tribe"]).height == 2
    assert pl.read_parquet(paths["rel_fact_customer_assignment"]).height == 6
    assert pl.read_parquet(paths["rel_fact_embedding_3d"]).height == 6
    assert pl.read_parquet(paths["rel_fact_embedding_3d_sample"]).height == 6
    assert pl.read_parquet(paths["rel_fact_tribe_metrics"]).height == 2
    assert pl.read_parquet(paths["rel_fact_coverage_group_metrics"]).height > 0
    assert pl.read_parquet(paths["rel_dim_dashboard_page"]).height == 9
    assert pl.read_parquet(paths["rel_dim_visualization"]).height > 0
    assert pl.read_parquet(paths["rel_mart_dashboard_story_page"]).height == 10
    assert pl.read_parquet(paths["rel_mart_executive_kpi"]).height >= 6
    assert pl.read_parquet(paths["rel_mart_tribe_scorecard"]).height == 2
    assert pl.read_parquet(paths["rel_mart_tribe_product_evidence"]).height == 2
    assert pl.read_parquet(paths["rel_mart_customer_landscape_sample"]).filter(pl.col("visualization_type") == "pca_3d").height == 6
    assert pl.read_parquet(paths["rel_mart_validation_report"]).filter((pl.col("severity") == "critical") & (pl.col("status") == "fail")).is_empty()
    rel_product_fact = pl.read_parquet(paths["rel_fact_tribe_product_affinity"])
    assert "tribe_name" not in rel_product_fact.columns
    assert "product_description" not in rel_product_fact.columns
    assert not paths["tribe_deep_dive"].exists()
    assert not paths["customer_coverage"].exists()
    assert not paths["embedding_3d_sample"].exists()


def test_stage8_manifest_has_no_markdown_inputs_and_expected_dashboard_products(tmp_path):
    cfg = _cfg(tmp_path)
    _write_fixture(cfg)

    write_stage8_dashboard_pack(cfg=cfg, embedding_sample_size=20)
    manifest = json.loads(stage8_artifact_paths(cfg=cfg)["dashboard_manifest"].read_text(encoding="utf-8"))
    product_names = {item["name"] for item in manifest["created_products"]}
    input_paths = manifest["input_paths"].values()

    assert "readme" in product_names
    assert "artifact_manifest" in product_names
    assert "completeness_audit" in product_names
    assert "relational_manifest" in product_names
    assert "relational_integrity_report" in product_names
    assert "relational_schema" in product_names
    assert "rel_dim_tribe" in product_names
    assert "rel_dim_product" in product_names
    assert "rel_dim_dashboard_page" in product_names
    assert "rel_dim_visualization" in product_names
    assert "rel_fact_executive_metric" in product_names
    assert "rel_fact_customer_assignment" in product_names
    assert "rel_fact_tribe_product_affinity" in product_names
    assert "rel_mart_dashboard_story_page" in product_names
    assert "rel_mart_executive_kpi" in product_names
    assert "rel_mart_tribe_scorecard" in product_names
    assert "rel_mart_tribe_product_evidence" in product_names
    assert "rel_mart_customer_landscape_sample" in product_names
    assert "rel_mart_data_dictionary" in product_names
    assert "rel_mart_validation_report" in product_names
    assert "tribe_deep_dive" not in product_names
    assert "customer_coverage" not in product_names
    assert "embedding_3d_sample" not in product_names
    assert all(not str(path).endswith(".md") for path in input_paths)
