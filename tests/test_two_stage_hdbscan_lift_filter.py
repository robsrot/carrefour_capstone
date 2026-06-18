import polars as pl

from src.config import PipelineConfig
from src.model_selection import _apply_lift_filter_to_assignments, _product_lift_filter_evidence


def _test_config(tmp_path):
    return PipelineConfig(
        values={
            "run": {"random_seed": 42},
            "paths": {
                "dev": "data/dev",
                "processed": "data/processed",
                "raw_csv": "data/raw/csv",
                "raw_parquet": "data/raw/parquet",
                "outputs": "outputs",
            },
            "profiling": {
                "strong_product_lift_threshold": 1.5,
                "significance_q_threshold": 0.05,
            },
            "quality_gates": {"min_strong_product_lifts_per_cluster": 1},
        },
        mode="dev",
        root=tmp_path,
    )


def test_lift_filter_keeps_stage1_and_stage2_clusters_with_product_lift(tmp_path):
    cfg = _test_config(tmp_path)
    profiles = pl.DataFrame(
        {
            "tribe_id": [0, 1, 2],
            "n_customers": [100, 80, 60],
            "top_product_ids": [["p1"], ["p2"], ["p3"]],
            "top_products": [["A"], ["B"], ["C"]],
            "top_product_lifts": [[2.0], [1.2], [1.8]],
            "top_product_q_values": [[0.01], [0.20], [0.30]],
        }
    )
    assignments = pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4"],
            "tribe_id": [0, 1, 2, -1],
            "model_name": ["m"] * 4,
            "model_variant": ["v"] * 4,
            "assignment_probability": [0.9, 0.8, 0.7, None],
            "assignment_confidence_score": [0.9, 0.8, 0.7, None],
            "assignment_confidence_type": ["hdbscan_membership_strength"] * 3 + [None],
            "assignment_source": [
                "two_stage_hdbscan_stage1_core",
                "two_stage_hdbscan_stage1_core",
                "two_stage_hdbscan_stage2_noise_core",
                "two_stage_hdbscan_noise_unassigned",
            ],
        }
    )

    evidence = _product_lift_filter_evidence(
        profiles,
        {"min_strong_product_lifts": 1, "require_significant_product_lift": False},
        cfg=cfg,
    )
    filtered, stats = _apply_lift_filter_to_assignments(assignments, evidence)

    assert evidence.select("tribe_id", "passes_lift_filter").to_dicts() == [
        {"tribe_id": 0, "passes_lift_filter": True},
        {"tribe_id": 1, "passes_lift_filter": False},
        {"tribe_id": 2, "passes_lift_filter": True},
    ]
    assert filtered.sort("cliente")["tribe_id"].to_list() == [0, -1, 1, -1]
    assert filtered.filter(pl.col("cliente") == "c2")[0, "assignment_source"] == "two_stage_hdbscan_lift_rejected"
    assert stats == {"kept_cluster_count": 2, "rejected_cluster_count": 1, "rejected_customer_count": 1}


def test_lift_filter_can_require_significant_product_lift(tmp_path):
    cfg = _test_config(tmp_path)
    profiles = pl.DataFrame(
        {
            "tribe_id": [0, 1],
            "n_customers": [100, 80],
            "top_product_ids": [["p1"], ["p2"]],
            "top_products": [["A"], ["B"]],
            "top_product_lifts": [[2.0], [2.0]],
            "top_product_q_values": [[0.20], [0.01]],
        }
    )

    evidence = _product_lift_filter_evidence(
        profiles,
        {"min_strong_product_lifts": 1, "require_significant_product_lift": True},
        cfg=cfg,
    )

    assert evidence.select("tribe_id", "passes_lift_filter").to_dicts() == [
        {"tribe_id": 0, "passes_lift_filter": False},
        {"tribe_id": 1, "passes_lift_filter": True},
    ]
