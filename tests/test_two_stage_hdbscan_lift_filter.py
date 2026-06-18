import polars as pl

from src.config import PipelineConfig
from src.model_selection import (
    _apply_lift_filter_to_assignments,
    _product_lift_filter_evidence,
    run_official_two_stage_hdbscan_lift_core,
    two_stage_hdbscan_artifact_paths,
)


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
            "cache": {"force": False, "use_cached": True},
            "data": {"prepared_transactions": "transactions.parquet"},
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


def test_two_stage_cache_backfills_lift_csv_and_rewrites_current_paths(tmp_path):
    cfg = _test_config(tmp_path)
    paths = two_stage_hdbscan_artifact_paths(cfg)
    for path in [
        paths["assignment_path"],
        paths["results_path"],
        paths["lift_evidence_path"],
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"cliente": ["c1"], "tribe_id": [0]}).write_parquet(paths["assignment_path"])
    pl.DataFrame(
        [
            {
                "model_name": "model_d_two_stage_hdbscan_lift_core",
                "model_variant": "cached_variant",
                "assignment_path": "/home/capstone06/carrefour_capstone/outputs/prod/models/model_selection/stale.parquet",
                "stage1_assignment_path": "/home/capstone06/carrefour_capstone/outputs/prod/models/model_selection/stale_stage1.parquet",
                "stage1_result_path": "/home/capstone06/carrefour_capstone/outputs/prod/models/model_selection/stale_stage1_results.parquet",
                "stage2_noise_feature_path": "/home/capstone06/carrefour_capstone/outputs/prod/models/model_selection/stale_noise.parquet",
                "stage2_assignment_path": None,
                "stage2_result_path": None,
                "unfiltered_assignment_path": "/home/capstone06/carrefour_capstone/outputs/prod/models/model_selection/stale_unfiltered.parquet",
                "unfiltered_profile_path": "/home/capstone06/carrefour_capstone/outputs/prod/profiles/stale.parquet",
                "lift_filter_evidence_path": "/home/capstone06/carrefour_capstone/outputs/prod/artifacts/stage6/stale.parquet",
                "lift_filter_evidence_csv_path": "/home/capstone06/carrefour_capstone/outputs/prod/artifacts/stage6/stale.csv",
            }
        ]
    ).write_parquet(paths["results_path"])
    pl.DataFrame(
        {
            "tribe_id": [0],
            "n_customers": [10],
            "strong_product_lift_threshold": [1.5],
            "min_strong_product_lifts": [1],
            "require_significant_product_lift": [True],
            "strong_product_lift_count": [2],
            "significant_strong_product_lift_count": [2],
            "passes_lift_filter": [True],
        }
    ).write_parquet(paths["lift_evidence_path"])

    assignment_path, result_path, result = run_official_two_stage_hdbscan_lift_core(
        tmp_path / "umap.parquet",
        cfg=cfg,
    )

    cached_result = pl.read_parquet(result_path).row(0, named=True)
    assert assignment_path == paths["assignment_path"]
    assert paths["lift_evidence_csv_path"].exists()
    assert result["lift_filter_evidence_csv_path"] == str(paths["lift_evidence_csv_path"])
    assert cached_result["lift_filter_evidence_csv_path"] == str(paths["lift_evidence_csv_path"])
    assert "/home/capstone06" not in cached_result["assignment_path"]
