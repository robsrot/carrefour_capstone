import polars as pl

from src.config import PipelineConfig
from src.model_selection import (
    build_stage6_hdbscan_diagnostics,
    build_stage6_representation_cluster_diagnostics,
    build_stage6_umap_diagnostics,
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
            "official_model_suite": {
                "include_umap_hdbscan": True,
                "umap_hdbscan": {
                    "trial_name": "tiny_core",
                    "model_name": "model_b_umap_hdbscan_core",
                    "output_prefix": "model_b_umap_hdbscan_core",
                    "algorithm_name": "UMAP_HDBSCAN",
                    "feature_space": "tiny_umap",
                    "variant_prefix": "tiny",
                    "umap": {"n_components": 2, "n_neighbors": 2, "min_dist": 0.0, "metric": "cosine"},
                    "hdbscan": {
                        "min_cluster_size": 2,
                        "min_samples": 1,
                        "cluster_selection_method": "eom",
                        "allow_noise_assignment": False,
                    },
                },
            },
            "umap": {"enabled": True, "n_components": 2, "n_neighbors": 2, "min_dist": 0.0, "metric": "cosine"},
            "hdbscan": {"min_cluster_size": 2, "min_samples": 1, "cluster_selection_method": "eom"},
            "quality_gates": {"min_clusters": 2},
            "stage6_quality": {"sample_size": 3, "neighbor_k": 1, "distance_sample_size": 3, "dbcv_sample_size": 3},
        },
        mode="dev",
        root=tmp_path,
    )


def _write_feature_and_umap(tmp_path):
    feature_path = tmp_path / "features.parquet"
    umap_path = tmp_path / "umap.parquet"
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3"],
            "emb_000": [1.0, 0.0, -1.0],
            "emb_001": [0.0, 1.0, -1.0],
        }
    ).write_parquet(feature_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3"],
            "umap_000": [0.1, 1.1, -0.8],
            "umap_001": [0.2, 1.0, -0.7],
        }
    ).write_parquet(umap_path)
    return feature_path, umap_path


def test_stage6_umap_and_hdbscan_diagnostics_pass_for_hard_core_assignments(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.ensure_directories()
    feature_path, umap_path = _write_feature_and_umap(tmp_path)
    assignment_path = tmp_path / "assignments.parquet"
    result_path = tmp_path / "results.parquet"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3"],
            "tribe_id": [0, 1, -1],
            "assignment_source": ["hdbscan_fit", "hdbscan_fit", "hdbscan_noise_unassigned"],
        }
    ).write_parquet(assignment_path)
    pl.DataFrame(
        [
            {
                "cluster_count": 2,
                "noise_pct": 33.3333,
                "core_coverage_pct": 66.6667,
                "assignment_policy": "hard_hdbscan_core_noise_retained",
                "soft_assignment_enabled": False,
                "soft_assigned_pct": 0.0,
                "passes_quality_gate": True,
                "quality_gate_reason": "pass",
            }
        ]
    ).write_parquet(result_path)

    umap_checks = pl.read_csv(build_stage6_umap_diagnostics(feature_path, umap_path, cfg=cfg)).row(0, named=True)
    hdbscan_checks = pl.read_csv(
        build_stage6_hdbscan_diagnostics(umap_path, assignment_path, result_path, cfg=cfg)
    ).row(0, named=True)

    assert umap_checks["check_status"] == "pass"
    assert umap_checks["umap_component_count"] == 2
    assert hdbscan_checks["check_status"] == "pass"
    assert hdbscan_checks["blocking_check_status"] == "pass"
    assert hdbscan_checks["quality_gate_status"] == "pass"
    assert hdbscan_checks["assignment_policy"] == "hard_hdbscan_core_noise_retained"
    assert hdbscan_checks["soft_assigned_pct"] == 0.0


def test_stage6_umap_diagnostics_include_pca_summary_when_available(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.ensure_directories()
    feature_path, umap_path = _write_feature_and_umap(tmp_path)
    pca_summary_path = cfg.artifacts / "stage6" / "stage6_1_pca_for_umap_summary.csv"
    pca_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {
                "stage": "pca_for_umap",
                "purpose": "Pre-reduce standardized customer product-behavior features before UMAP.",
                "feature_path": str(feature_path),
                "pca_path": str(tmp_path / "pca.parquet"),
                "input_dimension_count": 128,
                "requested_component_count": 40,
                "retained_component_count": 40,
                "dimension_reduction": "128 -> 40",
                "standardized_input": True,
                "retained_variance_ratio": 0.84,
                "retained_variance_pct": 84.0,
                "pc1_variance_pct": 12.0,
                "pc5_cumulative_variance_pct": 42.0,
                "pc10_cumulative_variance_pct": 61.0,
            }
        ]
    ).write_csv(pca_summary_path)

    checks = pl.read_csv(build_stage6_umap_diagnostics(feature_path, umap_path, cfg=cfg)).row(0, named=True)

    assert checks["pca_pre_reduction_summary_status"] == "present"
    assert checks["pca_dimension_reduction"] == "128 -> 40"
    assert checks["pca_requested_component_count"] == 40
    assert checks["pca_retained_component_count"] == 40
    assert checks["pca_retained_variance_pct"] == 84.0


def test_stage6_hdbscan_diagnostics_fail_when_soft_assignment_is_present(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.ensure_directories()
    _, umap_path = _write_feature_and_umap(tmp_path)
    assignment_path = tmp_path / "assignments_soft.parquet"
    result_path = tmp_path / "results_soft.parquet"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3"],
            "tribe_id": [0, 1, 0],
            "assignment_source": ["hdbscan_fit", "hdbscan_fit", "nearest_centroid_soft_noise_q95"],
        }
    ).write_parquet(assignment_path)
    pl.DataFrame(
        [
            {
                "cluster_count": 2,
                "noise_pct": 0.0,
                "core_coverage_pct": 66.6667,
                "assignment_policy": "hdbscan_core_plus_soft_noise_assignment",
                "soft_assignment_enabled": True,
                "soft_assigned_pct": 33.3333,
                "passes_quality_gate": True,
                "quality_gate_reason": "pass",
            }
        ]
    ).write_parquet(result_path)

    checks = pl.read_csv(build_stage6_hdbscan_diagnostics(umap_path, assignment_path, result_path, cfg=cfg)).row(
        0,
        named=True,
    )

    assert checks["check_status"] == "fail"
    assert checks["blocking_check_status"] == "fail"
    assert "soft_assignment_enabled" in checks["check_issues"]
    assert "soft_assigned_customers_present" in checks["check_issues"]


def test_stage6_hdbscan_diagnostics_pass_for_two_stage_hard_policy(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.ensure_directories()
    _, umap_path = _write_feature_and_umap(tmp_path)
    assignment_path = tmp_path / "assignments_two_stage.parquet"
    result_path = tmp_path / "results_two_stage.parquet"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3"],
            "tribe_id": [0, 1, -1],
            "assignment_source": [
                "two_stage_hdbscan_stage1_core",
                "two_stage_hdbscan_stage2_noise_core",
                "two_stage_hdbscan_noise_unassigned",
            ],
        }
    ).write_parquet(assignment_path)
    pl.DataFrame(
        [
            {
                "cluster_count": 2,
                "noise_pct": 33.3333,
                "core_coverage_pct": 66.6667,
                "assignment_policy": "hard_two_stage_hdbscan_lift_core_noise_retained",
                "soft_assignment_enabled": False,
                "soft_assigned_pct": 0.0,
                "passes_quality_gate": True,
                "quality_gate_reason": "pass",
            }
        ]
    ).write_parquet(result_path)

    checks = pl.read_csv(build_stage6_hdbscan_diagnostics(umap_path, assignment_path, result_path, cfg=cfg)).row(
        0,
        named=True,
    )

    assert checks["check_status"] == "pass"
    assert checks["blocking_check_status"] == "pass"
    assert checks["assignment_policy"] == "hard_two_stage_hdbscan_lift_core_noise_retained"


def test_stage6_hdbscan_diagnostics_quality_failure_does_not_block_evidence(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.ensure_directories()
    _, umap_path = _write_feature_and_umap(tmp_path)
    assignment_path = tmp_path / "assignments_high_noise.parquet"
    result_path = tmp_path / "results_high_noise.parquet"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3"],
            "tribe_id": [0, 1, -1],
            "assignment_source": ["hdbscan_fit", "hdbscan_fit", "hdbscan_noise_unassigned"],
        }
    ).write_parquet(assignment_path)
    pl.DataFrame(
        [
            {
                "cluster_count": 2,
                "noise_pct": 66.6667,
                "core_coverage_pct": 33.3333,
                "assignment_policy": "hard_hdbscan_core_noise_retained",
                "soft_assignment_enabled": False,
                "soft_assigned_pct": 0.0,
                "passes_quality_gate": False,
                "quality_gate_reason": "noise_pct>60.0",
            }
        ]
    ).write_parquet(result_path)

    checks = pl.read_csv(build_stage6_hdbscan_diagnostics(umap_path, assignment_path, result_path, cfg=cfg)).row(
        0,
        named=True,
    )

    assert checks["check_status"] == "fail"
    assert checks["blocking_check_status"] == "pass"
    assert checks["quality_gate_status"] == "fail"
    assert checks["quality_gate_issues"] == "noise_pct>60.0"


def test_stage6_representation_cluster_diagnostics_writes_umap_and_hdbscan_evidence(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.ensure_directories()
    feature_path, umap_path = _write_feature_and_umap(tmp_path)
    assignment_path = tmp_path / "assignments.parquet"
    result_path = tmp_path / "results.parquet"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3"],
            "tribe_id": [0, 1, -1],
            "assignment_source": ["hdbscan_fit", "hdbscan_fit", "hdbscan_noise_unassigned"],
            "assignment_confidence_score": [0.9, 0.8, None],
        }
    ).write_parquet(assignment_path)
    pl.DataFrame(
        [
            {
                "cluster_count": 2,
                "noise_pct": 33.3333,
                "core_coverage_pct": 66.6667,
                "silhouette": 0.25,
                "coverage_adjusted_silhouette": 0.1667,
                "davies_bouldin": 0.9,
                "calinski_harabasz": 3.0,
                "avg_assignment_confidence": 0.85,
                "passes_quality_gate": True,
                "quality_gate_reason": "pass",
            }
        ]
    ).write_parquet(result_path)

    path = build_stage6_representation_cluster_diagnostics(
        feature_path,
        umap_path,
        assignment_path,
        result_path,
        cfg=cfg,
    )
    row = pl.read_csv(path).row(0, named=True)

    assert row["stage"] == "6.5_representation_cluster_quality"
    assert row["aligned_rows"] == 3
    assert row["umap_component_count"] == 2
    assert row["umap_neighbor_k"] == 1
    assert row["cluster_count"] == 2
    assert row["silhouette_core_only"] == 0.25
    assert row["hdbscan_dbcv_status"] in {"not_run", "unavailable", "failed", "computed"}
