import polars as pl

from src.cluster_validation import build_cluster_promotion_audit, build_cluster_validity_stability_report
from src.config import PipelineConfig


def _test_config(tmp_path, *, use_cached=False):
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
            "cache": {"force": False, "use_cached": use_cached},
            "stability": {"sample_size": 4, "repeats": 1, "jitter_scale": 0.0},
            "model_selection": {"write_summary_markdown": False},
            "official_model_suite": {
                "umap_hdbscan": {
                    "model_name": "model_b_umap_hdbscan_core",
                    "hdbscan": {"min_cluster_size": 2},
                },
            },
            "hdbscan": {"min_cluster_size": 5},
        },
        mode="dev",
        root=tmp_path,
    )


def test_cluster_readiness_uses_promoted_candidate_min_cluster_size(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.ensure_directories()
    feature_path = tmp_path / "features.parquet"
    assignment_path = tmp_path / "assignments.parquet"
    output_path = tmp_path / "stage6_4_cluster_stability_readiness.parquet"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4"],
            "umap_000": [0.0, 0.1, 5.0, 5.1],
            "umap_001": [0.0, 0.1, 5.0, 5.1],
        }
    ).write_parquet(feature_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4"],
            "tribe_id": [0, 0, 1, 1],
            "model_name": ["model_b_umap_hdbscan_core"] * 4,
            "model_variant": ["u8_n75_leaf_mcs2_ms1_hdbscan_mcs2_ms1_leaf"] * 4,
            "assignment_confidence_score": [0.9, 0.8, 0.9, 0.8],
            "assignment_source": ["hdbscan_fit"] * 4,
        }
    ).write_parquet(assignment_path)

    candidate_key = "model_b_umap_hdbscan_core::u8_n75_leaf_mcs2_ms1_hdbscan_mcs2_ms1_leaf"
    build_cluster_validity_stability_report(
        {
            "candidate_results": [
                {
                    "model_name": "model_b_umap_hdbscan_core",
                    "model_variant": "u8_n75_leaf_mcs2_ms1_hdbscan_mcs2_ms1_leaf",
                }
            ],
            "assignment_paths": {candidate_key: assignment_path},
            "official_candidate_key": candidate_key,
        },
        feature_path,
        output_path=output_path,
        cfg=cfg,
    )

    clusters = pl.read_parquet(output_path.with_name(f"{output_path.stem}_clusters.parquet"))

    assert clusters["min_cluster_size_reference"].to_list() == [2, 2]
    assert not any("customers<5" in issue for issue in clusters["readiness_issues"].to_list())


def test_cluster_stability_report_reuses_cached_artifacts(tmp_path):
    cfg = _test_config(tmp_path, use_cached=True)
    cfg.ensure_directories()
    feature_path = tmp_path / "features.parquet"
    assignment_path = tmp_path / "assignments.parquet"
    output_path = tmp_path / "stage6_6_cluster_stability_readiness.parquet"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4"],
            "umap_000": [0.0, 0.1, 5.0, 5.1],
            "umap_001": [0.0, 0.1, 5.0, 5.1],
        }
    ).write_parquet(feature_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4"],
            "tribe_id": [0, 0, 1, 1],
            "model_name": ["model_b_umap_hdbscan_core"] * 4,
            "model_variant": ["u8_n75_leaf_mcs2_ms1_hdbscan_mcs2_ms1_leaf"] * 4,
            "assignment_confidence_score": [0.9, 0.8, 0.9, 0.8],
            "assignment_source": ["hdbscan_fit"] * 4,
        }
    ).write_parquet(assignment_path)

    candidate_key = "model_b_umap_hdbscan_core::u8_n75_leaf_mcs2_ms1_hdbscan_mcs2_ms1_leaf"
    model_suite = {
        "candidate_results": [
            {
                "model_name": "model_b_umap_hdbscan_core",
                "model_variant": "u8_n75_leaf_mcs2_ms1_hdbscan_mcs2_ms1_leaf",
            }
        ],
        "assignment_paths": {candidate_key: assignment_path},
        "official_candidate_key": candidate_key,
    }

    first = build_cluster_validity_stability_report(
        model_suite,
        feature_path,
        output_path=output_path,
        force=True,
        cfg=cfg,
    )
    feature_path.unlink()
    second = build_cluster_validity_stability_report(
        model_suite,
        feature_path,
        output_path=output_path,
        cfg=cfg,
    )

    assert second == first
    assert first["parquet"].exists()
    assert first["summary_csv"].exists()
    assert first["cluster_parquet"].exists()
    assert first["cluster_summary_csv"].exists()


def test_cluster_promotion_audit_explains_review_and_lift_rejected_clusters(tmp_path):
    cfg = PipelineConfig(
        values={
            "paths": {"outputs": "outputs"},
            "profiling": {"final_handoff_readiness_statuses": ["strong", "usable"]},
        },
        mode="dev",
        root=tmp_path,
    )
    cfg.ensure_directories()
    assignments_path = tmp_path / "unfiltered.parquet"
    lift_path = tmp_path / "lift.parquet"
    readiness_path = tmp_path / "readiness.parquet"
    output_csv = tmp_path / "audit.csv"
    output_md = tmp_path / "audit.md"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
            "tribe_id": [0, 0, 1, 1, 2, 2, -1],
            "assignment_confidence_score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, None],
            "assignment_source": [
                "three_stage_hdbscan_stage1_core",
                "three_stage_hdbscan_stage1_core",
                "three_stage_hdbscan_stage2_noise_core",
                "three_stage_hdbscan_stage2_noise_core",
                "three_stage_hdbscan_stage3_remaining_noise_core",
                "three_stage_hdbscan_stage3_remaining_noise_core",
                "three_stage_hdbscan_noise_unassigned",
            ],
        }
    ).write_parquet(assignments_path)
    pl.DataFrame(
        {
            "tribe_id": [0, 1, 2],
            "strong_product_lift_count": [3, 3, 0],
            "significant_strong_product_lift_count": [3, 3, 0],
            "passes_lift_filter": [True, True, False],
        }
    ).write_parquet(lift_path)
    pl.DataFrame(
        {
            "tribe_id": [0, 1],
            "profile_readiness": ["strong", "review"],
            "readiness_issues": ["pass", "jitter_recovery<0.60"],
            "jitter_label_recovery_accuracy_mean": [0.92, 0.51],
        }
    ).write_parquet(readiness_path)

    outputs = build_cluster_promotion_audit(
        assignments_path,
        lift_path,
        readiness_path,
        output_csv=output_csv,
        output_md=output_md,
        cfg=cfg,
    )
    audit = pl.read_csv(outputs["csv"])
    by_candidate = {row["candidate_tribe_id"]: row for row in audit.iter_rows(named=True)}

    assert by_candidate[0]["promotion_status"] == "promoted_to_stage7"
    assert by_candidate[1]["promotion_status"] == "review_not_promoted"
    assert by_candidate[1]["blocker_reason"] == "jitter_recovery<0.60"
    assert by_candidate[2]["promotion_status"] == "rejected_lift_filter"
    assert "insufficient" in by_candidate[2]["blocker_reason"]
    assert "Review-only clusters" in outputs["markdown"].read_text(encoding="utf-8")
