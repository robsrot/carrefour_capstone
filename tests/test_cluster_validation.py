import polars as pl

from src.cluster_validation import build_cluster_validity_stability_report
from src.config import PipelineConfig


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
            "cache": {"use_cached": False},
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
