import polars as pl

from src.config import PipelineConfig
from src.feature_engineering import build_feature_set_diagnostics


def _test_config(tmp_path, *, selection_feature_set="embeddings_only"):
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
            "feature_sets": {
                "default": "embeddings_only",
                "diagnostics": {
                    "output_dir": "stage5",
                    "output_csv": "feature_set_diagnostics.csv",
                    "neighbor_overlap_k": 1,
                    "neighbor_overlap_sample_size": 4,
                },
            },
            "modeling": {"feature_set_for_selection": selection_feature_set},
        },
        mode="dev",
        root=tmp_path,
    )


def _write_feature_sets(tmp_path):
    baseline_path = tmp_path / "embeddings_only.parquet"
    behavior_path = tmp_path / "embeddings_behavior.parquet"
    shifted_path = tmp_path / "shifted.parquet"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4"],
            "emb_000": [1.0, 0.9, -1.0, -0.9],
            "emb_001": [0.0, 0.1, 0.0, -0.1],
        }
    ).write_parquet(baseline_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4"],
            "emb_000": [1.0, 0.9, -1.0, -0.9],
            "emb_001": [0.0, 0.1, 0.0, -0.1],
            "beh_frequency": [10.0, 10.0, -10.0, -10.0],
            "beh_constant": [1.0, 1.0, 1.0, 1.0],
        }
    ).write_parquet(behavior_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3"],
            "emb_000": [1.0, None, -1.0],
            "emb_001": [0.0, 0.1, 0.0],
        }
    ).write_parquet(shifted_path)
    return {
        "embeddings_only": baseline_path,
        "embeddings_behavior": behavior_path,
        "shifted": shifted_path,
    }


def test_build_feature_set_diagnostics_writes_compact_health_and_topology_rows(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.ensure_directories()
    feature_paths = _write_feature_sets(tmp_path)

    output = build_feature_set_diagnostics(feature_paths, cfg=cfg)
    diagnostics = pl.read_csv(output).sort("feature_set_name")

    assert output == cfg.artifacts / "stage5" / "feature_set_diagnostics.csv"
    assert diagnostics.height == 3
    assert {
        "feature_set_name",
        "rows",
        "feature_count",
        "finite_pct",
        "null_pct",
        "zero_variance_feature_count",
        "customer_alignment_status",
        "neighbor_overlap_vs_baseline_pct",
    }.issubset(diagnostics.columns)

    baseline = diagnostics.filter(pl.col("feature_set_name") == "embeddings_only").row(0, named=True)
    behavior = diagnostics.filter(pl.col("feature_set_name") == "embeddings_behavior").row(0, named=True)
    shifted = diagnostics.filter(pl.col("feature_set_name") == "shifted").row(0, named=True)

    assert baseline["neighbor_overlap_vs_baseline_pct"] == 100.0
    assert behavior["zero_variance_feature_count"] == 1
    assert "beh_constant" in behavior["zero_variance_features"]
    assert shifted["customer_alignment_status"] == "warn"
    assert shifted["missing_vs_baseline_customers"] == 1
    assert shifted["null_pct"] > 0
    assert shifted["finite_pct"] < 100


def test_feature_set_diagnostics_warns_when_behavior_is_official_selection(tmp_path):
    cfg = _test_config(tmp_path, selection_feature_set="embeddings_behavior")
    cfg.ensure_directories()
    feature_paths = _write_feature_sets(tmp_path)

    output = build_feature_set_diagnostics(feature_paths, cfg=cfg)
    diagnostics = pl.read_csv(output)

    selected = diagnostics.filter(pl.col("feature_set_name") == "embeddings_behavior").row(0, named=True)
    assert selected["is_selection_feature_set"] is True
    assert "WARNING" in selected["selection_warning"]
