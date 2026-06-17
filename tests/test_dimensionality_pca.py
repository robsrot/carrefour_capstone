import polars as pl

from src.config import PipelineConfig
from src.dimensionality import build_pca_representation


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
            "pca": {"n_components": 2},
        },
        mode="dev",
        root=tmp_path,
    )


def test_build_pca_representation_writes_dimensionality_summary(tmp_path):
    cfg = _test_config(tmp_path)
    feature_path = tmp_path / "features.parquet"
    output_path = tmp_path / "feature_set_pca_for_umap.parquet"
    summary_path = tmp_path / "stage6_1_pca_for_umap_summary.csv"
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4"],
            "emb_000": [1.0, 0.0, -1.0, 0.5],
            "emb_001": [0.0, 1.0, -1.0, 0.25],
            "emb_002": [0.2, -0.5, 0.7, 1.0],
        }
    ).write_parquet(feature_path)

    result = build_pca_representation(
        feature_path,
        output_path=output_path,
        summary_path=summary_path,
        n_components=2,
        force=True,
        cfg=cfg,
    )

    summary = pl.read_csv(summary_path).row(0, named=True)
    assert result == output_path
    assert output_path.exists()
    assert summary["purpose"].startswith("Pre-reduce standardized customer product-behavior features")
    assert summary["input_dimension_count"] == 3
    assert summary["requested_component_count"] == 2
    assert summary["retained_component_count"] == 2
    assert summary["dimension_reduction"] == "3 -> 2"
    assert 0.0 < summary["retained_variance_pct"] <= 100.0
