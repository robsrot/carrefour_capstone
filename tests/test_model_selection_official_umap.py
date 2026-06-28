from pathlib import Path

from src.config import PipelineConfig
from src.model_selection import build_official_umap_core_representation


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
            "official_model_suite": {
                "include_umap_hdbscan": True,
                "umap_hdbscan": {
                    "trial_name": "u20_n75_eom_mcs550_ms2_core",
                    "pca_components": 64,
                    "umap": {
                        "output": "feature_set_umap_pca64_u20_n75.parquet",
                        "n_components": 20,
                        "n_neighbors": 75,
                        "min_dist": 0.0,
                        "metric": "cosine",
                    },
                    "hdbscan": {
                        "min_cluster_size": 550,
                        "min_samples": 2,
                        "cluster_selection_method": "eom",
                        "allow_noise_assignment": False,
                    },
                },
            },
            "umap": {"enabled": True, "n_components": 20, "n_neighbors": 75, "min_dist": 0.0, "metric": "cosine"},
            "hdbscan": {"min_cluster_size": 550, "min_samples": 2, "cluster_selection_method": "eom"},
        },
        mode="dev",
        root=tmp_path,
    )


def test_official_umap_core_prereduces_with_pca_before_umap(tmp_path, monkeypatch):
    cfg = _test_config(tmp_path)
    raw_feature_path = tmp_path / "feature_set_embeddings_only.parquet"
    calls = {}

    def fake_pca(feature_path, output_path=None, summary_path=None, n_components=None, force=None, cfg=None):
        calls["pca"] = {
            "feature_path": Path(feature_path),
            "output_path": Path(output_path),
            "summary_path": Path(summary_path),
            "n_components": n_components,
            "force": force,
        }
        return Path(output_path)

    def fake_umap(feature_path, output_path=None, umap_overrides=None, force=None, cfg=None):
        calls["umap"] = {
            "feature_path": Path(feature_path),
            "output_path": output_path,
            "umap_overrides": dict(umap_overrides or {}),
            "force": force,
        }
        return tmp_path / "umap.parquet"

    monkeypatch.setattr("src.model_selection.build_pca_representation", fake_pca)
    monkeypatch.setattr("src.model_selection.build_umap_representation", fake_umap)

    result = build_official_umap_core_representation(raw_feature_path, force=True, cfg=cfg)

    expected_pca_path = cfg.outputs / "features" / "feature_set_pca_for_umap.parquet"
    expected_pca_summary_path = cfg.artifacts / "stage6" / "stage6_1_pca_for_umap_summary.csv"
    assert result == tmp_path / "umap.parquet"
    assert calls["pca"] == {
        "feature_path": raw_feature_path,
        "output_path": expected_pca_path,
        "summary_path": expected_pca_summary_path,
        "n_components": 64,
        "force": True,
    }
    assert calls["umap"]["feature_path"] == expected_pca_path
    assert calls["umap"]["umap_overrides"]["n_components"] == 20
    assert calls["umap"]["umap_overrides"]["n_neighbors"] == 75
    assert calls["umap"]["force"] is True
