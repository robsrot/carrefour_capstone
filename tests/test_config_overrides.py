from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten(value, dotted_key))
        else:
            flattened[dotted_key] = value
    return flattened


def test_config_folder_contains_only_authoritative_mode_files():
    assert sorted(path.name for path in CONFIG_DIR.glob("*.y*ml")) == [
        "base.yaml",
        "dev.yaml",
        "prod.yaml",
    ]


def test_mode_configs_only_override_real_differences():
    base = _flatten(_read_yaml(CONFIG_DIR / "base.yaml"))
    failures: list[str] = []

    for mode_file in ["dev.yaml", "prod.yaml"]:
        overrides = _flatten(_read_yaml(CONFIG_DIR / mode_file))
        for key, value in sorted(overrides.items()):
            if key not in base:
                failures.append(f"{mode_file}: unknown override key {key}")
            elif base[key] == value:
                failures.append(f"{mode_file}: redundant override {key}")

    assert failures == []


def test_stage1_config_has_no_legacy_second_artifact_path():
    base = _read_yaml(CONFIG_DIR / "base.yaml")
    downsampling = base["baskets"]["downsampling"]

    assert "output" not in downsampling
    assert "summary_md" not in downsampling


def test_official_stage6_config_is_hard_umap_hdbscan_core_discovery():
    base = _read_yaml(CONFIG_DIR / "base.yaml")
    official = base["official_model_suite"]
    promoted = official["umap_hdbscan"]
    two_stage = official["two_stage_hdbscan"]
    hdbscan = promoted["hdbscan"]
    second_stage = two_stage["second_stage_hdbscan"]

    assert base["modeling"]["feature_set_for_selection"] == "embeddings_only"
    assert base["product_exposure_features"]["dimensions"] == {
        "themes": True,
        "sectors": True,
        "product_families": True,
    }
    assert base["product_exposure_features"]["enabled"] is False
    assert base["feature_sets"]["default"] == "embeddings_only"
    assert "embeddings_frequency" in base["feature_sets"]["outputs"]
    assert "embeddings_product_exposure" in base["feature_sets"]["outputs"]
    assert base["word2vec"]["full_basket_context"] is True
    assert base["word2vec"]["min_count"] == 60
    assert base["customer_embeddings"]["weight_strategy"] == "quantity_idf"
    assert base["customer_embeddings"]["gates"]["min_line_coverage_pct"] == 85.0
    assert base["customer_embeddings"]["gates"]["min_unit_coverage_pct"] == 85.0
    assert base["profiling"]["final_handoff_readiness_statuses"] == ["ready_strong"]
    assert base["profiling"]["final_handoff_require_actionability_proof"] is True
    assert base["profiling"]["final_handoff_require_theme_proof"] is False
    assert base["profiling"]["final_actionability_allow_theme_proof"] is False
    assert official["include_umap_hdbscan"] is True
    assert official["include_gmm"] is False
    assert official["include_pca_kmeans"] is False
    assert hdbscan["allow_noise_assignment"] is False
    assert promoted["pca_components"] == 64
    assert promoted["umap"]["n_components"] == 20
    assert promoted["umap"]["n_neighbors"] == 75
    assert "pca64" in promoted["feature_space"]
    assert "pca64" in promoted["umap"]["output"]
    assert "pca64" in two_stage["feature_space"]
    assert hdbscan["min_cluster_size"] == 550
    assert hdbscan["cluster_selection_method"] == "eom"
    assert two_stage["enabled"] is True
    assert second_stage["allow_noise_assignment"] is False
    assert second_stage["min_cluster_size"] >= hdbscan["min_cluster_size"]
    assert second_stage["cluster_selection_method"] == "eom"
    assert two_stage["lift_filter"]["min_strong_product_lifts"] >= 2
    assert two_stage["lift_filter"]["require_significant_product_lift"] is True
    assert "soft" not in promoted["trial_name"].lower()
    assert "soft" not in promoted["model_name"].lower()
    assert "soft" not in promoted["output_prefix"].lower()
    assert "SoftNoiseAssignment" not in promoted["algorithm_name"]
    assert "soft" not in two_stage["trial_name"].lower()
    assert "soft" not in two_stage["model_name"].lower()
    assert "soft" not in two_stage["output_prefix"].lower()
    assert "SoftNoiseAssignment" not in two_stage["algorithm_name"]

    prod = _read_yaml(CONFIG_DIR / "prod.yaml")
    prod_promoted = prod["official_model_suite"]["umap_hdbscan"]
    prod_two_stage = prod["official_model_suite"]["two_stage_hdbscan"]
    assert "soft" not in prod_promoted["trial_name"].lower()
    assert "soft" not in prod_two_stage["trial_name"].lower()
    assert (
        prod_two_stage["second_stage_hdbscan"]["min_cluster_size"]
        >= prod_promoted["hdbscan"]["min_cluster_size"]
    )
