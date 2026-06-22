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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    three_stage = official["three_stage_hdbscan"]
    hdbscan = promoted["hdbscan"]
    second_stage = two_stage["second_stage_hdbscan"]
    stage3_probe = two_stage["stage3_noise_probe"]
    third_stage = three_stage["third_stage_hdbscan"]

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
    assert base["profiling"]["final_handoff_readiness_statuses"] == ["strong", "usable", "ready_strong", "ready", "pass"]
    assert base["profiling"]["stage7_delivery_readiness"]["enabled"] is True
    assert base["profiling"]["stage7_delivery_readiness"]["fail_on_critical"] is True
    assert base["profiling"]["stage7_delivery_readiness"]["activation_customer_export_required"] is True
    assert base["profiling"]["tribe_name_lookup"] == {}
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
    assert hdbscan["min_cluster_size"] == 500
    assert hdbscan["min_samples"] == 12
    assert hdbscan["cluster_selection_method"] == "leaf"
    assert two_stage["enabled"] is True
    assert second_stage["allow_noise_assignment"] is False
    assert second_stage["min_cluster_size"] == 350
    assert second_stage["min_samples"] == 12
    assert second_stage["cluster_selection_method"] == "leaf"
    assert stage3_probe["enabled"] is True
    assert stage3_probe["run_candidate_hdbscan_after_visual_review"] is False
    assert stage3_probe["hdbscan"]["allow_noise_assignment"] is False
    assert stage3_probe["hdbscan"]["min_cluster_size"] < second_stage["min_cluster_size"]
    assert stage3_probe["lift_filter"]["min_strong_product_lifts"] >= 2
    assert stage3_probe["lift_filter"]["require_significant_product_lift"] is True
    assert "leaf_mcs500_ms12" in promoted["trial_name"]
    assert "leaf_mcs500_ms12" in promoted["variant_prefix"]
    assert "leaf_mcs500_ms12" in two_stage["trial_name"]
    assert "leaf_mcs500_ms12" in two_stage["variant_prefix"]
    assert two_stage["lift_filter"]["min_strong_product_lifts"] >= 2
    assert two_stage["lift_filter"]["require_significant_product_lift"] is True
    assert three_stage["enabled"] is True
    assert three_stage["model_name"] == "model_e_three_stage_hdbscan_lift_core"
    assert three_stage["output_prefix"] == "model_e_three_stage_hdbscan_lift_core"
    assert three_stage["min_noise_customers"] == stage3_probe["min_noise_customers"]
    assert "pca_components" not in three_stage
    assert "umap" not in three_stage
    assert third_stage["allow_noise_assignment"] is False
    assert third_stage["min_cluster_size"] == stage3_probe["hdbscan"]["min_cluster_size"]
    assert third_stage["min_cluster_size"] < second_stage["min_cluster_size"]
    assert three_stage["lift_filter"]["min_strong_product_lifts"] >= 2
    assert three_stage["lift_filter"]["require_significant_product_lift"] is True
    assert "soft" not in promoted["trial_name"].lower()
    assert "soft" not in promoted["model_name"].lower()
    assert "soft" not in promoted["output_prefix"].lower()
    assert "SoftNoiseAssignment" not in promoted["algorithm_name"]
    assert "soft" not in two_stage["trial_name"].lower()
    assert "soft" not in two_stage["model_name"].lower()
    assert "soft" not in two_stage["output_prefix"].lower()
    assert "SoftNoiseAssignment" not in two_stage["algorithm_name"]
    assert "soft" not in three_stage["trial_name"].lower()
    assert "soft" not in three_stage["model_name"].lower()
    assert "soft" not in three_stage["output_prefix"].lower()
    assert "SoftNoiseAssignment" not in three_stage["algorithm_name"]

    prod_overrides = _read_yaml(CONFIG_DIR / "prod.yaml")
    prod = _deep_merge(base, prod_overrides)
    prod_promoted = prod["official_model_suite"]["umap_hdbscan"]
    prod_two_stage = prod["official_model_suite"]["two_stage_hdbscan"]
    prod_three_stage = prod["official_model_suite"]["three_stage_hdbscan"]
    prod_two_stage_overrides = prod_overrides["official_model_suite"]["two_stage_hdbscan"]
    prod_three_stage_overrides = prod_overrides["official_model_suite"]["three_stage_hdbscan"]
    assert "leaf_mcs3000_ms6" in prod_promoted["trial_name"]
    assert "leaf_mcs3000_ms6" in prod_promoted["variant_prefix"]
    assert "leaf_mcs3000_ms6" in prod_two_stage["trial_name"]
    assert "leaf_mcs3000_ms6" in prod_two_stage["variant_prefix"]
    assert "leaf_mcs3000_ms6" in prod_three_stage["trial_name"]
    assert "leaf_mcs3000_ms6" in prod_three_stage["variant_prefix"]
    assert prod_promoted["hdbscan"]["min_cluster_size"] == 3000
    assert prod_promoted["hdbscan"]["min_samples"] == 6
    assert "balanced" in prod_two_stage["trial_name"]
    assert "balanced" in prod_three_stage["trial_name"]
    assert prod_two_stage["second_stage_hdbscan"]["min_cluster_size"] == 3000
    assert prod_two_stage["second_stage_hdbscan"]["min_samples"] == 6
    assert prod_two_stage["stage3_noise_probe"]["hdbscan"]["min_cluster_size"] < 3000
    assert prod_three_stage["third_stage_hdbscan"]["min_cluster_size"] == 2000
    assert prod_three_stage["third_stage_hdbscan"]["min_samples"] == 6
    assert prod_two_stage_overrides["second_stage_hdbscan"]["min_samples"] == 6
    assert prod_three_stage_overrides["third_stage_hdbscan"]["min_samples"] == 6
    assert "umap" not in prod_three_stage_overrides
    assert "soft" not in prod_promoted["trial_name"].lower()
    assert "soft" not in prod_two_stage["trial_name"].lower()
    assert "soft" not in prod_three_stage["trial_name"].lower()
