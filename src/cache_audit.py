"""Cache audit helpers for expensive official pipeline stages."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.customer_embeddings import _frequency_weighting_config, _recency_weighting_config, _uses_idf, _uses_quantity
from src.embedding_validation import _category_pattern_metadata
from src.item2vec import _corpus_limits, _effective_window
from src.utils import cache_status, file_fingerprint, write_artifact_metadata


def stage_1_6_cache_audit(cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    """Report whether expensive Stage 1-6 artifacts will be reused from cache."""

    rows = [
        _status_row(
            spec["stage"],
            spec["artifact"],
            spec["path"],
            spec["metadata"],
            cfg=spec.get("cfg", cfg),
        )
        for spec in _stage_1_6_cache_specs(cfg)
    ]
    return pl.DataFrame(rows).sort(["stage", "artifact"])


def adopt_existing_stage_1_6_cache_metadata(cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    """Write current metadata for existing Stage 1-6 artifacts.

    Use this only when the existing artifacts were produced from the current
    prepared data and current config, but their metadata manifest is missing,
    for example after an interrupted run or manual file copy.
    """

    rows = []
    for spec in _stage_1_6_cache_specs(cfg):
        path = spec["path"]
        metadata = spec["metadata"]
        adopted = bool(path.exists() and metadata is not None)
        reason = "metadata written" if adopted else "artifact missing or metadata unavailable"
        if adopted:
            write_artifact_metadata(path, metadata)
        rows.append(
            {
                "stage": spec["stage"],
                "artifact": spec["artifact"],
                "adopted": adopted,
                "reason": reason,
                "path": str(path),
            }
        )
    return pl.DataFrame(rows).sort(["stage", "artifact"])


def mode_path_audit(cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    """Check that official pipeline paths resolve to the active dev/prod namespace."""

    import src.config as config_module

    output_root = cfg.root / cfg.get("paths.outputs")
    current_output_root = cfg.outputs
    other_mode = "prod" if cfg.mode == "dev" else "dev"
    forbidden_output_root = output_root / other_mode
    forbidden_data_root = cfg.root / ("data/dev" if cfg.mode == "prod" else "data/processed")
    expected_data_root = cfg.data_processed

    rows: list[dict[str, Any]] = []
    env_name = str(cfg.get("run.mode_env", "CARREFOUR_MODE"))
    env_mode = (os.getenv(env_name) or "").strip().lower()
    rows.append(
        {
            "check": "environment_mode",
            "status": "pass" if env_mode == cfg.mode else "fail",
            "exists": True,
            "path": env_mode,
            "expected_root": cfg.mode,
            "forbidden_root": "",
            "reason": f"{env_name} matches CONFIG.mode" if env_mode == cfg.mode else f"{env_name}={env_mode!r}, CONFIG.mode={cfg.mode!r}",
        }
    )
    module_mode = getattr(config_module.CONFIG, "mode", None)
    rows.append(
        {
            "check": "src_config_module_mode",
            "status": "pass" if module_mode == cfg.mode else "fail",
            "exists": True,
            "path": str(module_mode),
            "expected_root": cfg.mode,
            "forbidden_root": "",
            "reason": "src.config.CONFIG matches notebook CONFIG"
            if module_mode == cfg.mode
            else f"src.config.CONFIG.mode={module_mode!r}, notebook CONFIG.mode={cfg.mode!r}",
        }
    )

    def add_path_check(
        check: str,
        path: Path,
        *,
        expected_root: Path | None,
        forbidden_roots: list[Path] | None = None,
        allow_missing: bool = True,
    ) -> None:
        forbidden_roots = forbidden_roots or []
        expected_ok = expected_root is None or _is_relative_to(path, expected_root)
        forbidden_hit = next((root for root in forbidden_roots if _is_relative_to(path, root)), None)
        exists = path.exists()
        missing_ok = allow_missing or exists
        status = "pass" if expected_ok and forbidden_hit is None and missing_ok else "fail"
        if not expected_ok:
            reason = "path is outside the expected active-mode root"
        elif forbidden_hit is not None:
            reason = "path points to the opposite mode namespace"
        elif not missing_ok:
            reason = "required path is missing"
        else:
            reason = "path is mode-safe"
        rows.append(
            {
                "check": check,
                "status": status,
                "exists": exists,
                "path": str(path),
                "expected_root": "" if expected_root is None else str(expected_root),
                "forbidden_root": "" if forbidden_hit is None else str(forbidden_hit),
                "reason": reason,
            }
        )

    add_path_check(
        "prepared_transactions",
        cfg.prepared_transactions_path,
        expected_root=expected_data_root,
        forbidden_roots=[forbidden_data_root] if cfg.mode == "prod" else [],
        allow_missing=False,
    )
    add_path_check(
        "outputs_root",
        current_output_root,
        expected_root=current_output_root,
        forbidden_roots=[forbidden_output_root],
        allow_missing=True,
    )
    for check, path in [
        ("models_root", cfg.models),
        ("reports_root", cfg.reports),
        ("artifacts_root", cfg.artifacts),
        ("figures_root", cfg.figures),
        ("model_selection_cache_root", cfg.model_selection_cache),
    ]:
        add_path_check(check, path, expected_root=current_output_root, forbidden_roots=[forbidden_output_root])

    for name, path in _flatten_path_specs(_official_stage_paths(cfg)).items():
        add_path_check(
            f"stage_1_6::{name}",
            path,
            expected_root=current_output_root,
            forbidden_roots=[forbidden_output_root],
        )

    for check, path in {
        "stage_7_profiles_root": cfg.outputs / "profiles",
        "stage_7_deep_profile_artifacts": cfg.artifacts / "stage7",
        "stage_7_lift_figures": cfg.figures / "tribe_lifts",
    }.items():
        add_path_check(check, path, expected_root=current_output_root, forbidden_roots=[forbidden_output_root])

    return pl.DataFrame(rows)


def assert_mode_path_audit(cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    """Raise if official pipeline paths could mix dev/prod namespaces."""

    audit = mode_path_audit(cfg)
    failures = audit.filter(pl.col("status") == "fail")
    if failures.height:
        preview = failures.select(["check", "path", "expected_root", "forbidden_root", "reason"]).to_dicts()
        raise RuntimeError(f"Mode/path audit failed for {cfg.mode!r}: {preview}")
    return audit


def _flatten_path_specs(spec: Any, prefix: str = "") -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if isinstance(spec, Path):
        paths[prefix or spec.name] = spec
    elif isinstance(spec, dict):
        for key, value in spec.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.update(_flatten_path_specs(value, child_prefix))
    return paths


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _stage_1_6_cache_specs(cfg: PipelineConfig) -> list[dict[str, Any]]:
    paths = _official_stage_paths(cfg)
    product_exposure_enabled = bool(cfg.get("product_exposure_features.enabled", False)) or str(
        cfg.get("modeling.feature_set_for_selection", "embeddings_only")
    ) == "embeddings_product_exposure"
    specs: list[dict[str, Any]] = [
        {
            "stage": "1",
            "artifact": "basket_sentences",
            "path": paths["basket_sentences"],
            "metadata": _basket_metadata(paths["basket_sentences"], cfg),
        },
        {
            "stage": "2",
            "artifact": "item2vec_model",
            "path": paths["item2vec_model"],
            "metadata": _item2vec_metadata(paths["basket_sentences"], cfg),
        },
        {
            "stage": "2",
            "artifact": "product_embeddings",
            "path": paths["product_embeddings"],
            "metadata": _product_embedding_metadata(paths["item2vec_model"], cfg),
        },
        {
            "stage": "3",
            "artifact": "embedding_validation_csv",
            "path": paths["embedding_validation_csv"],
            "metadata": _embedding_validation_metadata(paths["product_embeddings"], cfg),
        },
        {
            "stage": "3",
            "artifact": "embedding_hubness_csv",
            "path": paths["embedding_hubness_csv"],
            "metadata": _embedding_validation_metadata(paths["product_embeddings"], cfg),
        },
        {
            "stage": "4",
            "artifact": "customer_embeddings",
            "path": paths["customer_embeddings"],
            "metadata": _customer_embedding_metadata(paths["product_embeddings"], cfg),
        },
        {
            "stage": "5",
            "artifact": "behavioral_features",
            "path": paths["behavioral_features"],
            "metadata": _behavior_metadata(cfg),
        },
    ]
    if product_exposure_enabled:
        specs.append(
            {
                "stage": "5",
                "artifact": "product_exposure_features",
                "path": paths["product_exposure_features"],
                "metadata": _product_exposure_metadata(cfg),
            }
        )
    for variant, path in paths["feature_sets"].items():
        behavior_path = paths["behavioral_features"] if variant == "embeddings_behavior" else None
        product_exposure_path = (
            paths["product_exposure_features"] if variant == "embeddings_product_exposure" else None
        )
        specs.append(
            {
                "stage": "5",
                "artifact": f"feature_set_{variant}",
                "path": path,
                "metadata": _feature_set_metadata(
                    variant,
                    paths["customer_embeddings"],
                    behavior_path,
                    product_exposure_path,
                    cfg,
                ),
            }
        )
    if cfg.get("official_model_suite.include_gmm", True):
        gmm_cfg = _config_with_section_overrides(cfg, "gmm", dict(cfg.get("official_model_suite.gmm", {}) or {}))
        for artifact, path in paths["gmm"].items():
            specs.append(
                {
                    "stage": "6",
                    "artifact": artifact,
                    "path": path,
                    "metadata": _gmm_metadata(paths["selection_feature_set"], gmm_cfg),
                }
            )
    if cfg.get("official_model_suite.include_umap_hdbscan", True) and cfg.get("umap.enabled", True):
        promoted = cfg.get("official_model_suite.umap_hdbscan", {}) or {}
        trial_cfg = _config_with_section_overrides(cfg, "hdbscan", dict(promoted.get("hdbscan", {}) or {}))
        specs.append(
            {
                "stage": "6",
                "artifact": "model_b_umap_representation",
                "path": paths["umap_representation"],
                "metadata": _umap_metadata(paths["selection_feature_set"], dict(promoted.get("umap", {}) or {}), trial_cfg),
            }
        )
        for artifact, path in paths["hdbscan"].items():
            specs.append(
                {
                    "stage": "6",
                    "artifact": artifact,
                    "path": path,
                    "metadata": _hdbscan_metadata(paths["umap_representation"], promoted, trial_cfg),
                }
            )
    if cfg.get("official_model_suite.include_pca_kmeans", True):
        pca_kmeans_overrides = cfg.get("official_model_suite.pca_kmeans", {}) or {}
        pca_kmeans_cfg = _config_with_section_overrides(
            cfg,
            "pca",
            dict(pca_kmeans_overrides.get("pca", {}) or {}),
        )
        pca_kmeans_cfg = _config_with_section_overrides(
            pca_kmeans_cfg,
            "kmeans",
            dict(pca_kmeans_overrides.get("kmeans", {}) or {}),
        )
        for artifact, path in paths["pca_kmeans"].items():
            specs.append(
                {
                    "stage": "6",
                    "artifact": artifact,
                    "path": path,
                    "metadata": _pca_kmeans_metadata(paths["selection_feature_set"], pca_kmeans_cfg),
                }
            )
    return specs


def _official_stage_paths(cfg: PipelineConfig) -> dict[str, Any]:
    feature_outputs = cfg.get("feature_sets.outputs", {}) or {}
    selection_feature_set = str(cfg.get("modeling.feature_set_for_selection", "embeddings_only"))
    selection_feature_filename = str(
        feature_outputs.get(selection_feature_set, f"feature_set_{selection_feature_set}.parquet")
    )
    promoted = cfg.get("official_model_suite.umap_hdbscan", {}) or {}
    model_b_name = str(promoted.get("output_prefix", promoted.get("model_name", "model_b_umap_hdbscan")))
    embedding_validation_dir = cfg.artifacts / str(cfg.get("embedding_validation.output_dir", "stage3"))
    paths = {
        "basket_sentences": cfg.artifact_path("baskets", "output", directory=cfg.outputs / "embeddings"),
        "item2vec_model": cfg.models / str(cfg.get("word2vec.model_name")),
        "product_embeddings": cfg.artifact_path("word2vec", "embeddings_output", directory=cfg.outputs / "embeddings"),
        "embedding_validation_csv": embedding_validation_dir / str(cfg.get("embedding_validation.output_csv")),
        "embedding_hubness_csv": embedding_validation_dir / str(
            cfg.get("embedding_validation.hubness_output_csv", "embedding_hubness.csv")
        ),
        "customer_embeddings": cfg.artifact_path("customer_embeddings", "output", directory=cfg.outputs / "features"),
        "behavioral_features": cfg.artifact_path("behavioral_features", "output", directory=cfg.outputs / "features"),
        "product_exposure_features": cfg.artifact_path(
            "product_exposure_features",
            "output",
            directory=cfg.outputs / "features",
        ),
        "feature_sets": {
            selection_feature_set: cfg.outputs / "features" / selection_feature_filename
        },
        "selection_feature_set": cfg.outputs
        / "features"
        / selection_feature_filename,
        "gmm": {
            "model_a_gmm_assignments": cfg.model_selection_cache / "cluster_assignments_model_a_gmm.parquet",
            "model_a_gmm_results": cfg.model_selection_cache / "model_a_gmm_grid_results.parquet",
        },
        "umap_representation": cfg.outputs
        / "features"
        / str((promoted.get("umap", {}) or {}).get("output", cfg.get("umap.output", "feature_set_umap_cluster.parquet"))),
        "hdbscan": {
            "model_b_hdbscan_assignments": cfg.model_selection_cache / f"cluster_assignments_{model_b_name}.parquet",
            "model_b_hdbscan_results": cfg.model_selection_cache / f"{model_b_name}_results.parquet",
        },
        "pca_kmeans": {
            "model_c_pca_kmeans_assignments": cfg.model_selection_cache
            / "cluster_assignments_model_c_pca_kmeans.parquet",
            "model_c_pca_kmeans_results": cfg.model_selection_cache / "model_c_pca_kmeans_grid_results.parquet",
        },
    }
    if bool(cfg.get("embedding_validation.write_markdown_report", True)):
        paths["embedding_validation_md"] = embedding_validation_dir / str(cfg.get("embedding_validation.output_md"))
    return paths


def _status_row(
    stage: str,
    artifact: str,
    path: Path,
    metadata: dict[str, Any] | None,
    *,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    if metadata is None:
        return {
            "stage": stage,
            "artifact": artifact,
            "cache_hit": False,
            "exists": path.exists(),
            "metadata_status": "unavailable",
            "reason": "cache metadata could not be computed for audit",
            "path": str(path),
        }
    status = cache_status(
        path,
        force=bool(cfg.get("cache.force", False)),
        use_cached=bool(cfg.get("cache.use_cached", True)),
        metadata=metadata,
    )
    return {
        "stage": stage,
        "artifact": artifact,
        "cache_hit": bool(status["cache_hit"]),
        "exists": bool(status["exists"]),
        "metadata_status": status["metadata_status"],
        "reason": status["reason"],
        "path": str(status["path"]),
    }


def _basket_metadata(output: Path, cfg: PipelineConfig) -> dict[str, Any]:
    strategy = str(cfg.get("baskets.construction_strategy", "baseline"))
    return {
        "stage": "basket_sentences",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "construction_strategy": strategy,
        "repeat_product_by_quantity": bool(cfg.get("baskets.repeat_product_by_quantity", False)),
        "downsampling": _downsampling_metadata(cfg) if strategy == "common_downsampled" else None,
        "common_product_diagnostics": _diagnostic_threshold_metadata(cfg)
        if strategy == "common_downsampled"
        else None,
        "ordering": "deterministic_ticket_hash",
    }


def _downsampling_metadata(cfg: PipelineConfig) -> dict[str, Any]:
    legacy_manual_ids = cfg.get("baskets.downsampling.exclude_product_ids", [])
    manual_ids = cfg.get("baskets.downsampling.manual_exclude_product_ids", legacy_manual_ids) or []
    auto_exclude = cfg.get("baskets.downsampling.auto_exclude", {}) or {}
    return {
        "manual_exclude_product_ids": [str(value) for value in manual_ids],
        "auto_exclude": {
            "enabled": bool(auto_exclude.get("enabled", True)),
            "customer_penetration_threshold": _optional_float(
                auto_exclude.get("customer_penetration_threshold", 0.50)
            ),
            "basket_penetration_threshold": _optional_float(auto_exclude.get("basket_penetration_threshold", 0.10)),
            "line_share_threshold": _optional_float(auto_exclude.get("line_share_threshold")),
            "max_products": int(auto_exclude.get("max_products", 25)),
        },
        "target_customer_penetration": float(cfg.get("baskets.downsampling.target_customer_penetration", 0.01)),
        "keep_probability_exponent": float(cfg.get("baskets.downsampling.keep_probability_exponent", 0.5)),
        "min_keep_probability": float(cfg.get("baskets.downsampling.min_keep_probability", 0.15)),
        "sample_modulus": 1_000_000,
    }


def _diagnostic_threshold_metadata(cfg: PipelineConfig) -> dict[str, float]:
    return {
        "common_customer_penetration_threshold": float(
            cfg.get("baskets.diagnostics.common_customer_penetration_threshold", 0.01)
        ),
        "common_basket_penetration_threshold": float(
            cfg.get("baskets.diagnostics.common_basket_penetration_threshold", 0.005)
        ),
        "common_line_share_threshold": float(cfg.get("baskets.diagnostics.common_line_share_threshold", 0.005)),
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _item2vec_metadata(basket_path: Path, cfg: PipelineConfig) -> dict[str, Any]:
    effective_window = _effective_window(basket_path, cfg) if basket_path.exists() else int(cfg.get("word2vec.window"))
    corpus_limits = _corpus_limits(cfg)
    return {
        "stage": "item2vec_model",
        "mode": cfg.mode,
        "basket_sentences": file_fingerprint(basket_path),
        "word2vec": {
            "vector_size": int(cfg.get("word2vec.vector_size")),
            "window": effective_window,
            "configured_window": int(cfg.get("word2vec.window")),
            "full_basket_context": bool(cfg.get("word2vec.full_basket_context", False)),
            "min_tokens_per_basket": int(corpus_limits["min_tokens_per_basket"] or 1),
            "max_tokens_per_basket": corpus_limits["max_tokens_per_basket"],
            "basket_context_policy": "deterministic_local_context",
            "min_count": int(cfg.get("word2vec.min_count")),
            "negative": int(cfg.get("word2vec.negative")),
            "sample": float(cfg.get("word2vec.sample")),
            "epochs": int(cfg.get("word2vec.epochs")),
            "sg": int(cfg.get("word2vec.sg")),
            "workers": int(cfg.get("word2vec.workers")),
            "seed": cfg.random_seed,
        },
    }


def _product_embedding_metadata(model_path: Path, cfg: PipelineConfig) -> dict[str, Any] | None:
    if not model_path.exists():
        return None
    try:
        from gensim.models import Word2Vec

        model = Word2Vec.load(str(model_path))
    except Exception:
        return None
    return {
        "stage": "product_embeddings",
        "mode": cfg.mode,
        "model_vector_size": int(model.vector_size),
        "model_vocab": int(len(model.wv)),
        "model_keys_head": list(model.wv.index_to_key[:10]),
    }


def _embedding_validation_metadata(embeddings_path: Path, cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "stage": "embedding_validation",
        "mode": cfg.mode,
        "product_embeddings": file_fingerprint(embeddings_path),
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "sample_product_ids": None,
        "sample_size": int(cfg.get("embedding_validation.sample_size", 25)),
        "neighbors": int(cfg.get("embedding_validation.neighbors", 8)),
        "staple_sample_size": int(cfg.get("embedding_validation.staple_sample_size", 8)),
        "niche_sample_size": int(cfg.get("embedding_validation.niche_sample_size", 8)),
        "common_sample_size": int(
            cfg.get("embedding_validation.common_sample_size", cfg.get("embedding_validation.staple_sample_size", 8))
        ),
        "rare_sample_size": int(
            cfg.get("embedding_validation.rare_sample_size", cfg.get("embedding_validation.niche_sample_size", 8))
        ),
        "niche_category_sample_size": int(cfg.get("embedding_validation.niche_category_sample_size", 5)),
        "niche_categories": _category_pattern_metadata(cfg),
        "generic_neighbor_warning_share_threshold": float(
            cfg.get("embedding_validation.generic_neighbor_warning_share_threshold", 0.50)
        ),
        "guardrails": cfg.get("embedding_validation.guardrails", {}),
        "write_extract_figures": bool(cfg.get("embedding_validation.write_extract_figures", False)),
        "niche_basket_penetration_max": float(
            cfg.get("embedding_validation.niche_basket_penetration_max", 0.001)
        ),
        "common_neighbor_basket_penetration_threshold": float(
            cfg.get(
                "embedding_validation.common_neighbor_basket_penetration_threshold",
                cfg.get("baskets.diagnostics.common_basket_penetration_threshold", 0.005),
            )
        ),
        "hubness_neighbors": int(cfg.get("embedding_validation.hubness_neighbors", 10)),
        "hubness_sample_size": cfg.get("embedding_validation.hubness_sample_size", None),
        "hubness_chunk_size": int(cfg.get("embedding_validation.hubness_chunk_size", 256)),
        "random_seed": cfg.random_seed,
    }


def _customer_embedding_metadata(embeddings_path: Path, cfg: PipelineConfig) -> dict[str, Any]:
    selected_weight_strategy = str(cfg.get("customer_embeddings.weight_strategy", "quantity"))
    return {
        "stage": "customer_embeddings",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "product_embeddings": file_fingerprint(embeddings_path),
        "weight_strategy": selected_weight_strategy,
        "idf_weighting": _uses_idf(selected_weight_strategy),
        "quantity_transform": str(cfg.get("customer_embeddings.quantity_transform", "raw"))
        if _uses_quantity(selected_weight_strategy)
        else None,
        "max_customer_product_weight": cfg.get("customer_embeddings.max_customer_product_weight", None),
        "recency_weighting": _recency_weighting_config(cfg),
        "frequency_weighting": _frequency_weighting_config(cfg),
        "normalize_vectors": bool(cfg.get("customer_embeddings.normalize_vectors", False)),
    }


def _behavior_metadata(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "stage": "behavioral_features",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "promo_definition": "promo_line_share plus promo_basket_share; promo_share aliases line share",
        "reference_date": cfg.get("behavioral_features.reference_date"),
    }


def _product_exposure_metadata(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "stage": "product_exposure_features",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "settings": cfg.get("product_exposure_features", {}) or {},
    }


def _feature_set_metadata(
    variant: str,
    customer_embeddings_path: Path,
    behavior_path: Path | None,
    product_exposure_path: Path | None,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    return {
        "stage": "feature_set",
        "mode": cfg.mode,
        "variant": variant,
        "customer_embeddings": file_fingerprint(customer_embeddings_path),
        "behavior": file_fingerprint(behavior_path) if behavior_path else None,
        "product_exposure": file_fingerprint(product_exposure_path) if product_exposure_path else None,
        "standardize_behavior": bool(cfg.get("feature_sets.standardize_behavior", True)),
        "standardize_product_exposure": bool(cfg.get("feature_sets.standardize_product_exposure", True)),
        "product_exposure_weight": float(cfg.get("feature_sets.product_exposure_weight", 0.35)),
    }


def _gmm_metadata(feature_path: Path, cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "stage": "gmm_grid",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "output_prefix": "model_a_gmm",
        "model_label": "Model A",
        "model_name": "model_a_gmm",
        "algorithm_name": "GaussianMixture",
        "feature_space": "raw_customer_embeddings",
        "gmm": cfg.get("gmm", {}),
        "modeling": cfg.get("modeling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
        "assignment_schema_version": 2,
    }


def _umap_metadata(feature_path: Path, umap_overrides: dict[str, Any], cfg: PipelineConfig) -> dict[str, Any]:
    umap_cfg = {**(cfg.get("umap", {}) or {}), **umap_overrides}
    return {
        "stage": "umap_representation",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "umap": umap_cfg,
        "random_seed": cfg.random_seed,
        "purpose": "clustering_candidate",
    }


def _hdbscan_metadata(umap_path: Path, promoted: dict[str, Any], cfg: PipelineConfig) -> dict[str, Any]:
    hdbscan_overrides = dict(promoted.get("hdbscan", {}) or {})
    return {
        "stage": "hdbscan",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(umap_path),
        "output_prefix": str(promoted.get("output_prefix", promoted.get("model_name", "model_b_umap_hdbscan"))),
        "model_label": "Model B",
        "model_name": str(promoted.get("model_name", "model_b_umap_hdbscan")),
        "algorithm_name": str(promoted.get("algorithm_name", "UMAP_HDBSCAN")),
        "feature_space": str(promoted.get("feature_space", "umap_customer_embeddings")),
        "trial_name": str(promoted.get("trial_name", "official")),
        "scale_features": False,
        "allow_noise_assignment": bool(hdbscan_overrides.get("allow_noise_assignment", False)),
        "noise_assignment_strategy": str(hdbscan_overrides.get("noise_assignment_strategy", "q95")),
        "hdbscan": cfg.get("hdbscan", {}),
        "modeling": cfg.get("modeling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
        "hdbscan_backend_policy": "external_hdbscan_with_sklearn_full_fit_fallback_and_optional_soft_assignment",
        "assignment_schema_version": 2,
    }


def _pca_kmeans_metadata(feature_path: Path, cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "stage": "pca_kmeans_grid",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "output_prefix": "model_c_pca_kmeans",
        "model_label": "Model C",
        "model_name": "model_c_pca_kmeans",
        "algorithm_name": "PCA_MiniBatchKMeans",
        "pca": cfg.get("pca", {}),
        "kmeans": cfg.get("kmeans", {}),
        "modeling": cfg.get("modeling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
        "assignment_schema_version": 2,
    }


def _config_with_section_overrides(
    cfg: PipelineConfig,
    section: str,
    overrides: dict[str, Any],
) -> PipelineConfig:
    values = deepcopy(cfg.values)
    base_section = values.get(section, {}) or {}
    values[section] = {**base_section, **overrides}
    return replace(cfg, values=values)
