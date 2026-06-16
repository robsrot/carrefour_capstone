"""Orchestration helpers for candidate model comparison."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.autoencoder import build_autoencoder_latents
from src.clustering import run_gmm_grid, run_hdbscan, run_pca_kmeans_grid
from src.config import CONFIG, PipelineConfig
from src.dimensionality import build_umap_representation
from src.experiment_reporting import write_summary_artifacts
from src.progress import log_event, stage_timer
from src.utils import collect_streaming, deterministic_sample_indices, frame_to_numpy, numeric_feature_columns


def run_candidate_model_suite(
    feature_path: str | Path,
    force: bool | None = None,
    include_autoencoder: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Run the curated official Stage 6 candidate suite."""

    candidate_results: list[dict[str, Any]] = []
    assignment_paths: dict[str, Path] = {}
    result_paths: dict[str, Path] = {}

    with stage_timer("Stage 6 model suite", "running candidate model suite", cfg=cfg, feature_path=feature_path):
        if cfg.get("official_model_suite.include_gmm", True):
            gmm_overrides = dict(cfg.get("official_model_suite.gmm", {}) or {})
            gmm_cfg = _config_with_section_overrides(cfg, "gmm", gmm_overrides)
            log_event(
                "Stage 6 model suite",
                "starting Model A GMM benchmark",
                cfg=gmm_cfg,
                components=f"{gmm_cfg.get('gmm.components_min')}-{gmm_cfg.get('gmm.components_max')}",
            )
            assignment, results, best = run_gmm_grid(
                feature_path,
                output_prefix="model_a_gmm",
                model_label="Model A",
                model_name="model_a_gmm",
                algorithm_name="GaussianMixture",
                feature_space="raw_customer_embeddings",
                force=force,
                cfg=gmm_cfg,
            )
            _record_candidate(candidate_results, assignment_paths, result_paths, assignment, results, best)
        else:
            log_event("Stage 6 model suite", "Model A GMM disabled", cfg=cfg)

        if cfg.get("official_model_suite.include_raw_hdbscan", False):
            log_event("Stage 6 model suite", "starting optional raw HDBSCAN diagnostic", cfg=cfg)
            assignment, results, best = run_hdbscan(
                feature_path,
                output_prefix="model_x_raw_hdbscan",
                model_label="Model X",
                model_name="model_x_raw_hdbscan",
                algorithm_name="HDBSCAN",
                feature_space="raw_customer_embeddings",
                force=force,
                cfg=cfg,
            )
            _record_candidate(candidate_results, assignment_paths, result_paths, assignment, results, best)
        else:
            log_event("Stage 6 model suite", "raw HDBSCAN diagnostic disabled", cfg=cfg)

        if cfg.get("official_model_suite.include_umap_hdbscan", True) and cfg.get("umap.enabled", True):
            promoted = cfg.get("official_model_suite.umap_hdbscan", {}) or {}
            hdbscan_overrides = dict(promoted.get("hdbscan", {}) or {})
            umap_overrides = dict(promoted.get("umap", {}) or {})
            trial_cfg = _config_with_section_overrides(cfg, "hdbscan", hdbscan_overrides)
            trial_name = str(promoted.get("trial_name", "official"))
            model_name = str(promoted.get("model_name", "model_b_umap_hdbscan"))
            output_prefix = str(promoted.get("output_prefix", model_name))
            algorithm_name = str(promoted.get("algorithm_name", "UMAP_HDBSCAN"))
            variant_prefix = promoted.get("variant_prefix", "umap")
            feature_space = str(promoted.get("feature_space", "umap_customer_embeddings"))
            allow_noise_assignment = bool(hdbscan_overrides.get("allow_noise_assignment", False))
            noise_assignment_strategy = str(hdbscan_overrides.get("noise_assignment_strategy", "q95"))

            log_event(
                "Stage 6 model suite",
                "starting Model B UMAP-HDBSCAN representation",
                cfg=trial_cfg,
                trial=trial_name,
                components=umap_overrides.get("n_components", trial_cfg.get("umap.n_components")),
                n_neighbors=umap_overrides.get("n_neighbors", trial_cfg.get("umap.n_neighbors")),
            )
            umap_path = build_umap_representation(feature_path, umap_overrides=umap_overrides, force=force, cfg=trial_cfg)
            log_event(
                "Stage 6 model suite",
                "starting Model B UMAP-HDBSCAN clustering",
                cfg=trial_cfg,
                trial=trial_name,
                soft_assignment=allow_noise_assignment,
                strategy=noise_assignment_strategy if allow_noise_assignment else None,
            )
            assignment, results, best = run_hdbscan(
                umap_path,
                output_prefix=output_prefix,
                model_label="Model B",
                model_name=model_name,
                algorithm_name=algorithm_name,
                feature_space=feature_space,
                trial_name=trial_name,
                variant_prefix=str(variant_prefix) if variant_prefix else None,
                scale_features=False,
                force=force,
                allow_noise_assignment=allow_noise_assignment,
                noise_assignment_strategy=noise_assignment_strategy,
                cfg=trial_cfg,
            )
            best["trial_name"] = trial_name
            best["promoted_from_experiment"] = promoted.get("promoted_from_experiment")
            _record_candidate(candidate_results, assignment_paths, result_paths, assignment, results, best)
        elif cfg.get("official_model_suite.include_umap_hdbscan", True):
            log_event("Stage 6 model suite", "Model B UMAP-HDBSCAN disabled because UMAP is disabled", cfg=cfg)
        else:
            log_event("Stage 6 model suite", "Model B UMAP-HDBSCAN disabled", cfg=cfg)

        run_autoencoder = (
            bool(cfg.get("official_model_suite.include_autoencoder", False))
            if include_autoencoder is None
            else include_autoencoder
        )
        if run_autoencoder:
            for latent_size in cfg.get("autoencoder.latent_sizes", [8, 16, 32]):
                log_event("Stage 6 model suite", "starting optional autoencoder", cfg=cfg, latent_size=latent_size)
                latent_path = build_autoencoder_latents(feature_path, int(latent_size), force=force, cfg=cfg)
                log_event("Stage 6 model suite", "starting optional autoencoder-GMM", cfg=cfg, latent_size=latent_size)
                assignment, results, best = run_gmm_grid(
                    latent_path,
                    output_prefix=f"model_x_autoencoder_gmm_ae{latent_size}",
                    model_label="Model X",
                    model_name="model_x_autoencoder_gmm",
                    algorithm_name="Autoencoder_GaussianMixture",
                    feature_space=f"autoencoder_latent_{latent_size}",
                    variant_prefix=f"ae{latent_size}",
                    force=force,
                    cfg=cfg,
                )
                _record_candidate(candidate_results, assignment_paths, result_paths, assignment, results, best)
        else:
            log_event("Stage 6 model suite", "autoencoder candidates disabled", cfg=cfg)

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
            log_event(
                "Stage 6 model suite",
                "starting Model C PCA-KMeans benchmark",
                cfg=pca_kmeans_cfg,
                clusters=f"{pca_kmeans_cfg.get('kmeans.k_min')}-{pca_kmeans_cfg.get('kmeans.k_max')}",
            )
            assignment, results, best = run_pca_kmeans_grid(
                feature_path,
                output_prefix="model_c_pca_kmeans",
                model_label="Model C",
                model_name="model_c_pca_kmeans",
                algorithm_name="PCA_MiniBatchKMeans",
                force=force,
                cfg=pca_kmeans_cfg,
            )
            _record_candidate(candidate_results, assignment_paths, result_paths, assignment, results, best)
        else:
            log_event("Stage 6 model suite", "Model C PCA-KMeans disabled", cfg=cfg)

        if not candidate_results:
            raise RuntimeError("Official Stage 6 candidate suite did not run any models.")

        log_event("Stage 6 model suite", "candidate suite complete", cfg=cfg, candidates=len(candidate_results))

    return {
        "candidate_results": candidate_results,
        "assignment_paths": assignment_paths,
        "result_paths": result_paths,
    }


def run_official_umap_hdbscan_core(
    feature_path: str | Path,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Run the official Stage 6.1/6.2 hard UMAP-HDBSCAN core-tribe flow."""

    settings = official_umap_hdbscan_core_settings(cfg)
    candidate_results: list[dict[str, Any]] = []
    assignment_paths: dict[str, Path] = {}
    result_paths: dict[str, Path] = {}

    with stage_timer(
        "Stage 6 UMAP-HDBSCAN core",
        "discovering hard organic core tribes",
        cfg=settings["trial_cfg"],
        feature_path=feature_path,
        trial=settings["trial_name"],
    ):
        umap_path = build_official_umap_core_representation(feature_path, force=force, cfg=cfg)
        assignment, results, best = run_official_hdbscan_core(umap_path, force=force, cfg=cfg)
        _record_candidate(candidate_results, assignment_paths, result_paths, assignment, results, best)

    return {
        "candidate_results": candidate_results,
        "assignment_paths": assignment_paths,
        "result_paths": result_paths,
        "umap_path": umap_path,
        "official_candidate_key": _candidate_key(candidate_results[0]),
        "stage6_flow": "hard_umap_hdbscan_core",
    }


def official_umap_hdbscan_core_settings(cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    """Return validated official Stage 6 hard UMAP-HDBSCAN settings."""

    if not bool(cfg.get("official_model_suite.include_umap_hdbscan", True)):
        raise RuntimeError("Official Stage 6 core discovery requires official_model_suite.include_umap_hdbscan=true.")
    if not bool(cfg.get("umap.enabled", True)):
        raise RuntimeError("Official Stage 6 core discovery requires umap.enabled=true.")

    promoted = cfg.get("official_model_suite.umap_hdbscan", {}) or {}
    hdbscan_overrides = dict(promoted.get("hdbscan", {}) or {})
    if bool(hdbscan_overrides.get("allow_noise_assignment", False)):
        raise ValueError(
            "Official Stage 6 core discovery must keep HDBSCAN noise unassigned. "
            "Set official_model_suite.umap_hdbscan.hdbscan.allow_noise_assignment=false."
        )
    hdbscan_overrides["allow_noise_assignment"] = False
    hdbscan_overrides.pop("noise_assignment_strategy", None)

    umap_overrides = dict(promoted.get("umap", {}) or {})
    trial_cfg = _config_with_section_overrides(cfg, "hdbscan", hdbscan_overrides)
    trial_name = str(promoted.get("trial_name", "official_umap_hdbscan_core"))
    model_name = str(promoted.get("model_name", "model_b_umap_hdbscan_core"))
    output_prefix = str(promoted.get("output_prefix", model_name))
    algorithm_name = str(promoted.get("algorithm_name", "UMAP_HDBSCAN"))
    variant_prefix = promoted.get("variant_prefix", "umap_core")
    feature_space = str(promoted.get("feature_space", "umap_customer_embeddings"))

    return {
        "promoted": promoted,
        "trial_cfg": trial_cfg,
        "umap_overrides": umap_overrides,
        "hdbscan_overrides": hdbscan_overrides,
        "trial_name": trial_name,
        "model_name": model_name,
        "output_prefix": output_prefix,
        "algorithm_name": algorithm_name,
        "variant_prefix": variant_prefix,
        "feature_space": feature_space,
    }


def build_official_umap_core_representation(
    feature_path: str | Path,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Stage 6.1: build the official product-only UMAP clustering representation."""

    settings = official_umap_hdbscan_core_settings(cfg)
    log_event(
        "Stage 6.1 UMAP",
        "building product-only customer manifold",
        cfg=settings["trial_cfg"],
        trial=settings["trial_name"],
        components=settings["umap_overrides"].get("n_components", settings["trial_cfg"].get("umap.n_components")),
        n_neighbors=settings["umap_overrides"].get("n_neighbors", settings["trial_cfg"].get("umap.n_neighbors")),
    )
    return build_umap_representation(
        feature_path,
        umap_overrides=settings["umap_overrides"],
        force=force,
        cfg=settings["trial_cfg"],
    )


def run_official_hdbscan_core(
    umap_path: str | Path,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """Stage 6.2: run hard HDBSCAN on the official UMAP representation."""

    settings = official_umap_hdbscan_core_settings(cfg)
    log_event(
        "Stage 6.2 HDBSCAN",
        "fitting hard density clusters with noise retained",
        cfg=settings["trial_cfg"],
        trial=settings["trial_name"],
        soft_assignment=False,
    )
    assignment, results, best = run_hdbscan(
        umap_path,
        output_prefix=settings["output_prefix"],
        model_label="Model B",
        model_name=settings["model_name"],
        algorithm_name=settings["algorithm_name"],
        feature_space=settings["feature_space"],
        trial_name=settings["trial_name"],
        variant_prefix=str(settings["variant_prefix"]) if settings["variant_prefix"] else None,
        scale_features=False,
        force=force,
        allow_noise_assignment=False,
        cfg=settings["trial_cfg"],
    )
    best["trial_name"] = settings["trial_name"]
    best["promoted_from_experiment"] = settings["promoted"].get("promoted_from_experiment")
    best["stage6_flow"] = "6.1_umap_representation_then_6.2_hard_hdbscan_core"
    return assignment, results, best


def model_suite_from_single_candidate(
    assignment_path: str | Path,
    result_path: str | Path,
    best: dict[str, Any],
    *,
    umap_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the standard model-suite dictionary for one official candidate."""

    assignment = Path(assignment_path)
    results = Path(result_path)
    candidate_key = _candidate_key(best)
    suite = {
        "candidate_results": [best],
        "assignment_paths": {candidate_key: assignment},
        "result_paths": {candidate_key: results},
        "official_candidate_key": candidate_key,
        "stage6_flow": "hard_umap_hdbscan_core",
    }
    if umap_path is not None:
        suite["umap_path"] = Path(umap_path)
    return suite


def build_stage6_umap_diagnostics(
    feature_path: str | Path,
    umap_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Write Stage 6.1 checks for the official UMAP representation."""

    cfg.ensure_directories()
    settings = official_umap_hdbscan_core_settings(cfg)
    output = Path(output_path) if output_path else cfg.artifacts / "stage6" / "stage6_1_umap_checks.csv"
    feature_schema = pl.read_parquet(feature_path, n_rows=1)
    umap_schema = pl.read_parquet(umap_path, n_rows=1)
    feature_rows = _row_count(feature_path)
    umap_rows = _row_count(umap_path)
    source_feature_count = len(numeric_feature_columns(feature_schema))
    umap_cols = [col for col in numeric_feature_columns(umap_schema) if col.startswith("umap_")]
    health = _numeric_scan_health(umap_path, umap_cols)
    duplicate_customers = _duplicate_customer_count(umap_path)
    missing_feature_customers = _anti_join_customer_count(feature_path, umap_path)
    extra_umap_customers = _anti_join_customer_count(umap_path, feature_path)

    issues = []
    if feature_rows != umap_rows:
        issues.append("row_count_mismatch")
    if missing_feature_customers:
        issues.append("missing_feature_customers")
    if extra_umap_customers:
        issues.append("extra_umap_customers")
    if duplicate_customers:
        issues.append("duplicate_umap_customers")
    if len(umap_cols) < 2:
        issues.append("umap_components_lt_2")
    if health["finite_pct"] is None or float(health["finite_pct"]) < 100.0:
        issues.append("non_finite_umap_values")
    if health["null_pct"] is not None and float(health["null_pct"]) > 0.0:
        issues.append("null_umap_values")

    row = {
        "stage": "6.1_umap_representation",
        "check_status": "pass" if not issues else "fail",
        "check_issues": "; ".join(issues) if issues else "pass",
        "feature_path": str(feature_path),
        "umap_path": str(umap_path),
        "source_rows": int(feature_rows),
        "umap_rows": int(umap_rows),
        "missing_feature_customers": int(missing_feature_customers),
        "extra_umap_customers": int(extra_umap_customers),
        "duplicate_umap_customers": None if duplicate_customers is None else int(duplicate_customers),
        "source_feature_count": int(source_feature_count),
        "umap_component_count": int(len(umap_cols)),
        "requested_umap_components": int(
            settings["umap_overrides"].get("n_components", settings["trial_cfg"].get("umap.n_components"))
        ),
        "umap_n_neighbors": int(
            settings["umap_overrides"].get("n_neighbors", settings["trial_cfg"].get("umap.n_neighbors"))
        ),
        "umap_metric": str(settings["umap_overrides"].get("metric", settings["trial_cfg"].get("umap.metric"))),
        "umap_min_dist": float(settings["umap_overrides"].get("min_dist", settings["trial_cfg"].get("umap.min_dist"))),
        "finite_pct": health["finite_pct"],
        "null_pct": health["null_pct"],
        "zero_variance_component_count": health["zero_variance_count"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_csv(output)
    log_event("Stage 6.1 checks", "wrote UMAP representation diagnostics", cfg=cfg, path=output)
    return output


def build_stage6_hdbscan_diagnostics(
    umap_path: str | Path,
    assignment_path: str | Path,
    result_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Write Stage 6.2 checks for hard HDBSCAN assignments."""

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.artifacts / "stage6" / "stage6_2_hdbscan_checks.csv"
    umap_rows = _row_count(umap_path)
    assignments = pl.read_parquet(assignment_path).unique(subset=["cliente"], keep="first")
    result = pl.read_parquet(result_path).row(0, named=True)
    assignment_rows = assignments.height
    duplicate_customers = _duplicate_customer_count(assignment_path)
    missing_umap_customers = _anti_join_customer_count(umap_path, assignment_path)
    extra_assignment_customers = _anti_join_customer_count(assignment_path, umap_path)
    soft_source_count = (
        assignments.filter(
            pl.col("assignment_source").cast(pl.Utf8).str.contains("nearest_centroid|soft_noise")
        ).height
        if "assignment_source" in assignments.columns
        else 0
    )
    noise_count = assignments.filter(pl.col("tribe_id") < 0).height
    assigned_count = assignments.height - noise_count
    cluster_count = int(result.get("cluster_count") or assignments.filter(pl.col("tribe_id") >= 0)["tribe_id"].n_unique())

    blocking_issues = []
    if umap_rows != assignment_rows:
        blocking_issues.append("row_count_mismatch")
    if missing_umap_customers:
        blocking_issues.append("missing_umap_customers")
    if extra_assignment_customers:
        blocking_issues.append("extra_assignment_customers")
    if duplicate_customers:
        blocking_issues.append("duplicate_assignment_customers")
    if result.get("assignment_policy") != "hard_hdbscan_core_noise_retained":
        blocking_issues.append("assignment_policy_not_hard_core")
    if _as_bool(result.get("soft_assignment_enabled")):
        blocking_issues.append("soft_assignment_enabled")
    if float(result.get("soft_assigned_pct") or 0.0) > 0.0 or soft_source_count > 0:
        blocking_issues.append("soft_assigned_customers_present")

    quality_issues = []
    if not _as_bool(result.get("passes_quality_gate")):
        quality_issues = [
            issue.strip()
            for issue in str(result.get("quality_gate_reason") or "quality_gate_failed").split(";")
            if issue.strip() and issue.strip() != "pass"
        ]
    elif cluster_count < int(cfg.get("quality_gates.min_clusters", 2)):
        quality_issues.append(f"cluster_count<{cfg.get('quality_gates.min_clusters', 2)}")
    issues = blocking_issues + quality_issues

    row = {
        "stage": "6.2_hard_hdbscan",
        "check_status": "pass" if not issues else "fail",
        "check_issues": "; ".join(issues) if issues else "pass",
        "blocking_check_status": "pass" if not blocking_issues else "fail",
        "blocking_check_issues": "; ".join(blocking_issues) if blocking_issues else "pass",
        "quality_gate_status": "pass" if not quality_issues else "fail",
        "quality_gate_issues": "; ".join(quality_issues) if quality_issues else "pass",
        "umap_path": str(umap_path),
        "assignment_path": str(assignment_path),
        "result_path": str(result_path),
        "umap_rows": int(umap_rows),
        "assignment_rows": int(assignment_rows),
        "missing_umap_customers": int(missing_umap_customers),
        "extra_assignment_customers": int(extra_assignment_customers),
        "duplicate_assignment_customers": None if duplicate_customers is None else int(duplicate_customers),
        "cluster_count": cluster_count,
        "assigned_customers": int(assigned_count),
        "noise_customers": int(noise_count),
        "noise_pct": float(result.get("noise_pct") or 0.0),
        "core_coverage_pct": float(result.get("core_coverage_pct") or (100.0 - float(result.get("noise_pct") or 0.0))),
        "assignment_policy": result.get("assignment_policy"),
        "soft_assignment_enabled": _as_bool(result.get("soft_assignment_enabled")),
        "soft_assigned_pct": float(result.get("soft_assigned_pct") or 0.0),
        "soft_assignment_source_rows": int(soft_source_count),
        "passes_quality_gate": _as_bool(result.get("passes_quality_gate")),
        "quality_gate_reason": result.get("quality_gate_reason"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_csv(output)
    log_event("Stage 6.2 checks", "wrote hard HDBSCAN diagnostics", cfg=cfg, path=output)
    return output


def build_stage6_representation_cluster_diagnostics(
    feature_path: str | Path,
    umap_path: str | Path,
    assignment_path: str | Path,
    result_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Write Stage 6.3 evidence for UMAP retention and HDBSCAN cluster validity.

    UMAP does not expose PCA-style explained variance, so this diagnostic measures
    whether local neighborhoods and sampled distance ordering survive reduction.
    HDBSCAN validity is reported with density-aware DBCV when available; silhouette
    remains a supporting core-only compactness metric.
    """

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.artifacts / "stage6" / "stage6_3_representation_cluster_quality.csv"
    assignment_df = pl.read_parquet(assignment_path)
    assignments = (
        assignment_df.select(
            [col for col in ["cliente", "tribe_id", "assignment_confidence_score"] if col in assignment_df.columns]
        )
        .unique(subset=["cliente"], keep="first")
        .sort("cliente")
    )
    result = pl.read_parquet(result_path).row(0, named=True)

    feature_rows = _row_count(feature_path)
    umap_rows = _row_count(umap_path)
    assignment_rows = assignments.height
    feature_cols = numeric_feature_columns(pl.read_parquet(feature_path, n_rows=1))
    umap_cols = [col for col in numeric_feature_columns(pl.read_parquet(umap_path, n_rows=1)) if col.startswith("umap_")]
    aligned_rows = _aligned_customer_count(feature_path, umap_path, assignments)

    neighbor_k = int(cfg.get("stage6_quality.neighbor_k", 15))
    sample_size = int(cfg.get("stage6_quality.sample_size", 5000))
    distance_sample_size = int(cfg.get("stage6_quality.distance_sample_size", 1500))
    dbcv_sample_size = int(cfg.get("stage6_quality.dbcv_sample_size", 3000))
    input_metric = str(cfg.get("official_model_suite.umap_hdbscan.umap.metric", cfg.get("umap.metric", "cosine")))
    joined = _sample_stage6_diagnostic_rows(
        feature_path,
        umap_path,
        assignments,
        feature_cols,
        umap_cols,
        sample_size=max(sample_size, distance_sample_size, dbcv_sample_size),
        seed=cfg.random_seed,
    )
    labels = joined["tribe_id"].to_numpy().astype(int) if "tribe_id" in joined.columns else np.array([], dtype=int)
    X_source = frame_to_numpy(joined, feature_cols) if joined.height else np.empty((0, len(feature_cols)), dtype=np.float32)
    X_umap = frame_to_numpy(joined, umap_cols) if joined.height else np.empty((0, len(umap_cols)), dtype=np.float32)
    X_source = np.nan_to_num(X_source.astype(np.float32, copy=False), copy=False)
    X_umap = np.nan_to_num(X_umap.astype(np.float32, copy=False), copy=False)

    retention = _umap_retention_metrics(
        X_source,
        X_umap,
        sample_size=sample_size,
        distance_sample_size=distance_sample_size,
        neighbor_k=neighbor_k,
        input_metric=input_metric,
        seed=cfg.random_seed,
    )
    density_validity = _hdbscan_density_validity(
        X_umap,
        labels,
        sample_size=dbcv_sample_size,
        seed=cfg.random_seed,
    )

    confidence_mean = None
    if "assignment_confidence_score" in joined.columns:
        confidence = joined["assignment_confidence_score"].drop_nulls().to_numpy()
        if confidence.size:
            confidence_mean = float(np.nanmean(confidence.astype(float)))

    row = {
        "stage": "6.3_representation_cluster_quality",
        "feature_path": str(feature_path),
        "umap_path": str(umap_path),
        "assignment_path": str(assignment_path),
        "result_path": str(result_path),
        "source_rows": int(feature_rows),
        "umap_rows": int(umap_rows),
        "assignment_rows": int(assignment_rows),
        "aligned_rows": int(aligned_rows),
        "diagnostic_sample_rows": int(joined.height),
        "source_feature_count": int(len(feature_cols)),
        "umap_component_count": int(len(umap_cols)),
        "umap_input_metric": input_metric,
        **retention,
        "cluster_count": int(result.get("cluster_count") or 0),
        "noise_pct": _as_float(result.get("noise_pct"), default=0.0),
        "core_coverage_pct": _as_float(result.get("core_coverage_pct"), default=100.0 - _as_float(result.get("noise_pct"), default=0.0)),
        "silhouette_core_only": result.get("silhouette"),
        "coverage_adjusted_silhouette": result.get("coverage_adjusted_silhouette"),
        "davies_bouldin_core_only": result.get("davies_bouldin"),
        "calinski_harabasz_core_only": result.get("calinski_harabasz"),
        "avg_assignment_confidence": result.get("avg_assignment_confidence", confidence_mean),
        "hdbscan_cluster_persistence_mean": result.get("hdbscan_cluster_persistence_mean"),
        "hdbscan_cluster_persistence_min": result.get("hdbscan_cluster_persistence_min"),
        "hdbscan_cluster_persistence_max": result.get("hdbscan_cluster_persistence_max"),
        **density_validity,
        "umap_retention_note": (
            "UMAP has no explained-variance metric; use trustworthiness, kNN overlap, "
            "and sampled distance-rank correlation as neighborhood-retention evidence."
        ),
        "silhouette_note": (
            "Silhouette is computed on non-noise core customers only and is not density-aware; "
            "use it as supporting evidence, not as the main HDBSCAN quality metric."
        ),
        "stability_note": (
            "Stage 6.4 tests perturbation stability and cluster readiness. For final sign-off, prefer repeated-seed "
            "or bootstrap reruns of the promoted recipe."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_csv(output)
    log_event("Stage 6.3 diagnostics", "wrote representation and cluster quality evidence", cfg=cfg, path=output)
    return output


def _row_count(path: str | Path) -> int:
    return int(collect_streaming(pl.scan_parquet(path).select(pl.len().alias("n_rows")))[0, "n_rows"])


def _duplicate_customer_count(path: str | Path) -> int:
    stats = collect_streaming(
        pl.scan_parquet(path).select(
            [
                pl.len().alias("n_rows"),
                pl.col("cliente").n_unique().alias("n_customers"),
            ]
        )
    ).row(0, named=True)
    return int(stats["n_rows"] - stats["n_customers"])


def _anti_join_customer_count(left_path: str | Path, right_path: str | Path) -> int:
    left = pl.scan_parquet(left_path).select("cliente").unique()
    right = pl.scan_parquet(right_path).select("cliente").unique()
    return int(collect_streaming(left.join(right, on="cliente", how="anti").select(pl.len().alias("n")))[0, "n"])


def _aligned_customer_count(feature_path: str | Path, umap_path: str | Path, assignments: pl.DataFrame) -> int:
    feature_customers = pl.scan_parquet(feature_path).select("cliente").unique()
    umap_customers = pl.scan_parquet(umap_path).select("cliente").unique()
    assignment_customers = assignments.select("cliente").lazy().unique()
    return int(
        collect_streaming(
            feature_customers.join(umap_customers, on="cliente", how="inner")
            .join(assignment_customers, on="cliente", how="inner")
            .select(pl.len().alias("n"))
        )[0, "n"]
    )


def _sample_stage6_diagnostic_rows(
    feature_path: str | Path,
    umap_path: str | Path,
    assignments: pl.DataFrame,
    feature_cols: list[str],
    umap_cols: list[str],
    *,
    sample_size: int,
    seed: int,
) -> pl.DataFrame:
    if assignments.is_empty():
        return pl.DataFrame()
    sample_idx = deterministic_sample_indices(assignments.height, min(sample_size, assignments.height), seed)
    sampled_assignments = assignments[sample_idx].with_row_index("_sample_order")
    sampled_customers = sampled_assignments["cliente"].to_list()
    sampled_features = collect_streaming(
        pl.scan_parquet(feature_path)
        .select(["cliente", *feature_cols])
        .filter(pl.col("cliente").is_in(sampled_customers))
    )
    sampled_umap = collect_streaming(
        pl.scan_parquet(umap_path)
        .select(["cliente", *umap_cols])
        .filter(pl.col("cliente").is_in(sampled_customers))
    )
    return (
        sampled_assignments.join(sampled_features, on="cliente", how="inner")
        .join(sampled_umap, on="cliente", how="inner")
        .sort("_sample_order")
    )


def _numeric_scan_health(path: str | Path, feature_cols: list[str]) -> dict[str, float | int | None]:
    if not feature_cols:
        return {"finite_pct": None, "null_pct": None, "zero_variance_count": 0}
    exprs = []
    for col in feature_cols:
        finite = pl.col(col).is_finite().fill_null(False)
        exprs.extend(
            [
                pl.col(col).is_null().sum().alias(f"{col}__null"),
                finite.sum().alias(f"{col}__finite"),
                pl.col(col).filter(finite).min().alias(f"{col}__min"),
                pl.col(col).filter(finite).max().alias(f"{col}__max"),
            ]
        )
    row = collect_streaming(pl.scan_parquet(path).select(exprs)).row(0, named=True)
    n_rows = _row_count(path)
    total_cells = n_rows * len(feature_cols)
    null_count = sum(int(row.get(f"{col}__null") or 0) for col in feature_cols)
    finite_count = sum(int(row.get(f"{col}__finite") or 0) for col in feature_cols)
    zero_variance_count = 0
    for col in feature_cols:
        finite_count_col = int(row.get(f"{col}__finite") or 0)
        min_value = row.get(f"{col}__min")
        max_value = row.get(f"{col}__max")
        if finite_count_col == 0 or min_value == max_value:
            zero_variance_count += 1
    return {
        "finite_pct": float(finite_count / max(total_cells, 1) * 100.0),
        "null_pct": float(null_count / max(total_cells, 1) * 100.0),
        "zero_variance_count": int(zero_variance_count),
    }


def _umap_retention_metrics(
    X_source: np.ndarray,
    X_umap: np.ndarray,
    *,
    sample_size: int,
    distance_sample_size: int,
    neighbor_k: int,
    input_metric: str,
    seed: int,
) -> dict[str, Any]:
    n_rows = int(min(X_source.shape[0], X_umap.shape[0]))
    if n_rows <= max(neighbor_k + 1, 2) or X_source.shape[1] == 0 or X_umap.shape[1] == 0:
        return {
            "umap_quality_sample_size": n_rows,
            "umap_neighbor_k": neighbor_k,
            "umap_trustworthiness": None,
            "umap_mean_knn_overlap_pct": None,
            "umap_median_knn_overlap_pct": None,
            "umap_distance_rank_sample_size": 0,
            "umap_distance_spearman": None,
        }

    from sklearn.manifold import trustworthiness

    sample_idx = deterministic_sample_indices(n_rows, min(sample_size, n_rows), seed)
    X_src_sample = X_source[sample_idx]
    X_umap_sample = X_umap[sample_idx]
    max_trustworthy_k = max(1, (X_src_sample.shape[0] - 1) // 2)
    k = min(max(int(neighbor_k), 1), X_src_sample.shape[0] - 1, max_trustworthy_k)
    metrics: dict[str, Any] = {
        "umap_quality_sample_size": int(X_src_sample.shape[0]),
        "umap_neighbor_k": int(k),
        "umap_trustworthiness": None,
        "umap_mean_knn_overlap_pct": None,
        "umap_median_knn_overlap_pct": None,
        "umap_distance_rank_sample_size": 0,
        "umap_distance_spearman": None,
    }

    try:
        metrics["umap_trustworthiness"] = float(
            trustworthiness(X_src_sample, X_umap_sample, n_neighbors=k, metric=input_metric)
        )
    except (TypeError, ValueError):
        try:
            metrics["umap_trustworthiness"] = float(
                trustworthiness(X_src_sample, X_umap_sample, n_neighbors=k)
            )
        except ValueError:
            pass

    overlap = _knn_overlap_scores(X_src_sample, X_umap_sample, k=k, input_metric=input_metric)
    if overlap.size:
        metrics["umap_mean_knn_overlap_pct"] = float(np.mean(overlap) * 100.0)
        metrics["umap_median_knn_overlap_pct"] = float(np.median(overlap) * 100.0)

    rank_sample_size = min(int(distance_sample_size), X_src_sample.shape[0])
    if rank_sample_size > 2:
        rank_idx = deterministic_sample_indices(X_src_sample.shape[0], rank_sample_size, seed + 17)
        metrics["umap_distance_rank_sample_size"] = int(rank_sample_size)
        metrics["umap_distance_spearman"] = _sampled_distance_spearman(
            X_src_sample[rank_idx],
            X_umap_sample[rank_idx],
            input_metric=input_metric,
        )
    return metrics


def _knn_overlap_scores(X_source: np.ndarray, X_umap: np.ndarray, *, k: int, input_metric: str) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    if X_source.shape[0] <= k:
        return np.array([], dtype=np.float32)
    try:
        source_nn = NearestNeighbors(n_neighbors=k + 1, metric=input_metric).fit(X_source)
        source_indices = source_nn.kneighbors(return_distance=False)
    except ValueError:
        source_nn = NearestNeighbors(n_neighbors=k + 1).fit(X_source)
        source_indices = source_nn.kneighbors(return_distance=False)
    umap_nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_umap)
    umap_indices = umap_nn.kneighbors(return_distance=False)

    scores = []
    for row_idx in range(X_source.shape[0]):
        left = [int(idx) for idx in source_indices[row_idx] if int(idx) != row_idx][:k]
        right = [int(idx) for idx in umap_indices[row_idx] if int(idx) != row_idx][:k]
        if left and right:
            scores.append(len(set(left).intersection(right)) / float(k))
    return np.asarray(scores, dtype=np.float32)


def _sampled_distance_spearman(X_source: np.ndarray, X_umap: np.ndarray, *, input_metric: str) -> float | None:
    from sklearn.metrics import pairwise_distances

    try:
        source_distances = pairwise_distances(X_source, metric=input_metric)
    except ValueError:
        source_distances = pairwise_distances(X_source)
    umap_distances = pairwise_distances(X_umap, metric="euclidean")
    upper = np.triu_indices(source_distances.shape[0], k=1)
    source_values = source_distances[upper]
    umap_values = umap_distances[upper]
    finite = np.isfinite(source_values) & np.isfinite(umap_values)
    if int(np.sum(finite)) < 2:
        return None
    source_ranks = _ordinal_ranks(source_values[finite])
    umap_ranks = _ordinal_ranks(umap_values[finite])
    source_centered = source_ranks - float(np.mean(source_ranks))
    umap_centered = umap_ranks - float(np.mean(umap_ranks))
    denom = float(np.linalg.norm(source_centered) * np.linalg.norm(umap_centered))
    if denom <= 0.0:
        return None
    return float(np.dot(source_centered, umap_centered) / denom)


def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def _hdbscan_density_validity(
    X_umap: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    n_rows = int(min(X_umap.shape[0], labels.shape[0]))
    result: dict[str, Any] = {
        "hdbscan_dbcv_score": None,
        "hdbscan_dbcv_sample_size": 0,
        "hdbscan_dbcv_status": "not_run",
        "hdbscan_dbcv_note": None,
    }
    if n_rows == 0 or int(np.unique(labels[labels >= 0]).shape[0]) < 2:
        result["hdbscan_dbcv_note"] = "DBCV requires at least two non-noise clusters."
        return result

    sample_idx = deterministic_sample_indices(n_rows, min(int(sample_size), n_rows), seed + 29)
    X_sample = X_umap[sample_idx]
    labels_sample = labels[sample_idx]
    if int(np.unique(labels_sample[labels_sample >= 0]).shape[0]) < 2:
        result["hdbscan_dbcv_note"] = "The diagnostic sample contained fewer than two non-noise clusters."
        return result

    try:
        from hdbscan.validity import validity_index
    except ImportError:
        result["hdbscan_dbcv_status"] = "unavailable"
        result["hdbscan_dbcv_note"] = "hdbscan.validity.validity_index is not available in this environment."
        return result

    try:
        score = validity_index(X_sample.astype(np.float64), labels_sample.astype(np.int64), metric="euclidean")
    except Exception as exc:  # pragma: no cover - depends on optional hdbscan internals.
        result["hdbscan_dbcv_status"] = "failed"
        result["hdbscan_dbcv_note"] = f"DBCV failed: {type(exc).__name__}: {exc}"
        return result

    result["hdbscan_dbcv_score"] = float(score)
    result["hdbscan_dbcv_sample_size"] = int(X_sample.shape[0])
    result["hdbscan_dbcv_status"] = "pass"
    result["hdbscan_dbcv_note"] = "DBCV is density-aware and ranges from -1 to 1; higher is better."
    return result


def _numeric_frame_health(df: pl.DataFrame, feature_cols: list[str]) -> dict[str, float | int | None]:
    if not feature_cols or df.height == 0:
        return {"finite_pct": None, "null_pct": None, "zero_variance_count": 0}
    total_cells = df.height * len(feature_cols)
    null_count = sum(int(df.select(pl.col(col).is_null().sum())[0, 0]) for col in feature_cols)
    X = frame_to_numpy(df, feature_cols)
    finite_pct = float(np.isfinite(X).mean() * 100.0)
    zero_variance_count = 0
    for idx in range(X.shape[1]):
        values = X[:, idx]
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0 or float(np.nanstd(finite_values)) <= 1e-12:
            zero_variance_count += 1
    return {
        "finite_pct": finite_pct,
        "null_pct": float(null_count / max(total_cells, 1) * 100.0),
        "zero_variance_count": int(zero_variance_count),
    }


def run_umap_hdbscan_experiments(
    feature_path: str | Path,
    trials: list[dict[str, Any]] | None = None,
    experiment_name: str = "manual_umap_hdbscan",
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Run only UMAP-HDBSCAN trial specs against an existing feature set."""

    if not cfg.experiments_enabled:
        raise RuntimeError(
            f"UMAP-HDBSCAN experiments are disabled for mode {cfg.mode!r}. "
            "Run experiments in dev, promote the winning settings to YAML, then run the official prod pipeline."
        )

    candidate_results: list[dict[str, Any]] = []
    assignment_paths: dict[str, Path] = {}
    result_paths: dict[str, Path] = {}
    experiment_slug = _trial_slug(experiment_name)
    selected_trials = [dict(trial) for trial in trials] if trials is not None else _umap_hdbscan_trials(cfg)
    experiment_dir = cfg.experiments / experiment_slug

    with stage_timer(
        "UMAP-HDBSCAN experiments",
        "running focused UMAP-HDBSCAN trials",
        cfg=cfg,
        experiment=experiment_slug,
        trials=len(selected_trials),
    ):
        for trial in selected_trials:
            trial_name = _trial_slug(str(trial.get("name", "default")))
            model_name = str(trial.get("model_name", f"experiment_{experiment_slug}_{trial_name}"))
            model_label = str(trial.get("model_label", "UMAP-HDBSCAN Experiment"))
            trial_cfg = _config_with_section_overrides(cfg, "hdbscan", trial.get("hdbscan", {}) or {})
            umap_overrides = dict(trial.get("umap", {}) or {})
            umap_output = umap_overrides.pop("output", f"{trial_name}_umap.parquet")

            log_event(
                "UMAP-HDBSCAN experiments",
                "building trial UMAP",
                cfg=trial_cfg,
                experiment=experiment_slug,
                trial=trial_name,
                model=model_name,
            )
            umap_path = build_umap_representation(
                feature_path,
                output_path=experiment_dir / str(umap_output),
                umap_overrides=umap_overrides,
                force=force,
                cfg=trial_cfg,
            )
            log_event(
                "UMAP-HDBSCAN experiments",
                "clustering trial",
                cfg=trial_cfg,
                experiment=experiment_slug,
                trial=trial_name,
                model=model_name,
            )
            assignment, results, best = run_hdbscan(
                umap_path,
                output_prefix=model_name,
                model_label=model_label,
                model_name=model_name,
                algorithm_name="UMAP_HDBSCAN",
                feature_space=f"umap_customer_embeddings_{experiment_slug}_{trial_name}",
                trial_name=trial_name,
                variant_prefix=trial_name,
                scale_features=False,
                output_dir=experiment_dir,
                model_dir=experiment_dir,
                force=force,
                cfg=trial_cfg,
            )
            best["experiment_name"] = experiment_slug
            best["trial_name"] = trial_name
            candidate_results.append(best)
            assignment_paths[_candidate_key(best)] = assignment
            result_paths[_candidate_key(best)] = results

    suite = {
        "candidate_results": candidate_results,
        "assignment_paths": assignment_paths,
        "result_paths": result_paths,
    }
    diagnostics = build_candidate_model_diagnostics(
        suite,
        output_path=experiment_dir / f"{experiment_slug}_diagnostics.parquet",
        write_csv=False,
        cfg=cfg,
    )
    suite["diagnostics"] = diagnostics
    return suite


def _record_candidate(
    candidate_results: list[dict[str, Any]],
    assignment_paths: dict[str, Path],
    result_paths: dict[str, Path],
    assignment: Path,
    results: Path,
    best: dict[str, Any],
) -> None:
    candidate_results.append(best)
    assignment_paths[_candidate_key(best)] = assignment
    result_paths[_candidate_key(best)] = results


def _candidate_key(result: dict[str, Any]) -> str:
    return f"{result.get('model_name')}::{result.get('model_variant')}"


def _umap_hdbscan_trials(cfg: PipelineConfig) -> list[dict[str, Any]]:
    if not bool(cfg.get("umap_hdbscan_trials.enabled", True)):
        return []
    trials = cfg.get("umap_hdbscan_trials.trials", [])
    if trials:
        return [dict(trial) for trial in trials]
    return [
        {
            "name": "default",
            "model_label": "UMAP-HDBSCAN Experiment",
            "model_name": "experiment_umap_hdbscan_default",
            "umap": {},
            "hdbscan": {},
        }
    ]


def _trial_slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.strip().lower()).strip("_") or "default"


def _config_with_section_overrides(
    cfg: PipelineConfig,
    section: str,
    overrides: dict[str, Any],
) -> PipelineConfig:
    values = deepcopy(cfg.values)
    base_section = values.get(section, {}) or {}
    values[section] = {**base_section, **overrides}
    return replace(cfg, values=values)


def build_candidate_model_diagnostics(
    model_suite: dict[str, Any],
    output_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    write_csv: bool = False,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Combine all Stage 6 grid/single-model rows into one diagnostics artifact."""

    cfg.ensure_directories()
    result_paths = model_suite.get("result_paths", {})
    assignment_paths = model_suite.get("assignment_paths", {})
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    for _, result_path in sorted(result_paths.items(), key=lambda item: str(item[1])):
        path = Path(result_path)
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        for row in pl.read_parquet(path).iter_rows(named=True):
            item = dict(row)
            item["model_id"] = item.get("model_id") or item.get("model_name")
            item["model_name"] = item.get("model_name") or item.get("model_id")
            item["algorithm_name"] = item.get("algorithm_name") or item.get("model_name")
            item["candidate_id"] = f"{item.get('model_name')}::{item.get('model_variant')}"
            item["assignment_path"] = (
                str(assignment_paths[item["candidate_id"]]) if item["candidate_id"] in assignment_paths else None
            )
            item["source_result_path"] = str(path)
            item["stage6_score"] = _as_float(
                item.get("coverage_adjusted_silhouette"),
                default=_as_float(item.get("silhouette"), default=-999.0),
            )
            rows.append(item)

    ranked = sorted(rows, key=_diagnostic_rank_key)
    for rank, row in enumerate(ranked, start=1):
        row["stage6_rank"] = rank

    output = Path(output_path) if output_path else cfg.model_selection / "stage6_candidate_model_diagnostics.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = pl.from_dicts(ranked, infer_schema_length=None) if ranked else pl.DataFrame()
    diagnostics.write_parquet(output)
    result = {"parquet": output}
    log_kwargs: dict[str, Any] = {"parquet": output}

    summaries = write_summary_artifacts(
        diagnostics,
        output_base=output,
        title="Stage 6 Candidate Model Diagnostics",
        priority_columns=[
            "stage6_rank",
            "passes_quality_gate",
            "model_name",
            "trial_name",
            "model_variant",
            "assignment_policy",
            "cluster_count",
            "stage6_score",
            "coverage_adjusted_silhouette",
            "silhouette",
            "davies_bouldin",
            "noise_pct",
            "core_coverage_pct",
            "soft_assigned_pct",
            "cluster_size_cv",
        ],
        cfg=cfg,
        write_markdown=bool(cfg.get("model_selection.write_summary_markdown", True)),
    )
    result.update(summaries)
    log_kwargs.update(summary_csv=summaries["summary_csv"], summary_md=summaries.get("summary_md"))

    if write_csv:
        csv_output = Path(csv_path) if csv_path else cfg.model_selection / "stage6_candidate_model_diagnostics.csv"
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.write_csv(csv_output)
        result["csv"] = csv_output
        log_kwargs["csv"] = csv_output
    log_event(
        "Stage 6 diagnostics",
        "wrote candidate diagnostics",
        cfg=cfg,
        candidates=len(ranked),
        **log_kwargs,
    )
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _diagnostic_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        0.0 if _as_bool(row.get("passes_quality_gate")) else 1.0,
        -_as_float(row.get("stage6_score"), -999.0),
        _as_float(row.get("davies_bouldin"), 999.0),
        _as_float(row.get("cluster_size_cv"), 999.0),
        str(row.get("candidate_id")),
    )
