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
from src.dimensionality import build_pca_representation, build_umap_representation
from src.evaluation import cluster_size_summary, evaluate_labels, quality_gate_result
from src.experiment_reporting import write_summary_artifacts
from src.progress import log_event, stage_timer
from src.utils import (
    collect_streaming,
    deterministic_sample_indices,
    file_fingerprint,
    frame_to_numpy,
    numeric_feature_columns,
    should_use_cache,
    write_artifact_metadata,
)


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
            umap_path = build_official_umap_core_representation(feature_path, force=force, cfg=cfg)
            log_event(
                "Stage 6 model suite",
                "starting Model B UMAP-HDBSCAN clustering",
                cfg=trial_cfg,
                trial=trial_name,
                soft_assignment=allow_noise_assignment,
                strategy=noise_assignment_strategy if allow_noise_assignment else None,
            )
            if bool(cfg.get("official_model_suite.two_stage_hdbscan.enabled", False)):
                assignment, results, best = run_official_two_stage_hdbscan_lift_core(umap_path, force=force, cfg=cfg)
            else:
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
        "stage6_flow": candidate_results[0].get("stage6_flow", "hard_umap_hdbscan_core"),
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
    pca_components = int(promoted.get("pca_components", 64))

    return {
        "promoted": promoted,
        "trial_cfg": trial_cfg,
        "umap_overrides": umap_overrides,
        "hdbscan_overrides": hdbscan_overrides,
        "pca_components": pca_components,
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
        pca_components=settings["pca_components"],
        components=settings["umap_overrides"].get("n_components", settings["trial_cfg"].get("umap.n_components")),
        n_neighbors=settings["umap_overrides"].get("n_neighbors", settings["trial_cfg"].get("umap.n_neighbors")),
    )
    pca_output_path = cfg.outputs / "features" / "feature_set_pca_for_umap.parquet"
    pca_summary_path = _official_pca_for_umap_summary_path(cfg)
    pca_feature_path = build_pca_representation(
        feature_path,
        output_path=pca_output_path,
        summary_path=pca_summary_path,
        n_components=settings["pca_components"],
        force=force,
        cfg=cfg,
    )
    return build_umap_representation(
        pca_feature_path,
        umap_overrides=settings["umap_overrides"],
        force=force,
        cfg=settings["trial_cfg"],
    )


def _official_pca_for_umap_summary_path(cfg: PipelineConfig) -> Path:
    return cfg.artifacts / "stage6" / "stage6_1_pca_for_umap_summary.csv"


def run_official_hdbscan_core(
    umap_path: str | Path,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """Stage 6.2: run the official hard HDBSCAN core discovery flow."""

    if bool(cfg.get("official_model_suite.two_stage_hdbscan.enabled", False)):
        return run_official_two_stage_hdbscan_lift_core(umap_path, force=force, cfg=cfg)

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


def official_two_stage_hdbscan_settings(cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    """Return validated settings for hard two-stage HDBSCAN plus product-lift filtering."""

    base = official_umap_hdbscan_core_settings(cfg)
    two_stage = cfg.get("official_model_suite.two_stage_hdbscan", {}) or {}
    second_stage_overrides = {
        **dict(base["hdbscan_overrides"]),
        **dict(two_stage.get("second_stage_hdbscan", {}) or {}),
    }
    if bool(second_stage_overrides.get("allow_noise_assignment", False)):
        raise ValueError(
            "Two-stage official HDBSCAN must keep noise unassigned. "
            "Set official_model_suite.two_stage_hdbscan.second_stage_hdbscan.allow_noise_assignment=false."
        )
    second_stage_overrides["allow_noise_assignment"] = False
    second_stage_overrides.pop("noise_assignment_strategy", None)

    return {
        "base": base,
        "first_stage_cfg": base["trial_cfg"],
        "second_stage_cfg": _config_with_section_overrides(cfg, "hdbscan", second_stage_overrides),
        "second_stage_overrides": second_stage_overrides,
        "trial_name": str(two_stage.get("trial_name", f"{base['trial_name']}_two_stage_lift_core")),
        "model_name": str(two_stage.get("model_name", "model_d_two_stage_hdbscan_lift_core")),
        "output_prefix": str(two_stage.get("output_prefix", "model_d_two_stage_hdbscan_lift_core")),
        "algorithm_name": str(two_stage.get("algorithm_name", "UMAP_HDBSCAN_TwoStageLiftCore")),
        "feature_space": str(two_stage.get("feature_space", base["feature_space"])),
        "variant_prefix": str(two_stage.get("variant_prefix", f"{base['variant_prefix']}_two_stage_lift")),
        "lift_filter": dict(two_stage.get("lift_filter", {}) or {}),
        "promoted": base["promoted"],
        "two_stage": two_stage,
    }


def two_stage_hdbscan_artifact_paths(
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Return the standard official two-stage HDBSCAN artifact paths."""

    settings = official_two_stage_hdbscan_settings(cfg)
    output_prefix = settings["output_prefix"]
    lift_evidence_path = cfg.artifacts / "stage6" / f"{output_prefix}_lift_filter_evidence.parquet"
    return {
        "assignment_path": cfg.model_selection_cache / f"cluster_assignments_{output_prefix}.parquet",
        "results_path": cfg.model_selection_cache / f"{output_prefix}_results.parquet",
        "stage1_assignment_path": cfg.model_selection_cache / f"cluster_assignments_{output_prefix}_stage1.parquet",
        "stage1_results_path": cfg.model_selection_cache / f"{output_prefix}_stage1_results.parquet",
        "stage2_assignment_path": cfg.model_selection_cache / f"cluster_assignments_{output_prefix}_stage2_noise.parquet",
        "stage2_results_path": cfg.model_selection_cache / f"{output_prefix}_stage2_noise_results.parquet",
        "stage2_noise_feature_path": cfg.model_selection_cache / f"{output_prefix}_stage2_noise_features.parquet",
        "unfiltered_assignment_path": cfg.model_selection_cache / f"cluster_assignments_{output_prefix}_unfiltered.parquet",
        "unfiltered_profile_path": cfg.outputs / "profiles" / f"tribe_profiles_{output_prefix}_unfiltered.parquet",
        "lift_evidence_path": lift_evidence_path,
        "lift_evidence_csv_path": lift_evidence_path.with_suffix(".csv"),
    }


def _two_stage_cache_metadata(
    umap_path: str | Path,
    settings: dict[str, Any],
    behavior_path: Path | None,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    return {
        "stage": "two_stage_hdbscan_lift_core",
        "mode": cfg.mode,
        "umap_path": file_fingerprint(umap_path),
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "behavior": file_fingerprint(behavior_path) if behavior_path else None,
        "output_prefix": settings["output_prefix"],
        "model_name": settings["model_name"],
        "algorithm_name": settings["algorithm_name"],
        "feature_space": settings["feature_space"],
        "trial_name": settings["trial_name"],
        "first_stage_hdbscan": settings["base"]["hdbscan_overrides"],
        "second_stage_hdbscan": settings["second_stage_overrides"],
        "lift_filter": settings["lift_filter"],
        "profiling": cfg.get("profiling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
        "assignment_schema_version": 2,
    }


def run_official_two_stage_hdbscan_first_pass(
    umap_path: str | Path,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """Run the first hard HDBSCAN pass on the full UMAP representation."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    settings = official_two_stage_hdbscan_settings(cfg)
    stage1_prefix = f"{settings['output_prefix']}_stage1"
    return run_hdbscan(
        umap_path,
        output_prefix=stage1_prefix,
        model_label="Model D",
        model_name=f"{settings['model_name']}_stage1",
        algorithm_name="UMAP_HDBSCAN_Stage1",
        feature_space=settings["feature_space"],
        trial_name=f"{settings['trial_name']}_stage1",
        variant_prefix=f"{settings['variant_prefix']}_stage1",
        scale_features=False,
        force=force,
        allow_noise_assignment=False,
        cfg=settings["first_stage_cfg"],
    )


def run_official_two_stage_hdbscan_second_pass(
    umap_path: str | Path,
    stage1_assignment_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, int, Path | None, Path | None, dict[str, Any]]:
    """Run the second stricter hard HDBSCAN pass on first-pass noise only."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    settings = official_two_stage_hdbscan_settings(cfg)
    paths = two_stage_hdbscan_artifact_paths(cfg)
    stage1_assignment = Path(stage1_assignment_path) if stage1_assignment_path else paths["stage1_assignment_path"]
    noise_feature_path = paths["stage2_noise_feature_path"]
    noise_rows = _write_stage2_noise_feature_subset(
        umap_path,
        stage1_assignment,
        noise_feature_path,
        cfg=cfg,
    )
    min_stage2_rows = max(int(settings["second_stage_cfg"].get("hdbscan.min_cluster_size", 2)), 2)
    if noise_rows < min_stage2_rows:
        log_event(
            "Stage 6.3 second-pass HDBSCAN",
            "skipping second pass because first-pass noise is below min_cluster_size",
            cfg=settings["second_stage_cfg"],
            noise_customers=noise_rows,
            min_cluster_size=min_stage2_rows,
        )
        return (
            noise_feature_path,
            noise_rows,
            None,
            None,
            {"cluster_count": 0, "noise_pct": 100.0 if noise_rows else 0.0, "assignment_path": None},
        )

    stage2_prefix = f"{settings['output_prefix']}_stage2_noise"
    stage2_assignment, stage2_results, stage2_best = run_hdbscan(
        noise_feature_path,
        output_prefix=stage2_prefix,
        model_label="Model D",
        model_name=f"{settings['model_name']}_stage2",
        algorithm_name="UMAP_HDBSCAN_Stage2Noise",
        feature_space=f"{settings['feature_space']}_stage1_noise",
        trial_name=f"{settings['trial_name']}_stage2_noise",
        variant_prefix=f"{settings['variant_prefix']}_stage2",
        scale_features=False,
        force=force,
        allow_noise_assignment=False,
        cfg=settings["second_stage_cfg"],
    )
    return noise_feature_path, noise_rows, stage2_assignment, stage2_results, stage2_best


def merge_official_two_stage_hdbscan_lift_core(
    umap_path: str | Path,
    *,
    stage1_assignment_path: str | Path | None = None,
    stage1_results_path: str | Path | None = None,
    stage2_assignment_path: str | Path | None = None,
    stage2_results_path: str | Path | None = None,
    stage2_noise_feature_path: str | Path | None = None,
    stage2_noise_customers: int | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """Merge first/second hard HDBSCAN passes, then retain product-lift-supported clusters."""

    from src.profiling import profile_tribes

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    settings = official_two_stage_hdbscan_settings(cfg)
    paths = two_stage_hdbscan_artifact_paths(cfg)
    assignment_path = paths["assignment_path"]
    results_path = paths["results_path"]
    unfiltered_assignment_path = paths["unfiltered_assignment_path"]
    noise_feature_path = Path(stage2_noise_feature_path) if stage2_noise_feature_path else paths["stage2_noise_feature_path"]
    unfiltered_profile_path = paths["unfiltered_profile_path"]
    lift_evidence_path = paths["lift_evidence_path"]
    lift_evidence_csv_path = paths["lift_evidence_csv_path"]
    stage1_assignment = Path(stage1_assignment_path) if stage1_assignment_path else paths["stage1_assignment_path"]
    stage1_results = Path(stage1_results_path) if stage1_results_path else paths["stage1_results_path"]
    stage2_assignment = Path(stage2_assignment_path) if stage2_assignment_path is not None else None
    stage2_results = Path(stage2_results_path) if stage2_results_path is not None else None
    behavior_path = _optional_behavior_feature_path(cfg)
    cache_metadata = _two_stage_cache_metadata(umap_path, settings, behavior_path, cfg)
    if (
        should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        and should_use_cache(results_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
    ):
        log_event("Stage 6.4 two-stage merge", "cache hit", cfg=cfg, results=results_path)
        result = pl.read_parquet(results_path).row(0, named=True)
        result["assignment_path"] = str(assignment_path)
        return assignment_path, results_path, result

    stage1_best = pl.read_parquet(stage1_results).row(0, named=True) if stage1_results.exists() else {}
    stage2_best = (
        pl.read_parquet(stage2_results).row(0, named=True)
        if stage2_results is not None and stage2_results.exists()
        else {"cluster_count": 0, "noise_pct": 100.0 if stage2_noise_customers else 0.0, "assignment_path": None}
    )
    noise_rows = (
        int(stage2_noise_customers)
        if stage2_noise_customers is not None
        else _row_count(noise_feature_path) if noise_feature_path.exists() else 0
    )
    variant = _two_stage_model_variant(settings)
    unfiltered_assignments = _merge_two_stage_hdbscan_assignments(
        stage1_assignment,
        stage2_assignment,
        model_name=settings["model_name"],
        model_variant=variant,
    )
    unfiltered_assignment_path.parent.mkdir(parents=True, exist_ok=True)
    unfiltered_assignments.write_parquet(unfiltered_assignment_path)
    write_artifact_metadata(unfiltered_assignment_path, cache_metadata)

    profile_path = profile_tribes(
        unfiltered_assignment_path,
        output_path=unfiltered_profile_path,
        force=force,
        enable_copurchase=False,
        enable_temporal=False,
        enable_loyalty=False,
        enable_llm=False,
        cfg=cfg,
    )
    lift_evidence = _product_lift_filter_evidence(pl.read_parquet(profile_path), settings["lift_filter"], cfg=cfg)
    lift_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    lift_evidence.write_parquet(lift_evidence_path)
    _lift_evidence_csv_frame(lift_evidence).write_csv(lift_evidence_csv_path)
    write_artifact_metadata(lift_evidence_path, cache_metadata)
    write_artifact_metadata(lift_evidence_csv_path, cache_metadata)

    final_assignments, lift_stats = _apply_lift_filter_to_assignments(unfiltered_assignments, lift_evidence)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    final_assignments.write_parquet(assignment_path)
    metrics, confidence_mean = _evaluate_assignment_on_umap(umap_path, final_assignments, cfg=settings["first_stage_cfg"])
    passes_gate, gate_reason = quality_gate_result(metrics, cfg=settings["first_stage_cfg"])
    unfiltered_summary = cluster_size_summary(unfiltered_assignments["tribe_id"].to_numpy().astype(np.int32))
    result = {
        "model": "Model D",
        "model_id": settings["model_name"],
        "model_name": settings["model_name"],
        "algorithm_name": settings["algorithm_name"],
        "model_variant": variant,
        "feature_space": settings["feature_space"],
        "trial_name": settings["trial_name"],
        "promoted_from_experiment": settings["promoted"].get("promoted_from_experiment"),
        "hdbscan_backend": "two_stage_run_hdbscan",
        "scale_features": False,
        "assignment_policy": "hard_two_stage_hdbscan_lift_core_noise_retained",
        "soft_assignment_enabled": False,
        "soft_assignment_strategy": None,
        "soft_assignment_distance_threshold": None,
        "soft_assigned_customers": 0,
        "soft_assigned_pct": 0.0,
        "soft_assignment_confidence_mean": None,
        "soft_assignment_confidence_p10": None,
        "soft_assignment_confidence_min": None,
        "stage1_assignment_path": str(stage1_assignment),
        "stage1_result_path": str(stage1_results),
        "stage1_cluster_count": int(stage1_best.get("cluster_count") or 0),
        "stage1_noise_pct": stage1_best.get("noise_pct"),
        "stage1_core_coverage_pct": stage1_best.get("core_coverage_pct")
        or (100.0 - float(stage1_best.get("noise_pct") or 0.0)),
        "stage2_noise_feature_path": str(noise_feature_path),
        "stage2_noise_customers": int(noise_rows),
        "stage2_assignment_path": str(stage2_assignment) if stage2_assignment else None,
        "stage2_result_path": str(stage2_results) if stage2_results else None,
        "stage2_cluster_count": int(stage2_best.get("cluster_count") or 0),
        "stage2_noise_pct_within_stage1_noise": stage2_best.get("noise_pct"),
        "unfiltered_assignment_path": str(unfiltered_assignment_path),
        "unfiltered_profile_path": str(profile_path),
        "lift_filter_evidence_path": str(lift_evidence_path),
        "lift_filter_evidence_csv_path": str(lift_evidence_csv_path),
        "unfiltered_cluster_count": int(unfiltered_summary.get("cluster_count") or 0),
        "unfiltered_noise_pct": unfiltered_summary.get("noise_pct"),
        "lift_supported_cluster_count": int(lift_stats["kept_cluster_count"]),
        "lift_rejected_cluster_count": int(lift_stats["rejected_cluster_count"]),
        "lift_rejected_customers": int(lift_stats["rejected_customer_count"]),
        "strong_product_lift_threshold": float(cfg.get("profiling.strong_product_lift_threshold", 1.5)),
        "min_strong_product_lifts": int(
            settings["lift_filter"].get(
                "min_strong_product_lifts",
                cfg.get("quality_gates.min_strong_product_lifts_per_cluster", 1),
            )
        ),
        "require_significant_product_lift": bool(
            settings["lift_filter"].get("require_significant_product_lift", False)
        ),
        **metrics,
        "avg_assignment_confidence": confidence_mean,
        "core_coverage_pct": metrics.get("coverage_pct"),
        "core_noise_pct": metrics.get("noise_pct"),
        "final_noise_pct": metrics.get("noise_pct"),
        "passes_quality_gate": passes_gate,
        "quality_gate_reason": gate_reason,
        "assignment_path": str(assignment_path),
        "selection_rank": 1,
        "selected_within_family": passes_gate,
        "stage6_flow": "6.1_umap_6.2_first_hdbscan_6.3_second_noise_hdbscan_6.4_merge_lift_filter",
    }
    pl.DataFrame([result]).write_parquet(results_path)
    write_artifact_metadata(assignment_path, cache_metadata)
    write_artifact_metadata(results_path, cache_metadata)
    log_event(
        "Stage 6.4 two-stage merge",
        "model evaluated",
        cfg=cfg,
        clusters=result["cluster_count"],
        noise_pct=result["noise_pct"],
        kept_clusters=result["lift_supported_cluster_count"],
        rejected_clusters=result["lift_rejected_cluster_count"],
    )
    return assignment_path, results_path, result


def run_official_two_stage_hdbscan_lift_core(
    umap_path: str | Path,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """Run hard HDBSCAN on all customers, rerun hard HDBSCAN on noise, then keep lifted clusters."""

    from src.profiling import profile_tribes

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    settings = official_two_stage_hdbscan_settings(cfg)
    output_prefix = settings["output_prefix"]
    assignment_path = cfg.model_selection_cache / f"cluster_assignments_{output_prefix}.parquet"
    results_path = cfg.model_selection_cache / f"{output_prefix}_results.parquet"
    unfiltered_assignment_path = cfg.model_selection_cache / f"cluster_assignments_{output_prefix}_unfiltered.parquet"
    noise_feature_path = cfg.model_selection_cache / f"{output_prefix}_stage2_noise_features.parquet"
    unfiltered_profile_path = cfg.outputs / "profiles" / f"tribe_profiles_{output_prefix}_unfiltered.parquet"
    lift_evidence_path = cfg.artifacts / "stage6" / f"{output_prefix}_lift_filter_evidence.parquet"
    lift_evidence_csv_path = lift_evidence_path.with_suffix(".csv")
    behavior_path = _optional_behavior_feature_path(cfg)
    cache_metadata = {
        "stage": "two_stage_hdbscan_lift_core",
        "mode": cfg.mode,
        "umap_path": file_fingerprint(umap_path),
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "behavior": file_fingerprint(behavior_path) if behavior_path else None,
        "output_prefix": output_prefix,
        "model_name": settings["model_name"],
        "algorithm_name": settings["algorithm_name"],
        "feature_space": settings["feature_space"],
        "trial_name": settings["trial_name"],
        "first_stage_hdbscan": settings["base"]["hdbscan_overrides"],
        "second_stage_hdbscan": settings["second_stage_overrides"],
        "lift_filter": settings["lift_filter"],
        "profiling": cfg.get("profiling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
        "assignment_schema_version": 2,
    }
    if (
        should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        and should_use_cache(results_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
    ):
        log_event("Stage 6.2 two-stage HDBSCAN", "cache hit", cfg=cfg, results=results_path)
        result = pl.read_parquet(results_path).row(0, named=True)
        result["assignment_path"] = str(assignment_path)
        return assignment_path, results_path, result

    with stage_timer(
        "Stage 6.2 two-stage HDBSCAN",
        "fitting hard density clusters and product-lift filter",
        cfg=settings["first_stage_cfg"],
        trial=settings["trial_name"],
    ):
        stage1_prefix = f"{output_prefix}_stage1"
        stage1_assignment, stage1_results, stage1_best = run_hdbscan(
            umap_path,
            output_prefix=stage1_prefix,
            model_label="Model D",
            model_name=f"{settings['model_name']}_stage1",
            algorithm_name="UMAP_HDBSCAN_Stage1",
            feature_space=settings["feature_space"],
            trial_name=f"{settings['trial_name']}_stage1",
            variant_prefix=f"{settings['variant_prefix']}_stage1",
            scale_features=False,
            force=force,
            allow_noise_assignment=False,
            cfg=settings["first_stage_cfg"],
        )

        noise_rows = _write_stage2_noise_feature_subset(
            umap_path,
            stage1_assignment,
            noise_feature_path,
            cfg=cfg,
        )
        min_stage2_rows = max(int(settings["second_stage_cfg"].get("hdbscan.min_cluster_size", 2)), 2)
        stage2_assignment: Path | None = None
        stage2_results: Path | None = None
        stage2_best: dict[str, Any] = {
            "cluster_count": 0,
            "noise_pct": 100.0 if noise_rows else 0.0,
            "assignment_path": None,
        }
        if noise_rows >= min_stage2_rows:
            stage2_prefix = f"{output_prefix}_stage2_noise"
            stage2_assignment, stage2_results, stage2_best = run_hdbscan(
                noise_feature_path,
                output_prefix=stage2_prefix,
                model_label="Model D",
                model_name=f"{settings['model_name']}_stage2",
                algorithm_name="UMAP_HDBSCAN_Stage2Noise",
                feature_space=f"{settings['feature_space']}_stage1_noise",
                trial_name=f"{settings['trial_name']}_stage2_noise",
                variant_prefix=f"{settings['variant_prefix']}_stage2",
                scale_features=False,
                force=force,
                allow_noise_assignment=False,
                cfg=settings["second_stage_cfg"],
            )
        else:
            log_event(
                "Stage 6.2 two-stage HDBSCAN",
                "skipping second pass because first-pass noise is below min_cluster_size",
                cfg=settings["second_stage_cfg"],
                noise_customers=noise_rows,
                min_cluster_size=min_stage2_rows,
            )

        variant = _two_stage_model_variant(settings)
        unfiltered_assignments = _merge_two_stage_hdbscan_assignments(
            stage1_assignment,
            stage2_assignment,
            model_name=settings["model_name"],
            model_variant=variant,
        )
        unfiltered_assignment_path.parent.mkdir(parents=True, exist_ok=True)
        unfiltered_assignments.write_parquet(unfiltered_assignment_path)
        write_artifact_metadata(unfiltered_assignment_path, cache_metadata)

        profile_path = profile_tribes(
            unfiltered_assignment_path,
            output_path=unfiltered_profile_path,
            force=force,
            enable_copurchase=False,
            enable_temporal=False,
            enable_loyalty=False,
            enable_llm=False,
            cfg=cfg,
        )
        lift_evidence = _product_lift_filter_evidence(pl.read_parquet(profile_path), settings["lift_filter"], cfg=cfg)
        lift_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        lift_evidence.write_parquet(lift_evidence_path)
        _lift_evidence_csv_frame(lift_evidence).write_csv(lift_evidence_csv_path)
        write_artifact_metadata(lift_evidence_path, cache_metadata)
        write_artifact_metadata(lift_evidence_csv_path, cache_metadata)

        final_assignments, lift_stats = _apply_lift_filter_to_assignments(unfiltered_assignments, lift_evidence)
        final_assignments.write_parquet(assignment_path)
        metrics, confidence_mean = _evaluate_assignment_on_umap(umap_path, final_assignments, cfg=settings["first_stage_cfg"])
        passes_gate, gate_reason = quality_gate_result(metrics, cfg=settings["first_stage_cfg"])
        unfiltered_summary = cluster_size_summary(unfiltered_assignments["tribe_id"].to_numpy().astype(np.int32))
        result = {
            "model": "Model D",
            "model_id": settings["model_name"],
            "model_name": settings["model_name"],
            "algorithm_name": settings["algorithm_name"],
            "model_variant": variant,
            "feature_space": settings["feature_space"],
            "trial_name": settings["trial_name"],
            "promoted_from_experiment": settings["promoted"].get("promoted_from_experiment"),
            "hdbscan_backend": "two_stage_run_hdbscan",
            "scale_features": False,
            "assignment_policy": "hard_two_stage_hdbscan_lift_core_noise_retained",
            "soft_assignment_enabled": False,
            "soft_assignment_strategy": None,
            "soft_assignment_distance_threshold": None,
            "soft_assigned_customers": 0,
            "soft_assigned_pct": 0.0,
            "soft_assignment_confidence_mean": None,
            "soft_assignment_confidence_p10": None,
            "soft_assignment_confidence_min": None,
            "stage1_assignment_path": str(stage1_assignment),
            "stage1_result_path": str(stage1_results),
            "stage1_cluster_count": int(stage1_best.get("cluster_count") or 0),
            "stage1_noise_pct": stage1_best.get("noise_pct"),
            "stage1_core_coverage_pct": stage1_best.get("core_coverage_pct")
            or (100.0 - float(stage1_best.get("noise_pct") or 0.0)),
            "stage2_noise_feature_path": str(noise_feature_path),
            "stage2_noise_customers": int(noise_rows),
            "stage2_assignment_path": str(stage2_assignment) if stage2_assignment else None,
            "stage2_result_path": str(stage2_results) if stage2_results else None,
            "stage2_cluster_count": int(stage2_best.get("cluster_count") or 0),
            "stage2_noise_pct_within_stage1_noise": stage2_best.get("noise_pct"),
            "unfiltered_assignment_path": str(unfiltered_assignment_path),
            "unfiltered_profile_path": str(profile_path),
            "lift_filter_evidence_path": str(lift_evidence_path),
            "lift_filter_evidence_csv_path": str(lift_evidence_csv_path),
            "unfiltered_cluster_count": int(unfiltered_summary.get("cluster_count") or 0),
            "unfiltered_noise_pct": unfiltered_summary.get("noise_pct"),
            "lift_supported_cluster_count": int(lift_stats["kept_cluster_count"]),
            "lift_rejected_cluster_count": int(lift_stats["rejected_cluster_count"]),
            "lift_rejected_customers": int(lift_stats["rejected_customer_count"]),
            "strong_product_lift_threshold": float(cfg.get("profiling.strong_product_lift_threshold", 1.5)),
            "min_strong_product_lifts": int(
                settings["lift_filter"].get(
                    "min_strong_product_lifts",
                    cfg.get("quality_gates.min_strong_product_lifts_per_cluster", 1),
                )
            ),
            "require_significant_product_lift": bool(
                settings["lift_filter"].get("require_significant_product_lift", False)
            ),
            **metrics,
            "avg_assignment_confidence": confidence_mean,
            "core_coverage_pct": metrics.get("coverage_pct"),
            "core_noise_pct": metrics.get("noise_pct"),
            "final_noise_pct": metrics.get("noise_pct"),
            "passes_quality_gate": passes_gate,
            "quality_gate_reason": gate_reason,
            "assignment_path": str(assignment_path),
            "selection_rank": 1,
            "selected_within_family": passes_gate,
            "stage6_flow": "6.1_umap_representation_then_6.2_two_stage_hard_hdbscan_lift_core",
        }

    pl.DataFrame([result]).write_parquet(results_path)
    write_artifact_metadata(assignment_path, cache_metadata)
    write_artifact_metadata(results_path, cache_metadata)
    log_event(
        "Stage 6.2 two-stage HDBSCAN",
        "model evaluated",
        cfg=cfg,
        clusters=result["cluster_count"],
        noise_pct=result["noise_pct"],
        kept_clusters=result["lift_supported_cluster_count"],
        rejected_clusters=result["lift_rejected_cluster_count"],
    )
    return assignment_path, results_path, result


def _optional_behavior_feature_path(cfg: PipelineConfig) -> Path | None:
    try:
        return cfg.artifact_path("behavioral_features", "output", directory=cfg.outputs / "features")
    except KeyError:
        return None


def _two_stage_model_variant(settings: dict[str, Any]) -> str:
    first_cfg = settings["first_stage_cfg"]
    second_cfg = settings["second_stage_cfg"]
    return (
        f"{settings['variant_prefix']}_"
        f"stage1_mcs{first_cfg.get('hdbscan.min_cluster_size')}_ms{first_cfg.get('hdbscan.min_samples')}_"
        f"{first_cfg.get('hdbscan.cluster_selection_method')}_"
        f"stage2_mcs{second_cfg.get('hdbscan.min_cluster_size')}_ms{second_cfg.get('hdbscan.min_samples')}_"
        f"{second_cfg.get('hdbscan.cluster_selection_method')}_lift"
    )


def _write_stage2_noise_feature_subset(
    umap_path: str | Path,
    stage1_assignment_path: str | Path,
    output_path: str | Path,
    cfg: PipelineConfig,
) -> int:
    output = Path(output_path)
    umap = pl.read_parquet(umap_path).sort("cliente")
    feature_cols = numeric_feature_columns(umap)
    assignments = (
        pl.read_parquet(stage1_assignment_path)
        .select(["cliente", "tribe_id"])
        .unique(subset=["cliente"], keep="first")
    )
    noise_features = (
        umap.join(assignments, on="cliente", how="left")
        .with_columns(pl.col("tribe_id").fill_null(-1).cast(pl.Int32))
        .filter(pl.col("tribe_id") < 0)
        .select(["cliente", *feature_cols])
        .sort("cliente")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    noise_features.write_parquet(output)
    write_artifact_metadata(
        output,
        {
            "stage": "two_stage_hdbscan_noise_feature_subset",
            "mode": cfg.mode,
            "umap_path": file_fingerprint(umap_path),
            "stage1_assignment_path": file_fingerprint(stage1_assignment_path),
        },
    )
    return noise_features.height


def _assignment_stage_frame(path: str | Path | None, prefix: str) -> pl.DataFrame:
    schema = {
        "cliente": pl.Utf8,
        f"{prefix}_tribe_id": pl.Int32,
        f"{prefix}_assignment_probability": pl.Float32,
        f"{prefix}_assignment_confidence_score": pl.Float32,
        f"{prefix}_assignment_confidence_type": pl.Utf8,
        f"{prefix}_assignment_source": pl.Utf8,
    }
    if path is None:
        return pl.DataFrame(schema=schema)
    assignments = pl.read_parquet(path).unique(subset=["cliente"], keep="first")
    column_specs = {
        "tribe_id": pl.Int32,
        "assignment_probability": pl.Float32,
        "assignment_confidence_score": pl.Float32,
        "assignment_confidence_type": pl.Utf8,
        "assignment_source": pl.Utf8,
    }
    exprs = [pl.col("cliente").cast(pl.Utf8)]
    for column, dtype in column_specs.items():
        expr = pl.col(column) if column in assignments.columns else pl.lit(None).cast(dtype)
        exprs.append(expr.cast(dtype).alias(f"{prefix}_{column}"))
    return assignments.select(exprs).sort("cliente")


def _label_mapping(labels: list[int], offset: int = 0) -> pl.DataFrame:
    valid_labels = sorted({int(label) for label in labels if int(label) >= 0})
    if not valid_labels:
        return pl.DataFrame(schema={"source_tribe_id": pl.Int32, "final_tribe_id": pl.Int32})
    return pl.DataFrame(
        {
            "source_tribe_id": valid_labels,
            "final_tribe_id": [offset + idx for idx in range(len(valid_labels))],
        },
        schema={"source_tribe_id": pl.Int32, "final_tribe_id": pl.Int32},
    )


def _merge_two_stage_hdbscan_assignments(
    stage1_assignment_path: str | Path,
    stage2_assignment_path: str | Path | None,
    *,
    model_name: str,
    model_variant: str,
) -> pl.DataFrame:
    stage1 = _assignment_stage_frame(stage1_assignment_path, "stage1")
    stage2 = _assignment_stage_frame(stage2_assignment_path, "stage2")
    stage1_labels = stage1["stage1_tribe_id"].to_list() if "stage1_tribe_id" in stage1.columns else []
    stage2_labels = stage2["stage2_tribe_id"].to_list() if "stage2_tribe_id" in stage2.columns else []
    stage1_map = _label_mapping(stage1_labels, offset=0).rename(
        {"source_tribe_id": "stage1_tribe_id", "final_tribe_id": "stage1_final_tribe_id"}
    )
    stage2_offset = stage1_map.height
    stage2_map = _label_mapping(stage2_labels, offset=stage2_offset).rename(
        {"source_tribe_id": "stage2_tribe_id", "final_tribe_id": "stage2_final_tribe_id"}
    )
    joined = stage1.join(stage2, on="cliente", how="left").join(stage1_map, on="stage1_tribe_id", how="left")
    if stage2_map.height:
        joined = joined.join(stage2_map, on="stage2_tribe_id", how="left")
    else:
        joined = joined.with_columns(pl.lit(None).cast(pl.Int32).alias("stage2_final_tribe_id"))

    stage1_core = pl.col("stage1_final_tribe_id").is_not_null()
    stage2_core = pl.col("stage2_final_tribe_id").is_not_null()
    return (
        joined.with_columns(
            [
                pl.when(stage1_core)
                .then(pl.col("stage1_final_tribe_id"))
                .when(stage2_core)
                .then(pl.col("stage2_final_tribe_id"))
                .otherwise(pl.lit(-1))
                .cast(pl.Int32)
                .alias("tribe_id"),
                pl.lit(model_name).alias("model_name"),
                pl.lit(model_variant).alias("model_variant"),
                pl.when(stage1_core)
                .then(pl.col("stage1_assignment_probability"))
                .when(stage2_core)
                .then(pl.col("stage2_assignment_probability"))
                .otherwise(pl.lit(None).cast(pl.Float32))
                .alias("assignment_probability"),
                pl.when(stage1_core)
                .then(pl.col("stage1_assignment_confidence_score"))
                .when(stage2_core)
                .then(pl.col("stage2_assignment_confidence_score"))
                .otherwise(pl.lit(None).cast(pl.Float32))
                .alias("assignment_confidence_score"),
                pl.when(stage1_core)
                .then(pl.col("stage1_assignment_confidence_type"))
                .when(stage2_core)
                .then(pl.col("stage2_assignment_confidence_type"))
                .otherwise(pl.lit(None).cast(pl.Utf8))
                .alias("assignment_confidence_type"),
                pl.when(stage1_core)
                .then(pl.lit("two_stage_hdbscan_stage1_core"))
                .when(stage2_core)
                .then(pl.lit("two_stage_hdbscan_stage2_noise_core"))
                .otherwise(pl.lit("two_stage_hdbscan_noise_unassigned"))
                .alias("assignment_source"),
            ]
        )
        .select(
            [
                "cliente",
                "tribe_id",
                "model_name",
                "model_variant",
                "assignment_probability",
                "assignment_confidence_score",
                "assignment_confidence_type",
                "assignment_source",
            ]
        )
        .sort("cliente")
    )


def _product_lift_filter_evidence(
    profiles: pl.DataFrame,
    lift_filter: dict[str, Any],
    cfg: PipelineConfig,
) -> pl.DataFrame:
    threshold = float(cfg.get("profiling.strong_product_lift_threshold", 1.5))
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    min_strong = int(
        lift_filter.get("min_strong_product_lifts", cfg.get("quality_gates.min_strong_product_lifts_per_cluster", 1))
    )
    require_significant = bool(lift_filter.get("require_significant_product_lift", False))
    rows = []
    for row in profiles.iter_rows(named=True):
        lifts = [float(value) for value in (row.get("top_product_lifts") or []) if value is not None]
        q_values = row.get("top_product_q_values") or []
        strong_indices = [idx for idx, value in enumerate(lifts) if value >= threshold]
        significant_count = 0
        for idx in strong_indices:
            if idx < len(q_values) and q_values[idx] is not None and float(q_values[idx]) <= q_threshold:
                significant_count += 1
        evidence_count = significant_count if require_significant else len(strong_indices)
        rows.append(
            {
                "tribe_id": int(row.get("tribe_id")),
                "n_customers": int(row.get("n_customers") or 0),
                "strong_product_lift_threshold": threshold,
                "significance_q_threshold": q_threshold,
                "min_strong_product_lifts": min_strong,
                "require_significant_product_lift": require_significant,
                "strong_product_lift_count": len(strong_indices),
                "significant_strong_product_lift_count": significant_count,
                "passes_lift_filter": evidence_count >= min_strong,
                "top_product_ids": row.get("top_product_ids") or [],
                "top_products": row.get("top_products") or [],
                "top_product_lifts": row.get("top_product_lifts") or [],
                "top_product_q_values": row.get("top_product_q_values") or [],
            }
        )
    if rows:
        return pl.from_dicts(rows, infer_schema_length=None).sort("tribe_id")
    return pl.DataFrame(
        schema={
            "tribe_id": pl.Int32,
            "n_customers": pl.Int64,
            "strong_product_lift_threshold": pl.Float64,
            "significance_q_threshold": pl.Float64,
            "min_strong_product_lifts": pl.Int64,
            "require_significant_product_lift": pl.Boolean,
            "strong_product_lift_count": pl.Int64,
            "significant_strong_product_lift_count": pl.Int64,
            "passes_lift_filter": pl.Boolean,
            "top_product_ids": pl.List(pl.Utf8),
            "top_products": pl.List(pl.Utf8),
            "top_product_lifts": pl.List(pl.Float64),
            "top_product_q_values": pl.List(pl.Float64),
        }
    )


def _lift_evidence_csv_frame(evidence: pl.DataFrame) -> pl.DataFrame:
    columns = [
        "tribe_id",
        "n_customers",
        "strong_product_lift_threshold",
        "min_strong_product_lifts",
        "require_significant_product_lift",
        "strong_product_lift_count",
        "significant_strong_product_lift_count",
        "passes_lift_filter",
    ]
    return evidence.select([column for column in columns if column in evidence.columns])


def _apply_lift_filter_to_assignments(
    assignments: pl.DataFrame,
    lift_evidence: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    cluster_ids = sorted({int(label) for label in assignments["tribe_id"].to_list() if int(label) >= 0})
    kept_ids = (
        sorted(
            int(label)
            for label in lift_evidence.filter(pl.col("passes_lift_filter")).get_column("tribe_id").to_list()
        )
        if "passes_lift_filter" in lift_evidence.columns and lift_evidence.height
        else []
    )
    kept_ids = [label for label in kept_ids if label in cluster_ids]
    rejected_ids = sorted(set(cluster_ids) - set(kept_ids))
    mapping = (
        pl.DataFrame(
            {"tribe_id": kept_ids, "lift_filtered_tribe_id": list(range(len(kept_ids)))},
            schema={"tribe_id": pl.Int32, "lift_filtered_tribe_id": pl.Int32},
        )
        if kept_ids
        else pl.DataFrame(schema={"tribe_id": pl.Int32, "lift_filtered_tribe_id": pl.Int32})
    )
    joined = assignments.join(mapping, on="tribe_id", how="left")
    kept = pl.col("lift_filtered_tribe_id").is_not_null()
    clustered = pl.col("tribe_id") >= 0
    final = (
        joined.with_columns(
            [
                pl.when(kept)
                .then(pl.col("lift_filtered_tribe_id"))
                .otherwise(pl.lit(-1))
                .cast(pl.Int32)
                .alias("final_tribe_id"),
                pl.when(kept)
                .then(pl.col("assignment_probability"))
                .otherwise(pl.lit(None).cast(pl.Float32))
                .alias("final_assignment_probability"),
                pl.when(kept)
                .then(pl.col("assignment_confidence_score"))
                .otherwise(pl.lit(None).cast(pl.Float32))
                .alias("final_assignment_confidence_score"),
                pl.when(kept)
                .then(pl.col("assignment_confidence_type"))
                .otherwise(pl.lit(None).cast(pl.Utf8))
                .alias("final_assignment_confidence_type"),
                pl.when(kept)
                .then(pl.col("assignment_source"))
                .when(clustered)
                .then(pl.lit("two_stage_hdbscan_lift_rejected"))
                .otherwise(pl.col("assignment_source"))
                .alias("final_assignment_source"),
            ]
        )
        .select(
            [
                pl.col("cliente"),
                pl.col("final_tribe_id").alias("tribe_id"),
                pl.col("model_name"),
                pl.col("model_variant"),
                pl.col("final_assignment_probability").alias("assignment_probability"),
                pl.col("final_assignment_confidence_score").alias("assignment_confidence_score"),
                pl.col("final_assignment_confidence_type").alias("assignment_confidence_type"),
                pl.col("final_assignment_source").alias("assignment_source"),
            ]
        )
        .sort("cliente")
    )
    rejected_customers = (
        assignments.filter(pl.col("tribe_id").is_in(rejected_ids)).height if rejected_ids else 0
    )
    return final, {
        "kept_cluster_count": len(kept_ids),
        "rejected_cluster_count": len(rejected_ids),
        "rejected_customer_count": int(rejected_customers),
    }


def _evaluate_assignment_on_umap(
    umap_path: str | Path,
    assignments: pl.DataFrame,
    cfg: PipelineConfig,
) -> tuple[dict[str, Any], float | None]:
    umap = pl.read_parquet(umap_path).sort("cliente")
    feature_cols = numeric_feature_columns(umap)
    aligned = umap.join(assignments.select(["cliente", "tribe_id", "assignment_confidence_score"]), on="cliente", how="inner")
    aligned = aligned.sort("cliente")
    X = frame_to_numpy(aligned, feature_cols)
    labels = aligned["tribe_id"].to_numpy().astype(np.int32)
    confidence_values = None
    confidence_mean = None
    if "assignment_confidence_score" in aligned.columns:
        confidence_values = aligned["assignment_confidence_score"].to_numpy().astype(np.float32)
        finite = confidence_values[np.isfinite(confidence_values)]
        confidence_mean = float(np.mean(finite)) if finite.size else None
        if not finite.size:
            confidence_values = None
    metrics = evaluate_labels(X, labels, probabilities=confidence_values, cfg=cfg)
    return metrics, confidence_mean


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
        "stage6_flow": best.get("stage6_flow", "hard_umap_hdbscan_core"),
    }
    if umap_path is not None:
        suite["umap_path"] = Path(umap_path)
    return suite


def _stage6_pca_summary_fields(cfg: PipelineConfig) -> dict[str, Any]:
    summary_path = _official_pca_for_umap_summary_path(cfg)
    fields: dict[str, Any] = {
        "pca_pre_reduction_summary_status": "missing",
        "pca_summary_path": str(summary_path),
        "pca_purpose": None,
        "pca_dimension_reduction": None,
        "pca_input_dimension_count": None,
        "pca_requested_component_count": None,
        "pca_retained_component_count": None,
        "pca_retained_variance_pct": None,
        "pca_pc1_variance_pct": None,
        "pca_pc5_cumulative_variance_pct": None,
        "pca_pc10_cumulative_variance_pct": None,
    }
    if not summary_path.exists():
        return fields

    try:
        summary = pl.read_csv(summary_path).row(0, named=True)
    except Exception as exc:  # pragma: no cover - defensive metadata read path.
        fields["pca_pre_reduction_summary_status"] = f"unreadable:{exc.__class__.__name__}"
        return fields

    fields.update(
        {
            "pca_pre_reduction_summary_status": "present",
            "pca_purpose": summary.get("purpose"),
            "pca_dimension_reduction": summary.get("dimension_reduction"),
            "pca_input_dimension_count": _optional_int(summary.get("input_dimension_count")),
            "pca_requested_component_count": _optional_int(summary.get("requested_component_count")),
            "pca_retained_component_count": _optional_int(summary.get("retained_component_count")),
            "pca_retained_variance_pct": _optional_float(summary.get("retained_variance_pct")),
            "pca_pc1_variance_pct": _optional_float(summary.get("pc1_variance_pct")),
            "pca_pc5_cumulative_variance_pct": _optional_float(summary.get("pc5_cumulative_variance_pct")),
            "pca_pc10_cumulative_variance_pct": _optional_float(summary.get("pc10_cumulative_variance_pct")),
        }
    )
    return fields


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


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
    pca_summary_fields = _stage6_pca_summary_fields(cfg)

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
        **pca_summary_fields,
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
    stage_name: str = "6.2_hard_hdbscan",
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
    hard_core_policies = {
        "hard_hdbscan_core_noise_retained",
        "hard_two_stage_hdbscan_lift_core_noise_retained",
    }

    blocking_issues = []
    if umap_rows != assignment_rows:
        blocking_issues.append("row_count_mismatch")
    if missing_umap_customers:
        blocking_issues.append("missing_umap_customers")
    if extra_assignment_customers:
        blocking_issues.append("extra_assignment_customers")
    if duplicate_customers:
        blocking_issues.append("duplicate_assignment_customers")
    if result.get("assignment_policy") not in hard_core_policies:
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
        "stage": stage_name,
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
    """Write Stage 6.5 evidence for UMAP retention and HDBSCAN cluster validity.

    UMAP does not expose PCA-style explained variance, so this diagnostic measures
    whether local neighborhoods and sampled distance ordering survive reduction.
    HDBSCAN validity is reported with density-aware DBCV when available; silhouette
    remains a supporting core-only compactness metric.
    """

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.artifacts / "stage6" / "stage6_5_representation_cluster_quality.csv"
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
        "stage": "6.5_representation_cluster_quality",
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
            "Stage 6.6 tests perturbation stability and cluster readiness. For final sign-off, prefer repeated-seed "
            "or bootstrap reruns of the promoted recipe."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_csv(output)
    log_event("Stage 6.5 diagnostics", "wrote representation and cluster quality evidence", cfg=cfg, path=output)
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
    result["hdbscan_dbcv_status"] = "computed"
    result["hdbscan_dbcv_note"] = (
        "DBCV computation completed. DBCV is density-aware and ranges from -1 to 1; higher is better."
    )
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
