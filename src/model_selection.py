"""Orchestration helpers for candidate model comparison."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl

from src.autoencoder import build_autoencoder_latents
from src.clustering import run_gmm_grid, run_hdbscan, run_pca_kmeans_grid
from src.config import CONFIG, PipelineConfig
from src.dimensionality import build_umap_representation
from src.experiment_reporting import write_summary_artifacts
from src.progress import log_event, stage_timer


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
            "cluster_count",
            "stage6_score",
            "coverage_adjusted_silhouette",
            "silhouette",
            "davies_bouldin",
            "noise_pct",
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
