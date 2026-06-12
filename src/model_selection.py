"""Orchestration helpers for candidate model comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.autoencoder import build_autoencoder_latents
from src.clustering import run_gmm_grid, run_hdbscan, run_pca_kmeans_grid
from src.config import CONFIG, PipelineConfig
from src.dimensionality import build_umap_representation
from src.progress import log_event, stage_timer


def run_candidate_model_suite(
    feature_path: str | Path,
    force: bool | None = None,
    include_autoencoder: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Run candidate clustering approaches and return their artifacts."""

    candidate_results: list[dict[str, Any]] = []
    assignment_paths: dict[str, Path] = {}
    result_paths: dict[str, Path] = {}

    with stage_timer("Stage 6 model suite", "running candidate model suite", cfg=cfg, feature_path=feature_path):
        log_event("Stage 6 model suite", "starting Model A GMM", cfg=cfg)
        assignment, results, best = run_gmm_grid(
            feature_path,
            output_prefix="model_a_gmm",
            model_label="Model A",
            model_name="model_a_gmm",
            algorithm_name="GaussianMixture",
            feature_space="raw_customer_embeddings",
            force=force,
            cfg=cfg,
        )
        candidate_results.append(best)
        assignment_paths[_candidate_key(best)] = assignment
        result_paths[_candidate_key(best)] = results

        log_event("Stage 6 model suite", "starting Model B HDBSCAN", cfg=cfg)
        assignment, results, best = run_hdbscan(
            feature_path,
            output_prefix="model_b_hdbscan",
            model_label="Model B",
            model_name="model_b_hdbscan",
            algorithm_name="HDBSCAN",
            feature_space="raw_customer_embeddings",
            force=force,
            cfg=cfg,
        )
        candidate_results.append(best)
        assignment_paths[_candidate_key(best)] = assignment
        result_paths[_candidate_key(best)] = results

        if cfg.get("umap.enabled", True):
            log_event("Stage 6 model suite", "starting Model C UMAP representation", cfg=cfg)
            umap_path = build_umap_representation(feature_path, force=force, cfg=cfg)
            log_event("Stage 6 model suite", "starting Model C UMAP-HDBSCAN", cfg=cfg)
            assignment, results, best = run_hdbscan(
                umap_path,
                output_prefix="model_c_umap_hdbscan",
                model_label="Model C",
                model_name="model_c_umap_hdbscan",
                algorithm_name="UMAP_HDBSCAN",
                feature_space="umap_customer_embeddings",
                variant_prefix="umap",
                scale_features=False,
                force=force,
                cfg=cfg,
            )
            candidate_results.append(best)
            assignment_paths[_candidate_key(best)] = assignment
            result_paths[_candidate_key(best)] = results
        else:
            log_event("Stage 6 model suite", "UMAP-HDBSCAN candidate disabled", cfg=cfg)

        run_autoencoder = cfg.get("autoencoder.enabled", True) if include_autoencoder is None else include_autoencoder
        if run_autoencoder:
            for latent_size in cfg.get("autoencoder.latent_sizes", [8, 16, 32]):
                log_event("Stage 6 model suite", "starting Model D autoencoder", cfg=cfg, latent_size=latent_size)
                latent_path = build_autoencoder_latents(feature_path, int(latent_size), force=force, cfg=cfg)
                log_event("Stage 6 model suite", "starting Model D autoencoder-GMM", cfg=cfg, latent_size=latent_size)
                assignment, results, best = run_gmm_grid(
                    latent_path,
                    output_prefix=f"model_d_autoencoder_gmm_ae{latent_size}",
                    model_label="Model D",
                    model_name="model_d_autoencoder_gmm",
                    algorithm_name="Autoencoder_GaussianMixture",
                    feature_space=f"autoencoder_latent_{latent_size}",
                    variant_prefix=f"ae{latent_size}",
                    force=force,
                    cfg=cfg,
                )
                candidate_results.append(best)
                assignment_paths[_candidate_key(best)] = assignment
                result_paths[_candidate_key(best)] = results
        else:
            log_event("Stage 6 model suite", "autoencoder candidates disabled", cfg=cfg)

        log_event("Stage 6 model suite", "starting Model E PCA-KMeans", cfg=cfg)
        assignment, results, best = run_pca_kmeans_grid(
            feature_path,
            output_prefix="model_e_pca_kmeans",
            model_label="Model E",
            model_name="model_e_pca_kmeans",
            algorithm_name="PCA_MiniBatchKMeans",
            force=force,
            cfg=cfg,
        )
        candidate_results.append(best)
        assignment_paths[_candidate_key(best)] = assignment
        result_paths[_candidate_key(best)] = results
        log_event("Stage 6 model suite", "candidate suite complete", cfg=cfg, candidates=len(candidate_results))

    return {
        "candidate_results": candidate_results,
        "assignment_paths": assignment_paths,
        "result_paths": result_paths,
    }


def _candidate_key(result: dict[str, Any]) -> str:
    return f"{result.get('model_name')}::{result.get('model_variant')}"


def build_candidate_model_diagnostics(
    model_suite: dict[str, Any],
    output_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Combine all Stage 6 grid/single-model rows into one diagnostics artifact."""

    cfg.ensure_directories()
    result_paths = model_suite.get("result_paths", {})
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
            item["source_result_path"] = str(path)
            item["stage6_score"] = _as_float(
                item.get("coverage_adjusted_silhouette"),
                default=_as_float(item.get("silhouette"), default=-999.0),
            )
            rows.append(item)

    ranked = sorted(rows, key=_diagnostic_rank_key)
    for rank, row in enumerate(ranked, start=1):
        row["stage6_rank"] = rank

    output = Path(output_path) if output_path else cfg.outputs / "model_selection" / "stage6_candidate_model_diagnostics.parquet"
    csv_output = Path(csv_path) if csv_path else cfg.outputs / "model_selection" / "stage6_candidate_model_diagnostics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = pl.from_dicts(ranked, infer_schema_length=None) if ranked else pl.DataFrame()
    diagnostics.write_parquet(output)
    diagnostics.write_csv(csv_output)
    log_event(
        "Stage 6 diagnostics",
        "wrote candidate diagnostics",
        cfg=cfg,
        candidates=len(ranked),
        parquet=output,
        csv=csv_output,
    )
    return {"parquet": output, "csv": csv_output}


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
