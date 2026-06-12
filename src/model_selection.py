"""Orchestration helpers for candidate model comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.autoencoder import build_autoencoder_latents
from src.clustering import run_gmm_grid, run_hdbscan, run_pca_kmeans_grid
from src.config import CONFIG, PipelineConfig


def run_candidate_model_suite(
    feature_path: str | Path,
    force: bool | None = None,
    include_autoencoder: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Run the four requested candidate approaches and return their artifacts."""

    candidate_results: list[dict[str, Any]] = []
    assignment_paths: dict[str, Path] = {}
    result_paths: dict[str, Path] = {}

    assignment, results, best = run_gmm_grid(
        feature_path,
        output_prefix="model_a_gmm",
        model_label="Model A",
        model_name="GMM",
        feature_space="raw_customer_embeddings",
        force=force,
        cfg=cfg,
    )
    candidate_results.append(best)
    assignment_paths[_candidate_key(best)] = assignment
    result_paths[_candidate_key(best)] = results

    assignment, results, best = run_hdbscan(
        feature_path,
        output_prefix="model_b_hdbscan",
        force=force,
        cfg=cfg,
    )
    candidate_results.append(best)
    assignment_paths[_candidate_key(best)] = assignment
    result_paths[_candidate_key(best)] = results

    run_autoencoder = cfg.get("autoencoder.enabled", True) if include_autoencoder is None else include_autoencoder
    if run_autoencoder:
        for latent_size in cfg.get("autoencoder.latent_sizes", [8, 16, 32]):
            latent_path = build_autoencoder_latents(feature_path, int(latent_size), force=force, cfg=cfg)
            assignment, results, best = run_gmm_grid(
                latent_path,
                output_prefix=f"model_c_ae{latent_size}_gmm",
                model_label="Model C",
                model_name="AE_GMM",
                feature_space=f"autoencoder_latent_{latent_size}",
                variant_prefix=f"ae{latent_size}",
                force=force,
                cfg=cfg,
            )
            candidate_results.append(best)
            assignment_paths[_candidate_key(best)] = assignment
            result_paths[_candidate_key(best)] = results

    assignment, results, best = run_pca_kmeans_grid(
        feature_path,
        output_prefix="benchmark_pca_kmeans",
        force=force,
        cfg=cfg,
    )
    candidate_results.append(best)
    assignment_paths[_candidate_key(best)] = assignment
    result_paths[_candidate_key(best)] = results

    return {
        "candidate_results": candidate_results,
        "assignment_paths": assignment_paths,
        "result_paths": result_paths,
    }


def _candidate_key(result: dict[str, Any]) -> str:
    return f"{result.get('model_name')}::{result.get('model_variant')}"
