"""Cluster validity and lightweight stability diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.evaluation import cluster_size_summary
from src.experiment_reporting import write_summary_artifacts
from src.progress import log_event, stage_timer
from src.utils import deterministic_sample_indices, frame_to_numpy, numeric_feature_columns


def build_cluster_validity_stability_report(
    model_suite: dict[str, Any],
    feature_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write a compact report on candidate validity and label-geometry stability.

    This is intentionally not a substitute for a full repeated-run stability study.
    It checks whether each candidate's labels remain recoverable from the original
    feature geometry after small feature perturbations.
    """

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.model_selection / "cluster_validity_stability.parquet"
    sample_size = int(cfg.get("stability.sample_size", 10000))
    repeats = int(cfg.get("stability.repeats", 5))
    jitter_scale = float(cfg.get("stability.jitter_scale", 0.02))

    with stage_timer(
        "Cluster validity",
        "building validity and stability report",
        cfg=cfg,
        feature_path=feature_path,
        repeats=repeats,
    ):
        features = pl.read_parquet(feature_path).sort("cliente")
        feature_cols = numeric_feature_columns(features)
        X = frame_to_numpy(features, feature_cols)
        X = _standardize(X)
        base = features.select("cliente").with_columns(pl.Series("_row_idx", np.arange(features.height)))

        rows = []
        for candidate_key, assignment_path in sorted((model_suite.get("assignment_paths") or {}).items()):
            assignments = (
                pl.read_parquet(assignment_path)
                .select(["cliente", "tribe_id", "model_name", "model_variant"])
                .unique(subset=["cliente"], keep="first")
            )
            joined = base.join(assignments, on="cliente", how="inner").sort("_row_idx")
            labels = joined["tribe_id"].to_numpy().astype(int)
            row_indices = joined["_row_idx"].to_numpy().astype(int)
            metrics = cluster_size_summary(labels)
            stability = _jitter_stability(
                X[row_indices],
                labels,
                sample_size=sample_size,
                repeats=repeats,
                jitter_scale=jitter_scale,
                seed=cfg.random_seed,
            )
            rows.append(
                {
                    "candidate_id": candidate_key,
                    "model_name": joined["model_name"][0] if joined.height else None,
                    "model_variant": joined["model_variant"][0] if joined.height else None,
                    "rows_evaluated": int(joined.height),
                    "feature_count": int(len(feature_cols)),
                    "stability_sample_size": int(min(sample_size, max(int(np.sum(labels >= 0)), 0))),
                    "stability_repeats": repeats,
                    "jitter_scale": jitter_scale,
                    **metrics,
                    **stability,
                    "validity_note": (
                        "Checks internal geometry and perturbation sensitivity. "
                        "A final submission should still include repeated-seed or bootstrap model reruns."
                    ),
                }
            )

    report = pl.DataFrame(rows).sort("candidate_id") if rows else pl.DataFrame()
    output.parent.mkdir(parents=True, exist_ok=True)
    report.write_parquet(output)
    summaries = write_summary_artifacts(
        report,
        output_base=output,
        title="Cluster Validity And Stability Diagnostics",
        priority_columns=[
            "candidate_id",
            "cluster_count",
            "noise_pct",
            "cluster_size_cv",
            "jitter_ari_mean",
            "jitter_ari_std",
            "jitter_label_recovery_accuracy_mean",
            "stability_sample_size",
            "validity_note",
        ],
        cfg=cfg,
        write_markdown=bool(cfg.get("model_selection.write_summary_markdown", True)),
    )
    log_event("Cluster validity", "wrote validity and stability report", cfg=cfg, path=output)
    return {"parquet": output, **summaries}


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X.astype(np.float32, copy=False), copy=False)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return ((X - mean) / np.maximum(std, 1e-12)).astype(np.float32)


def _jitter_stability(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    repeats: int,
    jitter_scale: float,
    seed: int,
) -> dict[str, float | None]:
    from sklearn.metrics import adjusted_rand_score

    valid_idx = np.where(labels >= 0)[0]
    if len(valid_idx) == 0 or len(np.unique(labels[valid_idx])) < 2:
        return {
            "jitter_ari_mean": None,
            "jitter_ari_std": None,
            "jitter_label_recovery_accuracy_mean": None,
            "jitter_label_recovery_accuracy_std": None,
        }

    aris: list[float] = []
    accuracies: list[float] = []
    chosen_base = valid_idx[
        deterministic_sample_indices(len(valid_idx), min(sample_size, len(valid_idx)), seed)
    ]
    X_sample = X[chosen_base]
    y_sample = labels[chosen_base]
    unique_labels = np.sort(np.unique(y_sample))

    centroids = np.vstack([X_sample[y_sample == label].mean(axis=0) for label in unique_labels])
    for repeat in range(max(repeats, 1)):
        rng = np.random.default_rng(seed + repeat + 1)
        jittered = X_sample + rng.normal(0.0, jitter_scale, size=X_sample.shape).astype(np.float32)
        distances = ((jittered[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        predicted = unique_labels[np.argmin(distances, axis=1)]
        aris.append(float(adjusted_rand_score(y_sample, predicted)))
        accuracies.append(float(np.mean(predicted == y_sample)))

    return {
        "jitter_ari_mean": float(np.mean(aris)),
        "jitter_ari_std": float(np.std(aris)),
        "jitter_label_recovery_accuracy_mean": float(np.mean(accuracies)),
        "jitter_label_recovery_accuracy_std": float(np.std(accuracies)),
    }
