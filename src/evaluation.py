"""Model evaluation helpers for customer tribe discovery."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.utils import deterministic_sample_indices


def cluster_size_summary(labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    total = int(labels.shape[0])
    counts = Counter(int(label) for label in labels if int(label) >= 0)
    noise = int(np.sum(labels < 0))
    shares = np.array(list(counts.values()), dtype=np.float64) / max(total, 1)
    return {
        "cluster_count": int(len(counts)),
        "noise_count": noise,
        "noise_pct": round(noise / max(total, 1) * 100.0, 4),
        "min_cluster_share": float(shares.min()) if len(shares) else None,
        "max_cluster_share": float(shares.max()) if len(shares) else None,
        "cluster_size_cv": float(shares.std() / shares.mean()) if len(shares) and shares.mean() else None,
    }


def evaluate_labels(
    X: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray | None = None,
    sample_size: int | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Compute standard clustering metrics with deterministic sampling."""

    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    labels = np.asarray(labels).astype(int)
    valid = labels >= 0
    summary = cluster_size_summary(labels)
    metrics: dict[str, Any] = {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        **summary,
    }

    if int(np.unique(labels[valid]).shape[0]) >= 2:
        valid_indices = np.where(valid)[0]
        sample_indices = deterministic_sample_indices(
            len(valid_indices),
            sample_size or int(cfg.get("modeling.evaluation_sample_size", 25000)),
            cfg.random_seed,
        )
        chosen = valid_indices[sample_indices]
        X_eval = X[chosen]
        y_eval = labels[chosen]
        try:
            metrics["silhouette"] = float(silhouette_score(X_eval, y_eval))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X_eval, y_eval))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, y_eval))
        except ValueError:
            pass

    if probabilities is not None:
        metrics["avg_assignment_confidence"] = float(np.nanmean(probabilities))
    metrics["coverage_pct"] = 100.0 - float(metrics["noise_pct"])
    metrics["coverage_adjusted_silhouette"] = (
        None
        if metrics["silhouette"] is None
        else float(metrics["silhouette"]) * max(0.0, metrics["coverage_pct"] / 100.0)
    )
    return metrics


def quality_gate_result(metrics: dict[str, Any], cfg: PipelineConfig = CONFIG) -> tuple[bool, str]:
    reasons = []
    cluster_count = metrics.get("cluster_count") or 0
    silhouette = metrics.get("silhouette")
    noise_pct = metrics.get("noise_pct")
    cluster_size_cv = metrics.get("cluster_size_cv")

    if cluster_count < int(cfg.get("quality_gates.min_clusters", 2)):
        reasons.append(f"cluster_count<{cfg.get('quality_gates.min_clusters', 2)}")
    if silhouette is None or float(silhouette) < float(cfg.get("quality_gates.min_silhouette", 0.0)):
        reasons.append(f"silhouette<{cfg.get('quality_gates.min_silhouette', 0.0)}")
    if noise_pct is not None and float(noise_pct) > float(cfg.get("quality_gates.max_noise_pct", 60.0)):
        reasons.append(f"noise_pct>{cfg.get('quality_gates.max_noise_pct', 60.0)}")
    if cluster_size_cv is not None and float(cluster_size_cv) > float(cfg.get("quality_gates.max_cluster_size_cv", 1.5)):
        reasons.append(f"cluster_size_cv>{cfg.get('quality_gates.max_cluster_size_cv', 1.5)}")

    return not reasons, "pass" if not reasons else "; ".join(reasons)


def add_client_hypothesis_flag(rows: list[dict[str, Any]], cfg: PipelineConfig = CONFIG) -> list[dict[str, Any]]:
    low = int(cfg.get("modeling.client_hypothesis_min", 10))
    high = int(cfg.get("modeling.client_hypothesis_max", 15))
    for row in rows:
        clusters = row.get("cluster_count")
        row["client_hypothesis_range"] = f"{low}-{high}"
        row["agrees_with_client_hypothesis"] = bool(clusters is not None and low <= int(clusters) <= high)
    return rows


def comparison_to_frame(rows: list[dict[str, Any]], cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    return pl.DataFrame(add_client_hypothesis_flag(rows, cfg=cfg))
