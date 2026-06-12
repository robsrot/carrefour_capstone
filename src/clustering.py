"""Candidate clustering models for customer tribe discovery."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.evaluation import evaluate_labels
from src.utils import deterministic_sample_indices, frame_to_numpy, numeric_feature_columns, should_use_cache


def _save_model(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from joblib import dump

        dump(payload, path)
    except Exception:
        with path.open("wb") as f:
            pickle.dump(payload, f)


def _load_feature_matrix(feature_path: str | Path):
    from sklearn.preprocessing import StandardScaler

    df = pl.read_parquet(feature_path).sort("cliente")
    cols = numeric_feature_columns(df)
    X = frame_to_numpy(df, cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    return df, cols, X_scaled, scaler


def _assignment_frame(
    clientes: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    model_name: str,
    model_variant: str,
) -> pl.DataFrame:
    if probabilities is None:
        probabilities = np.where(np.asarray(labels) >= 0, 1.0, 0.0)
    return pl.DataFrame(
        {
            "cliente": clientes,
            "tribe_id": np.asarray(labels).astype(np.int32),
            "model_name": [model_name] * len(clientes),
            "model_variant": [model_variant] * len(clientes),
            "assignment_probability": np.asarray(probabilities).astype(np.float32),
        }
    ).sort("cliente")


def _fit_indices(n_rows: int, cfg: PipelineConfig) -> np.ndarray:
    return deterministic_sample_indices(n_rows, cfg.get("modeling.fit_sample_size"), cfg.random_seed)


def run_gmm_grid(
    feature_path: str | Path,
    output_prefix: str = "gmm",
    model_label: str = "Model A",
    model_name: str = "GMM",
    feature_space: str = "raw_customer_embeddings",
    variant_prefix: str | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """Model A: raw customer embeddings to Gaussian Mixture Model."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    assignment_path = cfg.data_processed / f"cluster_assignments_{output_prefix}.parquet"
    results_path = cfg.data_processed / f"{output_prefix}_grid_results.parquet"
    if should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True)) and results_path.exists():
        results = pl.read_parquet(results_path).sort("selection_rank")
        return assignment_path, results_path, results.row(0, named=True)

    from sklearn.mixture import GaussianMixture

    df, feature_cols, X, scaler = _load_feature_matrix(feature_path)
    clientes = df["cliente"].to_list()
    fit_idx = _fit_indices(X.shape[0], cfg)
    X_fit = X[fit_idx]

    rows = []
    best: dict[str, Any] | None = None
    best_payload: tuple[GaussianMixture, np.ndarray, np.ndarray] | None = None
    for k in range(int(cfg.get("gmm.components_min", 6)), int(cfg.get("gmm.components_max", 25)) + 1):
        if k >= X_fit.shape[0]:
            continue
        model = GaussianMixture(
            n_components=k,
            covariance_type=str(cfg.get("gmm.covariance_type", "diag")),
            max_iter=int(cfg.get("gmm.max_iter", 300)),
            n_init=int(cfg.get("gmm.n_init", 2)),
            random_state=cfg.random_seed,
        )
        model.fit(X_fit)
        labels = model.predict(X)
        probabilities = model.predict_proba(X).max(axis=1)
        metrics = evaluate_labels(X, labels, probabilities=probabilities, cfg=cfg)
        variant = f"{variant_prefix}_gmm_k{k}" if variant_prefix else f"gmm_k{k}"
        row = {
            "model": model_label,
            "model_name": model_name,
            "model_variant": variant,
            "feature_space": feature_space,
            "cluster_count": metrics["cluster_count"],
            "aic": float(model.aic(X_fit)),
            "bic": float(model.bic(X_fit)),
            **metrics,
            "assignment_path": str(assignment_path),
        }
        rows.append(row)
        if best is None or row["bic"] < best["bic"]:
            best = row
            best_payload = (model, labels, probabilities)

    if best is None or best_payload is None:
        raise RuntimeError("GMM grid did not produce a valid model.")

    ranked = sorted(rows, key=lambda row: row["bic"])
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
        row["selected_within_family"] = rank == 1

    model, labels, probabilities = best_payload
    _assignment_frame(clientes, labels, probabilities, model_name, best["model_variant"]).write_parquet(assignment_path)
    pl.DataFrame(ranked).write_parquet(results_path)
    _save_model(
        cfg.models / f"{output_prefix}_model.pkl",
        {"model": model, "scaler": scaler, "feature_columns": feature_cols, "selection": best},
    )
    return assignment_path, results_path, best


def _centroid_assign(X: np.ndarray, X_fit: np.ndarray, labels_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_labels = sorted({int(label) for label in labels_fit if int(label) >= 0})
    if not valid_labels:
        return np.full(X.shape[0], -1, dtype=np.int32), np.zeros(X.shape[0], dtype=np.float32)
    centroids = np.vstack([X_fit[labels_fit == label].mean(axis=0) for label in valid_labels])
    distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    labels = np.asarray([valid_labels[idx] for idx in nearest], dtype=np.int32)
    scale = np.maximum(np.median(np.sqrt(distances.min(axis=1))), 1e-6)
    probabilities = np.exp(-np.sqrt(distances.min(axis=1)) / scale).astype(np.float32)
    return labels, probabilities


def run_hdbscan(
    feature_path: str | Path,
    output_prefix: str = "hdbscan",
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """Model B: HDBSCAN organic tribe discovery."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    assignment_path = cfg.data_processed / f"cluster_assignments_{output_prefix}.parquet"
    results_path = cfg.data_processed / f"{output_prefix}_results.parquet"
    if should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True)) and results_path.exists():
        result = pl.read_parquet(results_path).row(0, named=True)
        return assignment_path, results_path, result

    df, feature_cols, X, scaler = _load_feature_matrix(feature_path)
    clientes = df["cliente"].to_list()
    fit_idx = _fit_indices(X.shape[0], cfg)
    X_fit = X[fit_idx]

    labels: np.ndarray
    probabilities: np.ndarray | None = None
    model: Any
    try:
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=int(cfg.get("hdbscan.min_cluster_size", 500)),
            min_samples=int(cfg.get("hdbscan.min_samples", 10)),
            cluster_selection_method=str(cfg.get("hdbscan.cluster_selection_method", "eom")),
            prediction_data=len(fit_idx) < X.shape[0],
        )
        labels_fit = model.fit_predict(X_fit)
        if len(fit_idx) < X.shape[0]:
            labels, probabilities = hdbscan.approximate_predict(model, X)
        else:
            labels = labels_fit
            probabilities = getattr(model, "probabilities_", None)
    except Exception:
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=int(cfg.get("hdbscan.min_cluster_size", 500)),
            min_samples=int(cfg.get("hdbscan.min_samples", 10)),
            cluster_selection_method=str(cfg.get("hdbscan.cluster_selection_method", "eom")),
        )
        labels_fit = model.fit_predict(X_fit)
        labels, probabilities = (
            _centroid_assign(X, X_fit, labels_fit) if len(fit_idx) < X.shape[0] else (labels_fit, None)
        )

    labels = np.asarray(labels).astype(np.int32)
    probabilities = None if probabilities is None else np.asarray(probabilities).astype(np.float32)
    metrics = evaluate_labels(X, labels, probabilities=probabilities, cfg=cfg)
    variant = (
        f"hdbscan_mcs{cfg.get('hdbscan.min_cluster_size')}_"
        f"ms{cfg.get('hdbscan.min_samples')}_{cfg.get('hdbscan.cluster_selection_method')}"
    )
    result = {
        "model": "Model B",
        "model_name": "HDBSCAN",
        "model_variant": variant,
        "feature_space": "raw_customer_embeddings",
        **metrics,
        "assignment_path": str(assignment_path),
        "selection_rank": 1,
        "selected_within_family": True,
    }
    _assignment_frame(clientes, labels, probabilities, "HDBSCAN", variant).write_parquet(assignment_path)
    pl.DataFrame([result]).write_parquet(results_path)
    _save_model(
        cfg.models / f"{output_prefix}_model.pkl",
        {"model": model, "scaler": scaler, "feature_columns": feature_cols, "selection": result},
    )
    return assignment_path, results_path, result


def run_pca_kmeans_grid(
    feature_path: str | Path,
    output_prefix: str = "pca_kmeans",
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """Benchmark model: PCA representation to MiniBatchKMeans."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    assignment_path = cfg.data_processed / f"cluster_assignments_{output_prefix}.parquet"
    results_path = cfg.data_processed / f"{output_prefix}_grid_results.parquet"
    if should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True)) and results_path.exists():
        results = pl.read_parquet(results_path).sort("selection_rank")
        return assignment_path, results_path, results.row(0, named=True)

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    df, feature_cols, X, scaler = _load_feature_matrix(feature_path)
    clientes = df["cliente"].to_list()
    fit_idx = _fit_indices(X.shape[0], cfg)
    X_fit = X[fit_idx]
    n_components = min(int(cfg.get("pca.n_components", 32)), X.shape[1], X_fit.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=cfg.random_seed)
    pca.fit(X_fit)
    Z = pca.transform(X).astype(np.float32)
    Z_fit = Z[fit_idx]

    rows = []
    best: dict[str, Any] | None = None
    best_payload: tuple[MiniBatchKMeans, np.ndarray] | None = None
    for k in range(int(cfg.get("kmeans.k_min", 6)), int(cfg.get("kmeans.k_max", 25)) + 1):
        if k >= X_fit.shape[0]:
            continue
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=cfg.random_seed,
            batch_size=int(cfg.get("kmeans.batch_size", 4096)),
            n_init=int(cfg.get("kmeans.n_init", 10)),
        )
        model.fit(Z_fit)
        labels = model.predict(Z)
        metrics = evaluate_labels(Z, labels, cfg=cfg)
        row = {
            "model": "Benchmark",
            "model_name": "PCA_KMeans",
            "model_variant": f"pca{n_components}_k{k}",
            "feature_space": "pca_customer_embeddings",
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            **metrics,
            "assignment_path": str(assignment_path),
        }
        rows.append(row)
        current_sil = row["silhouette"] if row["silhouette"] is not None else -999.0
        best_sil = best["silhouette"] if best and best["silhouette"] is not None else -999.0
        if best is None or current_sil > best_sil:
            best = row
            best_payload = (model, labels)

    if best is None or best_payload is None:
        raise RuntimeError("PCA-KMeans grid did not produce a valid model.")

    ranked = sorted(rows, key=lambda row: (row["silhouette"] is None, -(row["silhouette"] or -999)))
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
        row["selected_within_family"] = rank == 1

    model, labels = best_payload
    _assignment_frame(clientes, labels, None, "PCA_KMeans", best["model_variant"]).write_parquet(assignment_path)
    pl.DataFrame(ranked).write_parquet(results_path)
    _save_model(
        cfg.models / f"{output_prefix}_model.pkl",
        {
            "model": model,
            "pca": pca,
            "scaler": scaler,
            "feature_columns": feature_cols,
            "selection": best,
        },
    )
    return assignment_path, results_path, best
