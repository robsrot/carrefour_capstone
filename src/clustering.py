"""Candidate clustering models for customer tribe discovery."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.evaluation import evaluate_labels, quality_gate_result
from src.utils import (
    deterministic_sample_indices,
    file_fingerprint,
    frame_to_numpy,
    numeric_feature_columns,
    should_use_cache,
    write_artifact_metadata,
)


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
    assignment_confidence_type: str | None = None,
    assignment_source: str = "model_predict",
) -> pl.DataFrame:
    if probabilities is None:
        probability_values = [None] * len(clientes)
    else:
        probability_values = np.asarray(probabilities).astype(np.float32)
    return pl.DataFrame(
        {
            "cliente": clientes,
            "tribe_id": np.asarray(labels).astype(np.int32),
            "model_name": [model_name] * len(clientes),
            "model_variant": [model_variant] * len(clientes),
            "assignment_probability": probability_values,
            "assignment_confidence_type": [assignment_confidence_type] * len(clientes),
            "assignment_source": [assignment_source] * len(clientes),
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
    cache_metadata = {
        "stage": "gmm_grid",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "output_prefix": output_prefix,
        "model_label": model_label,
        "model_name": model_name,
        "feature_space": feature_space,
        "gmm": cfg.get("gmm", {}),
        "modeling": cfg.get("modeling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
    }
    if (
        should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        and should_use_cache(results_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
    ):
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
        passes_gate, gate_reason = quality_gate_result(metrics, cfg=cfg)
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
            "passes_quality_gate": passes_gate,
            "quality_gate_reason": gate_reason,
            "assignment_path": str(assignment_path),
        }
        rows.append(row)
        if best is None or row["bic"] < best["bic"]:
            best = row
            best_payload = (model, labels, probabilities)

    if best is None or best_payload is None:
        raise RuntimeError("GMM grid did not produce a valid model.")

    valid_rows = [row for row in rows if row["passes_quality_gate"]]
    ranked = sorted(valid_rows, key=lambda row: row["bic"]) + sorted(
        [row for row in rows if not row["passes_quality_gate"]],
        key=lambda row: row["bic"],
    )
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
        row["selected_within_family"] = rank == 1 and row["passes_quality_gate"]

    selected_row = ranked[0]
    selected_k = int(str(selected_row["model_variant"]).split("k")[-1])
    model = GaussianMixture(
        n_components=selected_k,
        covariance_type=str(cfg.get("gmm.covariance_type", "diag")),
        max_iter=int(cfg.get("gmm.max_iter", 300)),
        n_init=int(cfg.get("gmm.n_init", 2)),
        random_state=cfg.random_seed,
    )
    model.fit(X_fit)
    labels = model.predict(X)
    probabilities = model.predict_proba(X).max(axis=1)
    _assignment_frame(
        clientes,
        labels,
        probabilities,
        model_name,
        selected_row["model_variant"],
        assignment_confidence_type="gmm_max_posterior_probability",
    ).write_parquet(assignment_path)
    pl.DataFrame(ranked).write_parquet(results_path)
    write_artifact_metadata(assignment_path, cache_metadata)
    write_artifact_metadata(results_path, cache_metadata)
    _save_model(
        cfg.models / f"{output_prefix}_model.pkl",
        {"model": model, "scaler": scaler, "feature_columns": feature_cols, "selection": selected_row},
    )
    return assignment_path, results_path, selected_row


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
    cache_metadata = {
        "stage": "hdbscan",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "output_prefix": output_prefix,
        "hdbscan": cfg.get("hdbscan", {}),
        "modeling": cfg.get("modeling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
    }
    if (
        should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        and should_use_cache(results_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
    ):
        result = pl.read_parquet(results_path).row(0, named=True)
        return assignment_path, results_path, result

    df, feature_cols, X, scaler = _load_feature_matrix(feature_path)
    clientes = df["cliente"].to_list()
    fit_idx = _fit_indices(X.shape[0], cfg)
    X_fit = X[fit_idx]

    labels: np.ndarray
    probabilities: np.ndarray | None = None
    model: Any
    assignment_source = "hdbscan_fit"
    try:
        import hdbscan
    except ImportError:
        from sklearn.cluster import HDBSCAN

        if len(fit_idx) < X.shape[0]:
            raise RuntimeError(
                "The hdbscan package is required when HDBSCAN is fit on a sample and predicted for all customers. "
                "Install hdbscan or set modeling.fit_sample_size large enough to fit the full population."
            )

        model = HDBSCAN(
            min_cluster_size=int(cfg.get("hdbscan.min_cluster_size", 500)),
            min_samples=int(cfg.get("hdbscan.min_samples", 10)),
            cluster_selection_method=str(cfg.get("hdbscan.cluster_selection_method", "eom")),
        )
        labels = model.fit_predict(X)
        probabilities = None
        assignment_source = "sklearn_hdbscan_full_fit"
    else:
        model = hdbscan.HDBSCAN(
            min_cluster_size=int(cfg.get("hdbscan.min_cluster_size", 500)),
            min_samples=int(cfg.get("hdbscan.min_samples", 10)),
            cluster_selection_method=str(cfg.get("hdbscan.cluster_selection_method", "eom")),
            prediction_data=len(fit_idx) < X.shape[0],
        )
        labels_fit = model.fit_predict(X_fit)
        if len(fit_idx) < X.shape[0]:
            labels, probabilities = hdbscan.approximate_predict(model, X)
            assignment_source = "hdbscan_approximate_predict"
        else:
            labels = labels_fit
            probabilities = getattr(model, "probabilities_", None)

    labels = np.asarray(labels).astype(np.int32)
    probabilities = None if probabilities is None else np.asarray(probabilities).astype(np.float32)
    metrics = evaluate_labels(X, labels, probabilities=probabilities, cfg=cfg)
    passes_gate, gate_reason = quality_gate_result(metrics, cfg=cfg)
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
        "passes_quality_gate": passes_gate,
        "quality_gate_reason": gate_reason,
        "assignment_path": str(assignment_path),
        "selection_rank": 1,
        "selected_within_family": passes_gate,
    }
    _assignment_frame(
        clientes,
        labels,
        probabilities,
        "HDBSCAN",
        variant,
        assignment_confidence_type="hdbscan_membership_strength" if probabilities is not None else None,
        assignment_source=assignment_source,
    ).write_parquet(assignment_path)
    pl.DataFrame([result]).write_parquet(results_path)
    write_artifact_metadata(assignment_path, cache_metadata)
    write_artifact_metadata(results_path, cache_metadata)
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
    cache_metadata = {
        "stage": "pca_kmeans_grid",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "output_prefix": output_prefix,
        "pca": cfg.get("pca", {}),
        "kmeans": cfg.get("kmeans", {}),
        "modeling": cfg.get("modeling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
    }
    if (
        should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        and should_use_cache(results_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
    ):
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
        passes_gate, gate_reason = quality_gate_result(metrics, cfg=cfg)
        row = {
            "model": "Benchmark",
            "model_name": "PCA_KMeans",
            "model_variant": f"pca{n_components}_k{k}",
            "feature_space": "pca_customer_embeddings",
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            **metrics,
            "passes_quality_gate": passes_gate,
            "quality_gate_reason": gate_reason,
            "assignment_path": str(assignment_path),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("PCA-KMeans grid did not produce a valid model.")

    def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        silhouette = row.get("coverage_adjusted_silhouette")
        if silhouette is None:
            silhouette = row.get("silhouette")
        db = row.get("davies_bouldin")
        balance = row.get("cluster_size_cv")
        noise = row.get("noise_pct")
        return (
            -float(silhouette) if silhouette is not None else 999.0,
            float(db) if db is not None else 999.0,
            float(balance) if balance is not None else 999.0,
            float(noise) if noise is not None else 999.0,
        )

    valid_rows = [row for row in rows if row["passes_quality_gate"]]
    ranked = sorted(valid_rows, key=_rank_key) + sorted(
        [row for row in rows if not row["passes_quality_gate"]],
        key=_rank_key,
    )
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
        row["selected_within_family"] = rank == 1 and row["passes_quality_gate"]

    selected_row = ranked[0]
    selected_k = int(str(selected_row["model_variant"]).split("_k")[-1])
    model = MiniBatchKMeans(
        n_clusters=selected_k,
        random_state=cfg.random_seed,
        batch_size=int(cfg.get("kmeans.batch_size", 4096)),
        n_init=int(cfg.get("kmeans.n_init", 10)),
    )
    model.fit(Z_fit)
    labels = model.predict(Z)

    _assignment_frame(
        clientes,
        labels,
        None,
        "PCA_KMeans",
        selected_row["model_variant"],
        assignment_confidence_type=None,
        assignment_source="kmeans_predict",
    ).write_parquet(assignment_path)
    pl.DataFrame(ranked).write_parquet(results_path)
    write_artifact_metadata(assignment_path, cache_metadata)
    write_artifact_metadata(results_path, cache_metadata)
    _save_model(
        cfg.models / f"{output_prefix}_model.pkl",
        {
            "model": model,
            "pca": pca,
            "scaler": scaler,
            "feature_columns": feature_cols,
            "selection": selected_row,
        },
    )
    return assignment_path, results_path, selected_row
