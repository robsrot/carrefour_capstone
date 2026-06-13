"""Candidate clustering models for customer tribe discovery."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.evaluation import evaluate_labels, quality_gate_result
from src.progress import log_event, stage_timer
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
    except ImportError:
        with path.open("wb") as f:
            pickle.dump(payload, f)
    else:
        dump(payload, path)


def _load_feature_matrix(feature_path: str | Path, scale_features: bool = True):
    from sklearn.preprocessing import StandardScaler

    df = pl.read_parquet(feature_path).sort("cliente")
    cols = numeric_feature_columns(df)
    X = frame_to_numpy(df, cols)
    scaler = None
    X_scaled = X.astype(np.float32, copy=False)
    if scale_features:
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


def _fit_sklearn_hdbscan_full(X: np.ndarray, cfg: PipelineConfig) -> tuple[Any, np.ndarray]:
    from sklearn.cluster import HDBSCAN

    model = HDBSCAN(
        min_cluster_size=int(cfg.get("hdbscan.min_cluster_size", 500)),
        min_samples=int(cfg.get("hdbscan.min_samples", 10)),
        cluster_selection_method=str(cfg.get("hdbscan.cluster_selection_method", "eom")),
    )
    labels = model.fit_predict(X)
    return model, np.asarray(labels).astype(np.int32)


def run_gmm_grid(
    feature_path: str | Path,
    output_prefix: str = "gmm",
    model_label: str = "Model A",
    model_name: str = "model_a_gmm",
    algorithm_name: str = "GaussianMixture",
    feature_space: str = "raw_customer_embeddings",
    variant_prefix: str | None = None,
    output_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """Model A: raw customer embeddings to Gaussian Mixture Model."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    model_selection_dir = Path(output_dir) if output_dir else cfg.model_selection_cache
    fitted_model_dir = Path(model_dir) if model_dir else cfg.models
    assignment_path = model_selection_dir / f"cluster_assignments_{output_prefix}.parquet"
    results_path = model_selection_dir / f"{output_prefix}_grid_results.parquet"
    cache_metadata = {
        "stage": "gmm_grid",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "output_prefix": output_prefix,
        "model_label": model_label,
        "model_name": model_name,
        "algorithm_name": algorithm_name,
        "feature_space": feature_space,
        "gmm": cfg.get("gmm", {}),
        "modeling": cfg.get("modeling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
    }
    if (
        should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        and should_use_cache(results_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
    ):
        log_event("Stage 6 GMM", "cache hit", cfg=cfg, model=model_name, results=results_path)
        results = pl.read_parquet(results_path).sort("selection_rank")
        best = results.row(0, named=True)
        best["assignment_path"] = str(assignment_path)
        return assignment_path, results_path, best

    from sklearn.mixture import GaussianMixture

    with stage_timer("Stage 6 GMM", "running GMM grid", cfg=cfg, model=model_name, feature_space=feature_space):
        df, feature_cols, X, scaler = _load_feature_matrix(feature_path)
        clientes = df["cliente"].to_list()
        fit_idx = _fit_indices(X.shape[0], cfg)
        X_fit = X[fit_idx]

        rows = []
        best: dict[str, Any] | None = None
        best_payload: tuple[GaussianMixture, np.ndarray, np.ndarray] | None = None
        k_values = [
            k
            for k in range(int(cfg.get("gmm.components_min", 6)), int(cfg.get("gmm.components_max", 25)) + 1)
            if k < X_fit.shape[0]
        ]
        for idx, k in enumerate(k_values, start=1):
            log_event("Stage 6 GMM", "fitting candidate", cfg=cfg, model=model_name, k=k, progress=f"{idx}/{len(k_values)}")
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
                "model_id": model_name,
                "model_name": model_name,
                "algorithm_name": algorithm_name,
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
            log_event(
                "Stage 6 GMM",
                "candidate evaluated",
                cfg=cfg,
                model=model_name,
                k=k,
                silhouette=metrics.get("silhouette"),
                bic=row["bic"],
                passes_gate=passes_gate,
            )
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
        fitted_model_dir / f"{output_prefix}_model.pkl",
        {"model": model, "scaler": scaler, "feature_columns": feature_cols, "selection": selected_row},
    )
    log_event(
        "Stage 6 GMM",
        "selected candidate",
        cfg=cfg,
        model=model_name,
        variant=selected_row["model_variant"],
        passes_gate=selected_row["passes_quality_gate"],
    )
    return assignment_path, results_path, selected_row


def run_hdbscan(
    feature_path: str | Path,
    output_prefix: str = "model_x_raw_hdbscan",
    model_label: str = "Model X",
    model_name: str = "model_x_raw_hdbscan",
    algorithm_name: str = "HDBSCAN",
    feature_space: str = "raw_customer_embeddings",
    trial_name: str | None = None,
    variant_prefix: str | None = None,
    scale_features: bool = True,
    output_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """HDBSCAN organic tribe discovery on the supplied feature representation."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    model_selection_dir = Path(output_dir) if output_dir else cfg.model_selection_cache
    fitted_model_dir = Path(model_dir) if model_dir else cfg.models
    assignment_path = model_selection_dir / f"cluster_assignments_{output_prefix}.parquet"
    results_path = model_selection_dir / f"{output_prefix}_results.parquet"
    cache_metadata = {
        "stage": "hdbscan",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "output_prefix": output_prefix,
        "model_label": model_label,
        "model_name": model_name,
        "algorithm_name": algorithm_name,
        "feature_space": feature_space,
        "trial_name": trial_name,
        "scale_features": scale_features,
        "hdbscan": cfg.get("hdbscan", {}),
        "modeling": cfg.get("modeling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
        "hdbscan_backend_policy": "external_hdbscan_with_sklearn_full_fit_fallback",
    }
    if (
        should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        and should_use_cache(results_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
    ):
        log_event("Stage 6 HDBSCAN", "cache hit", cfg=cfg, results=results_path)
        result = pl.read_parquet(results_path).row(0, named=True)
        result["assignment_path"] = str(assignment_path)
        return assignment_path, results_path, result

    with stage_timer(
        "Stage 6 HDBSCAN",
        "fitting density model",
        cfg=cfg,
        model=model_name,
        feature_space=feature_space,
        min_cluster_size=cfg.get("hdbscan.min_cluster_size"),
        min_samples=cfg.get("hdbscan.min_samples"),
    ):
        df, feature_cols, X, scaler = _load_feature_matrix(feature_path, scale_features=scale_features)
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
            if len(fit_idx) < X.shape[0]:
                raise RuntimeError(
                    "The hdbscan package is required when HDBSCAN is fit on a sample and predicted for all customers. "
                    "Install hdbscan or set modeling.fit_sample_size large enough to fit the full population."
                )

            model, labels = _fit_sklearn_hdbscan_full(X, cfg)
            probabilities = None
            assignment_source = "sklearn_hdbscan_full_fit"
        else:
            model = hdbscan.HDBSCAN(
                min_cluster_size=int(cfg.get("hdbscan.min_cluster_size", 500)),
                min_samples=int(cfg.get("hdbscan.min_samples", 10)),
                cluster_selection_method=str(cfg.get("hdbscan.cluster_selection_method", "eom")),
                prediction_data=len(fit_idx) < X.shape[0],
            )
            try:
                labels_fit = model.fit_predict(X_fit)
            except TypeError as exc:
                if "force_all_finite" not in str(exc):
                    raise
                if len(fit_idx) < X.shape[0]:
                    raise RuntimeError(
                        "The installed hdbscan package is incompatible with the installed scikit-learn version "
                        "and sampled production HDBSCAN requires hdbscan.approximate_predict. Upgrade hdbscan "
                        "or pin scikit-learn to a compatible version before running sampled prod HDBSCAN."
                    ) from exc
                model, labels = _fit_sklearn_hdbscan_full(X, cfg)
                probabilities = None
                assignment_source = "sklearn_hdbscan_full_fit_hdbscan_api_mismatch"
                labels_fit = None
            if len(fit_idx) < X.shape[0]:
                log_event("Stage 6 HDBSCAN", "assigning full population with approximate_predict", cfg=cfg)
                labels, probabilities = hdbscan.approximate_predict(model, X)
                assignment_source = "hdbscan_approximate_predict"
            elif labels_fit is not None:
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
    if variant_prefix:
        variant = f"{variant_prefix}_{variant}"
    result = {
        "model": model_label,
        "model_id": model_name,
        "model_name": model_name,
        "algorithm_name": algorithm_name,
        "model_variant": variant,
        "feature_space": feature_space,
        "trial_name": trial_name,
        "hdbscan_backend": assignment_source,
        "scale_features": scale_features,
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
        model_name,
        variant,
        assignment_confidence_type="hdbscan_membership_strength" if probabilities is not None else None,
        assignment_source=assignment_source,
    ).write_parquet(assignment_path)
    pl.DataFrame([result]).write_parquet(results_path)
    write_artifact_metadata(assignment_path, cache_metadata)
    write_artifact_metadata(results_path, cache_metadata)
    _save_model(
        fitted_model_dir / f"{output_prefix}_model.pkl",
        {"model": model, "scaler": scaler, "feature_columns": feature_cols, "selection": result},
    )
    log_event(
        "Stage 6 HDBSCAN",
        "model evaluated",
        cfg=cfg,
        clusters=result["cluster_count"],
        noise_pct=result["noise_pct"],
        passes_gate=passes_gate,
        source=assignment_source,
    )
    return assignment_path, results_path, result


def run_pca_kmeans_grid(
    feature_path: str | Path,
    output_prefix: str = "model_c_pca_kmeans",
    model_label: str = "Model C",
    model_name: str = "model_c_pca_kmeans",
    algorithm_name: str = "PCA_MiniBatchKMeans",
    output_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """Benchmark model: PCA representation to MiniBatchKMeans."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    model_selection_dir = Path(output_dir) if output_dir else cfg.model_selection_cache
    fitted_model_dir = Path(model_dir) if model_dir else cfg.models
    assignment_path = model_selection_dir / f"cluster_assignments_{output_prefix}.parquet"
    results_path = model_selection_dir / f"{output_prefix}_grid_results.parquet"
    cache_metadata = {
        "stage": "pca_kmeans_grid",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "output_prefix": output_prefix,
        "model_label": model_label,
        "model_name": model_name,
        "algorithm_name": algorithm_name,
        "pca": cfg.get("pca", {}),
        "kmeans": cfg.get("kmeans", {}),
        "modeling": cfg.get("modeling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
    }
    if (
        should_use_cache(assignment_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        and should_use_cache(results_path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
    ):
        log_event("Stage 6 PCA-KMeans", "cache hit", cfg=cfg, results=results_path)
        results = pl.read_parquet(results_path).sort("selection_rank")
        best = results.row(0, named=True)
        best["assignment_path"] = str(assignment_path)
        return assignment_path, results_path, best

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    with stage_timer("Stage 6 PCA-KMeans", "running KMeans grid", cfg=cfg):
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
        k_values = [
            k
            for k in range(int(cfg.get("kmeans.k_min", 6)), int(cfg.get("kmeans.k_max", 25)) + 1)
            if k < X_fit.shape[0]
        ]
        for idx, k in enumerate(k_values, start=1):
            log_event("Stage 6 PCA-KMeans", "fitting candidate", cfg=cfg, k=k, progress=f"{idx}/{len(k_values)}")
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
                "model": model_label,
                "model_id": model_name,
                "model_name": model_name,
                "algorithm_name": algorithm_name,
                "model_variant": f"pca{n_components}_k{k}",
                "feature_space": "pca_customer_embeddings",
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                **metrics,
                "passes_quality_gate": passes_gate,
                "quality_gate_reason": gate_reason,
                "assignment_path": str(assignment_path),
            }
            rows.append(row)
            log_event(
                "Stage 6 PCA-KMeans",
                "candidate evaluated",
                cfg=cfg,
                k=k,
                silhouette=metrics.get("silhouette"),
                passes_gate=passes_gate,
            )

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
        model_name,
        selected_row["model_variant"],
        assignment_confidence_type=None,
        assignment_source="kmeans_predict",
    ).write_parquet(assignment_path)
    pl.DataFrame(ranked).write_parquet(results_path)
    write_artifact_metadata(assignment_path, cache_metadata)
    write_artifact_metadata(results_path, cache_metadata)
    _save_model(
        fitted_model_dir / f"{output_prefix}_model.pkl",
        {
            "model": model,
            "pca": pca,
            "scaler": scaler,
            "feature_columns": feature_cols,
            "selection": selected_row,
        },
    )
    log_event(
        "Stage 6 PCA-KMeans",
        "selected candidate",
        cfg=cfg,
        variant=selected_row["model_variant"],
        passes_gate=selected_row["passes_quality_gate"],
    )
    return assignment_path, results_path, selected_row
