"""Candidate clustering models for customer tribe discovery."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.evaluation import cluster_size_summary, evaluate_labels, quality_gate_result
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
    assignment_confidence_type: str | list[str | None] | np.ndarray | None = None,
    assignment_confidence_score: np.ndarray | list[float | None] | None = None,
    assignment_source: str | list[str] | np.ndarray = "model_predict",
) -> pl.DataFrame:
    if probabilities is None:
        probability_values = [None] * len(clientes)
    else:
        probability_values = np.asarray(probabilities).astype(np.float32)
    if assignment_confidence_score is None:
        confidence_score_values = probability_values if probabilities is not None else [None] * len(clientes)
    else:
        confidence_score_values = _as_repeated_column(assignment_confidence_score, len(clientes))
    confidence_values = _as_repeated_column(assignment_confidence_type, len(clientes))
    source_values = _as_repeated_column(assignment_source, len(clientes))
    return pl.DataFrame(
        {
            "cliente": clientes,
            "tribe_id": np.asarray(labels).astype(np.int32),
            "model_name": [model_name] * len(clientes),
            "model_variant": [model_variant] * len(clientes),
            "assignment_probability": probability_values,
            "assignment_confidence_score": confidence_score_values,
            "assignment_confidence_type": confidence_values,
            "assignment_source": source_values,
        }
    ).sort("cliente")


def _as_repeated_column(value: Any, n_rows: int) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return [value] * n_rows


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


def _soft_assign_noise_labels(
    X: np.ndarray,
    labels: np.ndarray,
    strategy: str = "q95",
) -> tuple[np.ndarray, np.ndarray, float | None, np.ndarray]:
    confidence_scores = np.full(labels.shape[0], np.nan, dtype=np.float32)
    valid_labels = sorted(int(label) for label in np.unique(labels) if label >= 0)
    if not valid_labels or not np.any(labels < 0):
        return labels.astype(np.int32, copy=True), np.zeros(labels.shape[0], dtype=bool), None, confidence_scores

    centroids = np.vstack([X[labels == label].mean(axis=0) for label in valid_labels]).astype(np.float32)
    valid_indices = np.where(labels >= 0)[0]
    valid_label_to_idx = {label: idx for idx, label in enumerate(valid_labels)}
    own_centroid_idx = np.array([valid_label_to_idx[int(label)] for label in labels[valid_indices]], dtype=np.int32)
    own_distances = np.linalg.norm(X[valid_indices] - centroids[own_centroid_idx], axis=1)

    threshold = None
    if strategy.startswith("q"):
        quantile = float(strategy[1:]) / 100.0
        threshold = float(np.quantile(own_distances, quantile))
    elif strategy != "all":
        raise ValueError(f"Unsupported soft assignment strategy: {strategy}")

    noise_indices = np.where(labels < 0)[0]
    distances = np.linalg.norm(X[noise_indices, None, :] - centroids[None, :, :], axis=2)
    nearest_centroid_idx = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(distances.shape[0]), nearest_centroid_idx]
    assign_mask = np.ones(noise_indices.shape[0], dtype=bool)
    if threshold is not None:
        assign_mask = nearest_distances <= threshold

    assigned_noise_indices = noise_indices[assign_mask]
    soft_labels = labels.astype(np.int32, copy=True)
    soft_labels[assigned_noise_indices] = np.array(valid_labels, dtype=np.int32)[nearest_centroid_idx[assign_mask]]
    core_distance_reference = np.sort(own_distances)
    if core_distance_reference.size:
        nearest_percentiles = np.searchsorted(core_distance_reference, nearest_distances, side="left") / float(
            core_distance_reference.size
        )
        nearest_confidence = np.clip(1.0 - nearest_percentiles, 0.0, 1.0).astype(np.float32)
        confidence_scores[assigned_noise_indices] = nearest_confidence[assign_mask]

    assigned_mask_full = np.zeros(labels.shape[0], dtype=bool)
    assigned_mask_full[assigned_noise_indices] = True
    return soft_labels.astype(np.int32), assigned_mask_full, threshold, confidence_scores


def _soft_assignment_metadata(
    labels: np.ndarray,
    core_probabilities: np.ndarray | None,
    assigned_mask: np.ndarray,
    soft_confidence_scores: np.ndarray | None,
    base_assignment_source: str,
    strategy: str,
) -> tuple[np.ndarray | None, np.ndarray, list[str | None], list[str]]:
    probability_values = None
    confidence_score_values = np.full(len(labels), np.nan, dtype=np.float32)
    confidence_values: list[str | None] = [None] * len(labels)
    source_values = [base_assignment_source if int(label) >= 0 else f"hdbscan_noise_unassigned_{strategy}" for label in labels]

    if core_probabilities is not None:
        probability_values = np.asarray(core_probabilities, dtype=np.float32).copy()
        probability_values[labels < 0] = np.nan
        confidence_score_values = probability_values.copy()
        for idx, label in enumerate(labels):
            if int(label) >= 0:
                confidence_values[idx] = "hdbscan_membership_strength"

    if soft_confidence_scores is not None:
        soft_scores = np.asarray(soft_confidence_scores, dtype=np.float32)
        confidence_score_values[assigned_mask] = soft_scores[assigned_mask]
    for idx in np.where(assigned_mask)[0]:
        source_values[int(idx)] = f"nearest_centroid_soft_noise_{strategy}"
        confidence_values[int(idx)] = f"nearest_centroid_distance_percentile_{strategy}"
        if probability_values is not None:
            probability_values[int(idx)] = np.nan

    return probability_values, confidence_score_values, confidence_values, source_values


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
        "assignment_schema_version": 2,
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
    allow_noise_assignment: bool | None = None,
    noise_assignment_strategy: str | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path, dict[str, Any]]:
    """HDBSCAN organic tribe discovery on the supplied feature representation."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    allow_noise_assignment = (
        bool(cfg.get("hdbscan.allow_noise_assignment", False))
        if allow_noise_assignment is None
        else bool(allow_noise_assignment)
    )
    noise_assignment_strategy = str(noise_assignment_strategy or cfg.get("hdbscan.noise_assignment_strategy", "q95"))
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
        "allow_noise_assignment": allow_noise_assignment,
        "noise_assignment_strategy": noise_assignment_strategy,
        "hdbscan": cfg.get("hdbscan", {}),
        "modeling": cfg.get("modeling", {}),
        "quality_gates": cfg.get("quality_gates", {}),
        "hdbscan_backend_policy": "external_hdbscan_with_sklearn_full_fit_fallback_and_optional_soft_assignment",
        "assignment_schema_version": 2,
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
        log_event(
            "Stage 6 HDBSCAN",
            "loaded dense feature matrix",
            cfg=cfg,
            rows=X.shape[0],
            features=len(feature_cols),
            fit_rows=X_fit.shape[0],
            full_matrix_mb=round(float(X.nbytes) / (1024**2), 1),
            fit_matrix_mb=round(float(X_fit.nbytes) / (1024**2), 1),
            sampled_fit=X_fit.shape[0] < X.shape[0],
        )

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
    core_labels = labels.copy()
    core_probabilities = None if probabilities is None else probabilities.copy()
    core_summary = cluster_size_summary(core_labels)
    assignment_confidence_type: str | list[str | None] | None = (
        "hdbscan_membership_strength" if probabilities is not None else None
    )
    assignment_confidence_score: np.ndarray | list[float | None] | None = probabilities
    assignment_sources: str | list[str] = assignment_source
    soft_assignment_info: dict[str, Any] = {
        "soft_assignment_strategy": None,
        "soft_assignment_distance_threshold": None,
        "core_noise_pct": core_summary.get("noise_pct"),
        "core_coverage_pct": 100.0 - float(core_summary.get("noise_pct") or 0.0),
        "soft_assigned_customers": 0,
        "soft_assigned_pct": 0.0,
        "soft_assignment_confidence_mean": None,
        "soft_assignment_confidence_p10": None,
        "soft_assignment_confidence_min": None,
    }

    if allow_noise_assignment and np.any(labels < 0) and np.unique(labels[labels >= 0]).shape[0] >= 2:
        labels, assigned_mask, threshold, soft_confidence_scores = _soft_assign_noise_labels(
            X,
            labels,
            strategy=noise_assignment_strategy,
        )
        probabilities, assignment_confidence_score, assignment_confidence_type, assignment_sources = _soft_assignment_metadata(
            labels,
            core_probabilities,
            assigned_mask,
            soft_confidence_scores,
            assignment_source,
            noise_assignment_strategy,
        )
        assigned_confidences = soft_confidence_scores[assigned_mask]
        assigned_confidences = assigned_confidences[np.isfinite(assigned_confidences)]
        soft_assignment_info.update(
            {
                "soft_assignment_strategy": noise_assignment_strategy,
                "soft_assignment_distance_threshold": threshold,
                "soft_assigned_customers": int(np.sum(assigned_mask)),
                "soft_assigned_pct": 100.0 * float(np.sum(assigned_mask)) / max(labels.shape[0], 1),
                "soft_assignment_confidence_mean": float(np.mean(assigned_confidences))
                if assigned_confidences.size
                else None,
                "soft_assignment_confidence_p10": float(np.quantile(assigned_confidences, 0.10))
                if assigned_confidences.size
                else None,
                "soft_assignment_confidence_min": float(np.min(assigned_confidences))
                if assigned_confidences.size
                else None,
            }
        )
        if "SoftNoiseAssignment" not in algorithm_name:
            algorithm_name = f"{algorithm_name}_SoftNoiseAssignment"
    elif np.any(labels < 0):
        assignment_sources = [
            assignment_source if int(label) >= 0 else "hdbscan_noise_unassigned"
            for label in labels
        ]

    metrics = evaluate_labels(X, labels, probabilities=probabilities, cfg=cfg)
    cluster_persistence = getattr(model, "cluster_persistence_", None)
    persistence_metrics = {
        "hdbscan_cluster_persistence_mean": None,
        "hdbscan_cluster_persistence_min": None,
        "hdbscan_cluster_persistence_max": None,
    }
    if cluster_persistence is not None and len(cluster_persistence):
        persistence_values = np.asarray(cluster_persistence, dtype=np.float64)
        persistence_values = persistence_values[np.isfinite(persistence_values)]
        if persistence_values.size:
            persistence_metrics = {
                "hdbscan_cluster_persistence_mean": float(np.mean(persistence_values)),
                "hdbscan_cluster_persistence_min": float(np.min(persistence_values)),
                "hdbscan_cluster_persistence_max": float(np.max(persistence_values)),
            }
    passes_gate, gate_reason = quality_gate_result(metrics, cfg=cfg)
    variant = (
        f"hdbscan_mcs{cfg.get('hdbscan.min_cluster_size')}_"
        f"ms{cfg.get('hdbscan.min_samples')}_{cfg.get('hdbscan.cluster_selection_method')}"
    )
    if variant_prefix:
        variant = f"{variant_prefix}_{variant}"
    if allow_noise_assignment and soft_assignment_info["soft_assignment_strategy"] is not None:
        variant = f"{variant}_soft_{noise_assignment_strategy}"
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
        "assignment_policy": "hdbscan_core_plus_soft_noise_assignment"
        if allow_noise_assignment and soft_assignment_info["soft_assignment_strategy"] is not None
        else "hard_hdbscan_core_noise_retained",
        "soft_assignment_enabled": bool(
            allow_noise_assignment and soft_assignment_info["soft_assignment_strategy"] is not None
        ),
        **soft_assignment_info,
        **metrics,
        **persistence_metrics,
        "final_noise_pct": metrics.get("noise_pct"),
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
        assignment_confidence_type=assignment_confidence_type,
        assignment_confidence_score=assignment_confidence_score,
        assignment_source=assignment_sources,
    ).write_parquet(assignment_path)
    pl.DataFrame([result]).write_parquet(results_path)
    write_artifact_metadata(assignment_path, cache_metadata)
    write_artifact_metadata(results_path, cache_metadata)
    _save_model(
        fitted_model_dir / f"{output_prefix}_model.pkl",
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_cols,
            "selection": result,
            "soft_assignment": {
                "enabled": allow_noise_assignment,
                "strategy": soft_assignment_info["soft_assignment_strategy"],
                "distance_threshold": soft_assignment_info["soft_assignment_distance_threshold"],
            },
        },
    )
    log_event(
        "Stage 6 HDBSCAN",
        "model evaluated",
        cfg=cfg,
        clusters=result["cluster_count"],
        noise_pct=result["noise_pct"],
        passes_gate=passes_gate,
        source=assignment_source,
        soft_assigned=soft_assignment_info["soft_assigned_customers"],
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
        "assignment_schema_version": 2,
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
