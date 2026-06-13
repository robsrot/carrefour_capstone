"""Optional experiment sandboxes for upstream pipeline choices."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.customer_embeddings import build_customer_embeddings
from src.data_loader import load_prepared_transactions
from src.experiment_reporting import write_summary_artifacts
from src.item2vec import save_product_embeddings, train_item2vec
from src.progress import log_event, stage_timer
from src.utils import collect_streaming, deterministic_sample_indices, frame_to_numpy, numeric_feature_columns


DEFAULT_EMBEDDING_EXPERIMENT_NAME = "item2vec_embedding_sandbox"
DEFAULT_EMBEDDING_SAMPLE_SIZE = 500
DEFAULT_EMBEDDING_NEIGHBORS = 8
DEFAULT_CUSTOMER_EMBEDDING_EXPERIMENT_NAME = "customer_embedding_sandbox"
DEFAULT_CUSTOMER_EMBEDDING_SAMPLE_SIZE = 12000
DEFAULT_FEATURE_SET_EXPERIMENT_NAME = "customer_feature_set_sandbox"
DEFAULT_FEATURE_SET_SAMPLE_SIZE = 12000
DEFAULT_FEATURE_SET_K_VALUES = [10, 12, 15]


def run_embedding_experiments(
    basket_path: str | Path,
    trials: list[dict[str, Any]] | None = None,
    experiment_name: str = DEFAULT_EMBEDDING_EXPERIMENT_NAME,
    sample_size: int = DEFAULT_EMBEDDING_SAMPLE_SIZE,
    neighbors: int = DEFAULT_EMBEDDING_NEIGHBORS,
    transactions: pl.LazyFrame | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Train Item2Vec variants, then rank their product-embedding quality."""

    _ensure_experiments_enabled(cfg)
    training = run_item2vec_training_experiments(
        basket_path,
        trials=trials,
        experiment_name=experiment_name,
        force=force,
        cfg=cfg,
    )
    evaluation = evaluate_product_embedding_experiments(
        training["embedding_paths"],
        experiment_name=experiment_name,
        sample_size=sample_size,
        neighbors=neighbors,
        transactions=transactions,
        cfg=cfg,
    )
    return {
        **training,
        "diagnostics": evaluation["diagnostics"],
        "best": evaluation["best"],
    }


def run_item2vec_training_experiments(
    basket_path: str | Path,
    trials: list[dict[str, Any]] | None = None,
    experiment_name: str = DEFAULT_EMBEDDING_EXPERIMENT_NAME,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Stage 2 sandbox: train Item2Vec trial variants and save embedding tables."""

    _ensure_experiments_enabled(cfg)
    experiment_slug = _trial_slug(experiment_name)
    selected_trials = [dict(trial) for trial in (trials or [])]
    if not selected_trials:
        raise ValueError("No Item2Vec experiment trials were provided.")

    rows: list[dict[str, Any]] = []
    model_paths: dict[str, Path] = {}
    embedding_paths: dict[str, Path] = {}
    experiment_dir = cfg.experiments / experiment_slug

    with stage_timer(
        "Item2Vec training sandbox",
        "training Item2Vec trials",
        cfg=cfg,
        experiment=experiment_slug,
        trials=len(selected_trials),
    ):
        for trial in selected_trials:
            trial_name = _trial_slug(str(trial.get("name", "trial")))
            overrides = dict(trial.get("word2vec", {}) or {})
            trial_cfg = _config_with_section_overrides(cfg, "word2vec", overrides)
            model_path = experiment_dir / f"{trial_name}_word2vec_product.model"
            embedding_path = experiment_dir / f"{trial_name}_product_embeddings.parquet"

            log_event(
                "Item2Vec training sandbox",
                "starting trial",
                cfg=trial_cfg,
                experiment=experiment_slug,
                trial=trial_name,
                overrides=json.dumps(overrides, sort_keys=True),
            )
            model = train_item2vec(
                basket_path,
                model_path=model_path,
                force=force,
                verbose=True,
                cfg=trial_cfg,
            )
            saved_embedding_path = save_product_embeddings(
                model,
                output_path=embedding_path,
                force=force,
                cfg=trial_cfg,
            )
            model_paths[trial_name] = model_path
            embedding_paths[trial_name] = saved_embedding_path
            rows.append(
                {
                    "experiment_name": experiment_slug,
                    "trial_name": trial_name,
                    "word2vec_overrides": json.dumps(overrides, sort_keys=True),
                    "model_path": str(model_path),
                    "product_embeddings_path": str(saved_embedding_path),
                }
            )

    manifest = _write_experiment_rows(
        rows,
        output_base=experiment_dir / f"{experiment_slug}_item2vec_training_manifest",
        cfg=cfg,
    )
    return {
        "manifest": manifest,
        "model_paths": model_paths,
        "embedding_paths": embedding_paths,
    }


def evaluate_product_embedding_experiments(
    embedding_paths: Mapping[str, str | Path],
    experiment_name: str = DEFAULT_EMBEDDING_EXPERIMENT_NAME,
    sample_size: int = DEFAULT_EMBEDDING_SAMPLE_SIZE,
    neighbors: int = DEFAULT_EMBEDDING_NEIGHBORS,
    transactions: pl.LazyFrame | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Stage 3 sandbox: rank existing product embedding tables."""

    _ensure_experiments_enabled(cfg)
    experiment_slug = _trial_slug(experiment_name)
    rows: list[dict[str, Any]] = []

    with stage_timer(
        "Product embedding validation sandbox",
        "evaluating embedding tables",
        cfg=cfg,
        experiment=experiment_slug,
        trials=len(embedding_paths),
    ):
        for trial_name, embedding_path in embedding_paths.items():
            path = Path(embedding_path)
            if not path.exists():
                raise FileNotFoundError(f"Product embedding table not found for {trial_name}: {path}")
            metrics = evaluate_product_embedding_quality(
                path,
                transactions=transactions,
                trial_name=_trial_slug(str(trial_name)),
                sample_size=sample_size,
                neighbors=neighbors,
                cfg=cfg,
            )
            metrics.update(
                {
                    "experiment_name": experiment_slug,
                    "product_embeddings_path": str(path),
                }
            )
            rows.append(metrics)

    diagnostics = _rank_and_write(
        rows,
        score_col="embedding_quality_score",
        output_base=cfg.experiments / experiment_slug / f"{experiment_slug}_embedding_diagnostics",
        rank_col="embedding_sandbox_rank",
        selected_col="selected_in_embedding_sandbox",
        cfg=cfg,
    )
    return {"diagnostics": diagnostics, "best": _first_row(diagnostics["parquet"])}


def evaluate_product_embedding_quality(
    embeddings_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    trial_name: str | None = None,
    sample_size: int = DEFAULT_EMBEDDING_SAMPLE_SIZE,
    neighbors: int = DEFAULT_EMBEDDING_NEIGHBORS,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Compute lightweight product-neighborhood diagnostics for one embedding table."""

    emb = pl.read_parquet(embeddings_path).sort("idarticu")
    feature_cols = numeric_feature_columns(emb, exclude=("idarticu",))
    if not feature_cols:
        raise ValueError(f"No embedding columns found in {embeddings_path}")

    matrix = frame_to_numpy(emb, feature_cols)
    ids = emb["idarticu"].to_numpy()
    norms = np.linalg.norm(matrix, axis=1)
    normalized = matrix / np.maximum(norms[:, None], 1e-12)
    n_neighbors = int(neighbors)
    sample_indices = deterministic_sample_indices(len(ids), int(sample_size), cfg.random_seed)
    sector_lookup, product_universe = _sector_lookup(transactions, cfg)

    mean_neighbor_scores: list[float] = []
    same_sector_at_1 = 0
    same_sector_pairs = 0
    eligible_sources = 0
    eligible_pairs = 0

    for idx in sample_indices:
        product_id = int(ids[idx])
        source_sector = sector_lookup.get(product_id)
        scores = normalized @ normalized[idx]
        order = np.argsort(-scores)
        neighbors = [int(neighbor_idx) for neighbor_idx in order if int(neighbor_idx) != int(idx)][:n_neighbors]
        if not neighbors:
            continue
        mean_neighbor_scores.extend(float(scores[neighbor_idx]) for neighbor_idx in neighbors)
        if source_sector is None:
            continue
        eligible_sources += 1
        neighbor_sectors = [sector_lookup.get(int(ids[neighbor_idx])) for neighbor_idx in neighbors]
        if neighbor_sectors and neighbor_sectors[0] == source_sector:
            same_sector_at_1 += 1
        for neighbor_sector in neighbor_sectors:
            if neighbor_sector is None:
                continue
            eligible_pairs += 1
            if neighbor_sector == source_sector:
                same_sector_pairs += 1

    coverage_pct = 100.0 * len(ids) / product_universe if product_universe else None
    same_sector_at_1_pct = 100.0 * same_sector_at_1 / eligible_sources if eligible_sources else None
    same_sector_neighbor_share_pct = 100.0 * same_sector_pairs / eligible_pairs if eligible_pairs else None
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))
    norm_stability = max(0.0, min(1.0, 1.0 - std_norm / max(mean_norm, 1e-12)))
    embedding_quality_score = _embedding_score(
        coverage_pct=coverage_pct,
        same_sector_at_1_pct=same_sector_at_1_pct,
        same_sector_neighbor_share_pct=same_sector_neighbor_share_pct,
        norm_stability=norm_stability,
    )

    return {
        "trial_name": trial_name,
        "product_count": int(len(ids)),
        "product_universe": int(product_universe) if product_universe else None,
        "product_vocab_coverage_pct": coverage_pct,
        "vector_dims": int(len(feature_cols)),
        "sampled_products": int(len(sample_indices)),
        "neighbor_k": int(n_neighbors),
        "mean_vector_norm": mean_norm,
        "std_vector_norm": std_norm,
        "norm_stability": norm_stability,
        "mean_neighbor_cosine": float(np.mean(mean_neighbor_scores)) if mean_neighbor_scores else None,
        "same_sector_at_1_pct": same_sector_at_1_pct,
        "same_sector_neighbor_share_pct": same_sector_neighbor_share_pct,
        "embedding_quality_score": embedding_quality_score,
        "selection_reason": (
            "Ranked by product vocabulary coverage, sampled same-sector nearest-neighbor "
            "coherence, and vector norm stability. Use the qualitative neighbor report as "
            "the final sanity check because good complements can cross sectors."
        ),
    }


def run_customer_embedding_experiments(
    embeddings_path: str | Path,
    trials: list[dict[str, Any]] | None = None,
    experiment_name: str = DEFAULT_CUSTOMER_EMBEDDING_EXPERIMENT_NAME,
    sample_size: int = DEFAULT_CUSTOMER_EMBEDDING_SAMPLE_SIZE,
    transactions: pl.LazyFrame | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Stage 4 sandbox: build and rank customer-vector aggregation variants."""

    _ensure_experiments_enabled(cfg)
    experiment_slug = _trial_slug(experiment_name)
    selected_trials = [dict(trial) for trial in (trials or [])]
    if not selected_trials:
        raise ValueError("No customer embedding experiment trials were provided.")

    experiment_dir = cfg.experiments / experiment_slug
    rows: list[dict[str, Any]] = []
    customer_embedding_paths: dict[str, Path] = {}

    with stage_timer(
        "Customer embedding sandbox",
        "building customer-vector trials",
        cfg=cfg,
        experiment=experiment_slug,
        trials=len(selected_trials),
    ):
        for trial in selected_trials:
            trial_name = _trial_slug(str(trial.get("name", "trial")))
            weight_strategy = str(trial.get("weight_strategy", "quantity"))
            normalize_vectors = bool(trial.get("normalize_vectors", False))
            output_path = experiment_dir / f"{trial_name}_customer_embeddings.parquet"

            log_event(
                "Customer embedding sandbox",
                "starting trial",
                cfg=cfg,
                experiment=experiment_slug,
                trial=trial_name,
                weight_strategy=weight_strategy,
                normalize_vectors=normalize_vectors,
            )
            customer_path = build_customer_embeddings(
                embeddings_path,
                transactions=transactions,
                output_path=output_path,
                weight_strategy=weight_strategy,
                normalize_vectors=normalize_vectors,
                force=force,
                cfg=cfg,
            )
            customer_embedding_paths[trial_name] = customer_path
            metrics = evaluate_customer_embedding_quality(
                customer_path,
                trial_name=trial_name,
                sample_size=sample_size,
                cfg=cfg,
            )
            metrics.update(
                {
                    "experiment_name": experiment_slug,
                    "customer_embeddings_path": str(customer_path),
                    "weight_strategy": weight_strategy,
                    "normalize_vectors": normalize_vectors,
                }
            )
            rows.append(metrics)

    diagnostics = _rank_and_write(
        rows,
        score_col="customer_embedding_quality_score",
        output_base=experiment_dir / f"{experiment_slug}_customer_embedding_diagnostics",
        rank_col="customer_embedding_sandbox_rank",
        selected_col="selected_in_customer_embedding_sandbox",
        cfg=cfg,
    )
    return {
        "diagnostics": diagnostics,
        "customer_embedding_paths": customer_embedding_paths,
        "best": _first_row(diagnostics["parquet"]),
    }


def evaluate_customer_embedding_quality(
    customer_embeddings_path: str | Path,
    trial_name: str | None = None,
    sample_size: int = DEFAULT_CUSTOMER_EMBEDDING_SAMPLE_SIZE,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Compute lightweight health diagnostics for one customer embedding table."""

    from sklearn.decomposition import PCA

    df = pl.read_parquet(customer_embeddings_path).sort("cliente")
    feature_cols = [col for col in df.columns if col.startswith("emb_")]
    if not feature_cols:
        raise ValueError(f"No customer embedding columns found in {customer_embeddings_path}")

    sample_idx = deterministic_sample_indices(df.height, int(sample_size), cfg.random_seed)
    raw = frame_to_numpy(df, feature_cols)[sample_idx]
    finite_pct = float(np.isfinite(raw).mean() * 100.0)
    X = np.nan_to_num(raw, copy=True)
    norms = np.linalg.norm(X, axis=1)
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))
    norm_cv = float(std_norm / max(mean_norm, 1e-12))
    zero_vector_pct = float(np.mean(norms <= 1e-12) * 100.0)
    dim_std = np.std(X, axis=0)
    dead_dimension_count = int(np.sum(dim_std <= 1e-12))

    pca_components = min(X.shape[1], X.shape[0] - 1, 50)
    if pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=cfg.random_seed)
        pca.fit(X)
        explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        cumulative = np.cumsum(explained)
        components_for_80pct = int(np.searchsorted(cumulative, 0.80) + 1) if len(cumulative) else None
        components_for_90pct = int(np.searchsorted(cumulative, 0.90) + 1) if len(cumulative) else None
        effective_dim = _effective_dimension(explained)
    else:
        components_for_80pct = None
        components_for_90pct = None
        effective_dim = 0.0

    mean_nearest_neighbor_cosine = _mean_nearest_neighbor_cosine(X, cfg)
    score = _customer_embedding_score(
        finite_pct=finite_pct,
        zero_vector_pct=zero_vector_pct,
        dead_dimension_count=dead_dimension_count,
        n_features=len(feature_cols),
        effective_dim=effective_dim,
        norm_cv=norm_cv,
        mean_nearest_neighbor_cosine=mean_nearest_neighbor_cosine,
    )

    return {
        "trial_name": trial_name,
        "customer_count": int(df.height),
        "sampled_customers": int(X.shape[0]),
        "vector_dims": int(len(feature_cols)),
        "finite_pct": finite_pct,
        "zero_vector_pct": zero_vector_pct,
        "dead_dimension_count": dead_dimension_count,
        "mean_vector_norm": mean_norm,
        "std_vector_norm": std_norm,
        "norm_cv": norm_cv,
        "effective_dimension": effective_dim,
        "pca_components_for_80pct": components_for_80pct,
        "pca_components_for_90pct": components_for_90pct,
        "mean_nearest_neighbor_cosine": mean_nearest_neighbor_cosine,
        "customer_embedding_quality_score": score,
        "selection_reason": (
            "Ranked by finite-value health, nonzero customer coverage, live embedding dimensions, "
            "retained effective dimensionality, vector norm stability, and sampled nearest-neighbor "
            "structure. This is an upstream diagnostic; Stage 5B and Stage 6 still decide whether the "
            "variant produces useful tribes."
        ),
    }


def run_feature_set_experiments(
    feature_paths: Mapping[str, str | Path],
    experiment_name: str = DEFAULT_FEATURE_SET_EXPERIMENT_NAME,
    k_values: list[int] | None = None,
    sample_size: int = DEFAULT_FEATURE_SET_SAMPLE_SIZE,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Compare customer-vector/feature-set variants with quick clustering diagnostics."""

    _ensure_experiments_enabled(cfg)
    experiment_slug = _trial_slug(experiment_name)
    selected_k_values = k_values or DEFAULT_FEATURE_SET_K_VALUES
    rows: list[dict[str, Any]] = []

    with stage_timer(
        "Feature-set sandbox",
        "running quick feature-set diagnostics",
        cfg=cfg,
        experiment=experiment_slug,
        feature_sets=len(feature_paths),
    ):
        for name, path in feature_paths.items():
            rows.extend(
                _evaluate_feature_set(
                    feature_set_name=str(name),
                    feature_path=Path(path),
                    k_values=[int(k) for k in selected_k_values],
                    sample_size=int(sample_size),
                    cfg=cfg,
                )
            )

    diagnostics = _rank_and_write(
        rows,
        score_col="feature_set_quality_score",
        output_base=cfg.experiments / experiment_slug / f"{experiment_slug}_diagnostics",
        rank_col="feature_set_sandbox_rank",
        selected_col="selected_in_feature_set_sandbox",
        cfg=cfg,
    )
    return {"diagnostics": diagnostics, "best": _first_row(diagnostics["parquet"])}


def _evaluate_feature_set(
    feature_set_name: str,
    feature_path: Path,
    k_values: list[int],
    sample_size: int,
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    df = pl.read_parquet(feature_path).sort("cliente")
    feature_cols = numeric_feature_columns(df)
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found in {feature_path}")

    X = np.nan_to_num(frame_to_numpy(df, feature_cols), copy=False)
    sample_idx = deterministic_sample_indices(X.shape[0], sample_size, cfg.random_seed)
    X_sample = X[sample_idx]
    raw_std = np.std(X_sample, axis=0)
    dead_feature_count = int(np.sum(raw_std <= 1e-12))
    finite_pct = float(np.isfinite(X_sample).mean() * 100.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample).astype(np.float32)
    pca_components = max(1, min(X_scaled.shape[1], X_scaled.shape[0] - 1, 50))
    pca = PCA(n_components=pca_components, random_state=cfg.random_seed)
    pca.fit(X_scaled)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative = np.cumsum(explained)
    components_for_80pct = int(np.searchsorted(cumulative, 0.80) + 1) if len(cumulative) else None
    components_for_90pct = int(np.searchsorted(cumulative, 0.90) + 1) if len(cumulative) else None
    effective_dim = _effective_dimension(explained)

    rows: list[dict[str, Any]] = []
    valid_k_values = [k for k in k_values if 1 < k < X_scaled.shape[0]]
    for progress_idx, k in enumerate(valid_k_values, start=1):
        log_event(
            "Feature-set sandbox",
            "fitting quick KMeans",
            cfg=cfg,
            feature_set=feature_set_name,
            k=k,
            progress=f"{progress_idx}/{len(valid_k_values)}",
        )
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=cfg.random_seed,
            batch_size=int(cfg.get("kmeans.batch_size", 4096)),
            n_init=int(cfg.get("kmeans.n_init", 10)),
        )
        labels = model.fit_predict(X_scaled)
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) <= 1:
            silhouette = None
            davies = None
        else:
            silhouette = float(silhouette_score(X_scaled, labels))
            davies = float(davies_bouldin_score(X_scaled, labels))
        cluster_size_cv = float(np.std(counts) / max(np.mean(counts), 1e-12))
        score = _feature_set_score(
            silhouette=silhouette,
            cluster_size_cv=cluster_size_cv,
            dead_feature_count=dead_feature_count,
            n_features=len(feature_cols),
            effective_dim=effective_dim,
            finite_pct=finite_pct,
        )
        rows.append(
            {
                "feature_set_name": feature_set_name,
                "feature_path": str(feature_path),
                "rows": int(df.height),
                "sampled_rows": int(X_sample.shape[0]),
                "feature_count": int(len(feature_cols)),
                "dead_feature_count": dead_feature_count,
                "finite_pct": finite_pct,
                "mean_feature_std": float(np.mean(raw_std)),
                "effective_dimension": effective_dim,
                "pca_components_for_80pct": components_for_80pct,
                "pca_components_for_90pct": components_for_90pct,
                "k": int(k),
                "cluster_size_cv": cluster_size_cv,
                "silhouette": silhouette,
                "davies_bouldin": davies,
                "feature_set_quality_score": score,
                "selection_reason": (
                    "Ranked by quick KMeans separability near the target tribe-count hypothesis, "
                    "cluster balance, finite-value health, and retained effective dimensionality. "
                    "This is a pre-clustering diagnostic; Stage 6 plus product-lift profiles still decide the final model."
                ),
            }
        )
    return rows


def _sector_lookup(
    transactions: pl.LazyFrame | None,
    cfg: PipelineConfig,
) -> tuple[dict[int, Any], int | None]:
    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    columns = set(lf.collect_schema().names())
    if "idsector" not in columns:
        return {}, None
    meta = collect_streaming(lf.select(["idarticu", "idsector"]).unique(subset=["idarticu"]))
    return {int(row["idarticu"]): row["idsector"] for row in meta.iter_rows(named=True)}, meta.height


def _embedding_score(
    coverage_pct: float | None,
    same_sector_at_1_pct: float | None,
    same_sector_neighbor_share_pct: float | None,
    norm_stability: float,
) -> float:
    coverage = 1.0 if coverage_pct is None else max(0.0, min(1.0, coverage_pct / 100.0))
    same_at_1 = 0.5 if same_sector_at_1_pct is None else max(0.0, min(1.0, same_sector_at_1_pct / 100.0))
    same_share = (
        0.5
        if same_sector_neighbor_share_pct is None
        else max(0.0, min(1.0, same_sector_neighbor_share_pct / 100.0))
    )
    return float(100.0 * (0.25 * coverage + 0.25 * same_at_1 + 0.35 * same_share + 0.15 * norm_stability))


def _mean_nearest_neighbor_cosine(
    X: np.ndarray,
    cfg: PipelineConfig,
    max_rows: int = 2000,
) -> float | None:
    if X.shape[0] < 2:
        return None
    sample_idx = deterministic_sample_indices(X.shape[0], min(max_rows, X.shape[0]), cfg.random_seed)
    sample = X[sample_idx]
    norms = np.linalg.norm(sample, axis=1)
    normalized = sample / np.maximum(norms[:, None], 1e-12)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    nearest = np.max(similarities, axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]
    if len(finite_nearest) == 0:
        return None
    return float(np.mean(finite_nearest))


def _customer_embedding_score(
    finite_pct: float,
    zero_vector_pct: float,
    dead_dimension_count: int,
    n_features: int,
    effective_dim: float,
    norm_cv: float,
    mean_nearest_neighbor_cosine: float | None,
) -> float:
    finite_share = max(0.0, min(1.0, finite_pct / 100.0))
    nonzero_share = 1.0 - max(0.0, min(1.0, zero_vector_pct / 100.0))
    live_dimension_share = 1.0 - dead_dimension_count / max(n_features, 1)
    dim_denominator = max(1, min(n_features, 50))
    effective_dimension_share = max(0.0, min(1.0, effective_dim / dim_denominator))
    norm_stability = max(0.0, min(1.0, 1.0 - min(norm_cv, 1.0)))
    neighbor_structure = (
        0.5
        if mean_nearest_neighbor_cosine is None
        else max(0.0, min(1.0, (mean_nearest_neighbor_cosine + 1.0) / 2.0))
    )
    return float(
        100.0
        * (
            0.25 * finite_share
            + 0.20 * nonzero_share
            + 0.15 * live_dimension_share
            + 0.15 * effective_dimension_share
            + 0.15 * norm_stability
            + 0.10 * neighbor_structure
        )
    )


def _feature_set_score(
    silhouette: float | None,
    cluster_size_cv: float,
    dead_feature_count: int,
    n_features: int,
    effective_dim: float,
    finite_pct: float,
) -> float:
    sil = -1.0 if silhouette is None else silhouette
    balance = max(0.0, 1.0 - min(cluster_size_cv, 2.0) / 2.0)
    live_feature_share = 1.0 - dead_feature_count / max(n_features, 1)
    finite_share = finite_pct / 100.0
    dim_share = min(effective_dim / max(n_features, 1), 1.0)
    return float(100.0 * (0.55 * ((sil + 1.0) / 2.0) + 0.20 * balance + 0.10 * live_feature_share + 0.10 * finite_share + 0.05 * dim_share))


def _effective_dimension(explained_variance_ratio: np.ndarray) -> float:
    total = float(np.sum(explained_variance_ratio))
    if not np.isfinite(total) or total <= 0:
        return 0.0
    probs = explained_variance_ratio / total
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return float(np.exp(entropy)) if np.isfinite(entropy) else 0.0


def _rank_and_write(
    rows: list[dict[str, Any]],
    score_col: str,
    output_base: Path,
    rank_col: str,
    selected_col: str,
    cfg: PipelineConfig,
) -> dict[str, Path]:
    if not rows:
        raise ValueError("No experiment rows were produced.")

    ranked = sorted(rows, key=lambda row: _score_value(row.get(score_col)), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row[rank_col] = rank
        row[selected_col] = rank == 1

    parquet_path = output_base.with_suffix(".parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.from_dicts(ranked, infer_schema_length=None)
    df.write_parquet(parquet_path)
    summaries = write_summary_artifacts(
        df,
        output_base=output_base,
        title=output_base.stem.replace("_", " ").title(),
        priority_columns=[rank_col, selected_col, score_col],
        cfg=cfg,
    )
    log_event(
        "Experiment sandbox",
        "wrote diagnostics",
        cfg=cfg,
        rows=len(ranked),
        parquet=parquet_path,
        summary_csv=summaries["summary_csv"],
        summary_md=summaries["summary_md"],
    )
    return {"parquet": parquet_path, **summaries}


def _write_experiment_rows(
    rows: list[dict[str, Any]],
    output_base: Path,
    cfg: PipelineConfig,
) -> dict[str, Path]:
    if not rows:
        raise ValueError("No experiment rows were produced.")

    parquet_path = output_base.with_suffix(".parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.from_dicts(rows, infer_schema_length=None)
    df.write_parquet(parquet_path)
    summaries = write_summary_artifacts(
        df,
        output_base=output_base,
        title=output_base.stem.replace("_", " ").title(),
        priority_columns=list(df.columns),
        cfg=cfg,
    )
    log_event(
        "Experiment sandbox",
        "wrote manifest",
        cfg=cfg,
        rows=len(rows),
        parquet=parquet_path,
        summary_csv=summaries["summary_csv"],
        summary_md=summaries["summary_md"],
    )
    return {"parquet": parquet_path, **summaries}


def _first_row(path: Path) -> dict[str, Any]:
    return pl.read_parquet(path).row(0, named=True)


def _score_value(value: Any) -> float:
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _ensure_experiments_enabled(cfg: PipelineConfig) -> None:
    if not cfg.experiments_enabled:
        raise RuntimeError(
            f"Experiment sandboxes are disabled for mode {cfg.mode!r}. "
            "Run experiments in dev, promote the winning settings to YAML, then run the official prod pipeline."
        )


def _trial_slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.strip().lower()).strip("_") or "trial"


def _config_with_section_overrides(
    cfg: PipelineConfig,
    section: str,
    overrides: dict[str, Any],
) -> PipelineConfig:
    values = deepcopy(cfg.values)
    base_section = values.get(section, {}) or {}
    values[section] = {**base_section, **overrides}
    return replace(cfg, values=values)
