"""Phase 3 — Dimensionality reduction for 1.48M customer vectors.

Two methods are compared:
  UMAP  (primary)   — preserves non-linear topology; better for density-based clustering
  PCA   (baseline)  — linear, fast, interpretable; run to measure structural loss

Scale strategy for UMAP (1.48M × 100 is too large for a single fit call):
  1. Fit UMAP on a stratified random sample of UMAP_FIT_SAMPLE rows.
  2. Transform the remaining customers in a single .transform() call.
  3. This is O(sample × log(sample)) for fit, O(n) for transform.
  PCA uses full-population SVD — sklearn handles 1.48M × 100 natively in ~30 s.

Public API
----------
reduce_umap_cluster()  → data/processed/umap_cluster.parquet
reduce_umap_viz()      → data/processed/umap_viz.parquet
reduce_pca()           → data/processed/pca_cluster.parquet

The number of clustering dimensions is controlled by umap.cluster_dims in base.yaml
(currently 50). Filenames intentionally omit the dimension count so they never
become stale when that config value changes.  Source-specific runs (e.g.
reduce_umap_cluster(cache_path=...umap_cluster_item2vec.parquet)) use explicit
paths supplied by the caller; the module-level defaults are only fallbacks.

All functions accept a Polars DataFrame with columns [cliente, vector, promo_rate]
and return one with [cliente, <dim columns>, promo_rate].
"""
from __future__ import annotations

import logging
from pathlib import Path

import pickle

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
import umap

from src.config import (
    DATA_PROCESSED,
    FEATURE_WEIGHT_KPI,
    FEATURE_WEIGHT_PROMO,
    FEATURE_WEIGHT_STORE,
    RANDOM_SEED,
    UMAP_CLUSTER_DIMS,
    UMAP_VIZ_DIMS,
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST_CLUSTER,
    UMAP_MIN_DIST_VIZ,
    UMAP_METRIC,
    UMAP_FIT_SAMPLE,
)

_log = logging.getLogger(__name__)

# UMAP_FIT_SAMPLE is loaded from config (base: 300k prod, 30k dev).

_UMAP_CLUSTER_CACHE = DATA_PROCESSED / "umap_cluster.parquet"
_UMAP_VIZ_CACHE     = DATA_PROCESSED / "umap_viz.parquet"
_PCA_CACHE          = DATA_PROCESSED / "pca_cluster.parquet"
_PCA_MODEL_CACHE    = DATA_PROCESSED / "pca_model.pkl"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _vectors_to_numpy(df: pl.DataFrame) -> np.ndarray:
    """Extract the 'vector' list column → (n, dims) float32 array."""
    return np.array(df["vector"].to_list(), dtype=np.float32)


def _df_from_embedding(
    cliente: pl.Series,
    embedding: np.ndarray,
    promo_rate: pl.Series,
    prefix: str,
) -> pl.DataFrame:
    """Build a Polars DataFrame from a numpy embedding array."""
    n_dims = embedding.shape[1]
    cols = {f"{prefix}{i}": pl.Series(embedding[:, i]) for i in range(n_dims)}
    return pl.DataFrame({"cliente": cliente, **cols, "promo_rate": promo_rate})


def _embedding_columns(df: pl.DataFrame) -> list[str]:
    """Return dimensionality-reduction columns regardless of reducer prefix."""
    return [c for c in df.columns if c not in ("cliente", "promo_rate")]


# ─── UMAP clustering embedding (20D) ─────────────────────────────────────────

def reduce_umap_cluster(
    customer_vectors: pl.DataFrame | None = None,
    *,
    force: bool = False,
    n_jobs: int = 1,
    cache_path: Path | None = None,
) -> pl.DataFrame:
    """UMAP 100D → 20D embedding for HDBSCAN clustering.

    Fits on UMAP_FIT_SAMPLE random customers, transforms the rest.
    Cached to umap_cluster.parquet, unless cache_path is supplied.

    Parameters
    ----------
    customer_vectors : DataFrame with [cliente, vector, promo_rate].
                       If None, loads customer_vectors_weighted.parquet.
    n_jobs           : parallel threads for UMAP fit (-1 = all cores).
    """
    cache = Path(cache_path) if cache_path is not None else _UMAP_CLUSTER_CACHE
    if cache.exists() and not force:
        n = pl.scan_parquet(cache).select(pl.len()).collect().item()
        _log.info("UMAP cluster cache hit — %s customers", f"{n:,}")
        return pl.read_parquet(cache)

    if customer_vectors is None:
        _log.info("Loading customer_vectors_weighted.parquet ...")
        customer_vectors = pl.read_parquet(DATA_PROCESSED / "customer_vectors_weighted.parquet")

    X = _vectors_to_numpy(customer_vectors)          # (N, 100)
    # Append promo_rate as 101st feature so promotional sensitivity influences UMAP topology,
    # not just post-hoc profiling — this is the key axis separating promo-surfers from loyalists
    if FEATURE_WEIGHT_PROMO > 0:
        promo = customer_vectors["promo_rate"].fill_null(0.0).to_numpy().reshape(-1, 1).astype(np.float32)
        X = np.hstack([X, promo * FEATURE_WEIGHT_PROMO])
        _log.info("  Promo feature appended (weight=%.2f)", FEATURE_WEIGHT_PROMO)
    else:
        _log.info("  Promo feature skipped (weight=0)")

    # Append store affinity features — separates store-format loyalists from cross-format shoppers
    # as a first-class UMAP dimension, not just a post-hoc label
    if FEATURE_WEIGHT_STORE > 0:
        _store_path = DATA_PROCESSED / "customer_store_features.parquet"
        if not _store_path.exists():
            from src.customer_vectors import build_store_features as _bsf
            _store_df = _bsf()
        else:
            _store_df = pl.read_parquet(_store_path)
        _store_cols = [c for c in _store_df.columns if c != "cliente"]
        _store_aligned = (
            customer_vectors.select("cliente")
            .join(_store_df, on="cliente", how="left")
            .fill_null(0.0)
            .select(_store_cols)
        )
        store_arr = _store_aligned.to_numpy().astype(np.float32)
        X = np.hstack([X, store_arr * FEATURE_WEIGHT_STORE])
        _log.info("  Store features appended: %d columns (weight=%.2f)", len(_store_cols), FEATURE_WEIGHT_STORE)
    else:
        _log.info("  Store features skipped (weight=0)")

    # Optional KPI features capture shopping intensity, but are disabled by default
    # so product type remains the primary clustering signal.
    _kpi_path = DATA_PROCESSED / "customer_kpis.parquet"
    if FEATURE_WEIGHT_KPI > 0 and _kpi_path.exists():
        _kpi_cols = ["total_spend_6m", "visit_count", "avg_basket_size", "unique_products"]
        _kpi_aligned = (
            customer_vectors.select("cliente")
            .join(
                pl.read_parquet(_kpi_path).select(["cliente"] + _kpi_cols),
                on="cliente", how="left",
            )
            .fill_null(0.0)
            .select(_kpi_cols)
        )
        _kpi_arr = np.log1p(_kpi_aligned.to_numpy().astype(np.float64))
        _kpi_arr = (_kpi_arr - _kpi_arr.mean(axis=0)) / (_kpi_arr.std(axis=0) + 1e-8)
        _kpi_arr = (_kpi_arr * FEATURE_WEIGHT_KPI).astype(np.float32)
        X = np.hstack([X, _kpi_arr])
        _log.info("  KPI features appended: %s (weight=%.2f)", _kpi_cols, FEATURE_WEIGHT_KPI)
    elif FEATURE_WEIGHT_KPI > 0:
        _log.warning("customer_kpis.parquet not found — KPI features skipped")
    else:
        _log.info("  KPI features skipped (weight=0)")

    N = len(X)

    # Sample for fit
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(N, size=min(UMAP_FIT_SAMPLE, N), replace=False)
    sample_idx.sort()
    X_sample = X[sample_idx]

    _log.info(
        "UMAP cluster fit: %s sample, input_dims=%d, n_neighbors=%d, n_components=%d, metric=%s ...",
        f"{len(X_sample):,}", X.shape[1], UMAP_N_NEIGHBORS, UMAP_CLUSTER_DIMS, UMAP_METRIC,
    )
    reducer = umap.UMAP(
        n_components=UMAP_CLUSTER_DIMS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST_CLUSTER,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
        low_memory=True,
    )
    reducer.fit(X_sample)
    _log.info("UMAP fit complete. Transforming all %s customers ...", f"{N:,}")

    embedding = reducer.transform(X).astype(np.float32)   # (N, 20)

    # UMAP transform can produce NaN for outlier points far from the training sample.
    nan_rows = np.isnan(embedding).any(axis=1)
    if nan_rows.any():
        col_means = np.nanmean(embedding, axis=0)
        embedding[nan_rows] = col_means
        _log.warning("  %d NaN embeddings replaced with column means", int(nan_rows.sum()))

    df = _df_from_embedding(
        customer_vectors["cliente"],
        embedding,
        customer_vectors["promo_rate"],
        prefix="u",
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache, compression="zstd")
    _log.info(
        "Saved %s UMAP cluster embeddings → %s  (%.1f MB on disk)",
        f"{len(df):,}", cache.name,
        cache.stat().st_size / 1024 ** 2,
    )
    return df


# ─── UMAP viz embedding (2D) ──────────────────────────────────────────────────

def reduce_umap_viz(
    umap_cluster: pl.DataFrame | None = None,
    *,
    force: bool = False,
    n_jobs: int = 1,
    cache_path: Path | None = None,
) -> pl.DataFrame:
    """UMAP 20D → 2D embedding for visualisation.

    Takes the 20D clustering embedding as input (not the raw 100D vectors) so
    the visualisation is geometrically consistent with the clustering.
    Cached to umap_viz.parquet.
    """
    cache = Path(cache_path) if cache_path is not None else _UMAP_VIZ_CACHE
    if cache.exists() and not force:
        n = pl.scan_parquet(cache).select(pl.len()).collect().item()
        _log.info("UMAP viz cache hit — %s customers", f"{n:,}")
        return pl.read_parquet(cache)

    if umap_cluster is None:
        _log.info("Loading umap_cluster.parquet ...")
        umap_cluster = pl.read_parquet(_UMAP_CLUSTER_CACHE)

    dim_cols = _embedding_columns(umap_cluster)
    if not dim_cols:
        raise ValueError("No embedding columns found for 2D visualisation.")
    X = umap_cluster.select(dim_cols).to_numpy().astype(np.float32)
    N = len(X)

    # Guard against NaN inherited from the cluster embedding (e.g. from a prior cached run)
    nan_rows = np.isnan(X).any(axis=1)
    if nan_rows.any():
        col_means = np.nanmean(X, axis=0)
        X[nan_rows] = col_means
        _log.warning("  %d NaN rows in input replaced with column means before viz fit", int(nan_rows.sum()))

    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(N, size=min(UMAP_FIT_SAMPLE, N), replace=False)
    sample_idx.sort()

    _log.info(
        "UMAP viz fit: %s sample, n_components=2, min_dist=%.2f ...",
        f"{len(sample_idx):,}", UMAP_MIN_DIST_VIZ,
    )
    reducer = umap.UMAP(
        n_components=UMAP_VIZ_DIMS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST_VIZ,
        metric="euclidean",
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
        low_memory=True,
    )
    reducer.fit(X[sample_idx])
    _log.info("UMAP viz fit complete. Transforming all %s customers ...", f"{N:,}")

    embedding = reducer.transform(X).astype(np.float32)   # (N, 2)

    nan_rows = np.isnan(embedding).any(axis=1)
    if nan_rows.any():
        col_means = np.nanmean(embedding, axis=0)
        embedding[nan_rows] = col_means
        _log.warning("  %d NaN viz embeddings replaced with column means", int(nan_rows.sum()))

    df = _df_from_embedding(
        umap_cluster["cliente"],
        embedding,
        umap_cluster["promo_rate"],
        prefix="viz_",
    )
    df = df.rename({"viz_0": "x", "viz_1": "y"})

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache, compression="zstd")
    _log.info(
        "Saved %s UMAP viz embeddings → %s  (%.1f MB on disk)",
        f"{len(df):,}", cache.name,
        cache.stat().st_size / 1024 ** 2,
    )
    return df


# ─── PCA baseline (20D) ───────────────────────────────────────────────────────

def reduce_pca(
    customer_vectors: pl.DataFrame | None = None,
    *,
    n_components: int = UMAP_CLUSTER_DIMS,
    force: bool = False,
    cache_path: Path | None = None,
    model_path: Path | None = None,
) -> pl.DataFrame:
    """PCA 100D → 20D baseline (linear, full population, no sampling needed).

    sklearn PCA on 1.48M × 100 with float32 uses ~2 GB RAM and finishes in ~30 s.
    Cached to pca_cluster.parquet.
    """
    cache = Path(cache_path) if cache_path is not None else _PCA_CACHE
    model_cache = Path(model_path) if model_path is not None else _PCA_MODEL_CACHE

    if cache.exists() and model_cache.exists() and not force:
        n = pl.scan_parquet(cache).select(pl.len()).collect().item()
        _log.info("PCA cache hit — %s customers", f"{n:,}")
        with open(model_cache, "rb") as fh:
            pca = pickle.load(fh)
        return pl.read_parquet(cache), pca

    if customer_vectors is None:
        _log.info("Loading customer_vectors_weighted.parquet ...")
        customer_vectors = pl.read_parquet(DATA_PROCESSED / "customer_vectors_weighted.parquet")

    X = _vectors_to_numpy(customer_vectors)

    _log.info("PCA fit+transform on %s × %d ...", f"{len(X):,}", X.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    embedding = pca.fit_transform(X).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()
    _log.info(
        "PCA complete — %d components explain %.1f%% of variance",
        n_components, explained * 100,
    )

    df = _df_from_embedding(
        customer_vectors["cliente"],
        embedding,
        customer_vectors["promo_rate"],
        prefix="pc",
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    model_cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache, compression="zstd")
    with open(model_cache, "wb") as fh:
        pickle.dump(pca, fh)
    _log.info(
        "Saved %s PCA embeddings → %s  (%.1f MB on disk)",
        f"{len(df):,}", cache.name,
        cache.stat().st_size / 1024 ** 2,
    )
    return df, pca
