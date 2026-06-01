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
reduce_umap_cluster()  → data/processed/umap_cluster_50d.parquet
reduce_umap_viz()      → data/processed/umap_viz_2d.parquet
reduce_pca()           → data/processed/pca_cluster_50d.parquet

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
    RANDOM_SEED,
    UMAP_CLUSTER_DIMS,
    UMAP_VIZ_DIMS,
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST_CLUSTER,
    UMAP_MIN_DIST_VIZ,
    UMAP_METRIC,
)

_log = logging.getLogger(__name__)

# How many customers to fit UMAP on — large enough to capture density structure,
# small enough that fit() finishes in a few minutes on 10 cores.
UMAP_FIT_SAMPLE = 300_000

_UMAP_CLUSTER_CACHE = DATA_PROCESSED / "umap_cluster_50d.parquet"
_UMAP_VIZ_CACHE     = DATA_PROCESSED / "umap_viz_2d.parquet"
_PCA_CACHE          = DATA_PROCESSED / "pca_cluster_50d.parquet"
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


# ─── UMAP clustering embedding (50D) ─────────────────────────────────────────

def reduce_umap_cluster(
    customer_vectors: pl.DataFrame | None = None,
    *,
    force: bool = False,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """UMAP 100D → 50D embedding for HDBSCAN clustering.

    Fits on UMAP_FIT_SAMPLE random customers, transforms the rest.
    Cached to umap_cluster_50d.parquet.

    Parameters
    ----------
    customer_vectors : DataFrame with [cliente, vector, promo_rate].
                       If None, loads customer_vectors_weighted.parquet.
    n_jobs           : parallel threads for UMAP fit (-1 = all cores).
    """
    if _UMAP_CLUSTER_CACHE.exists() and not force:
        n = pl.scan_parquet(_UMAP_CLUSTER_CACHE).select(pl.len()).collect().item()
        _log.info("UMAP cluster cache hit — %s customers", f"{n:,}")
        return pl.read_parquet(_UMAP_CLUSTER_CACHE)

    if customer_vectors is None:
        _log.info("Loading customer_vectors_weighted.parquet ...")
        customer_vectors = pl.read_parquet(DATA_PROCESSED / "customer_vectors_weighted.parquet")

    X = _vectors_to_numpy(customer_vectors)          # (N, 100)
    N = len(X)

    # Sample for fit
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(N, size=min(UMAP_FIT_SAMPLE, N), replace=False)
    sample_idx.sort()
    X_sample = X[sample_idx]

    _log.info(
        "UMAP cluster fit: %s sample, n_neighbors=%d, n_components=%d, metric=%s ...",
        f"{len(X_sample):,}", UMAP_N_NEIGHBORS, UMAP_CLUSTER_DIMS, UMAP_METRIC,
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

    embedding = reducer.transform(X).astype(np.float32)   # (N, 50)

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
    _UMAP_CLUSTER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_UMAP_CLUSTER_CACHE, compression="zstd")
    _log.info(
        "Saved %s UMAP cluster embeddings → %s  (%.1f MB on disk)",
        f"{len(df):,}", _UMAP_CLUSTER_CACHE.name,
        _UMAP_CLUSTER_CACHE.stat().st_size / 1024 ** 2,
    )
    return df


# ─── UMAP viz embedding (2D) ──────────────────────────────────────────────────

def reduce_umap_viz(
    umap_cluster: pl.DataFrame | None = None,
    *,
    force: bool = False,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """UMAP 50D → 2D embedding for visualisation.

    Takes the 50D clustering embedding as input (not the raw 100D vectors) so
    the visualisation is geometrically consistent with the clustering.
    Cached to umap_viz_2d.parquet.
    """
    if _UMAP_VIZ_CACHE.exists() and not force:
        n = pl.scan_parquet(_UMAP_VIZ_CACHE).select(pl.len()).collect().item()
        _log.info("UMAP viz cache hit — %s customers", f"{n:,}")
        return pl.read_parquet(_UMAP_VIZ_CACHE)

    if umap_cluster is None:
        _log.info("Loading umap_cluster_50d.parquet ...")
        umap_cluster = pl.read_parquet(_UMAP_CLUSTER_CACHE)

    dim_cols = [c for c in umap_cluster.columns if c.startswith("u")]
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

    df.write_parquet(_UMAP_VIZ_CACHE, compression="zstd")
    _log.info(
        "Saved %s UMAP viz embeddings → %s  (%.1f MB on disk)",
        f"{len(df):,}", _UMAP_VIZ_CACHE.name,
        _UMAP_VIZ_CACHE.stat().st_size / 1024 ** 2,
    )
    return df


# ─── PCA baseline (50D) ───────────────────────────────────────────────────────

def reduce_pca(
    customer_vectors: pl.DataFrame | None = None,
    *,
    n_components: int = UMAP_CLUSTER_DIMS,
    force: bool = False,
) -> pl.DataFrame:
    """PCA 100D → 50D baseline (linear, full population, no sampling needed).

    sklearn PCA on 1.48M × 100 with float32 uses ~2 GB RAM and finishes in ~30 s.
    Cached to pca_cluster_50d.parquet.
    """
    if _PCA_CACHE.exists() and _PCA_MODEL_CACHE.exists() and not force:
        n = pl.scan_parquet(_PCA_CACHE).select(pl.len()).collect().item()
        _log.info("PCA cache hit — %s customers", f"{n:,}")
        with open(_PCA_MODEL_CACHE, "rb") as fh:
            pca = pickle.load(fh)
        return pl.read_parquet(_PCA_CACHE), pca

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
    _PCA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_PCA_CACHE, compression="zstd")
    with open(_PCA_MODEL_CACHE, "wb") as fh:
        pickle.dump(pca, fh)
    _log.info(
        "Saved %s PCA embeddings → %s  (%.1f MB on disk)",
        f"{len(df):,}", _PCA_CACHE.name,
        _PCA_CACHE.stat().st_size / 1024 ** 2,
    )
    return df, pca
