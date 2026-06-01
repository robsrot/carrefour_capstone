"""Phase 4 — Customer clustering and tribe profiling.

Two methods are compared on the 50D UMAP embedding:
  HDBSCAN  (primary)   — density-based, no predefined K, handles noise
  K-Means  (baseline)  — K = n_tribes found by HDBSCAN (fair comparison)

Scale strategy for HDBSCAN (cannot run exact on 1.48M × 50):
  1. Fit HDBSCAN on HDBSCAN_FIT_SAMPLE random customers.
  2. Assign remaining customers via hdbscan.approximate_predict().
  3. Customers the model cannot assign confidently are labelled -1 (noise).
  K-Means uses MiniBatchKMeans which is O(n) and runs on the full 1.48M.

Public API
----------
cluster_hdbscan()   → data/processed/cluster_labels_hdbscan.parquet
cluster_kmeans()    → data/processed/cluster_labels_kmeans.parquet
profile_tribes()    → data/processed/tribe_profiles.parquet

Label schema for both:
    cliente   str
    cluster   int32   (-1 = noise / unassigned for HDBSCAN)
    promo_rate float32
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import hdbscan
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.config import (
    DATA_PROCESSED,
    RANDOM_SEED,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_METRIC,
    HDBSCAN_CLUSTER_METHOD,
)

_log = logging.getLogger(__name__)

# Fit HDBSCAN on this many customers (empirically safe limit for memory/time)
HDBSCAN_FIT_SAMPLE = 300_000
# Evaluate silhouette on this many customers (O(n²) metric — needs sampling)
SILHOUETTE_SAMPLE  = 50_000

_HDBSCAN_CACHE  = DATA_PROCESSED / "cluster_labels_hdbscan.parquet"
_KMEANS_CACHE   = DATA_PROCESSED / "cluster_labels_kmeans.parquet"
_PROFILES_CACHE = DATA_PROCESSED / "tribe_profiles.parquet"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _embedding_to_numpy(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract numeric embedding columns from a dimensionality-reduction DataFrame."""
    dim_cols = [c for c in df.columns if c not in ("cliente", "promo_rate")]
    X = df.select(dim_cols).to_numpy().astype(np.float32)
    return X, dim_cols


# ─── HDBSCAN (primary) ───────────────────────────────────────────────────────

def cluster_hdbscan(
    umap_cluster: pl.DataFrame | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Fit HDBSCAN on a 300k sample, assign remaining customers via approximate_predict.

    Parameters
    ----------
    umap_cluster : DataFrame from reduce_umap_cluster() — columns [cliente, u0…u49, promo_rate].
                   If None, loads umap_cluster_50d.parquet.

    Returns
    -------
    DataFrame: cliente str | cluster int32 | promo_rate float32
    Cached to cluster_labels_hdbscan.parquet.
    """
    if _HDBSCAN_CACHE.exists() and not force:
        n = pl.scan_parquet(_HDBSCAN_CACHE).select(pl.len()).collect().item()
        _log.info("HDBSCAN cache hit — %s customers", f"{n:,}")
        return pl.read_parquet(_HDBSCAN_CACHE)

    if umap_cluster is None:
        _log.info("Loading umap_cluster_50d.parquet ...")
        umap_cluster = pl.read_parquet(DATA_PROCESSED / "umap_cluster_50d.parquet")

    X, _ = _embedding_to_numpy(umap_cluster)
    N = len(X)

    # Random sample for fit
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(N, size=min(HDBSCAN_FIT_SAMPLE, N), replace=False)
    sample_idx.sort()
    X_sample = X[sample_idx]

    _log.info(
        "HDBSCAN fit: %s sample, min_cluster_size=%d, min_samples=%d, metric=%s ...",
        f"{len(X_sample):,}",
        HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_METRIC,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_METHOD,
        prediction_data=True,    # required for approximate_predict
        core_dist_n_jobs=-1,
    )
    clusterer.fit(X_sample)

    n_tribes  = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    noise_pct = (clusterer.labels_ == -1).mean() * 100
    _log.info(
        "HDBSCAN fit complete — %d tribes, %.1f%% noise in sample",
        n_tribes, noise_pct,
    )

    # Assign all customers (fit sample gets exact labels; rest get approximate)
    all_labels = np.full(N, -1, dtype=np.int32)
    all_labels[sample_idx] = clusterer.labels_.astype(np.int32)

    rest_idx = np.setdiff1d(np.arange(N), sample_idx)
    if len(rest_idx) > 0:
        _log.info("approximate_predict on remaining %s customers ...", f"{len(rest_idx):,}")
        approx_labels, _ = hdbscan.approximate_predict(clusterer, X[rest_idx])
        all_labels[rest_idx] = approx_labels.astype(np.int32)

    total_noise = (all_labels == -1).mean() * 100
    _log.info(
        "Full population — %d tribes, %.1f%% noise/unassigned",
        n_tribes, total_noise,
    )

    df = pl.DataFrame({
        "cliente":    umap_cluster["cliente"],
        "cluster":    pl.Series(all_labels),
        "promo_rate": umap_cluster["promo_rate"],
    })

    _HDBSCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_HDBSCAN_CACHE, compression="zstd")
    _log.info(
        "Saved %s HDBSCAN labels → %s",
        f"{len(df):,}", _HDBSCAN_CACHE.name,
    )
    return df


# ─── K-Means baseline ────────────────────────────────────────────────────────

def cluster_kmeans(
    umap_cluster: pl.DataFrame | None = None,
    *,
    n_clusters: int | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """MiniBatchKMeans on the full 1.48M customers.

    K = number of non-noise HDBSCAN tribes (loaded from cache if available).
    Uses MiniBatchKMeans so memory is O(K × dims + batch_size) not O(n).

    Returns
    -------
    DataFrame: cliente str | cluster int32 | promo_rate float32
    Cached to cluster_labels_kmeans.parquet.
    """
    if _KMEANS_CACHE.exists() and not force:
        n = pl.scan_parquet(_KMEANS_CACHE).select(pl.len()).collect().item()
        _log.info("K-Means cache hit — %s customers", f"{n:,}")
        return pl.read_parquet(_KMEANS_CACHE)

    if umap_cluster is None:
        _log.info("Loading umap_cluster_50d.parquet ...")
        umap_cluster = pl.read_parquet(DATA_PROCESSED / "umap_cluster_50d.parquet")

    # Derive K from HDBSCAN output if not provided
    if n_clusters is None:
        if _HDBSCAN_CACHE.exists():
            hdb_labels = pl.read_parquet(_HDBSCAN_CACHE)["cluster"]
            n_clusters = int(hdb_labels.filter(hdb_labels >= 0).n_unique())
            _log.info("K derived from HDBSCAN: K = %d", n_clusters)
        else:
            n_clusters = 20
            _log.warning(
                "HDBSCAN cache not found — defaulting to K=%d. "
                "Run cluster_hdbscan() first for a fair comparison.",
                n_clusters,
            )

    X, _ = _embedding_to_numpy(umap_cluster)
    _log.info(
        "MiniBatchKMeans: K=%d on %s × %d ...",
        n_clusters, f"{len(X):,}", X.shape[1],
    )

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        batch_size=10_000,
        n_init=5,
        max_iter=300,
    )
    labels = km.fit_predict(X).astype(np.int32)
    _log.info("K-Means complete — inertia: %.2e", km.inertia_)

    df = pl.DataFrame({
        "cliente":    umap_cluster["cliente"],
        "cluster":    pl.Series(labels),
        "promo_rate": umap_cluster["promo_rate"],
    })

    _KMEANS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_KMEANS_CACHE, compression="zstd")
    _log.info(
        "Saved %s K-Means labels → %s",
        f"{len(df):,}", _KMEANS_CACHE.name,
    )
    return df


# ─── Cluster quality metrics ──────────────────────────────────────────────────

def evaluate_clustering(
    umap_cluster: pl.DataFrame,
    labels: pl.DataFrame,
    method_name: str,
    *,
    sample_size: int = SILHOUETTE_SAMPLE,
) -> dict:
    """Compute silhouette score and Davies-Bouldin index on a sample.

    Both metrics use the 50D UMAP embedding (not the raw 100D vectors), which
    is the space the clustering actually operated in.

    Returns dict: method, n_clusters, noise_pct, silhouette, davies_bouldin
    """
    X, _ = _embedding_to_numpy(umap_cluster)
    lbl  = labels["cluster"].to_numpy()

    # Exclude noise for evaluation
    mask = lbl >= 0
    X_clean, lbl_clean = X[mask], lbl[mask]

    # Subsample for silhouette (O(n²) cost)
    rng = np.random.default_rng(RANDOM_SEED)
    if len(X_clean) > sample_size:
        idx = rng.choice(len(X_clean), sample_size, replace=False)
        X_eval, lbl_eval = X_clean[idx], lbl_clean[idx]
    else:
        X_eval, lbl_eval = X_clean, lbl_clean

    n_clusters  = int(np.unique(lbl_eval).size)
    noise_pct   = float((lbl == -1).mean() * 100)

    sil = float(silhouette_score(X_eval, lbl_eval, metric="euclidean", sample_size=None))
    db  = float(davies_bouldin_score(X_eval, lbl_eval))

    result = {
        "method":        method_name,
        "n_clusters":    n_clusters,
        "noise_pct":     round(noise_pct, 2),
        "silhouette":    round(sil, 4),
        "davies_bouldin": round(db, 4),
    }
    _log.info(
        "%s — %d clusters | %.1f%% noise | silhouette=%.4f | DB=%.4f",
        method_name, n_clusters, noise_pct, sil, db,
    )
    return result


# ─── Tribe profiling ──────────────────────────────────────────────────────────

def profile_tribes(
    cluster_labels: pl.DataFrame,
    method_name: str = "hdbscan",
    *,
    top_n_products: int = 20,
    force: bool = False,
) -> pl.DataFrame:
    """Build a commercial profile for each tribe.

    Joins cluster labels with:
      - customer_kpis.parquet     → avg_basket, visit_freq, total_spend
      - df_combined.parquet (lazy, streamed) → top N products by frequency per tribe

    Returns DataFrame per tribe:
        cluster        int32
        n_customers    int64
        noise_pct      float32  (only for HDBSCAN; 0 for K-Means)
        avg_basket     float32
        avg_visits     float32
        total_revenue  float64
        revenue_share  float32  (% of all revenue this tribe accounts for)
        avg_promo_rate float32
        top_products   list[str]  (product descriptions, sorted by purchase frequency)

    Cached to tribe_profiles.parquet.
    """
    cache = DATA_PROCESSED / f"tribe_profiles_{method_name}.parquet"
    if cache.exists() and not force:
        _log.info("Tribe profiles cache hit — %s", cache.name)
        return pl.read_parquet(cache)

    _log.info("Building tribe profiles for %s ...", method_name)

    # Load customer KPIs (1.46M rows, 117 MB — safe in RAM)
    kpis = pl.read_parquet(DATA_PROCESSED / "customer_kpis.parquet")

    # KPI aggregation per tribe
    joined = cluster_labels.join(kpis, on="cliente", how="left")

    kpi_profiles = (
        joined
        .group_by("cluster")
        .agg([
            pl.len().alias("n_customers"),
            pl.col("avg_basket_size").mean().cast(pl.Float32).alias("avg_basket"),
            pl.col("n_tickets").mean().cast(pl.Float32).alias("avg_visits"),
            pl.col("total_spend").sum().alias("total_revenue"),
            pl.col("promo_rate").mean().cast(pl.Float32).alias("avg_promo_rate"),
        ])
        .with_columns(
            (pl.col("total_revenue") / pl.col("total_revenue").sum() * 100)
            .cast(pl.Float32)
            .alias("revenue_share")
        )
        .sort("cluster")
    )

    # Top products per tribe — stream df_combined, join cluster labels, aggregate
    _log.info("  Computing top products per tribe (streaming df_combined) ...")
    combined_path = DATA_PROCESSED / "df_combined.parquet"

    top_products = (
        pl.scan_parquet(combined_path)
        .select(["cliente", "idarticu", "desc_larga_articulo"])
        .join(
            cluster_labels.select(["cliente", "cluster"]).lazy(),
            on="cliente",
            how="inner",
        )
        .group_by(["cluster", "idarticu", "desc_larga_articulo"])
        .agg(pl.len().alias("purchase_count"))
        .sort("purchase_count", descending=True)
        .group_by("cluster")
        .agg(
            pl.col("desc_larga_articulo").head(top_n_products).alias("top_products")
        )
        .collect(engine="streaming")
    )

    profiles = kpi_profiles.join(top_products, on="cluster", how="left")

    cache.parent.mkdir(parents=True, exist_ok=True)
    profiles.write_parquet(cache, compression="zstd")
    _log.info("Saved tribe profiles → %s  (%d tribes)", cache.name, len(profiles))
    return profiles
