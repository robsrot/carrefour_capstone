"""Phase 4 — Customer clustering and tribe profiling.

Two methods are compared on the 20D UMAP embedding:
  HDBSCAN  (discovery) — density-based, no predefined K, handles noise
  K-Means  (baseline)  — fixed-K runs for business-readable segment counts

Scale strategy for HDBSCAN (cannot run exact on 1.48M × 20):
  1. Fit HDBSCAN on HDBSCAN_FIT_SAMPLE random customers.
  2. Assign remaining customers via nearest-neighbour label lookup on the fit sample.
  3. Customers whose nearest sample neighbour is noise are labelled -1 (noise).
  K-Means uses MiniBatchKMeans which is O(n) and runs on the full 1.48M.

Public API
----------
cluster_hdbscan()   → data/processed/cluster_labels_hdbscan.parquet
cluster_kmeans()    → data/processed/cluster_labels_kmeans.parquet
run_kmeans_baselines() → data/processed/kmeans_baseline_results.parquet
grid_search_hdbscan()  → data/processed/hdbscan_grid_results.parquet
profile_tribes()    → data/processed/tribe_profiles.parquet

Label schema for both:
    cliente   str
    cluster   int32   (-1 = noise / unassigned for HDBSCAN)
    promo_rate float32
"""
from __future__ import annotations

import logging
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.config import (
    DATA_PROCESSED,
    ROOT,
    RANDOM_SEED,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_METRIC,
    HDBSCAN_CLUSTER_METHOD,
    HDBSCAN_FIT_SAMPLE,
    HDBSCAN_GRID_CLUSTER_METHOD,
    HDBSCAN_GRID_MIN_CLUSTER_SIZE,
    HDBSCAN_GRID_MIN_SAMPLES,
    KMEANS_BASELINE_CLUSTERS,
    KMEANS_BATCH_SIZE,
    KMEANS_MAX_ITER,
    KMEANS_N_INIT,
    SILHOUETTE_SAMPLE,
)
from src.product_themes import classify_product_themes

_log = logging.getLogger(__name__)

# HDBSCAN_FIT_SAMPLE and SILHOUETTE_SAMPLE are loaded from config (base: 100k/50k prod, 10k dev).

_HDBSCAN_CACHE  = DATA_PROCESSED / "cluster_labels_hdbscan.parquet"
_HDBSCAN_ASSIGNED_CACHE = DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet"
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
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    cluster_method: str | None = None,
    force: bool = False,
    cache_path: Path | None = None,
) -> pl.DataFrame:
    """Fit HDBSCAN on a 300k sample, assign remaining customers via approximate_predict.

    Parameters
    ----------
    umap_cluster : DataFrame from reduce_umap_cluster() — columns [cliente, u0…u19, promo_rate].
                   If None, loads umap_cluster.parquet.

    Returns
    -------
    DataFrame: cliente str | cluster int32 | promo_rate float32
    Cached to cluster_labels_hdbscan.parquet, unless cache_path is supplied.
    """
    cache = Path(cache_path) if cache_path is not None else _HDBSCAN_CACHE
    if cache.exists() and not force:
        n = pl.scan_parquet(cache).select(pl.len()).collect().item()
        _log.info("HDBSCAN cache hit — %s customers", f"{n:,}")
        return pl.read_parquet(cache)

    if umap_cluster is None:
        _log.info("Loading umap_cluster.parquet ...")
        umap_cluster = pl.read_parquet(DATA_PROCESSED / "umap_cluster.parquet")

    X, _ = _embedding_to_numpy(umap_cluster)
    N = len(X)

    min_cluster_size = int(min_cluster_size or HDBSCAN_MIN_CLUSTER_SIZE)
    min_samples = int(min_samples or HDBSCAN_MIN_SAMPLES)
    cluster_method = str(cluster_method or HDBSCAN_CLUSTER_METHOD)

    # Random sample for fit
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(N, size=min(HDBSCAN_FIT_SAMPLE, N), replace=False)
    sample_idx.sort()
    X_sample = X[sample_idx]

    _log.info(
        "HDBSCAN fit: %s sample, min_cluster_size=%d, min_samples=%d, metric=%s ...",
        f"{len(X_sample):,}",
        min_cluster_size, min_samples, HDBSCAN_METRIC,
    )
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=HDBSCAN_METRIC,
        algorithm="ball_tree",
        cluster_selection_method=cluster_method,
        n_jobs=1,
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
        _log.info("Assigning remaining %s customers via nearest-neighbour lookup ...", f"{len(rest_idx):,}")
        nn = NearestNeighbors(n_neighbors=1, metric=HDBSCAN_METRIC, n_jobs=1)
        nn.fit(X_sample)
        _, indices = nn.kneighbors(X[rest_idx])
        all_labels[rest_idx] = clusterer.labels_[indices.ravel()].astype(np.int32)

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

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache, compression="zstd")
    _log.info(
        "Saved %s HDBSCAN labels → %s",
        f"{len(df):,}", cache.name,
    )
    return df


def assign_hdbscan_noise_to_nearest_tribe(
    umap_cluster: pl.DataFrame | None = None,
    hdbscan_labels: pl.DataFrame | None = None,
    *,
    force: bool = False,
    cache_path: Path | None = None,
) -> pl.DataFrame:
    """Assign HDBSCAN noise customers to their nearest non-noise tribe.

    This keeps HDBSCAN as the discovery method while producing an all-customer
    operational segmentation. The original density result is preserved in
    `hdbscan_cluster`; the final all-customer label is stored in `cluster`.

    Returns
    -------
    DataFrame:
        cliente str
        cluster int32                 all customers assigned to a non-noise tribe
        hdbscan_cluster int32          original HDBSCAN label (-1 = density noise)
        was_hdbscan_noise bool
        assignment_source str          "hdbscan_core" | "nearest_hdbscan_tribe"
        assignment_distance float32    0 for core customers, nearest-core distance for assigned noise
        promo_rate float32
    """
    cache = Path(cache_path) if cache_path is not None else _HDBSCAN_ASSIGNED_CACHE
    if cache.exists() and not force:
        cached = pl.read_parquet(cache)
        required = {
            "cluster",
            "hdbscan_cluster",
            "was_hdbscan_noise",
            "assignment_source",
            "assignment_distance",
        }
        if required.issubset(set(cached.columns)):
            _log.info("HDBSCAN assigned cache hit — %s", cache.name)
            return cached
        _log.info("HDBSCAN assigned cache has older schema — rebuilding")

    if umap_cluster is None:
        _log.info("Loading umap_cluster.parquet ...")
        umap_cluster = pl.read_parquet(DATA_PROCESSED / "umap_cluster.parquet")
    if hdbscan_labels is None:
        _log.info("Loading cluster_labels_hdbscan.parquet ...")
        hdbscan_labels = pl.read_parquet(_HDBSCAN_CACHE)

    X, _ = _embedding_to_numpy(umap_cluster)
    aligned = (
        umap_cluster.select("cliente")
        .join(hdbscan_labels.select(["cliente", "cluster"]), on="cliente", how="left")
        .with_columns(pl.col("cluster").fill_null(-1).cast(pl.Int32))
    )
    original = aligned["cluster"].to_numpy().astype(np.int32)

    core_mask = original >= 0
    noise_mask = ~core_mask
    if not core_mask.any():
        raise ValueError("HDBSCAN produced no non-noise tribes; cannot assign noise customers.")

    final = original.copy()
    assignment_distance = np.zeros(len(original), dtype=np.float32)
    assignment_source = np.full(len(original), "hdbscan_core", dtype=object)

    if noise_mask.any():
        _log.info(
            "Assigning %s HDBSCAN noise customers to nearest non-noise tribe ...",
            f"{int(noise_mask.sum()):,}",
        )
        core_labels = original[core_mask]
        nn = NearestNeighbors(n_neighbors=1, metric=HDBSCAN_METRIC, n_jobs=1)
        nn.fit(X[core_mask])
        distances, indices = nn.kneighbors(X[noise_mask])
        final[noise_mask] = core_labels[indices.ravel()].astype(np.int32)
        assignment_distance[noise_mask] = distances.ravel().astype(np.float32)
        assignment_source[noise_mask] = "nearest_hdbscan_tribe"

    df = pl.DataFrame({
        "cliente": umap_cluster["cliente"],
        "cluster": pl.Series(final, dtype=pl.Int32),
        "hdbscan_cluster": pl.Series(original, dtype=pl.Int32),
        "was_hdbscan_noise": pl.Series(noise_mask),
        "assignment_source": pl.Series(assignment_source.tolist()),
        "assignment_distance": pl.Series(assignment_distance),
        "promo_rate": umap_cluster["promo_rate"],
    })

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache, compression="zstd")
    _log.info(
        "Saved %s HDBSCAN assigned labels → %s  (%d tribes, %.1f%% assigned from noise)",
        f"{len(df):,}",
        cache.name,
        int(df["cluster"].n_unique()),
        float(df["was_hdbscan_noise"].mean() * 100),
    )
    return df


# ─── K-Means baseline ────────────────────────────────────────────────────────

def cluster_kmeans(
    umap_cluster: pl.DataFrame | None = None,
    *,
    n_clusters: int | None = None,
    force: bool = False,
    cache_path: Path | None = None,
) -> pl.DataFrame:
    """MiniBatchKMeans on the full 1.48M customers.

    If n_clusters is provided, this is a fixed-K business baseline. If omitted,
    K is derived from the non-noise HDBSCAN tribes when available. MiniBatchKMeans
    keeps memory O(K × dims + batch_size), not O(n).

    Returns
    -------
    DataFrame: cliente str | cluster int32 | promo_rate float32
    Cached to cluster_labels_kmeans.parquet or cluster_labels_kmeans_k{K}.parquet.
    """
    explicit_k = n_clusters is not None
    cache = Path(cache_path) if cache_path is not None else (
        DATA_PROCESSED / f"cluster_labels_kmeans_k{int(n_clusters)}.parquet"
        if explicit_k
        else _KMEANS_CACHE
    )
    if cache.exists() and not force:
        n = pl.scan_parquet(cache).select(pl.len()).collect().item()
        _log.info("K-Means cache hit — %s customers (%s)", f"{n:,}", cache.name)
        return pl.read_parquet(cache)

    if umap_cluster is None:
        _log.info("Loading umap_cluster.parquet ...")
        umap_cluster = pl.read_parquet(DATA_PROCESSED / "umap_cluster.parquet")

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
        batch_size=KMEANS_BATCH_SIZE,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
    )
    labels = km.fit_predict(X).astype(np.int32)
    _log.info("K-Means complete — inertia: %.2e", km.inertia_)

    df = pl.DataFrame({
        "cliente":    umap_cluster["cliente"],
        "cluster":    pl.Series(labels),
        "promo_rate": umap_cluster["promo_rate"],
    })

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache, compression="zstd")
    _log.info(
        "Saved %s K-Means labels → %s",
        f"{len(df):,}", cache.name,
    )
    return df


# ─── Cluster quality metrics ──────────────────────────────────────────────────

def run_kmeans_baselines(
    umap_cluster: pl.DataFrame | None = None,
    *,
    k_values: list[int] | None = None,
    force: bool = False,
    profile: bool = False,
    cache_prefix: str | None = None,
    cache_path: Path | None = None,
) -> pl.DataFrame:
    """Run fixed-K MiniBatchKMeans baselines and compare cluster quality.

    This is the business-readable baseline for an expected segment range such as
    12-20 groups. HDBSCAN can still be used for density discovery, but it should
    not be the only source of the segment count.
    """
    if umap_cluster is None:
        _log.info("Loading umap_cluster.parquet ...")
        umap_cluster = pl.read_parquet(DATA_PROCESSED / "umap_cluster.parquet")

    if k_values is None:
        k_values = KMEANS_BASELINE_CLUSTERS

    results: list[dict] = []
    for k in k_values:
        label_cache = (
            DATA_PROCESSED / f"cluster_labels_kmeans_{cache_prefix}_k{int(k)}.parquet"
            if cache_prefix
            else None
        )
        labels = cluster_kmeans(
            umap_cluster,
            n_clusters=int(k),
            force=force,
            cache_path=label_cache,
        )
        result = evaluate_clustering(
            umap_cluster,
            labels,
            method_name=f"kmeans_k{int(k)}",
        )
        result["requested_k"] = int(k)
        results.append(result)

        if profile:
            profile_tribes(
                labels,
                method_name=f"kmeans_k{int(k)}",
                force=force,
            )

    df = pl.DataFrame(results).select([
        "method",
        "requested_k",
        "n_clusters",
        "noise_pct",
        "silhouette",
        "davies_bouldin",
    ])
    out = Path(cache_path) if cache_path is not None else DATA_PROCESSED / "kmeans_baseline_results.parquet"
    df.write_parquet(out, compression="zstd")
    _log.info("Saved K-Means baseline comparison → %s", out.name)
    return df


def evaluate_clustering(
    umap_cluster: pl.DataFrame,
    labels: pl.DataFrame,
    method_name: str,
    *,
    sample_size: int = SILHOUETTE_SAMPLE,
) -> dict:
    """Compute silhouette score and Davies-Bouldin index on a sample.

    Both metrics use the 20D UMAP embedding (not the raw 100D vectors), which
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

def _sample_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
) -> tuple[float | None, float | None]:
    """Compute clustering metrics on non-noise rows, returning None when invalid."""
    mask = labels >= 0
    X_clean, lbl_clean = X[mask], labels[mask]
    unique = np.unique(lbl_clean)
    if len(unique) < 2 or len(lbl_clean) <= len(unique):
        return None, None

    rng = np.random.default_rng(RANDOM_SEED)
    if len(X_clean) > sample_size:
        idx = rng.choice(len(X_clean), sample_size, replace=False)
        X_eval, lbl_eval = X_clean[idx], lbl_clean[idx]
    else:
        X_eval, lbl_eval = X_clean, lbl_clean

    try:
        sil = float(silhouette_score(X_eval, lbl_eval, metric="euclidean", sample_size=None))
        db = float(davies_bouldin_score(X_eval, lbl_eval))
    except ValueError:
        return None, None
    return round(sil, 4), round(db, 4)


def grid_search_hdbscan(
    umap_cluster: pl.DataFrame | None = None,
    *,
    min_cluster_sizes: list[int] | None = None,
    min_samples_values: list[int] | None = None,
    cluster_methods: list[str] | None = None,
    target_clusters: int = 15,
    metric_sample_size: int | None = None,
    force: bool = False,
    cache_path: Path | None = None,
) -> pl.DataFrame:
    """Evaluate HDBSCAN hyperparameter candidates on the configured fit sample.

    The grid intentionally scores only the HDBSCAN fit sample, not full-population
    nearest-neighbour assignment. Use the winning candidate with cluster_hdbscan()
    or update configs/base.yaml once a commercially readable option is chosen.
    """
    out = Path(cache_path) if cache_path is not None else DATA_PROCESSED / "hdbscan_grid_results.parquet"
    if out.exists() and not force:
        _log.info("HDBSCAN grid cache hit — %s", out.name)
        return pl.read_parquet(out)

    if umap_cluster is None:
        _log.info("Loading umap_cluster.parquet ...")
        umap_cluster = pl.read_parquet(DATA_PROCESSED / "umap_cluster.parquet")

    if min_cluster_sizes is None:
        min_cluster_sizes = HDBSCAN_GRID_MIN_CLUSTER_SIZE
    if min_samples_values is None:
        min_samples_values = HDBSCAN_GRID_MIN_SAMPLES
    if cluster_methods is None:
        cluster_methods = HDBSCAN_GRID_CLUSTER_METHOD
    if metric_sample_size is None:
        metric_sample_size = min(SILHOUETTE_SAMPLE, 10_000)

    X, _ = _embedding_to_numpy(umap_cluster)
    N = len(X)
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(N, size=min(HDBSCAN_FIT_SAMPLE, N), replace=False)
    sample_idx.sort()
    X_sample = X[sample_idx]

    results: list[dict] = []
    for min_cluster_size, min_samples, cluster_method in product(
        min_cluster_sizes,
        min_samples_values,
        cluster_methods,
    ):
        if int(min_cluster_size) >= len(X_sample):
            continue

        _log.info(
            "HDBSCAN grid: min_cluster_size=%d, min_samples=%d, method=%s",
            int(min_cluster_size),
            int(min_samples),
            str(cluster_method),
        )
        clusterer = HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            metric=HDBSCAN_METRIC,
            algorithm="ball_tree",
            cluster_selection_method=str(cluster_method),
            n_jobs=1,
        )
        labels = clusterer.fit_predict(X_sample).astype(np.int32)
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        noise_pct = float((labels == -1).mean() * 100)
        sil, db = _sample_metrics(
            X_sample,
            labels,
            sample_size=int(metric_sample_size),
        )
        results.append({
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
            "cluster_method": str(cluster_method),
            "n_clusters": n_clusters,
            "target_gap": abs(n_clusters - int(target_clusters)),
            "noise_pct": round(noise_pct, 2),
            "silhouette": sil,
            "davies_bouldin": db,
        })

    df = (
        pl.DataFrame(results)
        .sort(["target_gap", "noise_pct", "davies_bouldin"], nulls_last=True)
    )
    df.write_parquet(out, compression="zstd")
    _log.info("Saved HDBSCAN grid results → %s", out.name)
    return df


def profile_tribes(
    cluster_labels: pl.DataFrame,
    method_name: str = "hdbscan",
    *,
    top_n_products: int = 20,
    top_n_sectors: int = 5,
    top_n_themes: int = 10,
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
        top_sectors    list[str]   (sector names, sorted by lift descending)
        top_sector_lifts list[float]
        top_themes     list[str]   (editable product-name themes, sorted by lift)
        top_theme_lifts list[float]
        top_products   list[str]   (product descriptions, sorted by lift descending)
        top_lifts      list[float] (lift scores — tribe purchase rate ÷ overall purchase rate)

    Cached to tribe_profiles.parquet.
    """
    cache = DATA_PROCESSED / f"tribe_profiles_{method_name}.parquet"
    if cache.exists() and not force:
        cached = pl.read_parquet(cache)
        required_cols = {
            "top_sectors",
            "top_sector_lifts",
            "top_themes",
            "top_theme_lifts",
            "top_products",
            "top_lifts",
        }
        if required_cols.issubset(set(cached.columns)):
            _log.info("Tribe profiles cache hit — %s", cache.name)
            return cached
        _log.info("Profile cache %s uses an older schema — rebuilding", cache.name)

    _log.info("Building tribe profiles for %s ...", method_name)

    # Load customer KPIs. In dev mode the transaction subset lives in data/dev,
    # while customer_kpis.parquet remains the production-wide artifact.
    kpi_path = DATA_PROCESSED / "customer_kpis.parquet"
    if not kpi_path.exists():
        prod_kpi_path = ROOT / "data" / "processed" / "customer_kpis.parquet"
        if prod_kpi_path.exists():
            kpi_path = prod_kpi_path
        else:
            raise FileNotFoundError(
                "customer_kpis.parquet not found in the active data directory "
                f"or production fallback: {DATA_PROCESSED}, {prod_kpi_path}"
            )
    kpis = pl.read_parquet(kpi_path)

    # KPI aggregation per tribe
    joined = cluster_labels.join(kpis, on="cliente", how="left")

    kpi_profiles = (
        joined
        .group_by("cluster")
        .agg([
            pl.len().alias("n_customers"),
            pl.col("avg_basket_size").mean().cast(pl.Float32).alias("avg_basket"),
            pl.col("visit_count").mean().cast(pl.Float32).alias("avg_visits"),
            pl.col("total_spend_6m").sum().alias("total_revenue"),
            pl.col("avg_promo_rate").mean().cast(pl.Float32).alias("avg_promo_rate"),
        ])
        .with_columns(
            (pl.col("total_revenue") / pl.col("total_revenue").sum() * 100)
            .cast(pl.Float32)
            .alias("revenue_share")
        )
        .sort("cluster")
    )

    # Top products per tribe — ranked by LIFT, not raw frequency.
    # Lift = (product's share of tribe's transactions) / (product's share of all transactions).
    # Lift > 1 means the tribe buys that product more than a random customer would.
    # This surfaces the products that make each tribe *distinctive*, not just universal staples.
    _log.info("  Computing top products per tribe by lift (streaming df_combined) ...")
    combined_path = DATA_PROCESSED / "df_combined.parquet"

    # Step 1: per-(cluster, product) purchase counts via streaming scan
    raw_counts = (
        pl.scan_parquet(combined_path)
        .select(["cliente", "idarticu", "desc_larga_articulo", "desc_sector"])
        .join(
            cluster_labels.select(["cliente", "cluster"]).lazy(),
            on="cliente",
            how="inner",
        )
        .group_by(["cluster", "idarticu", "desc_larga_articulo", "desc_sector"])
        .agg(pl.len().alias("tribe_count"))
        .collect(engine="streaming")
    )

    # Step 2: overall purchase counts per product and grand total
    overall_counts = (
        raw_counts
        .group_by(["idarticu", "desc_larga_articulo", "desc_sector"])
        .agg(pl.col("tribe_count").sum().alias("overall_count"))
    )
    total_purchases = int(raw_counts["tribe_count"].sum())

    # Step 3: total purchases per tribe (denominator for tribe rate)
    tribe_totals = (
        raw_counts
        .group_by("cluster")
        .agg(pl.col("tribe_count").sum().alias("tribe_total"))
    )

    # Step 4: sector lift profiles
    sector_counts = (
        raw_counts
        .group_by(["cluster", "desc_sector"])
        .agg(pl.col("tribe_count").sum().alias("tribe_count"))
    )
    sector_overall = (
        sector_counts
        .group_by("desc_sector")
        .agg(pl.col("tribe_count").sum().alias("overall_count"))
    )
    sector_lift = (
        sector_counts
        .join(sector_overall, on="desc_sector", how="left")
        .join(tribe_totals, on="cluster", how="left")
        .with_columns(
            (
                (pl.col("tribe_count").cast(pl.Float64) / pl.col("tribe_total"))
                / (pl.col("overall_count").cast(pl.Float64) / total_purchases)
            ).alias("lift")
        )
    )
    top_sectors = (
        sector_lift
        .group_by("cluster")
        .agg([
            pl.col("desc_sector")
              .sort_by(pl.col("lift"), descending=True)
              .head(top_n_sectors)
              .alias("top_sectors"),
            pl.col("lift")
              .sort(descending=True)
              .head(top_n_sectors)
              .round(2)
              .alias("top_sector_lifts"),
        ])
    )

    # Step 5: editable product-name theme lift profiles.
    # A product can match several themes, so theme shares need not sum to 100%.
    theme_counts = (
        raw_counts
        .with_columns(
            pl.col("desc_larga_articulo")
            .map_elements(classify_product_themes, return_dtype=pl.List(pl.Utf8))
            .alias("theme")
        )
        .explode("theme")
        .filter(pl.col("theme").is_not_null())
        .group_by(["cluster", "theme"])
        .agg(pl.col("tribe_count").sum().alias("tribe_count"))
    )
    if len(theme_counts) > 0:
        theme_overall = (
            theme_counts
            .group_by("theme")
            .agg(pl.col("tribe_count").sum().alias("overall_count"))
        )
        theme_lift = (
            theme_counts
            .join(theme_overall, on="theme", how="left")
            .join(tribe_totals, on="cluster", how="left")
            .with_columns(
                (
                    (pl.col("tribe_count").cast(pl.Float64) / pl.col("tribe_total"))
                    / (pl.col("overall_count").cast(pl.Float64) / total_purchases)
                ).alias("lift")
            )
            .filter(pl.col("tribe_count") >= 30)
        )
        top_themes = (
            theme_lift
            .group_by("cluster")
            .agg([
                pl.col("theme")
                  .sort_by(pl.col("lift"), descending=True)
                  .head(top_n_themes)
                  .alias("top_themes"),
                pl.col("lift")
                  .sort(descending=True)
                  .head(top_n_themes)
                  .round(2)
                  .alias("top_theme_lifts"),
            ])
        )
    else:
        top_themes = pl.DataFrame(
            schema={
                "cluster": pl.Int32,
                "top_themes": pl.List(pl.Utf8),
                "top_theme_lifts": pl.List(pl.Float64),
            }
        )

    # Step 6: product lift = (tribe_count / tribe_total) / (overall_count / total_purchases)
    with_lift = (
        raw_counts
        .join(overall_counts, on=["idarticu", "desc_larga_articulo", "desc_sector"], how="left")
        .join(tribe_totals, on="cluster", how="left")
        .with_columns(
            (
                (pl.col("tribe_count").cast(pl.Float64) / pl.col("tribe_total"))
                / (pl.col("overall_count").cast(pl.Float64) / total_purchases)
            ).alias("lift")
        )
        .filter(pl.col("tribe_count") >= 30)   # ignore products bought by fewer than 30 customers in this tribe
    )

    # Step 7: top N products per tribe sorted by lift descending
    top_products = (
        with_lift
        .group_by("cluster")
        .agg([
            pl.col("desc_larga_articulo")
              .sort_by(pl.col("lift"), descending=True)
              .head(top_n_products)
              .alias("top_products"),
            pl.col("lift")
              .sort(descending=True)
              .head(top_n_products)
              .round(2)
              .alias("top_lifts"),
        ])
    )

    profiles = (
        kpi_profiles
        .join(top_sectors, on="cluster", how="left")
        .join(top_themes, on="cluster", how="left")
        .join(top_products, on="cluster", how="left")
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    profiles.write_parquet(cache, compression="zstd")
    _log.info("Saved tribe profiles → %s  (%d tribes)", cache.name, len(profiles))
    return profiles
