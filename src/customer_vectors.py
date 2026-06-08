"""Phase 2 — Customer vector aggregation (product embeddings → behavioral profiles).

Each customer's purchase history is compressed into a single 100-dim vector by
taking a weighted mean of the product embeddings for every product they bought.

Two variants are produced:

  weighted (primary)
    weight per (customer, product) pair = Σ recency_decay over all purchases of that product
    recency_decay(t) = exp(-ln(2)/halflife * days_before_reference)
    → Recent and frequently-purchased products dominate the customer's profile.

  mean (baseline)
    Binary product ownership average. If product-popularity IDF weighting is
    enabled, each purchased product contributes once and is then IDF-scaled.
    Used to quantify the signal gained from frequency+recency weighting.

A promo_rate field (fraction of purchase lines that were promotional) is attached to
both outputs so Phase 4 can profile promo-sensitive customers within each tribe.

Public API
----------
build_customer_vectors()      → data/processed/customer_vectors_weighted.parquet
build_customer_vectors_mean() → data/processed/customer_vectors_mean.parquet

Both return a Polars DataFrame:
    cliente    str            (matches df_combined schema)
    vector     list[float32]  (length = W2V_VECTOR_SIZE = 100)
    promo_rate float32        (0.0–1.0)
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.config import (
    CUSTOMER_VECTOR_WEIGHT_TRANSFORM,
    DATA_PROCESSED,
    HYBRID_WEIGHT_CATEGORY_SHARES,
    HYBRID_INCLUDE_STORE_SHARES,
    HYBRID_WEIGHT_ITEM2VEC,
    HYBRID_WEIGHT_TFIDF_SVD,
    NORMALIZE_PRODUCT_EMBEDDINGS,
    PRODUCT_IDF_WEIGHTING_ENABLED,
    PRODUCT_POPULARITY_ENABLED,
    RECENCY_HALFLIFE_DAYS,
    RECENCY_REFERENCE_DATE,
    RANDOM_SEED,
    TFIDF_SVD_DIMS,
    W2V_VECTOR_SIZE,
)
from src.product_filtering import build_product_popularity
from src.product_themes import STRATEGIC_THEME_PATTERNS, classify_product_themes

_log = logging.getLogger(__name__)

_REFERENCE_DATE       = date.fromisoformat(RECENCY_REFERENCE_DATE)  # from configs/*.yaml
_INTERACTIONS_CACHE   = DATA_PROCESSED / "customer_product_weights.parquet"
_WEIGHTED_CACHE       = DATA_PROCESSED / "customer_vectors_weighted.parquet"
_MEAN_CACHE           = DATA_PROCESSED / "customer_vectors_mean.parquet"
_TFIDF_SVD_CACHE      = DATA_PROCESSED / "customer_vectors_tfidf_svd.parquet"
_PRODUCT_SHARE_FEATURES_CACHE = DATA_PROCESSED / "customer_product_share_features.parquet"
_SHARE_FEATURES_WITH_STORE_CACHE = DATA_PROCESSED / "customer_share_features_with_store.parquet"
_HYBRID_PRODUCT_ONLY_CACHE = DATA_PROCESSED / "customer_vectors_hybrid_product_only.parquet"
_HYBRID_WITH_STORE_CACHE = DATA_PROCESSED / "customer_vectors_hybrid_with_store.parquet"
_STORE_FEATURES_CACHE = DATA_PROCESSED / "customer_store_features.parquet"

# Recency weights are floats; integer-quantise before streaming aggregation so
# the sum is associative across chunk orderings on different machines.
# 1e8 preserves 8 significant decimal digits — more than enough for exp(-λ·t).
_WEIGHT_SCALE = 100_000_000


# ─── 1. Stage 1: per-(customer, product) interaction weights ──────────────────

def _build_interactions(
    df_combined_path: Path,
    halflife_days: int,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Stream df_combined → per-(customer, product) weights + promo counts.

    Returns DataFrame:
        cliente          str
        idarticu         int64
        weight           float64   Σ exp(-λ·days_before_reference) across all purchases
        promo_purchases  int32     lines where idpromoc is non-null and non-"0"
        total_purchases  int32     total purchase lines for this (customer, product) pair

    Cached at customer_product_weights.parquet. Pass force=True when changing halflife_days.
    """
    if _INTERACTIONS_CACHE.exists() and not force:
        n = pl.scan_parquet(_INTERACTIONS_CACHE).select(pl.len()).collect().item()
        _log.info("Interactions cache hit — %s (customer, product) pairs", f"{n:,}")
        return pl.read_parquet(_INTERACTIONS_CACHE)

    decay_lambda = math.log(2) / halflife_days
    _log.info(
        "Stage 1 — streaming df_combined → interaction weights "
        "(halflife=%d d, λ=%.4f, reference=%s) ...",
        halflife_days, decay_lambda, _REFERENCE_DATE,
    )

    interactions = (
        pl.scan_parquet(df_combined_path)
        .select(["cliente", "idarticu", "fecha", "idpromoc"])
        .with_columns([
            # Quantise to Int64 before aggregation: integer addition is associative,
            # so the streaming sum is bit-identical regardless of chunk boundaries
            # or the order in which chunks are merged across machines.
            (
                (
                    pl.lit(-decay_lambda, dtype=pl.Float64)
                    * (pl.lit(_REFERENCE_DATE) - pl.col("fecha")).dt.total_days().cast(pl.Float64)
                ).exp() * _WEIGHT_SCALE
            ).round(0).cast(pl.Int64).alias("recency_weight_int"),
            # idpromoc is String in df_combined: "Promo" | "No promo" (never null)
            (pl.col("idpromoc") == "Promo")
            .cast(pl.Int32).alias("is_promo"),
        ])
        .group_by(["cliente", "idarticu"])
        .agg([
            (pl.col("recency_weight_int").sum().cast(pl.Float64) / _WEIGHT_SCALE).alias("weight"),
            pl.col("is_promo").sum().alias("promo_purchases"),
            pl.len().cast(pl.Int32).alias("total_purchases"),
        ])
        .sort(["cliente", "idarticu"])                 # canonical order → deterministic Categorical assignment downstream
        .collect(engine="streaming")
    )

    _INTERACTIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    interactions.write_parquet(_INTERACTIONS_CACHE, compression="zstd")
    _log.info(
        "Saved %s (customer, product) pairs → %s  (%.0f MB on disk)",
        f"{len(interactions):,}",
        _INTERACTIONS_CACHE.name,
        _INTERACTIONS_CACHE.stat().st_size / 1024 ** 2,
    )
    return interactions


def _apply_weight_transform(values: np.ndarray, transform: str = CUSTOMER_VECTOR_WEIGHT_TRANSFORM) -> np.ndarray:
    """Dampen repeated purchases before row-normalising customer-product weights."""
    transform = (transform or "none").lower()
    if transform == "none":
        return values
    if transform == "sqrt":
        return np.sqrt(np.maximum(values, 0.0)).astype(np.float32)
    if transform == "log1p":
        return np.log1p(np.maximum(values, 0.0)).astype(np.float32)
    raise ValueError(
        f"Unsupported customer vector weight_transform={transform!r}. "
        "Use one of: none, sqrt, log1p."
    )


def _promo_rate_df(interactions: pl.DataFrame) -> pl.DataFrame:
    """Per-customer promo rate from full interactions."""
    return (
        interactions
        .group_by("cliente")
        .agg([
            pl.col("promo_purchases").sum(),
            pl.col("total_purchases").sum(),
        ])
        .with_columns(
            (pl.col("promo_purchases") / pl.col("total_purchases"))
            .cast(pl.Float32)
            .alias("promo_rate")
        )
        .select(["cliente", "promo_rate"])
    )


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalise dense rows, preserving all-zero rows."""
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (values / norms).astype(np.float32)


def _safe_feature_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def _wide_share_from_counts(
    counts: pl.DataFrame,
    *,
    category_col: str,
    value_col: str,
    total_col: str,
    prefix: str,
) -> pl.DataFrame:
    """Pivot long customer/category totals to customer share features."""
    if len(counts) == 0:
        return pl.DataFrame(schema={"cliente": pl.Utf8})

    wide = counts.pivot(
        on=category_col,
        index="cliente",
        values=value_col,
        aggregate_function="sum",
    ).fill_null(0.0)

    value_cols = [c for c in wide.columns if c != "cliente"]
    wide = (
        wide
        .with_columns(pl.sum_horizontal([pl.col(c) for c in value_cols]).alias(total_col))
        .with_columns([
            (
                pl.when(pl.col(total_col) > 0)
                .then(pl.col(c) / pl.col(total_col))
                .otherwise(0.0)
            )
            .cast(pl.Float32)
            .alias(f"{prefix}_{_safe_feature_name(c)}")
            for c in value_cols
        ])
        .select(["cliente"] + [f"{prefix}_{_safe_feature_name(c)}" for c in value_cols])
    )
    return wide


def _wide_value_features(
    values: pl.DataFrame,
    *,
    category_col: str,
    value_col: str,
    prefix: str,
) -> pl.DataFrame:
    """Pivot already-normalised customer/category values to wide features."""
    if len(values) == 0:
        return pl.DataFrame(schema={"cliente": pl.Utf8})

    wide = values.pivot(
        on=category_col,
        index="cliente",
        values=value_col,
        aggregate_function="sum",
    ).fill_null(0.0)
    value_cols = [c for c in wide.columns if c != "cliente"]
    return wide.rename({
        c: f"{prefix}_{_safe_feature_name(c)}"
        for c in value_cols
    })


def _matrix_from_interactions(
    interactions: pl.DataFrame,
    *,
    use_recency_weight: bool = True,
) -> tuple[csr_matrix, pl.Series, pl.DataFrame]:
    """Build row-normalised sparse TF-IDF customer-product matrix."""
    popularity = build_product_popularity().select([
        "idarticu",
        "idf_weight",
        "is_popularity_removed",
    ])
    iact = (
        interactions
        .join(popularity, on="idarticu", how="left")
        .with_columns([
            pl.col("idf_weight").fill_null(1.0).cast(pl.Float32),
            pl.col("is_popularity_removed").fill_null(False),
        ])
    )
    if PRODUCT_POPULARITY_ENABLED:
        iact = iact.filter(~pl.col("is_popularity_removed"))

    iact_cat = iact.with_columns(pl.col("cliente").cast(pl.Categorical))
    row = iact_cat["cliente"].to_physical().to_numpy().astype(np.int32)
    unique_customers = iact_cat["cliente"].cat.get_categories()
    del iact_cat

    prod_arr = iact["idarticu"].to_numpy()
    unique_prods, col = np.unique(prod_arr, return_inverse=True)
    del prod_arr

    base = (
        iact["weight"].to_numpy().astype(np.float32)
        if use_recency_weight
        else iact["total_purchases"].to_numpy().astype(np.float32)
    )
    data = _apply_weight_transform(base)
    data *= iact["idf_weight"].to_numpy().astype(np.float32)

    matrix = csr_matrix(
        (data, (row, col.astype(np.int32))),
        shape=(len(unique_customers), len(unique_prods)),
        dtype=np.float32,
    )
    del data, row, col

    row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    row_norms[row_norms == 0] = 1.0
    matrix = matrix.multiply(1.0 / row_norms[:, None]).tocsr()

    product_index = pl.DataFrame({
        "idarticu": pl.Series(unique_prods, dtype=pl.Int64),
    })
    return matrix, unique_customers, product_index


# ─── 2. Stage 2: sparse matrix multiply → customer vectors ────────────────────

def _aggregate_vectors(
    interactions: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    use_weights: bool,
) -> pl.DataFrame:
    """(n_customers × n_products) sparse W @ (n_products × 100) dense E → (n_customers × 100).

    W is row-normalised so the result is a weighted mean, not a weighted sum.

    Parameters
    ----------
    use_weights : True → recency+frequency weights; False → binary (each product once)
    Product-popularity IDF is applied to both variants when configured.
    """
    emb_ids = embeddings["idarticu"].to_numpy()           # (V,)  int64
    emb_mat = np.array(
        embeddings["embedding"].to_list(), dtype=np.float32
    )                                                     # (V, 100)
    if NORMALIZE_PRODUCT_EMBEDDINGS:
        norms = np.linalg.norm(emb_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_mat = emb_mat / norms
        _log.info("  Product embeddings L2-normalised before customer aggregation")

    # Keep only interactions whose product has an embedding
    emb_id_set = pl.Series("idarticu", emb_ids)
    iact = interactions.filter(pl.col("idarticu").is_in(emb_id_set))
    if PRODUCT_IDF_WEIGHTING_ENABLED:
        popularity = build_product_popularity()
        iact = (
            iact
            .join(
                popularity.select(["idarticu", "idf_weight"]),
                on="idarticu",
                how="left",
            )
            .with_columns(pl.col("idf_weight").fill_null(1.0).cast(pl.Float32))
        )
        _log.info("  Applying product-popularity IDF weights to customer vectors")

    if len(iact) == 0:
        raise ValueError(
            "No (customer, product) interactions matched the embedding vocabulary. "
            "Check that product_embeddings.parquet and df_combined.parquet are aligned."
        )

    n_matched = len(iact)
    n_total   = len(interactions)
    _log.info(
        "  Pairs with embeddings: %s / %s  (%s unique customers covered)",
        f"{n_matched:,}", f"{n_total:,}",
        f"{iact['cliente'].n_unique():,}",
    )

    # Customer ID → row index via Polars Categorical (O(n), ~4 bytes/ID vs ~60 for numpy objects)
    # This saves ~9 GB of peak RAM versus np.unique on a Python string object array.
    iact_cat = iact.with_columns(pl.col("cliente").cast(pl.Categorical))
    row = iact_cat["cliente"].to_physical().to_numpy().astype(np.int32)  # uint32 → int32
    unique_customers = iact_cat["cliente"].cat.get_categories()           # pl.Series[str]
    del iact_cat

    # Product ID → column index via numpy unique (int64 — memory-efficient, no Python objects)
    prod_arr = iact["idarticu"].to_numpy()
    unique_prods, col_local = np.unique(prod_arr, return_inverse=True)
    del prod_arr

    # Map unique_prods (subset of vocab) → column indices in emb_mat
    prod_to_emb_col = {int(pid): i for i, pid in enumerate(emb_ids.tolist())}
    emb_cols = np.array([prod_to_emb_col[int(p)] for p in unique_prods.tolist()], dtype=np.int32)
    col = emb_cols[col_local]
    del unique_prods, col_local, emb_cols

    data = (
        iact["weight"].to_numpy().astype(np.float32)
        if use_weights
        else np.ones(len(iact), dtype=np.float32)
    )
    if use_weights:
        data = _apply_weight_transform(data)
    if PRODUCT_IDF_WEIGHTING_ENABLED:
        data *= iact["idf_weight"].to_numpy().astype(np.float32)
    del iact  # free the filtered DataFrame before building the sparse matrix

    n_cust = len(unique_customers)
    n_prod = len(emb_ids)
    _log.info(
        "Stage 2 — sparse @ dense  (%s × %s) @ (%s × %d) ...",
        f"{n_cust:,}", f"{n_prod:,}", f"{n_prod:,}", emb_mat.shape[1],
    )

    W = csr_matrix(
        (data, (row, col)),
        shape=(n_cust, n_prod),
        dtype=np.float32,
    )
    del data, row, col

    # Row-normalise → weighted mean rather than weighted sum
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W = W.multiply(1.0 / row_sums[:, None])

    vectors = (W @ emb_mat).astype(np.float32)            # (n_cust, 100)
    del W
    _log.info("  Done — %s customer vectors computed", f"{n_cust:,}")

    promo_df = _promo_rate_df(interactions)

    return (
        pl.DataFrame({
            "cliente": unique_customers,
            "vector":  pl.Series(vectors.tolist(), dtype=pl.List(pl.Float32)),
        })
        .join(promo_df, on="cliente", how="left")
        .sort("cliente")                               # explicit guarantee: downstream positional sampling is reproducible
    )


# ─── 3. Public API ────────────────────────────────────────────────────────────

def build_customer_vectors(
    df_combined_path: Path | None = None,
    embeddings_path: Path | None = None,
    *,
    halflife_days: int = RECENCY_HALFLIFE_DAYS,
    force: bool = False,
) -> pl.DataFrame:
    """Build frequency+recency-weighted customer vectors (primary Phase 2 output).

    Each product's contribution is weighted by how recently and how frequently
    the customer bought it (half-life = RECENCY_HALFLIFE_DAYS days).

    Returns
    -------
    DataFrame: cliente str | vector list[float32×100] | promo_rate float32
    Cached to data/processed/customer_vectors_weighted.parquet.
    """
    if df_combined_path is None:
        df_combined_path = DATA_PROCESSED / "df_combined.parquet"
    if embeddings_path is None:
        embeddings_path = DATA_PROCESSED / "product_embeddings.parquet"

    if _WEIGHTED_CACHE.exists() and not force:
        n = pl.scan_parquet(_WEIGHTED_CACHE).select(pl.len()).collect().item()
        _log.info("Weighted vectors cache hit — %s customers", f"{n:,}")
        return pl.read_parquet(_WEIGHTED_CACHE)

    interactions = _build_interactions(df_combined_path, halflife_days, force=force)
    embeddings   = pl.read_parquet(embeddings_path)
    df = _aggregate_vectors(interactions, embeddings, use_weights=True)

    df.write_parquet(_WEIGHTED_CACHE, compression="zstd")
    _log.info(
        "Saved %s weighted customer vectors → %s  (%.1f MB on disk)",
        f"{len(df):,}",
        _WEIGHTED_CACHE.name,
        _WEIGHTED_CACHE.stat().st_size / 1024 ** 2,
    )
    return df


def build_customer_vectors_tfidf_svd(
    df_combined_path: Path | None = None,
    *,
    halflife_days: int = RECENCY_HALFLIFE_DAYS,
    n_components: int = TFIDF_SVD_DIMS,
    force: bool = False,
) -> pl.DataFrame:
    """Build customer vectors from a sparse product TF-IDF matrix plus TruncatedSVD.

    This preserves product-affinity signal more directly than averaging Item2Vec
    embeddings. Values are log/sqrt/none transformed recency weights multiplied
    by product IDF, L2 row-normalised, then reduced to a dense vector.
    """
    if df_combined_path is None:
        df_combined_path = DATA_PROCESSED / "df_combined.parquet"

    if _TFIDF_SVD_CACHE.exists() and not force:
        n = pl.scan_parquet(_TFIDF_SVD_CACHE).select(pl.len()).collect().item()
        _log.info("TF-IDF SVD vectors cache hit - %s customers", f"{n:,}")
        return pl.read_parquet(_TFIDF_SVD_CACHE)

    interactions = _build_interactions(df_combined_path, halflife_days, force=force)
    matrix, unique_customers, product_index = _matrix_from_interactions(
        interactions,
        use_recency_weight=True,
    )

    n_components = min(int(n_components), max(2, matrix.shape[1] - 1))
    _log.info(
        "TF-IDF SVD fit: sparse matrix %s x %s -> %d dims",
        f"{matrix.shape[0]:,}",
        f"{matrix.shape[1]:,}",
        n_components,
    )
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    vectors = svd.fit_transform(matrix).astype(np.float32)
    vectors = _l2_normalize_rows(vectors)
    explained = float(svd.explained_variance_ratio_.sum())
    _log.info("  TF-IDF SVD explained variance: %.1f%%", explained * 100)

    del matrix, product_index
    df = (
        pl.DataFrame({
            "cliente": unique_customers,
            "vector": pl.Series(vectors.tolist(), dtype=pl.List(pl.Float32)),
        })
        .join(_promo_rate_df(interactions), on="cliente", how="left")
        .sort("cliente")
    )

    _TFIDF_SVD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_TFIDF_SVD_CACHE, compression="zstd")
    _log.info(
        "Saved %s TF-IDF SVD customer vectors -> %s",
        f"{len(df):,}",
        _TFIDF_SVD_CACHE.name,
    )
    return df


def build_store_features(
    df_combined_path: Path | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Per-customer store affinity features.

    Computes the fraction of each customer's total spend at each store format.
    A single-store loyalist gets share=1.0 for their store and 0.0 for all others.
    A cross-format shopper gets distributed shares.

    This is a first-class behavioural dimension: a customer who shops exclusively
    at a suburban hypermarket is structurally different from one who uses the city
    express format — not because of who they are but because of what each format
    stocks and in what basket context.

    Returns
    -------
    DataFrame: cliente str | spend_share_s{X} float32 for each store X
    Cached to customer_store_features.parquet.
    """
    if _STORE_FEATURES_CACHE.exists() and not force:
        n = pl.scan_parquet(_STORE_FEATURES_CACHE).select(pl.len()).collect().item()
        _log.info("Store features cache hit — %s customers", f"{n:,}")
        return pl.read_parquet(_STORE_FEATURES_CACHE)

    if df_combined_path is None:
        df_combined_path = DATA_PROCESSED / "df_combined.parquet"

    _log.info("Computing per-customer store spend shares from %s ...", df_combined_path.name)

    # Per-(customer, store) total spend — cast store ID to string for clean column names
    store_spend = (
        pl.scan_parquet(df_combined_path)
        .select(["cliente", "idempres", "importe"])
        .with_columns(pl.col("idempres").cast(pl.Utf8).alias("store"))
        .group_by(["cliente", "store"])
        .agg(pl.col("importe").sum().alias("spend"))
        .collect(engine="streaming")
    )

    stores = store_spend["store"].unique().sort().to_list()
    _log.info("  Stores found: %s", stores)

    # Pivot → one column per store; customers absent from a store get 0.0
    df_wide = (
        store_spend
        .pivot(on="store", index="cliente", values="spend", aggregate_function="sum")
        .fill_null(0.0)
    )
    raw_cols = [c for c in df_wide.columns if c != "cliente"]

    # Row-normalise to spend shares (each row sums to 1.0)
    df = (
        df_wide
        .with_columns(
            pl.sum_horizontal([pl.col(c) for c in raw_cols]).alias("_total")
        )
        .with_columns([
            (pl.col(c) / pl.col("_total")).cast(pl.Float32).alias(f"spend_share_s{c}")
            for c in raw_cols
        ])
        .select(["cliente"] + [f"spend_share_s{c}" for c in raw_cols])
        .sort("cliente")                # canonical order — pivot output row order is non-deterministic
    )

    _STORE_FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_STORE_FEATURES_CACHE, compression="zstd")
    _log.info(
        "Saved %s customer store features → %s  (%d stores, %.1f MB on disk)",
        f"{len(df):,}", _STORE_FEATURES_CACHE.name, len(stores),
        _STORE_FEATURES_CACHE.stat().st_size / 1024 ** 2,
    )
    return df


def build_customer_share_features(
    df_combined_path: Path | None = None,
    *,
    include_store_features: bool = False,
    force: bool = False,
) -> pl.DataFrame:
    """Build product/category share features from transaction fields.

    Includes:
      - sector line/spend shares from desc_sector
      - product-type line/spend shares from idtiprod
      - strategic product-theme line/spend shares from product name patterns

    Store spend shares are excluded by default so the Experiment 0 baseline is
    product-only. Pass include_store_features=True for an explicit store ablation.
    """
    cache = (
        _SHARE_FEATURES_WITH_STORE_CACHE
        if include_store_features
        else _PRODUCT_SHARE_FEATURES_CACHE
    )
    if cache.exists() and not force:
        n = pl.scan_parquet(cache).select(pl.len()).collect().item()
        _log.info("Customer share features cache hit - %s customers (%s)", f"{n:,}", cache.name)
        return pl.read_parquet(cache)

    if df_combined_path is None:
        df_combined_path = DATA_PROCESSED / "df_combined.parquet"

    _log.info(
        "Building customer product/category share features from %s (store_features=%s) ...",
        df_combined_path.name,
        include_store_features,
    )
    base = pl.scan_parquet(df_combined_path).select([
        "cliente",
        "idempres",
        "idarticu",
        "idtiprod",
        "desc_larga_articulo",
        "desc_sector",
        "importe",
    ])

    sector_counts = (
        base
        .group_by(["cliente", "desc_sector"])
        .agg([
            pl.len().alias("line_count"),
            pl.col("importe").sum().alias("spend"),
        ])
        .collect(engine="streaming")
    )
    sector_line = _wide_share_from_counts(
        sector_counts,
        category_col="desc_sector",
        value_col="line_count",
        total_col="_sector_lines",
        prefix="sector_line_share",
    )
    sector_spend = _wide_share_from_counts(
        sector_counts,
        category_col="desc_sector",
        value_col="spend",
        total_col="_sector_spend",
        prefix="sector_spend_share",
    )

    type_counts = (
        base
        .with_columns(pl.col("idtiprod").cast(pl.Utf8).alias("product_type"))
        .group_by(["cliente", "product_type"])
        .agg([
            pl.len().alias("line_count"),
            pl.col("importe").sum().alias("spend"),
        ])
        .collect(engine="streaming")
    )
    type_line = _wide_share_from_counts(
        type_counts,
        category_col="product_type",
        value_col="line_count",
        total_col="_type_lines",
        prefix="type_line_share",
    )
    type_spend = _wide_share_from_counts(
        type_counts,
        category_col="product_type",
        value_col="spend",
        total_col="_type_spend",
        prefix="type_spend_share",
    )

    product_theme = (
        base
        .select(["idarticu", "desc_larga_articulo"])
        .unique()
        .collect(engine="streaming")
        .with_columns(
            pl.col("desc_larga_articulo")
            .map_elements(classify_product_themes, return_dtype=pl.List(pl.Utf8))
            .alias("theme")
        )
        .explode("theme")
        .filter(pl.col("theme").is_not_null())
        .select(["idarticu", "theme"])
    )
    if len(product_theme) > 0:
        customer_totals = (
            base
            .group_by("cliente")
            .agg([
                pl.len().alias("total_lines"),
                pl.col("importe").sum().alias("total_spend"),
            ])
            .collect(engine="streaming")
        )
        theme_counts = (
            base
            .select(["cliente", "idarticu", "importe"])
            .join(product_theme.lazy(), on="idarticu", how="inner")
            .group_by(["cliente", "theme"])
            .agg([
                pl.len().alias("line_count"),
                pl.col("importe").sum().alias("spend"),
            ])
            .collect(engine="streaming")
            .join(customer_totals, on="cliente", how="left")
            .with_columns([
                (pl.col("line_count") / pl.col("total_lines")).cast(pl.Float32).alias("line_share"),
                (
                    pl.when(pl.col("total_spend") > 0)
                    .then(pl.col("spend") / pl.col("total_spend"))
                    .otherwise(0.0)
                )
                .cast(pl.Float32)
                .alias("spend_share"),
            ])
        )
    else:
        theme_counts = pl.DataFrame(
            schema={
                "cliente": pl.Utf8,
                "theme": pl.Utf8,
                "line_count": pl.Int64,
                "spend": pl.Float64,
                "line_share": pl.Float32,
                "spend_share": pl.Float32,
            }
        )
    theme_line = _wide_value_features(
        theme_counts,
        category_col="theme",
        value_col="line_share",
        prefix="theme_line_share",
    )
    theme_spend = _wide_value_features(
        theme_counts,
        category_col="theme",
        value_col="spend_share",
        prefix="theme_spend_share",
    )

    feature_frames = [
        sector_line,
        sector_spend,
        type_line,
        type_spend,
        theme_line,
        theme_spend,
    ]
    if include_store_features:
        feature_frames.append(build_store_features(df_combined_path, force=force))

    customers = (
        base
        .select("cliente")
        .unique()
        .collect(engine="streaming")
        .sort("cliente")
    )
    out = customers
    for features in feature_frames:
        out = out.join(features, on="cliente", how="left")

    out = out.fill_null(0.0).sort("cliente")
    out.write_parquet(cache, compression="zstd")
    _log.info(
        "Saved %s customer share feature rows -> %s (%d features)",
        f"{len(out):,}",
        cache.name,
        len(out.columns) - 1,
    )
    return out


def build_customer_vectors_mean(
    df_combined_path: Path | None = None,
    embeddings_path: Path | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Build simple mean customer vectors (baseline - no recency/frequency weighting).

    Each purchased product contributes once regardless of recency or frequency.
    If product-popularity IDF is enabled, the same IDF scaling used by the
    weighted vectors is applied here too.
    Compared against the weighted method in Section 6.4 to quantify the signal
    added by time-decay aggregation.

    Returns
    -------
    DataFrame: cliente str | vector list[float32×100] | promo_rate float32
    Cached to data/processed/customer_vectors_mean.parquet.
    """
    if df_combined_path is None:
        df_combined_path = DATA_PROCESSED / "df_combined.parquet"
    if embeddings_path is None:
        embeddings_path = DATA_PROCESSED / "product_embeddings.parquet"

    if _MEAN_CACHE.exists() and not force:
        n = pl.scan_parquet(_MEAN_CACHE).select(pl.len()).collect().item()
        _log.info("Mean vectors cache hit — %s customers", f"{n:,}")
        return pl.read_parquet(_MEAN_CACHE)

    # Reuse the cached Stage 1 interactions (halflife_days value doesn't affect mean)
    interactions = _build_interactions(
        df_combined_path, RECENCY_HALFLIFE_DAYS, force=False
    )
    embeddings = pl.read_parquet(embeddings_path)
    df = _aggregate_vectors(interactions, embeddings, use_weights=False)

    df.write_parquet(_MEAN_CACHE, compression="zstd")
    _log.info(
        "Saved %s mean customer vectors → %s  (%.1f MB on disk)",
        f"{len(df):,}",
        _MEAN_CACHE.name,
        _MEAN_CACHE.stat().st_size / 1024 ** 2,
    )
    return df


def build_customer_vectors_hybrid(
    df_combined_path: Path | None = None,
    embeddings_path: Path | None = None,
    *,
    include_store_features: bool = HYBRID_INCLUDE_STORE_SHARES,
    force: bool = False,
) -> pl.DataFrame:
    """Build a hybrid product-led vector for tribe discovery.

    Concatenates three row-normalised blocks:
      - enhanced Item2Vec weighted-average vector
      - product TF-IDF + SVD vector
      - product-only sector/type/theme share features

    The default is product-only. Pass include_store_features=True only for an
    explicit store-ablation experiment; this writes to a separate cache.
    """
    cache = (
        _HYBRID_WITH_STORE_CACHE
        if include_store_features
        else _HYBRID_PRODUCT_ONLY_CACHE
    )
    if cache.exists() and not force:
        n = pl.scan_parquet(cache).select(pl.len()).collect().item()
        _log.info("Hybrid customer vectors cache hit - %s customers (%s)", f"{n:,}", cache.name)
        return pl.read_parquet(cache)

    item2vec = build_customer_vectors(
        df_combined_path=df_combined_path,
        embeddings_path=embeddings_path,
        force=force,
    )
    tfidf = build_customer_vectors_tfidf_svd(
        df_combined_path=df_combined_path,
        force=force,
    ).rename({"vector": "vector_tfidf", "promo_rate": "promo_rate_tfidf"})
    shares = build_customer_share_features(
        df_combined_path=df_combined_path,
        include_store_features=include_store_features,
        force=force,
    )

    aligned = (
        item2vec
        .join(tfidf.select(["cliente", "vector_tfidf"]), on="cliente", how="inner")
        .join(shares, on="cliente", how="left")
        .fill_null(0.0)
        .sort("cliente")
    )

    share_cols = [
        c for c in aligned.columns
        if c not in {"cliente", "vector", "vector_tfidf", "promo_rate"}
    ]
    item_arr = _l2_normalize_rows(
        np.array(aligned["vector"].to_list(), dtype=np.float32)
    ) * HYBRID_WEIGHT_ITEM2VEC
    tfidf_arr = _l2_normalize_rows(
        np.array(aligned["vector_tfidf"].to_list(), dtype=np.float32)
    ) * HYBRID_WEIGHT_TFIDF_SVD
    share_arr = aligned.select(share_cols).to_numpy().astype(np.float32)
    share_arr = np.sqrt(np.maximum(share_arr, 0.0)).astype(np.float32)
    share_arr = _l2_normalize_rows(share_arr) * HYBRID_WEIGHT_CATEGORY_SHARES

    hybrid = np.hstack([item_arr, tfidf_arr, share_arr]).astype(np.float32)
    del item_arr, tfidf_arr, share_arr

    df = pl.DataFrame({
        "cliente": aligned["cliente"],
        "vector": pl.Series(hybrid.tolist(), dtype=pl.List(pl.Float32)),
        "promo_rate": aligned["promo_rate"],
    })
    df.write_parquet(cache, compression="zstd")
    _log.info(
        "Saved %s hybrid customer vectors -> %s (%d dims, store_features=%s)",
        f"{len(df):,}",
        cache.name,
        hybrid.shape[1],
        include_store_features,
    )
    return df


def build_customer_vectors_hybrid_with_store(
    df_combined_path: Path | None = None,
    embeddings_path: Path | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Build the non-baseline hybrid vector with store shares included."""
    return build_customer_vectors_hybrid(
        df_combined_path=df_combined_path,
        embeddings_path=embeddings_path,
        include_store_features=True,
        force=force,
    )
