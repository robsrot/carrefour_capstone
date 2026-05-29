"""Phase 2 — Customer vector aggregation (product embeddings → behavioral profiles).

Each customer's purchase history is compressed into a single 100-dim vector by
taking a weighted mean of the product embeddings for every product they bought.

Two variants are produced:

  weighted (primary)
    weight per (customer, product) pair = Σ recency_decay over all purchases of that product
    recency_decay(t) = exp(-ln(2)/halflife * days_before_reference)
    → Recent and frequently-purchased products dominate the customer's profile.

  mean (baseline)
    Simple unweighted average — every purchased product contributes equally once.
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
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from src.config import (
    DATA_PROCESSED,
    RECENCY_HALFLIFE_DAYS,
    W2V_VECTOR_SIZE,
)

_log = logging.getLogger(__name__)

_REFERENCE_DATE     = date(2022, 6, 30)      # last day in the dataset
_INTERACTIONS_CACHE = DATA_PROCESSED / "customer_product_weights.parquet"
_WEIGHTED_CACHE     = DATA_PROCESSED / "customer_vectors_weighted.parquet"
_MEAN_CACHE         = DATA_PROCESSED / "customer_vectors_mean.parquet"


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
            (
                pl.lit(-decay_lambda, dtype=pl.Float64)
                * (pl.lit(_REFERENCE_DATE) - pl.col("fecha")).dt.total_days().cast(pl.Float64)
            ).exp().alias("recency_weight"),
            # idpromoc is String in df_combined; null or "0" → non-promo
            (pl.col("idpromoc").is_not_null() & (pl.col("idpromoc") != "0"))
            .cast(pl.Int32).alias("is_promo"),
        ])
        .group_by(["cliente", "idarticu"])
        .agg([
            pl.col("recency_weight").sum().alias("weight"),
            pl.col("is_promo").sum().alias("promo_purchases"),
            pl.len().cast(pl.Int32).alias("total_purchases"),
        ])
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
    """
    emb_ids = embeddings["idarticu"].to_numpy()           # (V,)  int64
    emb_mat = np.array(
        embeddings["embedding"].to_list(), dtype=np.float32
    )                                                     # (V, 100)

    # Keep only interactions whose product has an embedding
    emb_id_set = pl.Series("idarticu", emb_ids)
    iact = interactions.filter(pl.col("idarticu").is_in(emb_id_set))

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

    # Map string/int IDs → dense row/col indices via numpy unique (O(n log n), no Python loops)
    cust_arr = iact["cliente"].to_numpy()
    prod_arr = iact["idarticu"].to_numpy()

    unique_customers, row = np.unique(cust_arr, return_inverse=True)  # row: per-interaction idx
    unique_prods, col_local = np.unique(prod_arr, return_inverse=True)

    # unique_prods is a subset of emb_ids — map each to its column in emb_mat
    prod_to_emb_col = {int(pid): i for i, pid in enumerate(emb_ids.tolist())}
    emb_cols = np.array([prod_to_emb_col[int(p)] for p in unique_prods.tolist()], dtype=np.int32)
    col = emb_cols[col_local]                              # embedding column per interaction

    data = (
        iact["weight"].to_numpy().astype(np.float32)
        if use_weights
        else np.ones(len(iact), dtype=np.float32)
    )

    n_cust = len(unique_customers)
    n_prod = len(emb_ids)
    _log.info(
        "Stage 2 — sparse @ dense  (%s × %s) @ (%s × %d) ...",
        f"{n_cust:,}", f"{n_prod:,}", f"{n_prod:,}", emb_mat.shape[1],
    )

    W = csr_matrix(
        (data, (row.astype(np.int32), col)),
        shape=(n_cust, n_prod),
        dtype=np.float32,
    )

    # Row-normalise → weighted mean rather than weighted sum
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W = W.multiply(1.0 / row_sums[:, None])

    vectors = (W @ emb_mat).astype(np.float32)            # (n_cust, 100)
    _log.info("  Done — %s customer vectors computed", f"{n_cust:,}")

    # Per-customer promo rate (from full interactions, not just embedded subset)
    promo_df = (
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

    return (
        pl.DataFrame({
            "cliente": pl.Series(unique_customers.tolist()),
            "vector":  pl.Series(vectors.tolist(), dtype=pl.List(pl.Float32)),
        })
        .join(promo_df, on="cliente", how="left")
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


def build_customer_vectors_mean(
    df_combined_path: Path | None = None,
    embeddings_path: Path | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Build simple mean customer vectors (baseline — no recency/frequency weighting).

    Each purchased product contributes equally regardless of recency or frequency.
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
