"""Product popularity diagnostics and filtering helpers.

This module supports product-led segmentation by identifying products that are
too common to be useful segment drivers. It separates two ideas:

- hard removal: products above configured basket/customer share thresholds
- soft weighting: IDF-style weights that mute common remaining products
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.config import (
    DATA_PROCESSED,
    PRODUCT_IDF_MAX_WEIGHT,
    PRODUCT_IDF_MIN_WEIGHT,
    PRODUCT_MAX_BASKET_SHARE,
    PRODUCT_MAX_CUSTOMER_SHARE,
    PRODUCT_POPULARITY_ENABLED,
)

_log = logging.getLogger(__name__)

_POPULARITY_CACHE = DATA_PROCESSED / "product_popularity.parquet"

_RAW_COLUMNS = {
    "idarticu",
    "n_purchase_lines",
    "n_baskets",
    "basket_share",
    "n_customers",
    "customer_share",
    "raw_idf",
}


def _decorate_popularity(df: pl.DataFrame) -> pl.DataFrame:
    """Add config-dependent filter and IDF columns to raw popularity metrics."""
    raw = pl.col("raw_idf")
    idf_weight = (
        pl.when(raw < PRODUCT_IDF_MIN_WEIGHT)
        .then(PRODUCT_IDF_MIN_WEIGHT)
        .when(raw > PRODUCT_IDF_MAX_WEIGHT)
        .then(PRODUCT_IDF_MAX_WEIGHT)
        .otherwise(raw)
        .cast(pl.Float32)
    )

    is_removed = (
        (pl.col("basket_share") > PRODUCT_MAX_BASKET_SHARE)
        | (pl.col("customer_share") > PRODUCT_MAX_CUSTOMER_SHARE)
    )
    if not PRODUCT_POPULARITY_ENABLED:
        is_removed = pl.lit(False)

    return df.with_columns([
        idf_weight.alias("idf_weight"),
        is_removed.alias("is_popularity_removed"),
    ])


def build_product_popularity(
    df_combined_path: Path | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Compute product basket/customer penetration and IDF weights.

    Returns one row per product:
        idarticu
        n_purchase_lines
        n_baskets
        basket_share
        n_customers
        customer_share
        raw_idf
        idf_weight
        is_popularity_removed
    """
    if df_combined_path is None:
        df_combined_path = DATA_PROCESSED / "df_combined.parquet"

    if _POPULARITY_CACHE.exists() and not force:
        cached = pl.read_parquet(_POPULARITY_CACHE)
        if _RAW_COLUMNS.issubset(set(cached.columns)):
            _log.info("Product popularity cache hit - %s", _POPULARITY_CACHE.name)
            return _decorate_popularity(cached)
        _log.info("Product popularity cache uses an older schema - rebuilding")

    _log.info("Computing product popularity from %s ...", df_combined_path.name)
    base = pl.scan_parquet(df_combined_path).select(["cliente", "ticket", "idarticu"])

    totals = base.select([
        pl.col("ticket").n_unique().alias("total_baskets"),
        pl.col("cliente").n_unique().alias("total_customers"),
    ]).collect()
    total_baskets = int(totals["total_baskets"][0])
    total_customers = int(totals["total_customers"][0])

    if total_baskets == 0 or total_customers == 0:
        raise ValueError("Cannot compute product popularity from an empty transaction file.")

    popularity = (
        base
        .group_by("idarticu")
        .agg([
            pl.len().alias("n_purchase_lines"),
            pl.col("ticket").n_unique().alias("n_baskets"),
            pl.col("cliente").n_unique().alias("n_customers"),
        ])
        .with_columns([
            (pl.col("n_baskets") / pl.lit(total_baskets)).cast(pl.Float32).alias("basket_share"),
            (pl.col("n_customers") / pl.lit(total_customers)).cast(pl.Float32).alias("customer_share"),
            (
                (pl.lit(1 + total_customers, dtype=pl.Float64)
                 / (pl.col("n_customers").cast(pl.Float64) + 1.0))
                .log()
            ).cast(pl.Float32).alias("raw_idf"),
        ])
        .select([
            "idarticu",
            "n_purchase_lines",
            "n_baskets",
            "basket_share",
            "n_customers",
            "customer_share",
            "raw_idf",
        ])
        .sort(["customer_share", "basket_share"], descending=True)
        .collect(engine="streaming")
    )

    _POPULARITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    popularity.write_parquet(_POPULARITY_CACHE, compression="zstd")
    _log.info(
        "Saved %s product popularity rows -> %s",
        f"{len(popularity):,}",
        _POPULARITY_CACHE.name,
    )
    return _decorate_popularity(popularity)


def allowed_product_ids(popularity: pl.DataFrame | None = None) -> pl.Series | None:
    """Return product IDs kept by the hard popularity filter, or None if disabled."""
    if not PRODUCT_POPULARITY_ENABLED:
        return None
    if popularity is None:
        popularity = build_product_popularity()
    return popularity.filter(~pl.col("is_popularity_removed"))["idarticu"]


def popularity_filter_summary(popularity: pl.DataFrame | None = None) -> dict[str, float | int]:
    """Compact summary for notebook logging."""
    if popularity is None:
        popularity = build_product_popularity()
    n_products = len(popularity)
    n_removed = int(popularity["is_popularity_removed"].sum())
    return {
        "enabled": int(PRODUCT_POPULARITY_ENABLED),
        "n_products": n_products,
        "n_removed": n_removed,
        "removed_pct": round(n_removed / n_products * 100, 2) if n_products else 0.0,
        "max_basket_share": PRODUCT_MAX_BASKET_SHARE,
        "max_customer_share": PRODUCT_MAX_CUSTOMER_SHARE,
    }
