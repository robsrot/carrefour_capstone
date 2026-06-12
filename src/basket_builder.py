"""Basket sentence construction for Item2Vec training."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.utils import collect_streaming, should_use_cache


def build_basket_sentences(
    transactions: pl.LazyFrame | None = None,
    output_path: str | Path | None = None,
    repeat_product_by_quantity: bool | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create one ticket-level product-token sentence per basket."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.artifact_path("baskets", "output")
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True)):
        return output

    repeat = (
        bool(cfg.get("baskets.repeat_product_by_quantity", False))
        if repeat_product_by_quantity is None
        else repeat_product_by_quantity
    )
    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)

    base = lf.select(["ticket", "idarticu"] + (["unidades"] if repeat else []))
    if repeat:
        token_lf = (
            base.with_columns(
                [
                    pl.col("idarticu").cast(pl.Utf8).alias("_product_token"),
                    pl.when(pl.col("unidades").cast(pl.Float64) > 0)
                    .then(pl.col("unidades").cast(pl.Int64))
                    .otherwise(1)
                    .clip(1, 20)
                    .cast(pl.UInt32)
                    .alias("_repeat_count"),
                ]
            )
            .with_columns(pl.col("_product_token").repeat_by("_repeat_count").alias("_tokens"))
            .select(["ticket", "_tokens"])
            .explode("_tokens")
        )
        basket_lf = token_lf.group_by("ticket").agg(
            [
                pl.col("_tokens").alias("products"),
                pl.len().alias("n_product_tokens"),
            ]
        )
    else:
        basket_lf = base.with_columns(pl.col("idarticu").cast(pl.Utf8).alias("_product_token")).group_by(
            "ticket"
        ).agg(
            [
                pl.col("_product_token").unique().sort().alias("products"),
                pl.col("_product_token").n_unique().alias("n_product_tokens"),
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    collect_streaming(basket_lf.sort("ticket")).write_parquet(output)
    return output


def basket_summary(basket_path: str | Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(basket_path)
        .select(
            [
                pl.len().alias("n_baskets"),
                pl.col("n_product_tokens").mean().alias("avg_products_per_basket"),
                pl.col("n_product_tokens").median().alias("median_products_per_basket"),
                pl.col("n_product_tokens").max().alias("max_products_per_basket"),
            ]
        )
        .collect()
    )
