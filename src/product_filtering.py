"""Product popularity diagnostics used to spot universal staples."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.utils import collect_streaming, file_fingerprint, schema_names, should_use_cache, write_artifact_metadata


def build_product_popularity(
    transactions: pl.LazyFrame | None = None,
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.data_processed / "product_popularity.parquet"
    cache_metadata = {
        "stage": "product_popularity",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
    }
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata):
        return output

    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    cols = set(schema_names(lf))
    product_cols = ["idarticu"]
    for col in ["desc_larga_articulo", "desc_sector"]:
        if col in cols:
            product_cols.append(col)

    totals = collect_streaming(
        lf.select(
            [
                pl.col("ticket").n_unique().alias("total_baskets"),
                pl.col("cliente").n_unique().alias("total_customers"),
            ]
        )
    ).row(0, named=True)
    product_customer = lf.select(["cliente", "idarticu"]).unique()
    product_basket = lf.select(["ticket", "idarticu"]).unique()
    popularity = (
        lf.select(product_cols)
        .unique(subset=["idarticu"])
        .join(product_customer.group_by("idarticu").agg(pl.len().alias("customer_count")), on="idarticu", how="left")
        .join(product_basket.group_by("idarticu").agg(pl.len().alias("basket_count")), on="idarticu", how="left")
        .with_columns(
            [
                (pl.col("customer_count") / totals["total_customers"]).alias("customer_share"),
                (pl.col("basket_count") / totals["total_baskets"]).alias("basket_share"),
            ]
        )
        .with_columns(
            (((pl.lit(totals["total_baskets"] + 1) / (pl.col("basket_count") + 1)).log()) + 1).alias(
                "idf_weight"
            )
        )
        .sort("basket_share", descending=True)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    collect_streaming(popularity).write_parquet(output)
    write_artifact_metadata(output, cache_metadata)
    return output
