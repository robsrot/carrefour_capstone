"""Customer behavioral features and feature-set assembly."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.utils import collect_streaming, numeric_feature_columns, schema_names, should_use_cache


def _promo_flag(columns: set[str]) -> pl.Expr:
    if "idpromoc" not in columns:
        return pl.lit(0).alias("_promo_flag")
    promo = pl.col("idpromoc").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    no_promo_values = ["", "0", "none", "null", "nan", "no promo", "no_promo", "sin promo", "sin promocion"]
    return (
        pl.when(promo.is_not_null() & (~promo.is_in(no_promo_values)))
        .then(1)
        .otherwise(0)
        .alias("_promo_flag")
    )


def build_behavioral_features(
    transactions: pl.LazyFrame | None = None,
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create one row per customer of non-demographic purchase behavior KPIs."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.artifact_path("behavioral_features", "output")
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True)):
        return output

    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    columns = set(schema_names(lf))
    reference_date_cfg = cfg.get("behavioral_features.reference_date")
    if reference_date_cfg:
        reference_date = date.fromisoformat(str(reference_date_cfg))
    else:
        reference_date = collect_streaming(lf.select(pl.col("fecha").max().alias("max_date")))[0, "max_date"]

    base = lf.with_columns(_promo_flag(columns))
    basket = base.group_by(["cliente", "ticket"]).agg(
        [
            pl.col("importe").sum().alias("_basket_spend"),
            pl.col("unidades").sum().alias("_basket_units"),
            pl.len().alias("_basket_lines"),
            pl.col("_promo_flag").sum().alias("_basket_promo_lines"),
            pl.col("fecha").max().alias("_basket_date"),
        ]
    )
    basket_features = basket.group_by("cliente").agg(
        [
            pl.len().alias("ticket_count"),
            pl.col("_basket_spend").sum().alias("total_spend"),
            pl.col("_basket_units").sum().alias("total_units"),
            pl.col("_basket_spend").mean().alias("avg_basket_value"),
            pl.col("_basket_units").mean().alias("avg_items_per_basket"),
            (pl.col("_basket_promo_lines").sum() / pl.col("_basket_lines").sum()).alias("promo_share"),
            pl.col("_basket_date").max().alias("last_purchase_date"),
            pl.col("_basket_date").min().alias("first_purchase_date"),
        ]
    )
    diversity_exprs = [pl.col("idarticu").n_unique().alias("unique_products")]
    if "idsector" in columns:
        diversity_exprs.append(pl.col("idsector").n_unique().alias("unique_sectors"))
    else:
        diversity_exprs.append(pl.lit(None, dtype=pl.UInt32).alias("unique_sectors"))

    diversity = base.group_by("cliente").agg(diversity_exprs)
    result = (
        basket_features.join(diversity, on="cliente", how="left")
        .with_columns(
            [
                (pl.lit(reference_date) - pl.col("last_purchase_date")).dt.total_days().alias("recency_days"),
                (
                    pl.col("ticket_count")
                    / ((pl.col("last_purchase_date") - pl.col("first_purchase_date")).dt.total_days() + 1)
                    * 30.0
                ).alias("frequency_per_30d"),
            ]
        )
        .drop(["first_purchase_date", "last_purchase_date"])
        .sort("cliente")
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    collect_streaming(result).write_parquet(output)
    return output


def build_feature_set(
    customer_embeddings_path: str | Path,
    behavior_path: str | Path | None = None,
    variant: str = "embeddings_only",
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create model-ready feature variants A and B."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    outputs = cfg.get("feature_sets.outputs", {})
    output = Path(output_path) if output_path else cfg.data_processed / outputs.get(variant, f"feature_set_{variant}.parquet")
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True)):
        return output

    embeddings = pl.read_parquet(customer_embeddings_path)
    emb_cols = [col for col in embeddings.columns if col.startswith("emb_")]
    if variant == "embeddings_only":
        result = embeddings.select(["cliente", *emb_cols]).sort("cliente")
    elif variant == "embeddings_behavior":
        if behavior_path is None:
            raise ValueError("behavior_path is required for embeddings_behavior feature set")
        behavior = pl.read_parquet(behavior_path)
        joined = embeddings.select(["cliente", *emb_cols]).join(behavior, on="cliente", how="inner")
        behavior_cols = [
            col
            for col in numeric_feature_columns(joined, exclude=("cliente", *emb_cols))
            if col not in {"embedding_weight_sum", "embedded_unique_products"}
        ]
        result = joined.select(["cliente", *emb_cols, *behavior_cols]).fill_null(0)
        if cfg.get("feature_sets.standardize_behavior", True):
            updates = []
            for col in behavior_cols:
                values = result[col].to_numpy().astype(np.float64)
                mean = float(np.nanmean(values))
                std = float(np.nanstd(values))
                denom = std if std > 1e-12 else 1.0
                updates.append(((pl.col(col) - mean) / denom).cast(pl.Float32).alias(f"beh_{col}"))
            result = result.with_columns(updates).drop(behavior_cols)
        result = result.sort("cliente")
    else:
        raise ValueError(f"Unknown feature set variant: {variant}")

    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output)
    return output
