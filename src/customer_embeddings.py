"""Customer-level product embedding aggregation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.progress import log_event, stage_timer
from src.utils import collect_streaming, file_fingerprint, schema_names, should_use_cache, write_artifact_metadata


def _weight_expression(columns: set[str]) -> pl.Expr:
    if "importe" in columns:
        return (
            pl.when(pl.col("importe").cast(pl.Float64) > 0)
            .then(pl.col("importe").cast(pl.Float64))
            .otherwise(
                pl.when(pl.col("unidades").cast(pl.Float64) > 0)
                .then(pl.col("unidades").cast(pl.Float64))
                .otherwise(1.0)
            )
            .alias("_weight")
        )
    if "unidades" in columns:
        return (
            pl.when(pl.col("unidades").cast(pl.Float64) > 0)
            .then(pl.col("unidades").cast(pl.Float64))
            .otherwise(1.0)
            .alias("_weight")
        )
    return pl.lit(1.0).alias("_weight")


def build_customer_embeddings(
    embeddings_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Aggregate product embeddings directly to one weighted vector per customer."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.artifact_path(
        "customer_embeddings",
        "output",
        directory=cfg.outputs / "features",
    )
    cache_metadata = {
        "stage": "customer_embeddings",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "product_embeddings": file_fingerprint(embeddings_path),
        "weight_priority": cfg.get("customer_embeddings.weight_priority", ["importe", "unidades", "equal"]),
    }
    if should_use_cache(
        output,
        force=force,
        use_cached=cfg.get("cache.use_cached", True),
        metadata=cache_metadata,
    ):
        log_event("Stage 4 customer embeddings", "cache hit", cfg=cfg, path=output)
        return output

    with stage_timer("Stage 4 customer embeddings", "aggregating product vectors", cfg=cfg, output=output):
        lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
        columns = set(schema_names(lf))
        embedding_lf = pl.scan_parquet(embeddings_path)
        emb_cols = [col for col in schema_names(embedding_lf) if col.startswith("emb_")]
        if not emb_cols:
            raise ValueError(f"No embedding columns found in {embeddings_path}")

        needed = ["cliente", "idarticu"] + [col for col in ["importe", "unidades"] if col in columns]
        joined = (
            lf.select(needed)
            .join(embedding_lf, on="idarticu", how="inner")
            .with_columns(_weight_expression(columns))
        )

        weight_sum = pl.col("_weight").sum()
        agg_exprs = [
            ((pl.col(col).cast(pl.Float64) * pl.col("_weight")).sum() / weight_sum)
            .cast(pl.Float32)
            .alias(col)
            for col in emb_cols
        ]
        agg_exprs.extend(
            [
                weight_sum.cast(pl.Float64).alias("embedding_weight_sum"),
                pl.col("idarticu").n_unique().alias("embedded_unique_products"),
            ]
        )

        result = collect_streaming(joined.group_by("cliente").agg(agg_exprs).sort("cliente"))
        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output)
        write_artifact_metadata(output, cache_metadata)
        log_event("Stage 4 customer embeddings", "wrote artifact", cfg=cfg, customers=result.height, path=output)
    return output


def load_customer_embeddings(path: str | Path | None = None, cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    embedding_path = Path(path) if path else cfg.artifact_path(
        "customer_embeddings",
        "output",
        directory=cfg.outputs / "features",
    )
    if not embedding_path.exists():
        raise FileNotFoundError(f"Customer embeddings not found: {embedding_path}")
    return pl.read_parquet(embedding_path)
