"""Customer-level product embedding aggregation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.progress import log_event, stage_timer
from src.utils import collect_streaming, file_fingerprint, schema_names, should_use_cache, write_artifact_metadata


def _uses_idf(strategy: str) -> bool:
    return strategy.strip().lower() in {"idf", "equal_idf", "quantity_idf", "units_idf", "unidades_idf"}


def _quantity_weight(columns: set[str]) -> pl.Expr:
    if "unidades" not in columns:
        raise ValueError("Quantity-weighted customer embeddings require the 'unidades' column.")
    return (
        pl.when(pl.col("unidades").cast(pl.Float64) > 0)
        .then(pl.col("unidades").cast(pl.Float64))
        .otherwise(1.0)
    )


def _base_weight_expression(columns: set[str], strategy: str = "quantity") -> pl.Expr:
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy in {"equal", "uniform", "idf", "equal_idf"}:
        return pl.lit(1.0).alias("_weight")

    if normalized_strategy in {"quantity", "units", "unidades", "quantity_idf", "units_idf", "unidades_idf"}:
        return _quantity_weight(columns).alias("_weight")

    raise ValueError(
        "Unknown customer embedding weight_strategy "
        f"{strategy!r}. Use one of: quantity, equal, quantity_idf, equal_idf."
    )


def _weight_expression(columns: set[str], strategy: str = "quantity") -> pl.Expr:
    base = _base_weight_expression(columns, strategy)
    if _uses_idf(strategy):
        return (base * pl.col("_idf_weight")).alias("_weight")
    return base


def _product_idf_weights(base_lf: pl.LazyFrame) -> pl.LazyFrame:
    total_customers = int(collect_streaming(base_lf.select(pl.col("cliente").n_unique().alias("n_customers")))[0, 0])
    return (
        base_lf.select(["cliente", "idarticu"])
        .unique()
        .group_by("idarticu")
        .agg(pl.col("cliente").n_unique().alias("_product_customer_count"))
        .with_columns(
            (((pl.lit(total_customers + 1) / (pl.col("_product_customer_count") + 1)).log()) + 1.0).alias(
                "_idf_weight"
            )
        )
        .select(["idarticu", "_idf_weight"])
    )


def _normalize_embedding_columns(df: pl.DataFrame, emb_cols: list[str]) -> pl.DataFrame:
    norm_sq = pl.sum_horizontal([pl.col(col).cast(pl.Float64) ** 2 for col in emb_cols])
    with_norm = df.with_columns(norm_sq.sqrt().alias("_vector_norm"))
    return (
        with_norm.with_columns(
            [
                (
                    pl.col(col).cast(pl.Float64)
                    / pl.when(pl.col("_vector_norm") > 0).then(pl.col("_vector_norm")).otherwise(1.0)
                )
                .cast(pl.Float32)
                .alias(col)
                for col in emb_cols
            ]
        )
        .drop("_vector_norm")
    )


def build_customer_embeddings(
    embeddings_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    output_path: str | Path | None = None,
    weight_strategy: str | None = None,
    normalize_vectors: bool = False,
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
    selected_weight_strategy = str(weight_strategy or cfg.get("customer_embeddings.weight_strategy", "quantity"))
    cache_metadata = {
        "stage": "customer_embeddings",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "product_embeddings": file_fingerprint(embeddings_path),
        "weight_strategy": selected_weight_strategy,
        "idf_weighting": _uses_idf(selected_weight_strategy),
        "normalize_vectors": bool(normalize_vectors),
    }
    if should_use_cache(
        output,
        force=force,
        use_cached=cfg.get("cache.use_cached", True),
        metadata=cache_metadata,
    ):
        log_event("Stage 4 customer embeddings", "cache hit", cfg=cfg, path=output)
        return output

    with stage_timer(
        "Stage 4 customer embeddings",
        "aggregating product vectors",
        cfg=cfg,
        output=output,
        weight_strategy=selected_weight_strategy,
        normalize_vectors=bool(normalize_vectors),
    ):
        lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
        columns = set(schema_names(lf))
        embedding_lf = pl.scan_parquet(embeddings_path)
        emb_cols = [col for col in schema_names(embedding_lf) if col.startswith("emb_")]
        if not emb_cols:
            raise ValueError(f"No embedding columns found in {embeddings_path}")

        needed = ["cliente", "idarticu"] + (["unidades"] if "unidades" in columns else [])
        base_lf = lf.select(needed)
        joined = base_lf.join(embedding_lf, on="idarticu", how="inner")
        if _uses_idf(selected_weight_strategy):
            joined = joined.join(_product_idf_weights(base_lf), on="idarticu", how="left")
        joined = joined.with_columns(_weight_expression(columns, selected_weight_strategy))

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
        if normalize_vectors:
            result = _normalize_embedding_columns(result, emb_cols)
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
