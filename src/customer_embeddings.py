"""Customer-level product embedding aggregation."""

from __future__ import annotations

import math
import shutil
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


def _default_partition_count(cfg: PipelineConfig) -> int:
    configured = int(cfg.get("customer_embeddings.partition_count", 32))
    return max(1, configured)


def _write_customer_embedding_partitions(
    joined: pl.LazyFrame,
    emb_cols: list[str],
    normalize_vectors: bool,
    parts_dir: Path,
    partition_count: int,
    cfg: PipelineConfig,
) -> tuple[list[Path], int]:
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

    cliente_type = joined.collect_schema().get("cliente")
    numeric_cliente = cliente_type in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }

    part_paths: list[Path] = []
    total_customers = 0

    if numeric_cliente:
        bounds = collect_streaming(
            joined.select(
                [
                    pl.col("cliente").min().alias("cliente_min"),
                    pl.col("cliente").max().alias("cliente_max"),
                ]
            )
        )
        cliente_min = int(bounds[0, "cliente_min"])
        cliente_max = int(bounds[0, "cliente_max"])
        total_span = (cliente_max - cliente_min) + 1
        partition_count = max(1, min(partition_count, total_span))
        step = max(1, math.ceil(total_span / partition_count))

        for idx in range(partition_count):
            start = cliente_min + idx * step
            end = min(cliente_max + 1, start + step)
            filter_expr = (
                (pl.col("cliente") >= pl.lit(start)) & (pl.col("cliente") < pl.lit(end))
                if idx < partition_count - 1
                else (pl.col("cliente") >= pl.lit(start)) & (pl.col("cliente") <= pl.lit(cliente_max))
            )
            part_df = collect_streaming(joined.filter(filter_expr).group_by("cliente").agg(agg_exprs).sort("cliente"))
            if part_df.is_empty():
                log_event(
                    "Stage 4 customer embeddings",
                    "partition complete",
                    cfg=cfg,
                    partition=idx + 1,
                    partitions=partition_count,
                    customer_min=start,
                    customer_max=end - 1,
                    customers=0,
                )
                continue
            if normalize_vectors:
                part_df = _normalize_embedding_columns(part_df, emb_cols)
            part_path = parts_dir / f"part_{idx:04d}.parquet"
            part_df.write_parquet(part_path)
            total_customers += part_df.height
            part_paths.append(part_path)
            log_event(
                "Stage 4 customer embeddings",
                "partition complete",
                cfg=cfg,
                partition=idx + 1,
                partitions=partition_count,
                customer_min=start,
                customer_max=end - 1,
                customers=part_df.height,
                path=part_path,
            )
    else:
        for idx in range(partition_count):
            filter_expr = (pl.col("cliente").hash(seed=0) % pl.lit(partition_count)) == pl.lit(idx)
            part_df = collect_streaming(joined.filter(filter_expr).group_by("cliente").agg(agg_exprs).sort("cliente"))
            if part_df.is_empty():
                log_event(
                    "Stage 4 customer embeddings",
                    "partition complete",
                    cfg=cfg,
                    partition=idx + 1,
                    partitions=partition_count,
                    hash_bucket=idx,
                    customers=0,
                )
                continue
            if normalize_vectors:
                part_df = _normalize_embedding_columns(part_df, emb_cols)
            part_path = parts_dir / f"part_{idx:04d}.parquet"
            part_df.write_parquet(part_path)
            total_customers += part_df.height
            part_paths.append(part_path)
            log_event(
                "Stage 4 customer embeddings",
                "partition complete",
                cfg=cfg,
                partition=idx + 1,
                partitions=partition_count,
                hash_bucket=idx,
                customers=part_df.height,
                path=part_path,
            )

    return part_paths, total_customers


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
        log_event(
            "Stage 4 customer embeddings",
            "pre-aggregating transaction weights by customer-product",
            cfg=cfg,
        )
        customer_product_weights = (
            base_lf.with_columns(_base_weight_expression(columns, selected_weight_strategy))
            .group_by(["cliente", "idarticu"])
            .agg(pl.col("_weight").sum().cast(pl.Float64).alias("_weight"))
        )
        if _uses_idf(selected_weight_strategy):
            log_event(
                "Stage 4 customer embeddings",
                "applying product IDF weights to customer-product rows",
                cfg=cfg,
            )
            customer_product_weights = (
                customer_product_weights.join(_product_idf_weights(base_lf), on="idarticu", how="left")
                .with_columns((pl.col("_weight") * pl.col("_idf_weight")).alias("_weight"))
                .drop("_idf_weight")
            )

        log_event(
            "Stage 4 customer embeddings",
            "joining compressed customer-product weights to product vectors",
            cfg=cfg,
            vector_dims=len(emb_cols),
        )
        joined = customer_product_weights.join(embedding_lf, on="idarticu", how="inner")

        output.parent.mkdir(parents=True, exist_ok=True)
        parts_dir = output.parent / f".{output.stem}_parts"
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        parts_dir.mkdir(parents=True, exist_ok=True)

        partition_count = _default_partition_count(cfg)
        log_event(
            "Stage 4 customer embeddings",
            "collecting weighted customer vectors in partitions",
            cfg=cfg,
            partitions=partition_count,
        )
        part_paths, total_customers = _write_customer_embedding_partitions(
            joined=joined,
            emb_cols=emb_cols,
            normalize_vectors=normalize_vectors,
            parts_dir=parts_dir,
            partition_count=partition_count,
            cfg=cfg,
        )
        if not part_paths:
            raise ValueError("No customer embeddings were produced from weighted product vectors.")

        # Range partitions are globally ordered by cliente; hash partitions are sorted within each part.
        pl.scan_parquet(str(parts_dir / "part_*.parquet")).sink_parquet(str(output))
        shutil.rmtree(parts_dir, ignore_errors=True)
        write_artifact_metadata(output, cache_metadata)
        log_event("Stage 4 customer embeddings", "wrote artifact", cfg=cfg, customers=total_customers, path=output)
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
