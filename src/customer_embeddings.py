"""Customer-level product embedding aggregation."""

from __future__ import annotations

import math
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.progress import log_event, stage_timer
from src.utils import collect_streaming, file_fingerprint, schema_names, should_use_cache, write_artifact_metadata


def _uses_idf(strategy: str) -> bool:
    return strategy.strip().lower() in {"idf", "equal_idf", "quantity_idf", "units_idf", "unidades_idf"}


def _uses_quantity(strategy: str) -> bool:
    return strategy.strip().lower() in {
        "quantity",
        "units",
        "unidades",
        "quantity_idf",
        "units_idf",
        "unidades_idf",
    }


def _quantity_weight(columns: set[str], quantity_transform: str = "raw") -> pl.Expr:
    if "unidades" not in columns:
        raise ValueError("Quantity-weighted customer embeddings require the 'unidades' column.")
    positive_units = (
        pl.when(pl.col("unidades").cast(pl.Float64) > 0)
        .then(pl.col("unidades").cast(pl.Float64))
        .otherwise(1.0)
    )
    normalized_transform = quantity_transform.strip().lower()
    if normalized_transform in {"raw", "identity", "none"}:
        return positive_units
    if normalized_transform in {"log1p", "log"}:
        return (positive_units + 1.0).log()
    if normalized_transform in {"sqrt", "square_root"}:
        return positive_units.sqrt()
    raise ValueError(
        "Unknown customer embedding quantity_transform "
        f"{quantity_transform!r}. Use one of: raw, log1p, sqrt."
    )


def _base_weight_expression(
    columns: set[str],
    strategy: str = "quantity",
    quantity_transform: str = "raw",
) -> pl.Expr:
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy in {"equal", "uniform", "idf", "equal_idf"}:
        return pl.lit(1.0).alias("_weight")

    if _uses_quantity(normalized_strategy):
        return _quantity_weight(columns, quantity_transform).alias("_weight")

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


def _configured_quantity_transform(cfg: PipelineConfig) -> str:
    return str(cfg.get("customer_embeddings.quantity_transform", "raw")).strip().lower()


def _configured_max_customer_product_weight(cfg: PipelineConfig) -> float | None:
    return _optional_positive_float(cfg.get("customer_embeddings.max_customer_product_weight", None))


def _recency_weighting_config(cfg: PipelineConfig) -> dict[str, Any]:
    settings = cfg.get("customer_embeddings.recency_weighting", {}) or {}
    return {
        "enabled": bool(settings.get("enabled", False)),
        "date_column": str(settings.get("date_column", "fecha")),
        "reference_date": settings.get("reference_date"),
        "half_life_days": _optional_positive_float(settings.get("half_life_days", 180.0)) or 180.0,
        "min_multiplier": max(0.0, min(1.0, float(settings.get("min_multiplier", 0.25)))),
    }


def _frequency_weighting_config(cfg: PipelineConfig) -> dict[str, Any]:
    settings = cfg.get("customer_embeddings.frequency_weighting", {}) or {}
    return {
        "enabled": bool(settings.get("enabled", False)),
        "ticket_column": str(settings.get("ticket_column", "ticket")),
        "transform": str(settings.get("transform", "log1p")).strip().lower(),
        "exponent": _optional_positive_float(settings.get("exponent", 1.0)) or 1.0,
        "max_multiplier": _optional_positive_float(settings.get("max_multiplier", 3.0)),
    }


def _customer_embedding_required_columns(
    columns: set[str],
    *,
    weight_strategy: str,
    cfg: PipelineConfig,
) -> list[str]:
    needed = ["cliente", "idarticu"]
    if _uses_quantity(weight_strategy):
        if "unidades" not in columns:
            raise ValueError("Quantity-weighted customer embeddings require the 'unidades' column.")
        needed.append("unidades")
    elif "unidades" in columns:
        needed.append("unidades")

    recency_cfg = _recency_weighting_config(cfg)
    if recency_cfg["enabled"]:
        date_column = str(recency_cfg["date_column"])
        if date_column not in columns:
            raise ValueError(
                "customer_embeddings.recency_weighting.enabled=true requires "
                f"date column {date_column!r} in prepared transactions."
            )
        needed.append(date_column)

    frequency_cfg = _frequency_weighting_config(cfg)
    if frequency_cfg["enabled"]:
        ticket_column = str(frequency_cfg["ticket_column"])
        if ticket_column not in columns:
            raise ValueError(
                "customer_embeddings.frequency_weighting.enabled=true requires "
                f"ticket column {ticket_column!r} in prepared transactions."
            )
        needed.append(ticket_column)

    return list(dict.fromkeys(needed))


def _optional_positive_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    numeric = float(value)
    if numeric <= 0:
        return None
    return numeric


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _units_expression(columns: set[str]) -> pl.Expr:
    if "unidades" not in columns:
        return pl.lit(1.0).alias("_units")
    return (
        pl.when(pl.col("unidades").cast(pl.Float64) > 0)
        .then(pl.col("unidades").cast(pl.Float64))
        .otherwise(1.0)
        .alias("_units")
    )


def _coerce_date(value: Any, *, setting_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{setting_name} must be an ISO date string, got {value!r}.") from exc


def _resolve_recency_reference_date(
    base_lf: pl.LazyFrame,
    date_column: str,
    recency_cfg: dict[str, Any],
) -> date:
    configured = recency_cfg.get("reference_date")
    if configured:
        return _coerce_date(configured, setting_name="customer_embeddings.recency_weighting.reference_date")
    reference = collect_streaming(
        base_lf.select(pl.col(date_column).cast(pl.Date, strict=False).max().alias("_reference_date"))
    )[0, "_reference_date"]
    if reference is None:
        raise ValueError(
            "customer_embeddings.recency_weighting.enabled=true but no non-null "
            f"values were found in date column {date_column!r}."
        )
    return _coerce_date(reference, setting_name=f"max({date_column})")


def _recency_multiplier_expression(
    date_column: str,
    reference_date: date,
    *,
    half_life_days: float,
    min_multiplier: float,
) -> pl.Expr:
    transaction_date = pl.col(date_column).cast(pl.Date, strict=False)
    days_ago = (pl.lit(reference_date) - transaction_date).dt.total_days().cast(pl.Float64)
    decay = (-math.log(2.0) * days_ago / float(half_life_days)).exp()
    return (
        pl.when(days_ago.is_null())
        .then(pl.lit(float(min_multiplier)))
        .when(days_ago < 0)
        .then(pl.lit(1.0))
        .otherwise(pl.max_horizontal([decay, pl.lit(float(min_multiplier))]))
        .cast(pl.Float64)
        .alias("_recency_multiplier")
    )


def _frequency_multiplier_expression(
    basket_count_col: str,
    *,
    transform: str,
    exponent: float,
    max_multiplier: float | None,
) -> pl.Expr:
    basket_count = pl.col(basket_count_col).cast(pl.Float64)
    normalized_transform = transform.strip().lower()
    if normalized_transform in {"none", "off", "identity", "binary"}:
        multiplier = pl.lit(1.0)
    elif normalized_transform in {"log", "log1p"}:
        multiplier = (basket_count + 1.0).log() / math.log(2.0)
    elif normalized_transform in {"sqrt", "square_root"}:
        multiplier = basket_count.sqrt()
    elif normalized_transform in {"raw", "count", "basket_count"}:
        multiplier = basket_count
    else:
        raise ValueError(
            "Unknown customer_embeddings.frequency_weighting.transform "
            f"{transform!r}. Use one of: log1p, sqrt, raw, identity."
        )

    if exponent != 1.0:
        multiplier = multiplier ** float(exponent)
    if max_multiplier is not None:
        multiplier = pl.min_horizontal([multiplier, pl.lit(float(max_multiplier))])
    return multiplier.cast(pl.Float64).alias("_frequency_multiplier")


def _apply_customer_product_weight_cap(
    customer_product_weights: pl.LazyFrame,
    max_customer_product_weight: float | None,
) -> pl.LazyFrame:
    if max_customer_product_weight is None:
        return customer_product_weights
    return customer_product_weights.with_columns(
        pl.when(pl.col("_weight") > max_customer_product_weight)
        .then(pl.lit(max_customer_product_weight))
        .otherwise(pl.col("_weight"))
        .cast(pl.Float64)
        .alias("_weight")
    )


def _build_customer_product_weights(
    base_lf: pl.LazyFrame,
    columns: set[str],
    *,
    weight_strategy: str,
    quantity_transform: str,
    max_customer_product_weight: float | None,
    cfg: PipelineConfig,
) -> pl.LazyFrame:
    recency_cfg = _recency_weighting_config(cfg)
    frequency_cfg = _frequency_weighting_config(cfg)
    log_event(
        "Stage 4 customer embeddings",
        "pre-aggregating transaction weights by customer-product",
        cfg=cfg,
        quantity_transform=quantity_transform if _uses_quantity(weight_strategy) else "not_applicable",
        max_customer_product_weight=max_customer_product_weight,
        recency_weighting=recency_cfg["enabled"],
        frequency_weighting=frequency_cfg["enabled"],
    )

    weighted_lines = base_lf.with_columns(_base_weight_expression(columns, weight_strategy, quantity_transform))
    recency_reference_date: date | None = None
    if recency_cfg["enabled"]:
        date_column = str(recency_cfg["date_column"])
        recency_reference_date = _resolve_recency_reference_date(base_lf, date_column, recency_cfg)
        weighted_lines = weighted_lines.with_columns(
            _recency_multiplier_expression(
                date_column,
                recency_reference_date,
                half_life_days=float(recency_cfg["half_life_days"]),
                min_multiplier=float(recency_cfg["min_multiplier"]),
            )
        ).with_columns((pl.col("_weight") * pl.col("_recency_multiplier")).alias("_weight"))
    else:
        weighted_lines = weighted_lines.with_columns(pl.lit(1.0).cast(pl.Float64).alias("_recency_multiplier"))

    aggregate_exprs = [
        pl.col("_weight").sum().cast(pl.Float64).alias("_weight"),
        pl.col("_recency_multiplier").mean().cast(pl.Float64).alias("_mean_recency_multiplier"),
    ]
    if frequency_cfg["enabled"]:
        aggregate_exprs.append(
            pl.col(str(frequency_cfg["ticket_column"])).n_unique().cast(pl.UInt32).alias("_basket_count")
        )

    customer_product_weights = weighted_lines.group_by(["cliente", "idarticu"]).agg(aggregate_exprs)
    if frequency_cfg["enabled"]:
        customer_product_weights = customer_product_weights.with_columns(
            _frequency_multiplier_expression(
                "_basket_count",
                transform=str(frequency_cfg["transform"]),
                exponent=float(frequency_cfg["exponent"]),
                max_multiplier=frequency_cfg["max_multiplier"],
            )
        ).with_columns((pl.col("_weight") * pl.col("_frequency_multiplier")).alias("_weight"))
    else:
        customer_product_weights = customer_product_weights.with_columns(
            [
                pl.lit(None, dtype=pl.UInt32).alias("_basket_count"),
                pl.lit(1.0).cast(pl.Float64).alias("_frequency_multiplier"),
            ]
        )

    customer_product_weights = customer_product_weights.with_columns(
        [
            pl.lit(bool(recency_cfg["enabled"])).alias("recency_weighting_enabled"),
            pl.lit(
                recency_reference_date.isoformat() if recency_reference_date is not None else None,
                dtype=pl.Utf8,
            ).alias("recency_reference_date"),
            pl.lit(float(recency_cfg["half_life_days"]) if recency_cfg["enabled"] else None, dtype=pl.Float64).alias(
                "recency_half_life_days"
            ),
            pl.lit(float(recency_cfg["min_multiplier"]) if recency_cfg["enabled"] else None, dtype=pl.Float64).alias(
                "recency_min_multiplier"
            ),
            pl.lit(bool(frequency_cfg["enabled"])).alias("frequency_weighting_enabled"),
            pl.lit(str(frequency_cfg["transform"]) if frequency_cfg["enabled"] else None, dtype=pl.Utf8).alias(
                "frequency_transform"
            ),
            pl.lit(float(frequency_cfg["exponent"]) if frequency_cfg["enabled"] else None, dtype=pl.Float64).alias(
                "frequency_exponent"
            ),
            pl.lit(frequency_cfg["max_multiplier"] if frequency_cfg["enabled"] else None, dtype=pl.Float64).alias(
                "frequency_max_multiplier"
            ),
        ]
    )
    if _uses_idf(weight_strategy):
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
    return _apply_customer_product_weight_cap(customer_product_weights, max_customer_product_weight)


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


def _common_product_diagnostics_path(cfg: PipelineConfig) -> Path:
    return cfg.artifacts / str(cfg.get("baskets.diagnostics.output_dir", "stage1")) / str(
        cfg.get("baskets.diagnostics.product_ubiquity_output", "product_ubiquity_diagnostics.parquet")
    )


def _customer_embedding_diagnostics_paths(
    cfg: PipelineConfig,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
) -> dict[str, Path]:
    diagnostics_cfg = cfg.get("customer_embeddings.diagnostics", {}) or {}
    directory = Path(output_dir) if output_dir else cfg.artifacts / str(diagnostics_cfg.get("output_dir", "stage4"))
    prefix = output_prefix or str(diagnostics_cfg.get("output_prefix", "customer_embedding"))
    paths = {"summary_csv": directory / f"{prefix}_weight_diagnostics.csv"}
    if bool(diagnostics_cfg.get("write_top_products_csv", False)):
        paths["top_products_csv"] = directory / f"{prefix}_top_weighted_products.csv"
    return paths


def _product_label_frame(transactions: pl.LazyFrame) -> pl.LazyFrame | None:
    columns = set(schema_names(transactions))
    label_cols = [col for col in ["desc_larga_articulo", "desc_sector", "idsector"] if col in columns]
    if not label_cols:
        return None
    return transactions.select(["idarticu", *label_cols]).group_by("idarticu").agg(
        [pl.col(col).drop_nulls().first().alias(col) for col in label_cols]
    )


def _common_product_flags(cfg: PipelineConfig) -> pl.LazyFrame | None:
    path = _common_product_diagnostics_path(cfg)
    if not path.exists():
        return None
    diagnostics = pl.scan_parquet(path)
    columns = set(schema_names(diagnostics))
    if "idarticu" not in columns or "common_product_candidate" not in columns:
        return None
    return (
        diagnostics.select(
            [
                pl.col("idarticu").cast(pl.Utf8).alias("_idarticu_key"),
                pl.col("common_product_candidate").fill_null(False).cast(pl.Boolean),
            ]
        )
        .unique(subset=["_idarticu_key"])
    )


def _safe_pct(value: float | None) -> float | None:
    return None if value is None else float(value * 100.0)


def _customer_embedding_coverage_metrics(
    base_lf: pl.LazyFrame,
    embeddings_path: str | Path | None,
    columns: set[str],
) -> dict[str, Any]:
    if embeddings_path is None:
        return {
            "total_transaction_lines": None,
            "embedded_transaction_lines": None,
            "line_coverage_pct": None,
            "total_transaction_units": None,
            "embedded_transaction_units": None,
            "unit_coverage_pct": None,
            "total_transaction_products": None,
            "embedded_transaction_products": None,
            "product_coverage_pct": None,
            "total_transaction_customers": None,
            "embedded_transaction_customers": None,
            "customer_coverage_pct": None,
            "zero_embedded_customers": None,
            "zero_embedded_customer_pct": None,
            "mean_embedded_unique_products_per_customer": None,
            "p05_embedded_unique_products_per_customer": None,
            "median_embedded_unique_products_per_customer": None,
            "p95_embedded_unique_products_per_customer": None,
        }

    embedded_products = (
        pl.scan_parquet(embeddings_path)
        .select(
            [
                pl.col("idarticu").cast(pl.Utf8).alias("_idarticu_key"),
                pl.lit(True).alias("_has_product_embedding"),
            ]
        )
        .unique(subset=["_idarticu_key"])
    )
    base = base_lf.with_columns(
        [
            pl.col("idarticu").cast(pl.Utf8).alias("_idarticu_key"),
            _units_expression(columns),
        ]
    )
    matched = (
        base.join(embedded_products, on="_idarticu_key", how="left")
        .with_columns(pl.col("_has_product_embedding").fill_null(False).cast(pl.Boolean))
    )
    totals = collect_streaming(
        matched.select(
            [
                pl.len().alias("total_transaction_lines"),
                pl.when(pl.col("_has_product_embedding"))
                .then(1)
                .otherwise(0)
                .sum()
                .alias("embedded_transaction_lines"),
                pl.col("_units").sum().alias("total_transaction_units"),
                pl.when(pl.col("_has_product_embedding"))
                .then(pl.col("_units"))
                .otherwise(0.0)
                .sum()
                .alias("embedded_transaction_units"),
                pl.col("idarticu").n_unique().alias("total_transaction_products"),
                pl.col("idarticu")
                .filter(pl.col("_has_product_embedding"))
                .n_unique()
                .alias("embedded_transaction_products"),
                pl.col("cliente").n_unique().alias("total_transaction_customers"),
                pl.col("cliente")
                .filter(pl.col("_has_product_embedding"))
                .n_unique()
                .alias("embedded_transaction_customers"),
            ]
        )
    ).row(0, named=True)

    customer_embedding_counts = matched.group_by("cliente").agg(
        pl.col("idarticu")
        .filter(pl.col("_has_product_embedding"))
        .n_unique()
        .alias("_embedded_unique_products")
    )
    customer_summary = collect_streaming(
        customer_embedding_counts.select(
            [
                (pl.col("_embedded_unique_products") == 0).sum().alias("zero_embedded_customers"),
                pl.col("_embedded_unique_products").mean().alias("mean_embedded_unique_products_per_customer"),
                pl.col("_embedded_unique_products").quantile(0.05).alias("p05_embedded_unique_products_per_customer"),
                pl.col("_embedded_unique_products").median().alias("median_embedded_unique_products_per_customer"),
                pl.col("_embedded_unique_products").quantile(0.95).alias("p95_embedded_unique_products_per_customer"),
            ]
        )
    ).row(0, named=True)

    total_lines = int(totals["total_transaction_lines"] or 0)
    total_units = float(totals["total_transaction_units"] or 0.0)
    total_products = int(totals["total_transaction_products"] or 0)
    total_customers = int(totals["total_transaction_customers"] or 0)
    zero_customers = int(customer_summary["zero_embedded_customers"] or 0)
    return {
        **totals,
        **customer_summary,
        "line_coverage_pct": _safe_pct(
            float(totals["embedded_transaction_lines"] or 0) / total_lines if total_lines else None
        ),
        "unit_coverage_pct": _safe_pct(
            float(totals["embedded_transaction_units"] or 0.0) / total_units if total_units else None
        ),
        "product_coverage_pct": _safe_pct(
            float(totals["embedded_transaction_products"] or 0) / total_products if total_products else None
        ),
        "customer_coverage_pct": _safe_pct(
            float(totals["embedded_transaction_customers"] or 0) / total_customers if total_customers else None
        ),
        "zero_embedded_customer_pct": _safe_pct(zero_customers / total_customers if total_customers else None),
    }


def _customer_embedding_weight_metrics(
    customer_product_weights: pl.LazyFrame,
    *,
    weight_strategy: str,
    quantity_transform: str,
    max_customer_product_weight: float | None,
    normalize_vectors: bool,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    weight_columns = set(schema_names(customer_product_weights))
    total_exprs = [
        pl.len().alias("customer_product_rows"),
        pl.col("cliente").n_unique().alias("weighted_customers"),
        pl.col("idarticu").n_unique().alias("weighted_products"),
        pl.col("_weight").sum().alias("total_weight"),
    ]
    if "recency_weighting_enabled" in weight_columns:
        total_exprs.extend(
            [
                pl.col("recency_weighting_enabled").first().alias("recency_weighting_enabled"),
                pl.col("recency_reference_date").first().alias("recency_reference_date"),
                pl.col("recency_half_life_days").first().alias("recency_half_life_days"),
                pl.col("recency_min_multiplier").first().alias("recency_min_multiplier"),
                pl.col("_mean_recency_multiplier").mean().alias("mean_recency_multiplier"),
            ]
        )
    if "frequency_weighting_enabled" in weight_columns:
        total_exprs.extend(
            [
                pl.col("frequency_weighting_enabled").first().alias("frequency_weighting_enabled"),
                pl.col("frequency_transform").first().alias("frequency_transform"),
                pl.col("frequency_exponent").first().alias("frequency_exponent"),
                pl.col("frequency_max_multiplier").first().alias("frequency_max_multiplier"),
                pl.col("_frequency_multiplier").mean().alias("mean_frequency_multiplier"),
                pl.col("_frequency_multiplier").max().alias("max_frequency_multiplier_observed"),
            ]
        )
    if "_basket_count" in weight_columns:
        total_exprs.extend(
            [
                pl.col("_basket_count").mean().alias("mean_customer_product_basket_count"),
                pl.col("_basket_count").quantile(0.95).alias("p95_customer_product_basket_count"),
                pl.col("_basket_count").max().alias("max_customer_product_basket_count"),
            ]
        )

    totals = collect_streaming(
        customer_product_weights.select(total_exprs)
    ).row(0, named=True)
    total_weight = float(totals["total_weight"] or 0.0)

    if total_weight <= 0:
        return {
            **totals,
            "weight_strategy": weight_strategy,
            "quantity_transform": quantity_transform,
            "max_customer_product_weight": max_customer_product_weight,
            "normalize_vectors": normalize_vectors,
            "top_product_weight_share_pct": None,
            "top_10_product_weight_share_pct": None,
            "product_weight_hhi": None,
            "effective_products_by_weight": None,
            "mean_customer_top_product_weight_share_pct": None,
            "p95_customer_top_product_weight_share_pct": None,
            "common_product_weight_share_pct": None,
            "mean_customer_common_product_weight_share_pct": None,
            "p95_customer_common_product_weight_share_pct": None,
        }

    product_weights = customer_product_weights.group_by("idarticu").agg(
        [
            pl.col("_weight").sum().alias("total_weight"),
            pl.col("cliente").n_unique().alias("weighted_customers"),
        ]
    )
    product_shares = product_weights.with_columns((pl.col("total_weight") / pl.lit(total_weight)).alias("weight_share"))
    product_summary = collect_streaming(
        product_shares.select(
            [
                pl.col("weight_share").max().alias("top_product_weight_share"),
                (pl.col("weight_share") ** 2).sum().alias("product_weight_hhi"),
            ]
        )
    ).row(0, named=True)
    top_10_share = collect_streaming(product_shares.sort("weight_share", descending=True).head(10).select(pl.col("weight_share").sum()))[
        0,
        0,
    ]

    customer_totals = customer_product_weights.group_by("cliente").agg(pl.col("_weight").sum().alias("_customer_weight"))
    customer_top = (
        customer_product_weights.join(customer_totals, on="cliente", how="inner")
        .with_columns((pl.col("_weight") / pl.col("_customer_weight")).alias("_customer_product_weight_share"))
        .group_by("cliente")
        .agg(pl.col("_customer_product_weight_share").max().alias("_customer_top_product_weight_share"))
    )
    customer_top_summary = collect_streaming(
        customer_top.select(
            [
                pl.col("_customer_top_product_weight_share").mean().alias("mean_customer_top_product_weight_share"),
                pl.col("_customer_top_product_weight_share")
                .quantile(0.95)
                .alias("p95_customer_top_product_weight_share"),
            ]
        )
    ).row(0, named=True)

    common_weight_share = None
    mean_customer_common_share = None
    p95_customer_common_share = None
    common_flags = _common_product_flags(cfg)
    if common_flags is not None:
        flagged = (
            customer_product_weights.with_columns(pl.col("idarticu").cast(pl.Utf8).alias("_idarticu_key"))
            .join(common_flags, on="_idarticu_key", how="left")
            .with_columns(pl.col("common_product_candidate").fill_null(False))
        )
        common_summary = collect_streaming(
            flagged.select(
                pl.when(pl.col("common_product_candidate"))
                .then(pl.col("_weight"))
                .otherwise(0.0)
                .sum()
                .alias("common_weight")
            )
        )
        common_weight_share = float(common_summary[0, "common_weight"] or 0.0) / total_weight
        customer_common = (
            flagged.group_by("cliente")
            .agg(
                [
                    pl.col("_weight").sum().alias("_customer_weight"),
                    pl.when(pl.col("common_product_candidate"))
                    .then(pl.col("_weight"))
                    .otherwise(0.0)
                    .sum()
                    .alias("_customer_common_weight"),
                ]
            )
            .with_columns(
                (pl.col("_customer_common_weight") / pl.col("_customer_weight")).alias("_customer_common_weight_share")
            )
        )
        common_customer_summary = collect_streaming(
            customer_common.select(
                [
                    pl.col("_customer_common_weight_share").mean().alias("mean_customer_common_weight_share"),
                    pl.col("_customer_common_weight_share").quantile(0.95).alias("p95_customer_common_weight_share"),
                ]
            )
        ).row(0, named=True)
        mean_customer_common_share = common_customer_summary["mean_customer_common_weight_share"]
        p95_customer_common_share = common_customer_summary["p95_customer_common_weight_share"]

    product_weight_hhi = product_summary["product_weight_hhi"]
    effective_products_by_weight = (
        float(1.0 / product_weight_hhi) if product_weight_hhi is not None and product_weight_hhi > 0 else None
    )
    return {
        **totals,
        "weight_strategy": weight_strategy,
        "quantity_transform": quantity_transform,
        "max_customer_product_weight": max_customer_product_weight,
        "normalize_vectors": normalize_vectors,
        "top_product_weight_share_pct": _safe_pct(product_summary["top_product_weight_share"]),
        "top_10_product_weight_share_pct": _safe_pct(float(top_10_share or 0.0)),
        "product_weight_hhi": float(product_weight_hhi) if product_weight_hhi is not None else None,
        "effective_products_by_weight": effective_products_by_weight,
        "mean_customer_top_product_weight_share_pct": _safe_pct(
            customer_top_summary["mean_customer_top_product_weight_share"]
        ),
        "p95_customer_top_product_weight_share_pct": _safe_pct(
            customer_top_summary["p95_customer_top_product_weight_share"]
        ),
        "common_product_weight_share_pct": _safe_pct(common_weight_share),
        "mean_customer_common_product_weight_share_pct": _safe_pct(mean_customer_common_share),
        "p95_customer_common_product_weight_share_pct": _safe_pct(p95_customer_common_share),
    }


def customer_embedding_gate_status(metrics: dict[str, Any], cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    """Evaluate Stage 4 coverage and dominance gates from diagnostic metrics."""

    gates = cfg.get("customer_embeddings.gates", {}) or {}
    if not bool(gates.get("enabled", False)):
        return {
            "passes_stage4_gates": True,
            "stage4_gate_status": "disabled",
            "stage4_gate_issues": "Stage 4 gates are disabled.",
        }

    checks = [
        ("line_coverage_pct", "min_line_coverage_pct", ">="),
        ("unit_coverage_pct", "min_unit_coverage_pct", ">="),
        ("product_coverage_pct", "min_product_coverage_pct", ">="),
        ("customer_coverage_pct", "min_customer_coverage_pct", ">="),
        ("zero_embedded_customer_pct", "max_zero_embedded_customer_pct", "<="),
        ("top_product_weight_share_pct", "max_top_product_weight_share_pct", "<="),
        ("top_10_product_weight_share_pct", "max_top_10_product_weight_share_pct", "<="),
        ("mean_customer_top_product_weight_share_pct", "max_mean_customer_top_product_weight_share_pct", "<="),
        ("p95_customer_top_product_weight_share_pct", "max_p95_customer_top_product_weight_share_pct", "<="),
        ("common_product_weight_share_pct", "max_common_product_weight_share_pct", "<="),
        (
            "mean_customer_common_product_weight_share_pct",
            "max_mean_customer_common_product_weight_share_pct",
            "<=",
        ),
        ("p95_customer_common_product_weight_share_pct", "max_p95_customer_common_product_weight_share_pct", "<="),
    ]

    issues: list[str] = []
    for metric_name, threshold_name, operator in checks:
        threshold = _optional_float(gates.get(threshold_name))
        if threshold is None:
            continue
        raw_value = metrics.get(metric_name)
        if raw_value is None:
            issues.append(f"{metric_name} unavailable for configured gate {threshold_name}")
            continue
        value = float(raw_value)
        if operator == ">=" and value < threshold:
            issues.append(f"{metric_name}={value:.2f} below {threshold_name}={threshold:.2f}")
        elif operator == "<=" and value > threshold:
            issues.append(f"{metric_name}={value:.2f} above {threshold_name}={threshold:.2f}")

    passes = not issues
    return {
        "passes_stage4_gates": passes,
        "stage4_gate_status": "pass" if passes else "fail",
        "stage4_gate_issues": "None" if passes else " | ".join(issues),
    }


def _enforce_customer_embedding_gates(metrics: dict[str, Any], cfg: PipelineConfig) -> None:
    gates = cfg.get("customer_embeddings.gates", {}) or {}
    if (
        bool(gates.get("enabled", False))
        and bool(gates.get("fail_on_violation", False))
        and not bool(metrics.get("passes_stage4_gates", True))
    ):
        raise ValueError(f"Stage 4 customer embedding gates failed: {metrics.get('stage4_gate_issues')}")


def _write_customer_embedding_weight_diagnostics(
    customer_product_weights: pl.LazyFrame,
    transactions: pl.LazyFrame,
    *,
    weight_strategy: str,
    quantity_transform: str,
    max_customer_product_weight: float | None,
    normalize_vectors: bool,
    embeddings_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    paths = _customer_embedding_diagnostics_paths(cfg, output_dir=output_dir, output_prefix=output_prefix)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    transaction_columns = set(schema_names(transactions))
    needed = _customer_embedding_required_columns(transaction_columns, weight_strategy=weight_strategy, cfg=cfg)
    base_lf = transactions.select(needed)
    metrics = _customer_embedding_weight_metrics(
        customer_product_weights,
        weight_strategy=weight_strategy,
        quantity_transform=quantity_transform,
        max_customer_product_weight=max_customer_product_weight,
        normalize_vectors=normalize_vectors,
        cfg=cfg,
    )
    metrics.update(_customer_embedding_coverage_metrics(base_lf, embeddings_path, transaction_columns))
    metrics.update(customer_embedding_gate_status(metrics, cfg=cfg))
    pl.DataFrame([metrics]).write_csv(paths["summary_csv"])

    diagnostics_cfg = cfg.get("customer_embeddings.diagnostics", {}) or {}
    if "top_products_csv" not in paths:
        return {**paths, "metrics": metrics}

    top_n = int(diagnostics_cfg.get("top_n_products", 15))
    total_weight = float(metrics.get("total_weight") or 0.0)
    product_weights = (
        customer_product_weights.group_by("idarticu")
        .agg(
            [
                pl.col("_weight").sum().alias("total_weight"),
                pl.col("cliente").n_unique().alias("weighted_customers"),
            ]
        )
        .with_columns((pl.col("total_weight") / pl.lit(max(total_weight, 1e-12)) * 100.0).alias("weight_share_pct"))
    )
    common_flags = _common_product_flags(cfg)
    if common_flags is not None:
        product_weights = (
            product_weights.with_columns(pl.col("idarticu").cast(pl.Utf8).alias("_idarticu_key"))
            .join(common_flags, on="_idarticu_key", how="left")
            .drop("_idarticu_key")
            .with_columns(pl.col("common_product_candidate").fill_null(False))
        )
    label_frame = _product_label_frame(transactions)
    if label_frame is not None:
        product_weights = product_weights.join(label_frame, on="idarticu", how="left")
    top_products = collect_streaming(product_weights.sort("total_weight", descending=True).head(top_n))
    top_products.write_csv(paths["top_products_csv"])
    return {**paths, "metrics": metrics}


def evaluate_customer_embedding_weight_profile(
    transactions: pl.LazyFrame | None = None,
    embeddings_path: str | Path | None = None,
    weight_strategy: str | None = None,
    quantity_transform: str | None = None,
    max_customer_product_weight: float | None = None,
    normalize_vectors: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Compute Stage 4 weight concentration metrics without writing artifacts."""

    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    columns = set(schema_names(lf))
    selected_weight_strategy = str(weight_strategy or cfg.get("customer_embeddings.weight_strategy", "quantity"))
    selected_quantity_transform = str(quantity_transform or _configured_quantity_transform(cfg))
    needed = _customer_embedding_required_columns(columns, weight_strategy=selected_weight_strategy, cfg=cfg)
    base_lf = lf.select(needed)
    selected_max_customer_product_weight = (
        _configured_max_customer_product_weight(cfg)
        if max_customer_product_weight is None
        else _optional_positive_float(max_customer_product_weight)
    )
    selected_normalize_vectors = (
        bool(cfg.get("customer_embeddings.normalize_vectors", False))
        if normalize_vectors is None
        else bool(normalize_vectors)
    )
    customer_product_weights = _build_customer_product_weights(
        base_lf,
        columns,
        weight_strategy=selected_weight_strategy,
        quantity_transform=selected_quantity_transform,
        max_customer_product_weight=selected_max_customer_product_weight,
        cfg=cfg,
    )
    metrics = _customer_embedding_weight_metrics(
        customer_product_weights,
        weight_strategy=selected_weight_strategy,
        quantity_transform=selected_quantity_transform,
        max_customer_product_weight=selected_max_customer_product_weight,
        normalize_vectors=selected_normalize_vectors,
        cfg=cfg,
    )
    metrics.update(_customer_embedding_coverage_metrics(base_lf, embeddings_path, columns))
    metrics.update(customer_embedding_gate_status(metrics, cfg=cfg))
    return metrics


def build_customer_embedding_weight_diagnostics(
    transactions: pl.LazyFrame | None = None,
    embeddings_path: str | Path | None = None,
    weight_strategy: str | None = None,
    quantity_transform: str | None = None,
    max_customer_product_weight: float | None = None,
    normalize_vectors: bool | None = None,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Write Stage 4 coverage and weight-dominance diagnostics without rebuilding vectors."""

    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    columns = set(schema_names(lf))
    selected_weight_strategy = str(weight_strategy or cfg.get("customer_embeddings.weight_strategy", "quantity"))
    selected_quantity_transform = str(quantity_transform or _configured_quantity_transform(cfg))
    needed = _customer_embedding_required_columns(columns, weight_strategy=selected_weight_strategy, cfg=cfg)
    base_lf = lf.select(needed)
    selected_max_customer_product_weight = (
        _configured_max_customer_product_weight(cfg)
        if max_customer_product_weight is None
        else _optional_positive_float(max_customer_product_weight)
    )
    selected_normalize_vectors = (
        bool(cfg.get("customer_embeddings.normalize_vectors", False))
        if normalize_vectors is None
        else bool(normalize_vectors)
    )
    customer_product_weights = _build_customer_product_weights(
        base_lf,
        columns,
        weight_strategy=selected_weight_strategy,
        quantity_transform=selected_quantity_transform,
        max_customer_product_weight=selected_max_customer_product_weight,
        cfg=cfg,
    )
    return _write_customer_embedding_weight_diagnostics(
        customer_product_weights,
        lf,
        weight_strategy=selected_weight_strategy,
        quantity_transform=selected_quantity_transform,
        max_customer_product_weight=selected_max_customer_product_weight,
        normalize_vectors=selected_normalize_vectors,
        embeddings_path=embeddings_path,
        cfg=cfg,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )


def build_customer_embedding_variant_diagnostics(
    embeddings_path: str | Path | None = None,
    transactions: pl.LazyFrame | None = None,
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path | None:
    """Compare Stage 4 weighting variants without promoting them to the official pipeline."""

    cfg.ensure_directories()
    diagnostics_cfg = cfg.get("customer_embeddings.variant_diagnostics", {}) or {}
    if not bool(diagnostics_cfg.get("enabled", False)):
        return None
    trials = [dict(trial) for trial in diagnostics_cfg.get("trials", []) or []]
    if not trials:
        raise ValueError("customer_embeddings.variant_diagnostics.enabled=true but no trials are configured.")

    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.artifacts / str(
        diagnostics_cfg.get("output_dir", "stage4")
    ) / str(diagnostics_cfg.get("output_csv", "customer_embedding_variant_diagnostics.csv"))
    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    cache_metadata = {
        "stage": "customer_embedding_variant_diagnostics",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "product_embeddings": file_fingerprint(embeddings_path) if embeddings_path else None,
        "trials": trials,
        "recency_weighting": _recency_weighting_config(cfg),
        "frequency_weighting": _frequency_weighting_config(cfg),
        "gates": cfg.get("customer_embeddings.gates", {}),
    }
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata):
        log_event("Stage 4 customer embeddings", "variant diagnostics cache hit", cfg=cfg, path=output)
        return output

    official_strategy = str(cfg.get("customer_embeddings.weight_strategy", "quantity"))
    official_quantity_transform = _configured_quantity_transform(cfg)
    official_cap = _configured_max_customer_product_weight(cfg)
    official_normalize = bool(cfg.get("customer_embeddings.normalize_vectors", False))
    common_share_gate = _optional_float(
        (cfg.get("customer_embeddings.gates", {}) or {}).get("max_common_product_weight_share_pct")
    )

    rows = []
    for trial in trials:
        trial_name = str(trial.get("name", "trial"))
        weight_strategy = str(trial.get("weight_strategy", official_strategy))
        quantity_transform = str(trial.get("quantity_transform", official_quantity_transform))
        max_customer_product_weight = (
            official_cap
            if "max_customer_product_weight" not in trial
            else _optional_positive_float(trial.get("max_customer_product_weight"))
        )
        normalize_vectors = bool(trial.get("normalize_vectors", official_normalize))
        metrics = evaluate_customer_embedding_weight_profile(
            transactions=lf,
            embeddings_path=embeddings_path,
            weight_strategy=weight_strategy,
            quantity_transform=quantity_transform,
            max_customer_product_weight=max_customer_product_weight,
            normalize_vectors=normalize_vectors,
            cfg=cfg,
        )
        common_share = metrics.get("common_product_weight_share_pct")
        rows.append(
            {
                "trial_name": trial_name,
                "is_official_stage4_recipe": bool(
                    weight_strategy == official_strategy
                    and quantity_transform == official_quantity_transform
                    and max_customer_product_weight == official_cap
                    and normalize_vectors == official_normalize
                ),
                "weight_strategy": weight_strategy,
                "quantity_transform": quantity_transform,
                "max_customer_product_weight": max_customer_product_weight,
                "normalize_vectors": normalize_vectors,
                "common_product_gate_margin_pct": (
                    common_share_gate - float(common_share)
                    if common_share_gate is not None and common_share is not None
                    else None
                ),
                **metrics,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(output)
    write_artifact_metadata(output, cache_metadata)
    log_event("Stage 4 customer embeddings", "wrote variant diagnostics", cfg=cfg, rows=len(rows), path=output)
    return output


def build_customer_embeddings(
    embeddings_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    output_path: str | Path | None = None,
    weight_strategy: str | None = None,
    quantity_transform: str | None = None,
    max_customer_product_weight: float | None = None,
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
    selected_quantity_transform = str(quantity_transform or _configured_quantity_transform(cfg))
    selected_max_customer_product_weight = (
        _configured_max_customer_product_weight(cfg)
        if max_customer_product_weight is None
        else _optional_positive_float(max_customer_product_weight)
    )
    cache_metadata = {
        "stage": "customer_embeddings",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "product_embeddings": file_fingerprint(embeddings_path),
        "weight_strategy": selected_weight_strategy,
        "idf_weighting": _uses_idf(selected_weight_strategy),
        "quantity_transform": selected_quantity_transform if _uses_quantity(selected_weight_strategy) else None,
        "max_customer_product_weight": selected_max_customer_product_weight,
        "recency_weighting": _recency_weighting_config(cfg),
        "frequency_weighting": _frequency_weighting_config(cfg),
        "normalize_vectors": bool(normalize_vectors),
    }
    if should_use_cache(
        output,
        force=force,
        use_cached=cfg.get("cache.use_cached", True),
        metadata=cache_metadata,
    ):
        diagnostics_cfg = cfg.get("customer_embeddings.diagnostics", {}) or {}
        gates_cfg = cfg.get("customer_embeddings.gates", {}) or {}
        gates_apply = output_path is None or bool(gates_cfg.get("apply_to_custom_outputs", False))
        if (output_path is None and bool(diagnostics_cfg.get("enabled", True))) or (
            gates_apply and bool(gates_cfg.get("enabled", False))
        ):
            lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
            columns = set(schema_names(lf))
            needed = _customer_embedding_required_columns(columns, weight_strategy=selected_weight_strategy, cfg=cfg)
            base_lf = lf.select(needed)
            diagnostic_result = _write_customer_embedding_weight_diagnostics(
                _build_customer_product_weights(
                    base_lf,
                    columns,
                    weight_strategy=selected_weight_strategy,
                    quantity_transform=selected_quantity_transform,
                    max_customer_product_weight=selected_max_customer_product_weight,
                    cfg=cfg,
                ),
                lf,
                weight_strategy=selected_weight_strategy,
                quantity_transform=selected_quantity_transform,
                max_customer_product_weight=selected_max_customer_product_weight,
                normalize_vectors=bool(normalize_vectors),
                embeddings_path=embeddings_path,
                cfg=cfg,
            )
            if gates_apply:
                _enforce_customer_embedding_gates(diagnostic_result["metrics"], cfg)
        log_event("Stage 4 customer embeddings", "cache hit", cfg=cfg, path=output)
        return output

    with stage_timer(
        "Stage 4 customer embeddings",
        "aggregating product vectors",
        cfg=cfg,
        output=output,
        weight_strategy=selected_weight_strategy,
        quantity_transform=selected_quantity_transform if _uses_quantity(selected_weight_strategy) else "not_applicable",
        max_customer_product_weight=selected_max_customer_product_weight,
        recency_weighting=_recency_weighting_config(cfg)["enabled"],
        frequency_weighting=_frequency_weighting_config(cfg)["enabled"],
        normalize_vectors=bool(normalize_vectors),
    ):
        lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
        columns = set(schema_names(lf))
        embedding_lf = pl.scan_parquet(embeddings_path)
        emb_cols = [col for col in schema_names(embedding_lf) if col.startswith("emb_")]
        if not emb_cols:
            raise ValueError(f"No embedding columns found in {embeddings_path}")

        needed = _customer_embedding_required_columns(columns, weight_strategy=selected_weight_strategy, cfg=cfg)
        base_lf = lf.select(needed)
        customer_product_weights = _build_customer_product_weights(
            base_lf,
            columns,
            weight_strategy=selected_weight_strategy,
            quantity_transform=selected_quantity_transform,
            max_customer_product_weight=selected_max_customer_product_weight,
            cfg=cfg,
        )

        diagnostics_cfg = cfg.get("customer_embeddings.diagnostics", {}) or {}
        gates_cfg = cfg.get("customer_embeddings.gates", {}) or {}
        gates_apply = output_path is None or bool(gates_cfg.get("apply_to_custom_outputs", False))
        if output_path is None and bool(diagnostics_cfg.get("enabled", True)):
            diagnostic_paths = _write_customer_embedding_weight_diagnostics(
                customer_product_weights,
                lf,
                weight_strategy=selected_weight_strategy,
                quantity_transform=selected_quantity_transform,
                max_customer_product_weight=selected_max_customer_product_weight,
                normalize_vectors=bool(normalize_vectors),
                embeddings_path=embeddings_path,
                cfg=cfg,
            )
            _enforce_customer_embedding_gates(diagnostic_paths["metrics"], cfg)
            log_event(
                "Stage 4 customer embeddings",
                "wrote coverage and dominance diagnostics",
                cfg=cfg,
                summary=diagnostic_paths["summary_csv"],
                top_products=diagnostic_paths.get("top_products_csv"),
                gate_status=diagnostic_paths["metrics"].get("stage4_gate_status"),
            )
        elif gates_apply and bool(gates_cfg.get("enabled", False)):
            gate_metrics = evaluate_customer_embedding_weight_profile(
                transactions=lf,
                embeddings_path=embeddings_path,
                weight_strategy=selected_weight_strategy,
                quantity_transform=selected_quantity_transform,
                max_customer_product_weight=selected_max_customer_product_weight,
                normalize_vectors=bool(normalize_vectors),
                cfg=cfg,
            )
            gate_metrics.update(customer_embedding_gate_status(gate_metrics, cfg=cfg))
            _enforce_customer_embedding_gates(gate_metrics, cfg)

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
