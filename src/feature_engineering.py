"""Customer behavioral features and feature-set assembly."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.product_themes import STRATEGIC_THEME_PATTERNS, detect_product_themes, normalize_product_text
from src.progress import log_event, stage_timer
from src.utils import (
    collect_streaming,
    deterministic_sample_indices,
    file_fingerprint,
    frame_to_numpy,
    numeric_feature_columns,
    schema_names,
    should_use_cache,
    write_artifact_metadata,
)


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


def _positive_units_expression(columns: set[str]) -> pl.Expr:
    if "unidades" not in columns:
        return pl.lit(1.0).alias("_units")
    return (
        pl.when(pl.col("unidades").cast(pl.Float64) > 0)
        .then(pl.col("unidades").cast(pl.Float64))
        .otherwise(1.0)
        .alias("_units")
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
    output = Path(output_path) if output_path else cfg.artifact_path(
        "behavioral_features",
        "output",
        directory=cfg.outputs / "features",
    )
    cache_metadata = {
        "stage": "behavioral_features",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "promo_definition": "promo_line_share plus promo_basket_share; promo_share aliases line share",
        "reference_date": cfg.get("behavioral_features.reference_date"),
    }
    if should_use_cache(
        output,
        force=force,
        use_cached=cfg.get("cache.use_cached", True),
        metadata=cache_metadata,
    ):
        log_event("Stage 5 behavior", "cache hit", cfg=cfg, path=output)
        return output

    with stage_timer("Stage 5 behavior", "building behavioral features", cfg=cfg, output=output):
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
                (pl.col("_promo_flag").sum() > 0).cast(pl.UInt8).alias("_basket_has_promo"),
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
                (pl.col("_basket_promo_lines").sum() / pl.col("_basket_lines").sum()).alias("promo_line_share"),
                pl.col("_basket_has_promo").mean().alias("promo_basket_share"),
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
                    pl.col("promo_line_share").alias("promo_share"),
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
        features = collect_streaming(result)
        features.write_parquet(output)
        write_artifact_metadata(output, cache_metadata)
        log_event("Stage 5 behavior", "wrote artifact", cfg=cfg, customers=features.height, path=output)
    return output


PRODUCT_FAMILY_STOPWORDS = {
    "carrefour",
    "marca",
    "producto",
    "productos",
    "pack",
    "lote",
    "bolsa",
    "bandeja",
    "caja",
    "unidad",
    "unidades",
    "und",
    "uds",
    "peso",
    "escurrido",
    "aprox",
    "granel",
    "sabor",
    "formato",
    "gr",
    "grs",
    "gramo",
    "gramos",
    "litro",
    "litros",
    "plastico",
    "reciclado",
    "reciclada",
    "reciclables",
    "tarrina",
    "envase",
    "cdc",
}

PRODUCT_FAMILY_UNIGRAM_STOPWORDS = PRODUCT_FAMILY_STOPWORDS | {
    "con",
    "sin",
    "para",
    "extra",
    "mini",
    "maxi",
    "super",
    "nuevo",
    "nueva",
    "gran",
}


def build_product_exposure_features(
    transactions: pl.LazyFrame | None = None,
    output_path: str | Path | None = None,
    catalog_output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create customer-level product-derived exposure features.

    Features are multi-label and product-first: one product can contribute to
    several tags, such as organic_bio plus lactose_free, or baby plus baby_food.
    No spend or demographic fields are used.
    """

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.artifact_path(
        "product_exposure_features",
        "output",
        directory=cfg.outputs / "features",
    )
    catalog_output = (
        Path(catalog_output_path)
        if catalog_output_path
        else cfg.artifacts
        / str(cfg.get("product_exposure_features.catalog_output_dir", "stage5"))
        / str(cfg.get("product_exposure_features.catalog_output_csv", "product_exposure_feature_catalog.csv"))
    )
    settings = cfg.get("product_exposure_features", {}) or {}
    cache_metadata = {
        "stage": "product_exposure_features",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "settings": settings,
        "theme_pattern_hash": _product_theme_pattern_hash(),
    }
    if should_use_cache(
        output,
        force=force,
        use_cached=cfg.get("cache.use_cached", True),
        metadata=cache_metadata,
    ):
        log_event("Stage 5 product exposure", "cache hit", cfg=cfg, path=output)
        return output

    with stage_timer("Stage 5 product exposure", "building product-derived exposure features", cfg=cfg, output=output):
        lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
        columns = set(schema_names(lf))
        if "cliente" not in columns or "idarticu" not in columns:
            raise ValueError("Product exposure features require 'cliente' and 'idarticu' columns.")

        feature_map = _build_product_exposure_feature_map(lf, cfg=cfg)
        if feature_map.is_empty():
            customers = collect_streaming(lf.select("cliente").unique()).sort("cliente")
            output.parent.mkdir(parents=True, exist_ok=True)
            customers.write_parquet(output)
            write_artifact_metadata(output, cache_metadata)
            return output

        catalog_output.parent.mkdir(parents=True, exist_ok=True)
        feature_map.select(
            [
                "feature_type",
                "feature_key",
                "feature_label",
                "feature_name",
                "customer_count",
                "product_count",
                "feature_rank",
            ]
        ).unique(subset=["feature_name"]).sort(["feature_type", "feature_rank"]).write_csv(catalog_output)

        ticket_col = str(settings.get("ticket_column", "ticket"))
        metrics = _product_exposure_metrics(columns, cfg)
        base_cols = ["cliente", "idarticu"]
        if ticket_col in columns and "basket_share" in metrics:
            base_cols.append(ticket_col)
        if "unidades" in columns:
            base_cols.append("unidades")

        base_cols = list(dict.fromkeys(base_cols))
        base = lf.select(base_cols).with_columns(_positive_units_expression(set(base_cols)))
        total_exprs = [
            pl.len().alias("_total_lines"),
            pl.col("_units").sum().alias("_total_units"),
            pl.col("idarticu").n_unique().alias("_total_distinct_products"),
        ]
        if ticket_col in base_cols:
            total_exprs.append(pl.col(ticket_col).n_unique().alias("_total_baskets"))
        totals = base.group_by("cliente").agg(total_exprs)

        feature_events = base.join(
            feature_map.lazy().select(["idarticu", "feature_name"]),
            on="idarticu",
            how="inner",
        )
        agg_exprs = [
            pl.len().alias("_feature_lines"),
            pl.col("_units").sum().alias("_feature_units"),
            pl.col("idarticu").n_unique().alias("_feature_distinct_products"),
        ]
        if ticket_col in base_cols:
            agg_exprs.append(pl.col(ticket_col).n_unique().alias("_feature_baskets"))

        long = (
            feature_events.group_by(["cliente", "feature_name"])
            .agg(agg_exprs)
            .join(totals, on="cliente", how="left")
            .with_columns(_product_exposure_metric_exprs(metrics))
            .select(["cliente", "feature_name", *[_product_exposure_metric_col(metric) for metric in metrics]])
        )
        long_df = collect_streaming(long)
        customers = collect_streaming(lf.select("cliente").unique()).sort("cliente")
        result = customers
        for metric in metrics:
            value_col = _product_exposure_metric_col(metric)
            metric_frame = _pivot_product_exposure_metric(long_df, value_col, metric)
            result = result.join(metric_frame, on="cliente", how="left")

        feature_cols = [col for col in result.columns if col != "cliente"]
        result = result.with_columns([pl.col(col).fill_null(0.0).cast(pl.Float32).alias(col) for col in feature_cols])
        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output)
        write_artifact_metadata(output, cache_metadata)
        log_event(
            "Stage 5 product exposure",
            "wrote artifact",
            cfg=cfg,
            customers=result.height,
            features=len(feature_cols),
            catalog=catalog_output,
            path=output,
        )
    return output


def build_feature_set(
    customer_embeddings_path: str | Path,
    behavior_path: str | Path | None = None,
    product_exposure_path: str | Path | None = None,
    variant: str = "embeddings_only",
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create model-ready feature variants."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    if variant == "embeddings_frequency" and behavior_path is None:
        behavior_path = cfg.artifact_path(
            "behavioral_features",
            "output",
            directory=cfg.outputs / "features",
        )
    outputs = cfg.get("feature_sets.outputs", {})
    output = Path(output_path) if output_path else cfg.outputs / "features" / outputs.get(
        variant,
        f"feature_set_{variant}.parquet",
    )
    cache_metadata = {
        "stage": "feature_set",
        "mode": cfg.mode,
        "variant": variant,
        "customer_embeddings": file_fingerprint(customer_embeddings_path),
        "behavior": file_fingerprint(behavior_path) if behavior_path else None,
        "product_exposure": file_fingerprint(product_exposure_path) if product_exposure_path else None,
        "standardize_behavior": bool(cfg.get("feature_sets.standardize_behavior", True)),
        "standardize_product_exposure": bool(cfg.get("feature_sets.standardize_product_exposure", True)),
        "product_exposure_weight": float(cfg.get("feature_sets.product_exposure_weight", 0.35)),
        "frequency_anchor_weight": float(cfg.get("feature_sets.frequency_anchor_weight", 0.12)),
    }
    if should_use_cache(
        output,
        force=force,
        use_cached=cfg.get("cache.use_cached", True),
        metadata=cache_metadata,
    ):
        log_event("Stage 5 feature set", "cache hit", cfg=cfg, variant=variant, path=output)
        return output

    with stage_timer("Stage 5 feature set", "building model-ready features", cfg=cfg, variant=variant, output=output):
        embeddings = pl.read_parquet(customer_embeddings_path)
        emb_cols = [col for col in embeddings.columns if col.startswith("emb_")]
        if variant == "embeddings_only":
            result = embeddings.select(["cliente", *emb_cols]).sort("cliente")
        elif variant == "embeddings_product_exposure":
            if product_exposure_path is None:
                product_exposure_path = cfg.artifact_path(
                    "product_exposure_features",
                    "output",
                    directory=cfg.outputs / "features",
                )
            exposure = pl.read_parquet(product_exposure_path)
            joined = embeddings.select(["cliente", *emb_cols]).join(exposure, on="cliente", how="inner")
            exposure_cols = [
                col
                for col in numeric_feature_columns(joined, exclude=("cliente", *emb_cols))
                if col.startswith("pdx_")
            ]
            result = joined.select(["cliente", *emb_cols, *exposure_cols]).fill_null(0)
            if cfg.get("feature_sets.standardize_product_exposure", True):
                result = _standardize_numeric_columns(
                    result,
                    exposure_cols,
                    multiplier=float(cfg.get("feature_sets.product_exposure_weight", 0.35)),
                    rename_prefix=None,
                )
            result = result.sort("cliente")
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
                result = _standardize_numeric_columns(result, behavior_cols, rename_prefix="beh_")
            result = result.sort("cliente")
        elif variant == "embeddings_frequency":
            if behavior_path is None:
                raise ValueError("behavior_path is required for embeddings_frequency feature set")
            behavior = pl.read_parquet(behavior_path)
            if "frequency_per_30d" not in behavior.columns:
                raise ValueError("embeddings_frequency feature set requires 'frequency_per_30d' in behavior features")
            frequency = _frequency_anchor_features(
                behavior,
                multiplier=float(cfg.get("feature_sets.frequency_anchor_weight", 0.12)),
            )
            result = (
                embeddings.select(["cliente", *emb_cols])
                .join(frequency, on="cliente", how="left")
                .with_columns(pl.col("freq_anchor").fill_null(0.0).cast(pl.Float32))
                .sort("cliente")
            )
        else:
            raise ValueError(f"Unknown feature set variant: {variant}")

        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output)
        write_artifact_metadata(output, cache_metadata)
        log_event("Stage 5 feature set", "wrote artifact", cfg=cfg, variant=variant, rows=result.height, path=output)
    return output


def _product_theme_pattern_hash() -> str:
    payload = repr(sorted((key, tuple(patterns)) for key, patterns in STRATEGIC_THEME_PATTERNS.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _product_exposure_metrics(columns: set[str], cfg: PipelineConfig) -> list[str]:
    requested = list(cfg.get("product_exposure_features.metrics", ["basket_share", "distinct_product_share"]) or [])
    normalized = []
    ticket_col = str(cfg.get("product_exposure_features.ticket_column", "ticket"))
    for metric in requested:
        value = str(metric).strip().lower()
        if value == "basket_share" and ticket_col not in columns:
            continue
        if value == "unit_share" and "unidades" not in columns:
            continue
        if value not in {"basket_share", "line_share", "unit_share", "distinct_product_share"}:
            raise ValueError(
                "Unknown product exposure metric "
                f"{metric!r}. Use basket_share, line_share, unit_share, or distinct_product_share."
            )
        if value not in normalized:
            normalized.append(value)
    return normalized or ["line_share"]


def _product_exposure_metric_col(metric: str) -> str:
    return f"_{metric}"


def _product_exposure_metric_exprs(metrics: list[str]) -> list[pl.Expr]:
    exprs = []
    if "basket_share" in metrics:
        exprs.append(
            (
                pl.col("_feature_baskets")
                / pl.when(pl.col("_total_baskets") > 0).then(pl.col("_total_baskets")).otherwise(1)
            ).alias(_product_exposure_metric_col("basket_share"))
        )
    if "line_share" in metrics:
        exprs.append(
            (
                pl.col("_feature_lines")
                / pl.when(pl.col("_total_lines") > 0).then(pl.col("_total_lines")).otherwise(1)
            ).alias(_product_exposure_metric_col("line_share"))
        )
    if "unit_share" in metrics:
        exprs.append(
            (
                pl.col("_feature_units")
                / pl.when(pl.col("_total_units") > 0).then(pl.col("_total_units")).otherwise(1.0)
            ).alias(_product_exposure_metric_col("unit_share"))
        )
    if "distinct_product_share" in metrics:
        exprs.append(
            (
                pl.col("_feature_distinct_products")
                / pl.when(pl.col("_total_distinct_products") > 0).then(pl.col("_total_distinct_products")).otherwise(1)
            ).alias(_product_exposure_metric_col("distinct_product_share"))
        )
    return exprs


def _pivot_product_exposure_metric(long_df: pl.DataFrame, value_col: str, metric: str) -> pl.DataFrame:
    if long_df.is_empty():
        return pl.DataFrame(schema={"cliente": pl.Utf8})
    try:
        pivoted = long_df.pivot(index="cliente", on="feature_name", values=value_col, aggregate_function="first")
    except TypeError:
        pivoted = long_df.pivot(index="cliente", columns="feature_name", values=value_col, aggregate_function="first")
    rename = {
        col: f"pdx_{metric}_{col}"
        for col in pivoted.columns
        if col != "cliente"
    }
    return pivoted.rename(rename)


def _build_product_exposure_feature_map(
    lf: pl.LazyFrame,
    *,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    columns = set(schema_names(lf))
    product_meta = _product_exposure_product_metadata(lf, columns)
    raw_map = _raw_product_exposure_feature_map(product_meta, cfg=cfg)
    if raw_map.is_empty():
        return raw_map
    customer_product = lf.select(["cliente", "idarticu"]).unique()
    counts = collect_streaming(
        customer_product.join(raw_map.lazy(), on="idarticu", how="inner")
        .group_by(["feature_type", "feature_key", "feature_label", "feature_name"])
        .agg(
            [
                pl.col("cliente").n_unique().alias("customer_count"),
                pl.col("idarticu").n_unique().alias("product_count"),
            ]
        )
    )
    selected = _select_product_exposure_features(counts, cfg=cfg)
    if selected.is_empty():
        return pl.DataFrame(schema=raw_map.schema)
    return raw_map.join(
        selected.select(
            [
                "feature_type",
                "feature_key",
                "feature_label",
                "feature_name",
                "customer_count",
                "product_count",
                "feature_rank",
            ]
        ),
        on=["feature_type", "feature_key", "feature_label", "feature_name"],
        how="inner",
    )


def _product_exposure_product_metadata(lf: pl.LazyFrame, columns: set[str]) -> pl.DataFrame:
    agg_exprs = []
    if "desc_larga_articulo" in columns:
        agg_exprs.append(pl.col("desc_larga_articulo").drop_nulls().first().alias("product_description"))
    else:
        agg_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("product_description"))
    if "idsector" in columns:
        agg_exprs.append(pl.col("idsector").drop_nulls().first().cast(pl.Utf8).alias("sector_id"))
    else:
        agg_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("sector_id"))
    if "desc_sector" in columns:
        agg_exprs.append(pl.col("desc_sector").drop_nulls().first().alias("sector_label"))
    else:
        agg_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("sector_label"))
    return collect_streaming(lf.group_by("idarticu").agg(agg_exprs))


def _raw_product_exposure_feature_map(product_meta: pl.DataFrame, *, cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    settings = cfg.get("product_exposure_features", {}) or {}
    dimensions = settings.get("dimensions", {}) or {}
    include_themes = bool(dimensions.get("themes", True))
    include_sectors = bool(dimensions.get("sectors", True))
    include_families = bool(dimensions.get("product_families", True))
    max_families_per_product = int(settings.get("max_families_per_product", 6))
    ngram_sizes = [int(value) for value in settings.get("family_ngram_sizes", [2, 1]) or [2, 1]]

    rows = []
    for row in product_meta.iter_rows(named=True):
        product_id = row.get("idarticu")
        description = "" if row.get("product_description") is None else str(row.get("product_description"))
        if include_themes:
            for theme in detect_product_themes(description):
                rows.append(
                    _product_exposure_feature_row(
                        product_id,
                        feature_type="theme",
                        feature_key=theme,
                        feature_label=theme.replace("_", " ").title(),
                    )
                )
        if include_sectors:
            sector_label = str(row.get("sector_label") or row.get("sector_id") or "").strip()
            if sector_label:
                rows.append(
                    _product_exposure_feature_row(
                        product_id,
                        feature_type="sector",
                        feature_key=sector_label,
                        feature_label=sector_label,
                    )
                )
        if include_families:
            for term in _extract_product_family_terms(
                description,
                max_terms=max_families_per_product,
                ngram_sizes=ngram_sizes,
            ):
                rows.append(
                    _product_exposure_feature_row(
                        product_id,
                        feature_type="family",
                        feature_key=term,
                        feature_label=term,
                    )
                )
    schema = {
        "idarticu": product_meta.schema.get("idarticu", pl.Int64),
        "feature_type": pl.Utf8,
        "feature_key": pl.Utf8,
        "feature_label": pl.Utf8,
        "feature_name": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def _product_exposure_feature_row(
    product_id: object,
    *,
    feature_type: str,
    feature_key: str,
    feature_label: str,
) -> dict[str, object]:
    safe_key = _safe_feature_token(feature_key)
    return {
        "idarticu": product_id,
        "feature_type": feature_type,
        "feature_key": feature_key,
        "feature_label": feature_label,
        "feature_name": f"{feature_type}_{safe_key}",
    }


def _select_product_exposure_features(counts: pl.DataFrame, *, cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    if counts.is_empty():
        return counts
    settings = cfg.get("product_exposure_features", {}) or {}
    min_customers = settings.get("min_customers", {}) or {}
    max_features = settings.get("max_features", {}) or {}
    frames = []
    for feature_type in ["theme", "sector", "family"]:
        frame = counts.filter(pl.col("feature_type") == feature_type)
        if frame.is_empty():
            continue
        threshold = int(min_customers.get(feature_type, min_customers.get(f"{feature_type}s", 50)))
        limit = max_features.get(feature_type, max_features.get(f"{feature_type}s", None))
        filtered = frame.filter(pl.col("customer_count") >= threshold).sort(
            ["customer_count", "product_count", "feature_key"],
            descending=[True, True, False],
        )
        if limit is not None:
            filtered = filtered.head(int(limit))
        filtered = filtered.with_row_index("feature_rank", offset=1)
        frames.append(filtered)
    return pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema=counts.schema)


def _extract_product_family_terms(
    description: str | None,
    *,
    max_terms: int,
    ngram_sizes: list[int],
) -> list[str]:
    text = normalize_product_text(description)
    raw_tokens = re.findall(r"[a-z0-9]+", text)
    tokens = []
    for token in raw_tokens:
        if token in PRODUCT_FAMILY_STOPWORDS:
            continue
        if len(token) < 3:
            continue
        if any(char.isdigit() for char in token):
            continue
        if token.isdigit() or re.fullmatch(r"\d+(?:kg|g|gr|ml|cl|l|uds?)?", token):
            continue
        tokens.append(token)

    terms: list[str] = []
    seen: set[str] = set()
    for n in ngram_sizes:
        if n <= 0:
            continue
        for start in range(0, max(len(tokens) - n + 1, 0)):
            candidate_tokens = tokens[start : start + n]
            if n == 1 and candidate_tokens[0] in PRODUCT_FAMILY_UNIGRAM_STOPWORDS:
                continue
            if n > 1 and candidate_tokens[0] in {"con", "para"}:
                continue
            if candidate_tokens[-1] in PRODUCT_FAMILY_UNIGRAM_STOPWORDS:
                continue
            term = " ".join(candidate_tokens)
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= max_terms:
                return terms
    return terms


def _safe_feature_token(value: str, *, max_length: int = 64) -> str:
    token = normalize_product_text(value)
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    if not token:
        token = hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).hexdigest()
    if len(token) > max_length:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).hexdigest()
        token = f"{token[: max_length - 9].rstrip('_')}_{digest}"
    return token


def _standardize_numeric_columns(
    frame: pl.DataFrame,
    columns: list[str],
    *,
    multiplier: float = 1.0,
    rename_prefix: str | None,
) -> pl.DataFrame:
    updates = []
    for col in columns:
        values = frame[col].to_numpy().astype(np.float64)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        denom = std if std > 1e-12 else 1.0
        alias = f"{rename_prefix}{col}" if rename_prefix is not None else col
        updates.append((((pl.col(col) - mean) / denom) * multiplier).cast(pl.Float32).alias(alias))
    result = frame.with_columns(updates)
    if rename_prefix is not None:
        result = result.drop(columns)
    return result


def _frequency_anchor_features(frame: pl.DataFrame, *, multiplier: float) -> pl.DataFrame:
    frequency = pl.col("frequency_per_30d").cast(pl.Float64).fill_null(0.0)
    result = frame.select(["cliente", "frequency_per_30d"]).with_columns(
        pl.when(frequency > 0.0).then(frequency).otherwise(0.0).log1p().alias("freq_log")
    )
    result = _standardize_numeric_columns(result, ["freq_log"], multiplier=multiplier, rename_prefix=None)
    return result.rename({"freq_log": "freq_anchor"}).select(["cliente", "freq_anchor"])


def build_feature_set_diagnostics(
    feature_paths: Mapping[str, str | Path],
    baseline_name: str | None = None,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Write compact Stage 5 diagnostics for feature health and topology drift."""

    cfg.ensure_directories()
    diagnostics_cfg = cfg.get("feature_sets.diagnostics", {}) or {}
    output = Path(output_path) if output_path else cfg.artifacts / str(
        diagnostics_cfg.get("output_dir", "stage5")
    ) / str(diagnostics_cfg.get("output_csv", "feature_set_diagnostics.csv"))
    baseline = str(baseline_name or cfg.get("feature_sets.default", "embeddings_only"))
    if baseline not in feature_paths:
        raise ValueError(f"Baseline feature set {baseline!r} is not present in feature_paths.")

    frames = {name: pl.read_parquet(path).sort("cliente") for name, path in feature_paths.items()}
    baseline_frame = frames[baseline]
    baseline_customers = baseline_frame["cliente"].to_list()
    baseline_customer_set = set(baseline_customers)
    baseline_features = numeric_feature_columns(baseline_frame)
    neighbor_k = int(diagnostics_cfg.get("neighbor_overlap_k", 10))
    sample_size = int(diagnostics_cfg.get("neighbor_overlap_sample_size", 2000))
    sample_customers = _diagnostic_sample_customers(baseline_customers, sample_size, cfg)
    baseline_neighbors = _nearest_neighbor_indices(
        _aligned_feature_matrix(baseline_frame, sample_customers, baseline_features),
        neighbor_k,
    )

    selection_feature_set = str(cfg.get("modeling.feature_set_for_selection", baseline))
    if selection_feature_set == "embeddings_only":
        selection_warning = "OK: official selection uses product embeddings only."
    elif selection_feature_set == "embeddings_product_exposure":
        selection_warning = (
            "REVIEW: official selection uses opt-in product-exposure challenger features; compare against "
            "embeddings_only before promoting."
        )
    else:
        selection_warning = (
            f"WARNING: official selection uses {selection_feature_set!r}; confirm Stage 6/8 evidence "
            "before allowing behavior or auxiliary features to drive organic tribes."
        )

    rows = []
    for name, path in feature_paths.items():
        df = frames[name]
        feature_cols = numeric_feature_columns(df)
        customer_set = set(df["cliente"].to_list()) if "cliente" in df.columns else set()
        missing_vs_baseline = len(baseline_customer_set - customer_set)
        extra_vs_baseline = len(customer_set - baseline_customer_set)
        duplicate_customer_count = df.height - df["cliente"].n_unique() if "cliente" in df.columns else None
        health = _feature_health_metrics(df, feature_cols)
        overlap = None
        if name == baseline:
            overlap = 100.0
        elif missing_vs_baseline == 0 and feature_cols and baseline_neighbors is not None:
            target_neighbors = _nearest_neighbor_indices(
                _aligned_feature_matrix(df, sample_customers, feature_cols),
                neighbor_k,
            )
            overlap = _neighbor_overlap_pct(baseline_neighbors, target_neighbors)
        rows.append(
            {
                "feature_set_name": str(name),
                "feature_path": str(path),
                "is_selection_feature_set": name == selection_feature_set,
                "selection_feature_set": selection_feature_set,
                "selection_warning": selection_warning,
                "rows": int(df.height),
                "baseline_rows": int(baseline_frame.height),
                "missing_vs_baseline_customers": int(missing_vs_baseline),
                "extra_vs_baseline_customers": int(extra_vs_baseline),
                "duplicate_customer_count": None if duplicate_customer_count is None else int(duplicate_customer_count),
                "customer_alignment_status": "pass"
                if missing_vs_baseline == 0 and extra_vs_baseline == 0 and duplicate_customer_count == 0
                else "warn",
                "feature_count": int(len(feature_cols)),
                "null_pct": health["null_pct"],
                "finite_pct": health["finite_pct"],
                "zero_variance_feature_count": health["zero_variance_feature_count"],
                "zero_variance_features": health["zero_variance_features"],
                "neighbor_overlap_vs_baseline_pct": overlap,
                "neighbor_overlap_k": int(neighbor_k),
                "neighbor_overlap_sample_size": int(len(sample_customers)),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = pl.DataFrame(rows, infer_schema_length=None)
    diagnostics.write_csv(output)
    log_event("Stage 5 diagnostics", "wrote feature-set diagnostics", cfg=cfg, rows=diagnostics.height, path=output)
    return output


def _feature_health_metrics(df: pl.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    if not feature_cols or df.height == 0:
        return {
            "null_pct": None,
            "finite_pct": None,
            "zero_variance_feature_count": 0,
            "zero_variance_features": "",
        }
    total_cells = df.height * len(feature_cols)
    null_count = sum(int(df.select(pl.col(col).is_null().sum())[0, 0]) for col in feature_cols)
    X = frame_to_numpy(df, feature_cols)
    finite_pct = float(np.isfinite(X).mean() * 100.0)
    zero_variance = []
    for idx, col in enumerate(feature_cols):
        values = X[:, idx]
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0 or float(np.nanstd(finite_values)) <= 1e-12:
            zero_variance.append(col)
    preview = ", ".join(zero_variance[:12])
    if len(zero_variance) > 12:
        preview = f"{preview}, ..."
    return {
        "null_pct": float(null_count / max(total_cells, 1) * 100.0),
        "finite_pct": finite_pct,
        "zero_variance_feature_count": int(len(zero_variance)),
        "zero_variance_features": preview,
    }


def _diagnostic_sample_customers(customers: list[object], sample_size: int, cfg: PipelineConfig) -> list[object]:
    if not customers:
        return []
    indices = deterministic_sample_indices(len(customers), sample_size, cfg.random_seed)
    return [customers[int(idx)] for idx in indices]


def _aligned_feature_matrix(df: pl.DataFrame, customers: list[object], feature_cols: list[str]) -> np.ndarray:
    if not customers or not feature_cols:
        return np.empty((0, 0), dtype=np.float32)
    order = pl.DataFrame({"cliente": customers, "_diagnostic_order": list(range(len(customers)))})
    aligned = order.join(df.select(["cliente", *feature_cols]), on="cliente", how="inner").sort("_diagnostic_order")
    return np.nan_to_num(frame_to_numpy(aligned, feature_cols), copy=False)


def _nearest_neighbor_indices(X: np.ndarray, k: int) -> np.ndarray | None:
    if X.shape[0] <= 1 or X.shape[1] == 0:
        return None
    n_neighbors = max(1, min(int(k), X.shape[0] - 1))
    norms = np.linalg.norm(X, axis=1)
    normalized = X / np.maximum(norms[:, None], 1e-12)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    return np.argsort(-similarities, axis=1)[:, :n_neighbors]


def _neighbor_overlap_pct(baseline_neighbors: np.ndarray | None, target_neighbors: np.ndarray | None) -> float | None:
    if baseline_neighbors is None or target_neighbors is None:
        return None
    if baseline_neighbors.shape != target_neighbors.shape or baseline_neighbors.shape[0] == 0:
        return None
    overlaps = []
    for base_row, target_row in zip(baseline_neighbors, target_neighbors):
        overlaps.append(len(set(base_row.tolist()) & set(target_row.tolist())) / max(len(base_row), 1))
    return float(np.mean(overlaps) * 100.0)
