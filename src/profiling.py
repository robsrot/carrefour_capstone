"""Commercial tribe profiling with product and sector lift evidence."""

from __future__ import annotations

import json
import hashlib
import os
import re
from html import escape
from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.product_themes import STRATEGIC_THEME_PATTERNS, detect_product_themes, normalize_product_text
from src.progress import log_event, stage_timer
from src.tribe_namer import THEME_LABELS, fallback_tribe_name
from src.utils import collect_streaming, file_fingerprint, schema_names, should_use_cache, write_artifact_metadata


def _join_list(values: list[Any], fmt: str = "{}") -> str:
    return "; ".join(fmt.format(value) for value in values)


def _csv_safe_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return _join_list(["" if item is None else item for item in value])
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _theme_pattern_fingerprint() -> str:
    payload = json.dumps(STRATEGIC_THEME_PATTERNS, sort_keys=True)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


def _product_term_fingerprint() -> str:
    payload = json.dumps(
        {
            "stopwords": sorted(PRODUCT_TERM_STOPWORDS),
            "unigram_stopwords": sorted(PRODUCT_TERM_UNIGRAM_STOPWORDS),
        },
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


CAMPAIGN_SIGNAL_THEMES = {
    "baby",
    "halal",
    "kids_general",
    "kids_girls",
    "kids_boys",
    "pet",
    "pet_dog",
    "pet_cat",
    "world_foods_asian",
    "world_foods_mexican",
    "world_foods_middle_eastern",
    "world_foods_latin",
}

PRODUCT_TERM_STOPWORDS = {
    "aceite",
    "agua",
    "alimentacion",
    "alta",
    "alto",
    "amarillo",
    "ano",
    "anos",
    "azul",
    "bajo",
    "barra",
    "blanca",
    "blanco",
    "bolsa",
    "botella",
    "caja",
    "carrefour",
    "clasico",
    "color",
    "con",
    "centralizado",
    "cortada",
    "cortado",
    "coste",
    "crf",
    "de",
    "del",
    "dia",
    "el",
    "en",
    "especial",
    "extra",
    "fresco",
    "fresca",
    "frescos",
    "franquicia",
    "frq",
    "gr",
    "granel",
    "gramos",
    "gran",
    "grande",
    "grandes",
    "kg",
    "la",
    "las",
    "lata",
    "litro",
    "litros",
    "los",
    "marca",
    "maxi",
    "ml",
    "natural",
    "negro",
    "nueva",
    "nuevo",
    "pack",
    "para",
    "peq",
    "pequena",
    "pequenas",
    "pequeno",
    "pequenos",
    "peso",
    "pieza",
    "piezas",
    "premium",
    "producto",
    "aprox",
    "roja",
    "rojo",
    "sabor",
    "seleccion",
    "sobre",
    "talla",
    "tallas",
    "tarrina",
    "tex",
    "und",
    "unds",
    "unidad",
    "unidades",
    "uds",
    "verde",
    "venta",
    "y",
}

PRODUCT_TERM_UNIGRAM_STOPWORDS = {
    "sin",
}


def profile_tribes(
    assignments_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    behavior_path: str | Path | None = None,
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create one row per tribe with KPI, product-lift, and sector-lift evidence."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    assignments_file = Path(assignments_path)
    stem = assignments_file.stem.replace("cluster_assignments_", "")
    output = Path(output_path) if output_path else cfg.outputs / "profiles" / f"tribe_profiles_{stem}.parquet"
    candidate_behavior_path = (
        Path(behavior_path)
        if behavior_path
        else cfg.artifact_path("behavioral_features", "output", directory=cfg.outputs / "features")
    )
    cache_metadata = {
        "stage": "tribe_profiles",
        "mode": cfg.mode,
        "assignments": file_fingerprint(assignments_file),
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "behavior": file_fingerprint(candidate_behavior_path),
        "profiling": cfg.get("profiling", {}),
        "product_theme_patterns": _theme_pattern_fingerprint(),
        "product_term_rules": _product_term_fingerprint(),
        "population_definition": "all_assigned_customers_including_noise",
    }
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata):
        log_event("Stage 8 profiling", "cache hit", cfg=cfg, assignments=assignments_file, path=output)
        return output

    with stage_timer("Stage 8 profiling", "building tribe profile", cfg=cfg, assignments=assignments_file):
        lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
        columns = set(schema_names(lf))
        product_cols = ["cliente", "idarticu"]
        for col in ["desc_larga_articulo", "idsector", "desc_sector", "importe", "unidades", "ticket"]:
            if col in columns:
                product_cols.append(col)

        assignment_columns = set(schema_names(pl.scan_parquet(assignments_file)))
        assignment_select = [pl.col("cliente"), pl.col("tribe_id")]
        if "model_name" in assignment_columns:
            assignment_select.append(pl.col("model_name"))
        else:
            assignment_select.append(pl.lit(stem).alias("model_name"))
        if "model_variant" in assignment_columns:
            assignment_select.append(pl.col("model_variant"))
        else:
            assignment_select.append(pl.lit(None).cast(pl.Utf8).alias("model_variant"))
        if "assignment_source" in assignment_columns:
            assignment_select.append(pl.col("assignment_source"))
        else:
            assignment_select.append(pl.lit("unknown").alias("assignment_source"))
        if "assignment_confidence_score" in assignment_columns:
            assignment_select.append(pl.col("assignment_confidence_score").cast(pl.Float64))
        else:
            assignment_select.append(pl.lit(None).cast(pl.Float64).alias("assignment_confidence_score"))
        assignments_all = (
            pl.scan_parquet(assignments_file)
            .select(assignment_select)
            .unique(subset=["cliente"], keep="first")
        )
        clustered_assignments = assignments_all.filter(pl.col("tribe_id") >= 0)
        assignment_customers = assignments_all.select("cliente")
        assignment_meta = collect_streaming(
            assignments_all.select(
                [
                    pl.col("model_name").drop_nulls().first().alias("model_name"),
                    pl.col("model_variant").drop_nulls().first().alias("model_variant"),
                ]
            )
        ).row(0, named=True)

        cluster_sizes = collect_streaming(
            clustered_assignments.group_by("tribe_id").agg(pl.col("cliente").n_unique().alias("n_customers")).sort("tribe_id")
        )
        total_customers = int(
            collect_streaming(assignments_all.select(pl.col("cliente").n_unique().alias("n_customers")))[0, "n_customers"]
        )
        clustered_customers = int(cluster_sizes["n_customers"].sum()) if cluster_sizes.height else 0
        noise_customers = max(total_customers - clustered_customers, 0)
        assignment_source_summary = collect_streaming(
            clustered_assignments.group_by(["tribe_id", "assignment_source"])
            .agg(pl.col("cliente").n_unique().alias("source_customers"))
            .sort(["tribe_id", "source_customers"], descending=[False, True])
        )
        soft_confidence_summary = collect_streaming(
            clustered_assignments.filter(pl.col("assignment_source").cast(pl.Utf8).str.contains("soft_noise"))
            .group_by("tribe_id")
            .agg(
                [
                    pl.col("assignment_confidence_score").mean().alias("soft_assignment_confidence_mean"),
                    pl.col("assignment_confidence_score").quantile(0.10).alias("soft_assignment_confidence_p10"),
                    pl.col("assignment_confidence_score").min().alias("soft_assignment_confidence_min"),
                ]
            )
        )
        log_event(
            "Stage 8 profiling",
            "assignment population",
            cfg=cfg,
            clusters=cluster_sizes.height,
            customers=total_customers,
            noise_customers=noise_customers,
            model=assignment_meta.get("model_name"),
            variant=assignment_meta.get("model_variant"),
        )

        min_product_customers = int(cfg.get("profiling.min_product_customers", 10))
        min_term_customers = int(cfg.get("profiling.min_term_customers", 20))
        min_term_population_customers = int(cfg.get("profiling.min_term_population_customers", 30))
        # Reduce transaction rows to customer-product evidence before adding tribe labels.
        profile_line_cols = ["cliente", "idarticu"] + [
            col for col in ["desc_larga_articulo", "desc_sector"] if col in product_cols
        ]
        assigned_line_items = lf.select(profile_line_cols).join(assignment_customers, on="cliente", how="inner")
        clustered_keys = clustered_assignments.select(["cliente", "tribe_id"])
        customer_product = assigned_line_items.select(["cliente", "idarticu"]).unique(subset=["cliente", "idarticu"])

        log_event(
            "Stage 8 profiling",
            "aggregating population product counts",
            cfg=cfg,
            min_product_customers=min_product_customers,
        )
        population_product = collect_streaming(
            customer_product.group_by("idarticu").agg(pl.len().alias("population_customers"))
        )
        if "desc_larga_articulo" in product_cols or "desc_sector" in product_cols:
            metadata_cols = []
            metadata_exprs = []
            if "desc_larga_articulo" in product_cols:
                metadata_cols.append("desc_larga_articulo")
                metadata_exprs.append(pl.col("desc_larga_articulo").first().alias("product_description"))
            if "desc_sector" in product_cols:
                metadata_cols.append("desc_sector")
                metadata_exprs.append(pl.col("desc_sector").first().alias("sector_description"))
            log_event("Stage 8 profiling", "aggregating product metadata", cfg=cfg)
            product_metadata = collect_streaming(
                assigned_line_items.select(["idarticu"] + metadata_cols)
                .group_by("idarticu")
                .agg(metadata_exprs)
            )
            population_product = population_product.join(product_metadata, on="idarticu", how="left")
        if "product_description" not in population_product.columns:
            population_product = population_product.with_columns(pl.lit(None).alias("product_description"))
        if "sector_description" not in population_product.columns:
            population_product = population_product.with_columns(pl.lit(None).alias("sector_description"))
        log_event(
            "Stage 8 profiling",
            "population product counts ready",
            cfg=cfg,
            products=population_product.height,
        )

        log_event("Stage 8 profiling", "aggregating tribe product lifts", cfg=cfg)
        cluster_product = (
            customer_product.join(clustered_keys, on="cliente", how="inner")
            .group_by(["tribe_id", "idarticu"])
            .agg(pl.len().alias("cluster_customers"))
            .filter(pl.col("cluster_customers") >= min_product_customers)
        )
        product_lifts = collect_streaming(
            cluster_product.join(population_product.lazy(), on="idarticu", how="left")
            .join(cluster_sizes.lazy(), on="tribe_id", how="left")
            .with_columns(
                [
                    (pl.col("cluster_customers") / pl.col("n_customers")).alias("cluster_rate"),
                    (pl.col("population_customers") / max(total_customers, 1)).alias("population_rate"),
                ]
            )
            .with_columns((pl.col("cluster_rate") / pl.col("population_rate")).alias("lift"))
            .sort(["tribe_id", "lift", "cluster_customers"], descending=[False, True, True])
        )
        log_event("Stage 8 profiling", "tribe product lifts ready", cfg=cfg, rows=product_lifts.height)

        theme_lifts = pl.DataFrame()
        term_lifts = pl.DataFrame()
        if "desc_larga_articulo" in product_cols:
            log_event("Stage 8 profiling", "building product theme map", cfg=cfg)
            product_theme_map = _build_product_theme_map(population_product.lazy())
            log_event("Stage 8 profiling", "product theme map ready", cfg=cfg, rows=product_theme_map.height)
            if product_theme_map.height:
                population_theme_events = customer_product.join(
                    product_theme_map.lazy(), on="idarticu", how="inner"
                ).select(["cliente", "strategic_theme"])
                log_event("Stage 8 profiling", "aggregating population theme counts", cfg=cfg)
                population_theme_counts = collect_streaming(
                    population_theme_events.group_by("strategic_theme").agg(
                        pl.col("cliente").n_unique().alias("population_customers")
                    )
                )
                log_event(
                    "Stage 8 profiling",
                    "population theme counts ready",
                    cfg=cfg,
                    rows=population_theme_counts.height,
                )
                log_event("Stage 8 profiling", "aggregating tribe theme lifts", cfg=cfg)
                cluster_theme = (
                    population_theme_events.join(clustered_keys, on="cliente", how="inner")
                    .group_by(["tribe_id", "strategic_theme"])
                    .agg(pl.col("cliente").n_unique().alias("cluster_customers"))
                    .filter(pl.col("cluster_customers") >= min_product_customers)
                )
                theme_lifts = collect_streaming(
                    cluster_theme.join(population_theme_counts.lazy(), on="strategic_theme", how="left")
                    .join(cluster_sizes.lazy(), on="tribe_id", how="left")
                    .with_columns(
                        [
                            (pl.col("cluster_customers") / pl.col("n_customers")).alias("cluster_rate"),
                            (pl.col("population_customers") / max(total_customers, 1)).alias("population_rate"),
                        ]
                    )
                    .with_columns((pl.col("cluster_rate") / pl.col("population_rate")).alias("lift"))
                    .sort(["tribe_id", "lift", "cluster_customers"], descending=[False, True, True])
                )
                log_event("Stage 8 profiling", "tribe theme lifts ready", cfg=cfg, rows=theme_lifts.height)

            log_event("Stage 8 profiling", "building product term map", cfg=cfg)
            product_term_map = _build_product_term_map(population_product.lazy(), cfg=cfg)
            log_event("Stage 8 profiling", "product term map ready", cfg=cfg, rows=product_term_map.height)
            if product_term_map.height:
                population_term_events = customer_product.join(product_term_map.lazy(), on="idarticu", how="inner").select(
                    ["cliente", "product_term"]
                )
                log_event(
                    "Stage 8 profiling",
                    "aggregating population term counts",
                    cfg=cfg,
                    min_population_customers=min_term_population_customers,
                )
                population_term_counts = collect_streaming(
                    population_term_events.group_by("product_term")
                    .agg(pl.col("cliente").n_unique().alias("population_customers"))
                    .filter(pl.col("population_customers") >= min_term_population_customers)
                )
                log_event(
                    "Stage 8 profiling",
                    "population term counts ready",
                    cfg=cfg,
                    rows=population_term_counts.height,
                )
                log_event(
                    "Stage 8 profiling",
                    "aggregating tribe term lifts",
                    cfg=cfg,
                    min_term_customers=min_term_customers,
                )
                cluster_term = (
                    population_term_events.join(clustered_keys, on="cliente", how="inner")
                    .group_by(["tribe_id", "product_term"])
                    .agg(pl.col("cliente").n_unique().alias("cluster_customers"))
                    .filter(pl.col("cluster_customers") >= min_term_customers)
                )
                term_lifts = collect_streaming(
                    cluster_term.join(population_term_counts.lazy(), on="product_term", how="inner")
                    .join(cluster_sizes.lazy(), on="tribe_id", how="left")
                    .with_columns(
                        [
                            (pl.col("cluster_customers") / pl.col("n_customers")).alias("cluster_rate"),
                            (pl.col("population_customers") / max(total_customers, 1)).alias("population_rate"),
                        ]
                    )
                    .with_columns((pl.col("cluster_rate") / pl.col("population_rate")).alias("lift"))
                    .sort(["tribe_id", "lift", "cluster_customers"], descending=[False, True, True])
                )
                log_event("Stage 8 profiling", "tribe term lifts ready", cfg=cfg, rows=term_lifts.height)

        if "desc_sector" in product_cols:
            log_event("Stage 8 profiling", "aggregating population sector line counts", cfg=cfg)
            population_sector = collect_streaming(
                assigned_line_items.select("desc_sector").group_by("desc_sector").agg(pl.len().alias("population_lines"))
            )
            population_total = int(population_sector["population_lines"].sum()) if population_sector.height else 0
            log_event(
                "Stage 8 profiling",
                "population sector counts ready",
                cfg=cfg,
                rows=population_sector.height,
                population_lines=population_total,
            )
            log_event(
                "Stage 8 profiling",
                "aggregating tribe sector line counts",
                cfg=cfg,
            )
            cluster_sector = collect_streaming(
                lf.select(["cliente", "desc_sector"])
                .join(clustered_keys, on="cliente", how="inner")
                .group_by(["tribe_id", "desc_sector"])
                .agg(pl.len().alias("cluster_lines"))
            )
            if cluster_sector.height:
                cluster_sector_totals = cluster_sector.group_by("tribe_id").agg(
                    pl.col("cluster_lines").sum().alias("cluster_lines_total")
                )
                sector_lifts = (
                    cluster_sector.join(population_sector, on="desc_sector", how="left")
                    .join(cluster_sector_totals, on="tribe_id", how="left")
                    .with_columns(
                        [
                            (pl.col("cluster_lines") / pl.col("cluster_lines_total")).alias("cluster_sector_share"),
                            (pl.col("population_lines") / max(population_total, 1)).alias("population_sector_share"),
                        ]
                    )
                    .with_columns((pl.col("cluster_sector_share") / pl.col("population_sector_share")).alias("lift"))
                    .sort(["tribe_id", "lift", "cluster_lines"], descending=[False, True, True])
                )
            else:
                sector_lifts = pl.DataFrame({"tribe_id": [], "desc_sector": [], "lift": []})
            log_event("Stage 8 profiling", "tribe sector lifts ready", cfg=cfg, rows=sector_lifts.height)
        else:
            sector_lifts = pl.DataFrame({"tribe_id": [], "desc_sector": [], "lift": []})

        behavior_summary = None
        if candidate_behavior_path.exists():
            behavior = pl.scan_parquet(candidate_behavior_path)
            behavior_cols = [col for col in schema_names(behavior) if col != "cliente"]
            aggregations = [pl.col(col).mean().alias(f"avg_{col}") for col in behavior_cols if col != "tribe_id"]
            log_event("Stage 8 profiling", "aggregating tribe KPI profiles", cfg=cfg, fields=len(aggregations))
            behavior_summary = collect_streaming(
                clustered_assignments.join(behavior, on="cliente", how="left").group_by("tribe_id").agg(aggregations)
            )
            log_event("Stage 8 profiling", "tribe KPI profiles ready", cfg=cfg, rows=behavior_summary.height)

        log_event("Stage 8 profiling", "assembling tribe profile rows", cfg=cfg, clusters=cluster_sizes.height)
        rows = []
        for cluster_row in cluster_sizes.iter_rows(named=True):
            tribe_id = int(cluster_row["tribe_id"])
            products = product_lifts.filter(pl.col("tribe_id") == tribe_id).head(int(cfg.get("profiling.top_n_products", 15)))
            sectors = sector_lifts.filter(pl.col("tribe_id") == tribe_id).head(int(cfg.get("profiling.top_n_sectors", 10)))
            themes = (
                theme_lifts.filter(pl.col("tribe_id") == tribe_id).head(int(cfg.get("profiling.top_n_themes", 8)))
                if not theme_lifts.is_empty() and "tribe_id" in theme_lifts.columns
                else pl.DataFrame()
            )
            terms = (
                term_lifts.filter(pl.col("tribe_id") == tribe_id).head(int(cfg.get("profiling.top_n_terms", 10)))
                if not term_lifts.is_empty() and "tribe_id" in term_lifts.columns
                else pl.DataFrame()
            )
            sources = assignment_source_summary.filter(pl.col("tribe_id") == tribe_id)
            source_names = sources["assignment_source"].to_list() if sources.height else []
            source_counts = sources["source_customers"].to_list() if sources.height else []
            soft_customers = sum(
                int(count)
                for source, count in zip(source_names, source_counts)
                if "soft_noise" in str(source)
            )
            soft_confidence = soft_confidence_summary.filter(pl.col("tribe_id") == tribe_id)
            core_customers = int(cluster_row["n_customers"]) - soft_customers
            row: dict[str, Any] = {
                "tribe_id": tribe_id,
                "n_customers": int(cluster_row["n_customers"]),
                "population_share": float(cluster_row["n_customers"] / max(total_customers, 1)),
                "profile_population": "all_assigned_customers_including_noise",
                "profile_population_customers": total_customers,
                "profile_noise_customers": noise_customers,
                "assignment_sources": source_names,
                "assignment_source_customer_counts": source_counts,
                "core_customers": core_customers,
                "soft_assigned_customers": soft_customers,
                "soft_assigned_share": float(soft_customers / max(int(cluster_row["n_customers"]), 1)),
                "soft_assignment_confidence_mean": soft_confidence[0, "soft_assignment_confidence_mean"]
                if soft_confidence.height
                else None,
                "soft_assignment_confidence_p10": soft_confidence[0, "soft_assignment_confidence_p10"]
                if soft_confidence.height
                else None,
                "soft_assignment_confidence_min": soft_confidence[0, "soft_assignment_confidence_min"]
                if soft_confidence.height
                else None,
                "top_product_ids": products["idarticu"].to_list() if products.height else [],
                "top_products": products["product_description"].to_list() if "product_description" in products.columns else [],
                "top_product_lifts": products["lift"].round(3).to_list() if products.height else [],
                "top_product_customer_counts": products["cluster_customers"].to_list() if products.height else [],
                "top_sectors": sectors["desc_sector"].to_list() if "desc_sector" in sectors.columns else [],
                "top_sector_lifts": sectors["lift"].round(3).to_list() if sectors.height else [],
                "top_themes": themes["strategic_theme"].to_list() if "strategic_theme" in themes.columns else [],
                "top_theme_lifts": themes["lift"].round(3).to_list() if themes.height else [],
                "top_theme_customer_counts": themes["cluster_customers"].to_list() if themes.height else [],
                "top_product_terms": terms["product_term"].to_list() if "product_term" in terms.columns else [],
                "top_product_term_lifts": terms["lift"].round(3).to_list() if terms.height else [],
                "top_product_term_customer_counts": terms["cluster_customers"].to_list() if terms.height else [],
            }
            if behavior_summary is not None:
                behavior_match = behavior_summary.filter(pl.col("tribe_id") == tribe_id)
                if behavior_match.height:
                    for col in behavior_match.columns:
                        if col != "tribe_id":
                            row[col] = behavior_match[0, col]
            row["suggested_tribe_name"] = fallback_tribe_name(row)
            row["profile_evidence_note"] = _profile_evidence_note(row, cfg=cfg)
            rows.append(row)

        if rows:
            profiles = pl.DataFrame(rows).sort("tribe_id")
        else:
            profiles = pl.DataFrame(
                schema={
                    "tribe_id": pl.Int32,
                    "n_customers": pl.Int64,
                    "population_share": pl.Float64,
                    "profile_population": pl.Utf8,
                    "profile_population_customers": pl.Int64,
                    "profile_noise_customers": pl.Int64,
                    "assignment_sources": pl.List(pl.Utf8),
                    "assignment_source_customer_counts": pl.List(pl.Int64),
                    "core_customers": pl.Int64,
                    "soft_assigned_customers": pl.Int64,
                    "soft_assigned_share": pl.Float64,
                    "soft_assignment_confidence_mean": pl.Float64,
                    "soft_assignment_confidence_p10": pl.Float64,
                    "soft_assignment_confidence_min": pl.Float64,
                    "top_product_ids": pl.List(pl.Utf8),
                    "top_products": pl.List(pl.Utf8),
                    "top_product_lifts": pl.List(pl.Float64),
                    "top_product_customer_counts": pl.List(pl.Int64),
                    "top_sectors": pl.List(pl.Utf8),
                    "top_sector_lifts": pl.List(pl.Float64),
                    "top_themes": pl.List(pl.Utf8),
                    "top_theme_lifts": pl.List(pl.Float64),
                    "top_theme_customer_counts": pl.List(pl.Int64),
                    "top_product_terms": pl.List(pl.Utf8),
                    "top_product_term_lifts": pl.List(pl.Float64),
                    "top_product_term_customer_counts": pl.List(pl.Int64),
                    "suggested_tribe_name": pl.Utf8,
                    "profile_evidence_note": pl.Utf8,
                }
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        profiles.write_parquet(output)
        write_artifact_metadata(output, cache_metadata)
        log_event("Stage 8 profiling", "wrote profile", cfg=cfg, tribes=profiles.height, path=output)
    return output


def _build_product_theme_map(population_product: pl.LazyFrame, allowed_themes: set[str] | None = None) -> pl.DataFrame:
    products = collect_streaming(population_product.select(["idarticu", "product_description"]).unique())
    rows = []
    for row in products.iter_rows(named=True):
        description = "" if row.get("product_description") is None else str(row.get("product_description"))
        for theme in detect_product_themes(description):
            if allowed_themes is not None and theme not in allowed_themes:
                continue
            rows.append({"idarticu": row["idarticu"], "strategic_theme": theme})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _build_product_term_map(
    population_product: pl.LazyFrame,
    allowed_terms: set[str] | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    products = collect_streaming(population_product.select(["idarticu", "product_description"]).unique())
    max_terms = int(cfg.get("profiling.max_terms_per_product", 18))
    rows = []
    for row in products.iter_rows(named=True):
        description = "" if row.get("product_description") is None else str(row.get("product_description"))
        for term in _extract_product_terms(description, max_terms=max_terms):
            if allowed_terms is not None and term not in allowed_terms:
                continue
            rows.append({"idarticu": row["idarticu"], "product_term": term})
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"idarticu": pl.Int64, "product_term": pl.Utf8})


def _extract_product_terms(description: str | None, max_terms: int = 18) -> list[str]:
    text = normalize_product_text(description)
    raw_tokens = re.findall(r"[a-z0-9]+", text)
    tokens = []
    for token in raw_tokens:
        if token in PRODUCT_TERM_STOPWORDS:
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
    for n in (2, 1):
        for start in range(0, max(len(tokens) - n + 1, 0)):
            if n == 1 and tokens[start] in PRODUCT_TERM_UNIGRAM_STOPWORDS:
                continue
            if n == 2 and tokens[start + 1] in PRODUCT_TERM_UNIGRAM_STOPWORDS:
                continue
            term = " ".join(tokens[start : start + n])
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= max_terms:
                return terms
    return terms


def _profile_evidence_note(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> str:
    product_lifts = row.get("top_product_lifts") or []
    sector_lifts = row.get("top_sector_lifts") or []
    theme_lifts = row.get("top_theme_lifts") or []
    term_lifts = row.get("top_product_term_lifts") or []
    strong_product = sum(
        1 for value in product_lifts if value and value >= float(cfg.get("profiling.strong_product_lift_threshold", 1.5))
    )
    strong_sector = sum(
        1 for value in sector_lifts if value and value >= float(cfg.get("profiling.strong_sector_lift_threshold", 1.2))
    )
    strong_theme = sum(
        1 for value in theme_lifts if value and value >= float(cfg.get("profiling.strong_theme_lift_threshold", 1.2))
    )
    strong_term = sum(
        1 for value in term_lifts if value and value >= float(cfg.get("profiling.strong_term_lift_threshold", 1.25))
    )
    return (
        f"{strong_product} strong product lifts; {strong_sector} strong sector lifts; "
        f"{strong_theme} strong theme lifts; {strong_term} strong data-driven term lifts; "
        f"{row.get('soft_assigned_share', 0.0):.1%} soft-assigned customers."
    )


def flatten_profiles_for_csv(profile_path: str | Path, output_csv: str | Path) -> Path:
    profiles = pl.read_parquet(profile_path)
    rows = []
    for row in profiles.iter_rows(named=True):
        flat = {key: _csv_safe_value(value) for key, value in row.items()}
        rows.append(flat)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(output)
    return output


def write_cluster_summary_artifacts(
    profile_path: str | Path,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write compact cluster-summary artifacts for presentation and appendix use."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    csv_output = (
        Path(output_csv)
        if output_csv
        else cfg.reports / str(cfg.get("exports.cluster_summary_template", "cluster_summary_{mode}.csv")).format(mode=cfg.mode)
    )
    write_companions = bool(cfg.get("exports.write_table_companions", False))
    md_output = Path(output_md) if output_md else (csv_output.with_suffix(".md") if write_companions else None)
    rows = []
    for row in profiles.iter_rows(named=True):
        rows.append(
            {
                "tribe_id": row.get("tribe_id"),
                "suggested_tribe_name": fallback_tribe_name(row),
                "n_customers": row.get("n_customers"),
                "population_share_pct": round(float(row.get("population_share") or 0.0) * 100.0, 2),
                "core_customers": row.get("core_customers"),
                "soft_assigned_customers": row.get("soft_assigned_customers"),
                "soft_assigned_share_pct": round(float(row.get("soft_assigned_share") or 0.0) * 100.0, 2),
                "soft_assignment_confidence_mean": row.get("soft_assignment_confidence_mean"),
                "soft_assignment_confidence_p10": row.get("soft_assignment_confidence_p10"),
                "top_themes": _csv_safe_value(row.get("top_themes") or []),
                "top_theme_lifts": _csv_safe_value(row.get("top_theme_lifts") or []),
                "top_product_terms": _csv_safe_value(row.get("top_product_terms") or []),
                "top_product_term_lifts": _csv_safe_value(row.get("top_product_term_lifts") or []),
                "top_products": _csv_safe_value((row.get("top_products") or [])[:5]),
                "top_product_lifts": _csv_safe_value((row.get("top_product_lifts") or [])[:5]),
                "top_sectors": _csv_safe_value((row.get("top_sectors") or [])[:5]),
                "profile_evidence_note": row.get("profile_evidence_note"),
            }
        )
    summary = pl.DataFrame(rows) if rows else pl.DataFrame()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(csv_output)
    paths = {"csv": csv_output}
    log_kwargs: dict[str, Path] = {"csv": csv_output}
    if md_output is not None:
        md_output.write_text(_cluster_summary_markdown(summary, cfg=cfg), encoding="utf-8")
        paths["markdown"] = md_output
        log_kwargs["markdown"] = md_output
    log_event("Stage 8 exports", "wrote cluster summary artifacts", cfg=cfg, **log_kwargs)
    return paths


def profile_overview_table(profile_path: str | Path, cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    """Return a compact, notebook-friendly explanation table for selected tribes."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    rows = [_comparison_row(row, cfg=cfg) for row in profiles.iter_rows(named=True)]
    if not rows:
        return pl.DataFrame()
    table = pl.DataFrame(rows)
    display_cols = [
        "tribe_id",
        "working_tribe_name",
        "label_confidence",
        "n_customers",
        "population_share_pct",
        "soft_assigned_share_pct",
        "top_theme_evidence",
        "top_data_driven_term_evidence",
        "top_product_evidence",
        "interpretation_note",
    ]
    return table.select([col for col in display_cols if col in table.columns])


def profile_evidence_metrics_table(profile_path: str | Path, cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    """Return numeric evidence diagnostics for each selected tribe."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    rows = []
    product_threshold = float(cfg.get("profiling.strong_product_lift_threshold", 1.5))
    sector_threshold = float(cfg.get("profiling.strong_sector_lift_threshold", 1.2))
    theme_threshold = float(cfg.get("profiling.strong_theme_lift_threshold", 1.2))
    term_threshold = float(cfg.get("profiling.strong_term_lift_threshold", 1.25))
    for row in profiles.iter_rows(named=True):
        product_lifts = [float(value) for value in (row.get("top_product_lifts") or []) if value is not None]
        sector_lifts = [float(value) for value in (row.get("top_sector_lifts") or []) if value is not None]
        theme_lifts = [float(value) for value in (row.get("top_theme_lifts") or []) if value is not None]
        term_lifts = [float(value) for value in (row.get("top_product_term_lifts") or []) if value is not None]
        rows.append(
            {
                "tribe_id": row.get("tribe_id"),
                "working_tribe_name": fallback_tribe_name(row),
                "n_customers": int(row.get("n_customers") or 0),
                "population_share_pct": round(float(row.get("population_share") or 0.0) * 100.0, 2),
                "soft_assigned_share_pct": round(float(row.get("soft_assigned_share") or 0.0) * 100.0, 2),
                "strong_product_lift_count": sum(1 for value in product_lifts if value >= product_threshold),
                "strong_sector_lift_count": sum(1 for value in sector_lifts if value >= sector_threshold),
                "strong_theme_lift_count": sum(1 for value in theme_lifts if value >= theme_threshold),
                "strong_term_lift_count": sum(1 for value in term_lifts if value >= term_threshold),
                "max_product_lift": round(max(product_lifts), 3) if product_lifts else None,
                "max_theme_lift": round(max(theme_lifts), 3) if theme_lifts else None,
                "max_term_lift": round(max(term_lifts), 3) if term_lifts else None,
                "label_confidence": _label_confidence(row, cfg=cfg),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def profile_subsegment_opportunity_table(
    profile_path: str | Path,
    *,
    max_rows: int = 60,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    """Return evidence-backed within-tribe subgroup opportunities without writing artifacts."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    definitions = _subsegment_definitions(profiles, cfg=cfg)
    if definitions.is_empty():
        return definitions
    return (
        definitions.rename(
            {
                "profile_subsegment_customers": "subsegment_customers",
                "profile_subsegment_coverage_pct": "subsegment_share_pct",
            }
        )
        .select(
            [
                "tribe_id",
                "working_tribe_name",
                "subsegment_type",
                "subsegment_label",
                "subsegment_customers",
                "subsegment_share_pct",
                "subsegment_lift",
                "subsegment_rank_in_tribe",
            ]
        )
        .sort(
            ["tribe_id", "subsegment_lift", "subsegment_customers"],
            descending=[False, True, True],
        )
        .head(max_rows)
    )


def write_tribe_comparison_artifacts(
    profile_path: str | Path,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write side-by-side tribe comparison artifacts for review and presentation."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    base_output = (
        Path(output_csv)
        if output_csv
        else cfg.reports / str(cfg.get("exports.tribe_comparison_template", "tribe_side_by_side_{mode}.csv")).format(mode=cfg.mode)
    )
    md_output = Path(output_md) if output_md else base_output.with_suffix(".md")
    html_output = Path(output_html) if output_html else base_output.with_suffix(".html")
    rows = [_comparison_row(row, cfg=cfg) for row in profiles.iter_rows(named=True)]
    comparison = pl.DataFrame(rows) if rows else pl.DataFrame()

    base_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(base_output)
    md_output.write_text(_tribe_comparison_markdown(comparison, cfg=cfg), encoding="utf-8")
    html_output.write_text(_tribe_comparison_html(comparison, cfg=cfg), encoding="utf-8")
    log_event(
        "Stage 8 exports",
        "wrote side-by-side tribe comparison",
        cfg=cfg,
        csv=base_output,
        markdown=md_output,
        html=html_output,
    )
    return {"csv": base_output, "markdown": md_output, "html": html_output}


def write_campaign_signal_artifacts(
    profile_path: str | Path,
    campaign_themes: set[str] | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write activation-oriented tribe overlays from product-theme evidence."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    campaign_set = CAMPAIGN_SIGNAL_THEMES if campaign_themes is None else campaign_themes
    base_output = (
        Path(output_csv)
        if output_csv
        else cfg.reports / f"tribe_campaign_signals_{cfg.mode}.csv"
    )
    md_output = Path(output_md) if output_md else base_output.with_suffix(".md")
    html_output = Path(output_html) if output_html else base_output.with_suffix(".html")
    rows = []
    for row in profiles.iter_rows(named=True):
        n_customers = max(int(row.get("n_customers") or 0), 1)
        themes = row.get("top_themes") or []
        lifts = row.get("top_theme_lifts") or []
        counts = row.get("top_theme_customer_counts") or []
        for rank, theme in enumerate(themes, start=1):
            if theme not in campaign_set:
                continue
            lift = float(lifts[rank - 1]) if rank - 1 < len(lifts) and lifts[rank - 1] is not None else None
            count = int(counts[rank - 1]) if rank - 1 < len(counts) and counts[rank - 1] is not None else None
            coverage = (count or 0) / n_customers
            rows.append(
                {
                    "campaign_signal": THEME_LABELS.get(str(theme), str(theme).replace("_", " ").title()),
                    "theme_key": theme,
                    "tribe_id": row.get("tribe_id"),
                    "working_tribe_name": fallback_tribe_name(row),
                    "n_customers": row.get("n_customers"),
                    "signal_customers": count,
                    "signal_coverage_pct": round(coverage * 100.0, 1),
                    "signal_lift": _round_optional(lift),
                    "theme_rank_in_tribe": rank,
                    "activation_read": _campaign_activation_read(lift, coverage),
                }
            )
    campaign = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "campaign_signal": pl.Utf8,
            "theme_key": pl.Utf8,
            "tribe_id": pl.Int64,
            "working_tribe_name": pl.Utf8,
            "n_customers": pl.Int64,
            "signal_customers": pl.Int64,
            "signal_coverage_pct": pl.Float64,
            "signal_lift": pl.Float64,
            "theme_rank_in_tribe": pl.Int64,
            "activation_read": pl.Utf8,
        }
    )
    if not campaign.is_empty():
        campaign = campaign.sort(["campaign_signal", "signal_lift", "signal_coverage_pct"], descending=[False, True, True])

    base_output.parent.mkdir(parents=True, exist_ok=True)
    campaign.write_csv(base_output)
    md_output.write_text(_campaign_signal_markdown(campaign, cfg=cfg), encoding="utf-8")
    html_output.write_text(_campaign_signal_html(campaign, cfg=cfg), encoding="utf-8")
    log_event(
        "Stage 8 exports",
        "wrote campaign signal overlays",
        cfg=cfg,
        csv=base_output,
        markdown=md_output,
        html=html_output,
    )
    return {"csv": base_output, "markdown": md_output, "html": html_output}


def write_discovered_product_term_artifacts(
    profile_path: str | Path,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write data-driven product phrase evidence surfaced from each tribe."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    base_output = Path(output_csv) if output_csv else cfg.reports / f"tribe_discovered_product_terms_{cfg.mode}.csv"
    md_output = Path(output_md) if output_md else base_output.with_suffix(".md")
    html_output = Path(output_html) if output_html else base_output.with_suffix(".html")
    rows = []
    for row in profiles.iter_rows(named=True):
        n_customers = max(int(row.get("n_customers") or 0), 1)
        terms = row.get("top_product_terms") or []
        lifts = row.get("top_product_term_lifts") or []
        counts = row.get("top_product_term_customer_counts") or []
        for rank, term in enumerate(terms, start=1):
            lift = float(lifts[rank - 1]) if rank - 1 < len(lifts) and lifts[rank - 1] is not None else None
            count = int(counts[rank - 1]) if rank - 1 < len(counts) and counts[rank - 1] is not None else None
            coverage = (count or 0) / n_customers
            rows.append(
                {
                    "tribe_id": row.get("tribe_id"),
                    "working_tribe_name": fallback_tribe_name(row),
                    "discovered_product_term": _repair_display_text(str(term)),
                    "term_lift": _round_optional(lift),
                    "term_coverage_pct": round(coverage * 100.0, 1),
                    "term_customers": count,
                    "term_rank_in_tribe": rank,
                    "evidence_read": _discovered_term_evidence_read(lift, coverage),
                }
            )
    discovered = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "tribe_id": pl.Int64,
            "working_tribe_name": pl.Utf8,
            "discovered_product_term": pl.Utf8,
            "term_lift": pl.Float64,
            "term_coverage_pct": pl.Float64,
            "term_customers": pl.Int64,
            "term_rank_in_tribe": pl.Int64,
            "evidence_read": pl.Utf8,
        }
    )

    base_output.parent.mkdir(parents=True, exist_ok=True)
    discovered.write_csv(base_output)
    md_output.write_text(_discovered_terms_markdown(discovered, cfg=cfg), encoding="utf-8")
    html_output.write_text(_discovered_terms_html(discovered, cfg=cfg), encoding="utf-8")
    log_event(
        "Stage 8 exports",
        "wrote discovered product-term evidence",
        cfg=cfg,
        csv=base_output,
        markdown=md_output,
        html=html_output,
    )
    return {"csv": base_output, "markdown": md_output, "html": html_output}


def write_customer_subsegment_artifacts(
    assignments_path: str | Path,
    profile_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    output_parquet: str | Path | None = None,
    output_summary_csv: str | Path | None = None,
    output_summary_md: str | Path | None = None,
    output_summary_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write customer-level subsegment tags inside each tribe."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    definitions = _subsegment_definitions(profiles, cfg=cfg)
    parquet_output = (
        Path(output_parquet)
        if output_parquet
        else cfg.reports / f"customer_subsegment_tags_{cfg.mode}.parquet"
    )
    summary_csv = (
        Path(output_summary_csv)
        if output_summary_csv
        else cfg.reports / f"tribe_subsegment_summary_{cfg.mode}.csv"
    )
    summary_md = Path(output_summary_md) if output_summary_md else summary_csv.with_suffix(".md")
    summary_html = Path(output_summary_html) if output_summary_html else summary_csv.with_suffix(".html")

    assignments_file = Path(assignments_path)
    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    columns = set(schema_names(lf))
    if "desc_larga_articulo" not in columns:
        customer_tags = _empty_subsegment_tags()
    elif definitions.is_empty():
        customer_tags = _empty_subsegment_tags()
    else:
        assignments = (
            pl.scan_parquet(assignments_file)
            .select(["cliente", "tribe_id"])
            .unique(subset=["cliente"], keep="first")
            .filter(pl.col("tribe_id") >= 0)
        )
        customer_product = (
            lf.select(["cliente", "idarticu", "desc_larga_articulo"])
            .join(assignments.select("cliente"), on="cliente", how="inner")
            .unique(subset=["cliente", "idarticu"])
        )
        population_product = customer_product.group_by("idarticu").agg(
            pl.col("desc_larga_articulo").first().alias("product_description")
        )

        tag_frames = []
        theme_keys = set(
            definitions.filter(pl.col("subsegment_type") == "strategic_theme")["subsegment_key"].to_list()
        )
        if theme_keys:
            product_theme_map = _build_product_theme_map(population_product, allowed_themes=theme_keys)
            if product_theme_map.height:
                tag_frames.append(
                    _customer_subsegment_facts(
                        customer_product,
                        product_theme_map.lazy().rename({"strategic_theme": "subsegment_key"}),
                        "strategic_theme",
                    )
                )

        term_keys = set(
            definitions.filter(pl.col("subsegment_type") == "data_driven_product_term")["subsegment_key"].to_list()
        )
        if term_keys:
            product_term_map = _build_product_term_map(population_product, allowed_terms=term_keys, cfg=cfg)
            if product_term_map.height:
                tag_frames.append(
                    _customer_subsegment_facts(
                        customer_product,
                        product_term_map.lazy().rename({"product_term": "subsegment_key"}),
                        "data_driven_product_term",
                    )
                )

        if tag_frames:
            facts = pl.concat(tag_frames, how="diagonal_relaxed")
            customer_tags = collect_streaming(
                facts.join(assignments, on="cliente", how="inner")
                .join(definitions.lazy(), on=["tribe_id", "subsegment_type", "subsegment_key"], how="inner")
                .select(
                    [
                        "cliente",
                        "tribe_id",
                        "working_tribe_name",
                        "subsegment_type",
                        "subsegment_key",
                        "subsegment_label",
                        "subsegment_lift",
                        "subsegment_rank_in_tribe",
                        "matching_product_count",
                    ]
                )
                .sort(["tribe_id", "subsegment_type", "subsegment_lift", "cliente"], descending=[False, False, True, False])
            )
        else:
            customer_tags = _empty_subsegment_tags()

    summary = _subsegment_summary(customer_tags, profiles)
    parquet_output.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    customer_tags.write_parquet(parquet_output)
    summary.write_csv(summary_csv)
    summary_md.write_text(_subsegment_summary_markdown(summary, cfg=cfg), encoding="utf-8")
    summary_html.write_text(_subsegment_summary_html(summary, cfg=cfg), encoding="utf-8")
    log_event(
        "Stage 8 exports",
        "wrote customer subsegment tags",
        cfg=cfg,
        parquet=parquet_output,
        summary_csv=summary_csv,
        summary_markdown=summary_md,
        summary_html=summary_html,
    )
    return {"parquet": parquet_output, "summary_csv": summary_csv, "summary_markdown": summary_md, "summary_html": summary_html}


def write_tribe_profile_report(
    profile_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Write a narrative-ready evidence report for the selected tribe profile."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    output = (
        Path(output_path)
        if output_path
        else cfg.reports / str(cfg.get("exports.profile_report_template", "tribe_profile_evidence_{mode}.md")).format(mode=cfg.mode)
    )
    total_customers = int(profiles["profile_population_customers"][0]) if profiles.height else 0
    noise_customers = int(profiles["profile_noise_customers"][0]) if profiles.height else 0
    clustered_customers = int(profiles["n_customers"].sum()) if profiles.height else 0

    lines = [
        "# Tribe Profile Evidence Report",
        "",
        f"Run mode: `{cfg.mode}`",
        f"Profile artifact: `{profile_path}`",
        "",
        "## Population Summary",
        "",
        f"- Profiled tribes: {profiles.height}",
        f"- Profiled customers: {total_customers:,}",
        f"- Assigned customers: {clustered_customers:,}",
        f"- Remaining noise customers: {noise_customers:,}",
        "",
        "## Tribe Evidence",
        "",
    ]

    for row in profiles.iter_rows(named=True):
        name = fallback_tribe_name(row) or row.get("suggested_tribe_name")
        lines.extend(
            [
                f"### Tribe {row.get('tribe_id')}: {name}",
                "",
                f"- Customers: {int(row.get('n_customers') or 0):,} ({_fmt_pct(row.get('population_share'))})",
                (
                    "- Assignment provenance: "
                    f"{int(row.get('core_customers') or 0):,} core / "
                    f"{int(row.get('soft_assigned_customers') or 0):,} soft-assigned "
                    f"({_fmt_pct(row.get('soft_assigned_share'))})"
                ),
                f"- Soft-assignment confidence: {_format_soft_assignment_confidence(row)}",
                f"- Evidence note: {row.get('profile_evidence_note') or 'n/a'}",
                f"- Top themes: {_format_lift_items(row.get('top_themes'), row.get('top_theme_lifts'), row.get('top_theme_customer_counts'))}",
                f"- Data-driven product terms: {_format_lift_items(row.get('top_product_terms'), row.get('top_product_term_lifts'), row.get('top_product_term_customer_counts'))}",
                f"- Top products: {_format_lift_items(row.get('top_products'), row.get('top_product_lifts'), row.get('top_product_customer_counts'))}",
                f"- Top sectors: {_format_lift_items(row.get('top_sectors'), row.get('top_sector_lifts'))}",
                f"- Behavioral context: {_format_behavior_context(row)}",
                "",
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    log_event("Stage 8 exports", "wrote tribe profile evidence report", cfg=cfg, path=output)
    return output


def write_clustering_atlas_artifacts(
    profile_path: str | Path,
    subsegment_summary_path: str | Path | None = None,
    figure_paths: dict[str, Any] | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write a one-stop atlas of core tribes, subtribes, evidence, and visual links."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    subsegments = _read_subsegment_summary(subsegment_summary_path)
    html_output = (
        Path(output_html)
        if output_html
        else cfg.reports / str(cfg.get("exports.clustering_atlas_template", "clustering_atlas_{mode}.html")).format(mode=cfg.mode)
    )
    md_output = Path(output_md) if output_md else html_output.with_suffix(".md")
    csv_output = Path(output_csv) if output_csv else html_output.with_suffix(".csv")
    atlas_index = _clustering_atlas_index(profiles, subsegments, cfg=cfg)

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    atlas_index.write_csv(csv_output)
    md_output.write_text(
        _clustering_atlas_markdown(
            profiles,
            atlas_index,
            subsegments,
            profile_path=Path(profile_path),
            figure_paths=figure_paths or {},
            output_path=md_output,
            cfg=cfg,
        ),
        encoding="utf-8",
    )
    html_output.write_text(
        _clustering_atlas_html(
            profiles,
            atlas_index,
            subsegments,
            profile_path=Path(profile_path),
            figure_paths=figure_paths or {},
            output_path=html_output,
            cfg=cfg,
        ),
        encoding="utf-8",
    )
    log_event(
        "Stage 8 exports",
        "wrote clustering atlas",
        cfg=cfg,
        csv=csv_output,
        markdown=md_output,
        html=html_output,
    )
    return {"csv": csv_output, "markdown": md_output, "html": html_output}


def _read_subsegment_summary(subsegment_summary_path: str | Path | None) -> pl.DataFrame:
    if subsegment_summary_path is None:
        return _empty_subsegment_summary()
    path = Path(subsegment_summary_path)
    if not path.exists():
        return _empty_subsegment_summary()
    return pl.read_csv(path)


def _empty_subsegment_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "tribe_id": pl.Int64,
            "working_tribe_name": pl.Utf8,
            "subsegment_type": pl.Utf8,
            "subsegment_label": pl.Utf8,
            "subsegment_key": pl.Utf8,
            "subsegment_customers": pl.Int64,
            "tribe_customers": pl.Int64,
            "subsegment_share_pct": pl.Float64,
            "subsegment_lift": pl.Float64,
            "avg_matching_product_count": pl.Float64,
        }
    )


def _clustering_atlas_index(
    profiles: pl.DataFrame,
    subsegments: pl.DataFrame,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    size_rank_by_tribe = {
        int(row["tribe_id"]): idx
        for idx, row in enumerate(
            profiles.sort("n_customers", descending=True).iter_rows(named=True),
            start=1,
        )
    }
    rows = []
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row.get("tribe_id"))
        tribe_subsegments = _tribe_subsegments(subsegments, tribe_id)
        largest = tribe_subsegments.head(1).row(0, named=True) if tribe_subsegments.height else {}
        share = float(row.get("population_share") or 0.0)
        rows.append(
            {
                "tribe_id": tribe_id,
                "size_rank": size_rank_by_tribe.get(tribe_id),
                "size_tier": _atlas_size_tier(size_rank_by_tribe.get(tribe_id), share),
                "working_tribe_name": fallback_tribe_name(row),
                "label_confidence": _label_confidence(row, cfg=cfg),
                "n_customers": int(row.get("n_customers") or 0),
                "population_share_pct": round(share * 100.0, 2),
                "core_customers": int(row.get("core_customers") or 0),
                "soft_assigned_customers": int(row.get("soft_assigned_customers") or 0),
                "soft_assigned_share_pct": round(float(row.get("soft_assigned_share") or 0.0) * 100.0, 2),
                "subtribe_count": tribe_subsegments.height,
                "largest_subtribe": largest.get("subsegment_label"),
                "largest_subtribe_customers": largest.get("subsegment_customers"),
                "top_theme_evidence": _format_theme_evidence(row),
                "top_data_driven_term_evidence": _format_term_evidence(row, cfg=cfg),
                "top_product_evidence": _format_lift_items(
                    row.get("top_products"),
                    row.get("top_product_lifts"),
                    row.get("top_product_customer_counts"),
                    limit=3,
                ),
                "interpretation_note": _interpretation_note(row, cfg=cfg),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _atlas_size_tier(size_rank: int | None, population_share: float) -> str:
    if size_rank is not None and size_rank <= 3:
        return "large core tribe"
    if population_share >= 0.05:
        return "core tribe"
    return "niche core tribe"


def _tribe_subsegments(subsegments: pl.DataFrame, tribe_id: int, limit: int | None = None) -> pl.DataFrame:
    if subsegments.is_empty():
        return subsegments
    result = (
        subsegments.filter(pl.col("tribe_id") == tribe_id)
        .sort(["subsegment_customers", "subsegment_lift"], descending=[True, True])
    )
    return result.head(limit) if limit else result


def _clustering_atlas_markdown(
    profiles: pl.DataFrame,
    atlas_index: pl.DataFrame,
    subsegments: pl.DataFrame,
    *,
    profile_path: Path,
    figure_paths: dict[str, Any],
    output_path: Path,
    cfg: PipelineConfig = CONFIG,
) -> str:
    total_customers = int(profiles["profile_population_customers"][0]) if profiles.height else 0
    assigned_customers = int(profiles["n_customers"].sum()) if profiles.height else 0
    subtribe_count = int(atlas_index["subtribe_count"].sum()) if atlas_index.height else 0
    lines = [
        "# Clustering Atlas",
        "",
        f"Run mode: `{cfg.mode}`",
        f"Profile artifact: `{profile_path}`",
        "",
        "This atlas connects the main/core tribes to their evidence-backed subtribes. "
        "Subtribes are activation overlays derived from lifted product themes and discovered product terms.",
        "",
        "Use all labels as purchase-behavior evidence, not as demographic, religious, household, or identity truth.",
        "",
        "## Discovery Progression",
        "",
        "| Layer | What it means | How to read it |",
        "|---|---|---|",
        "| 1. Organic core tribes | Dense product-purchase groups found by the clustering model. | Highest-confidence evidence that the tribe exists. |",
        "| 2. Confidence-scored soft assignment | Nearby non-core customers attached for operational coverage. | Useful for campaigns, but always review assignment source and confidence. |",
        "| 3. Evidence-backed subtribes | Within-tribe overlays from product/theme/term lift. | Campaign and loyalty-program hypotheses, not demographic identity claims. |",
        "",
        "## Run Snapshot",
        "",
        f"- Core tribes: {profiles.height}",
        f"- Profiled customers: {total_customers:,}",
        f"- Assigned customers: {assigned_customers:,}",
        f"- Subtribes surfaced: {subtribe_count:,}",
        "",
        "## Visual Outputs",
        "",
    ]
    visual_links = _atlas_visual_links_markdown(figure_paths, output_path.parent)
    lines.extend(visual_links if visual_links else ["No visualization links were provided."])
    lines.extend(["", "## Core Tribe Index", ""])
    if atlas_index.is_empty():
        lines.append("No atlas rows were produced.")
        return "\n".join(lines) + "\n"
    display_cols = [
        "tribe_id",
        "size_tier",
        "working_tribe_name",
        "label_confidence",
        "n_customers",
        "population_share_pct",
        "soft_assigned_share_pct",
        "subtribe_count",
        "largest_subtribe",
        "top_theme_evidence",
        "top_data_driven_term_evidence",
        "interpretation_note",
    ]
    lines.append(_markdown_table(atlas_index.select([col for col in display_cols if col in atlas_index.columns])))
    lines.extend(["", "## Tribe Pages", ""])
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row.get("tribe_id"))
        name = fallback_tribe_name(row)
        tribe_subsegments = _tribe_subsegments(subsegments, tribe_id, limit=12)
        lines.extend(
            [
                f"### Tribe {tribe_id}: {name}",
                "",
                f"- Customers: {int(row.get('n_customers') or 0):,} ({_fmt_pct(row.get('population_share'))})",
                f"- Assignment: {int(row.get('core_customers') or 0):,} core / {int(row.get('soft_assigned_customers') or 0):,} soft-assigned ({_fmt_pct(row.get('soft_assigned_share'))})",
                f"- Label confidence: {_label_confidence(row, cfg=cfg)}",
                f"- Evidence note: {row.get('profile_evidence_note') or 'n/a'}",
                f"- Themes: {_format_lift_items(row.get('top_themes'), row.get('top_theme_lifts'), row.get('top_theme_customer_counts'))}",
                f"- Data-driven terms: {_format_term_evidence(row, cfg=cfg)}",
                f"- Products: {_format_lift_items(row.get('top_products'), row.get('top_product_lifts'), row.get('top_product_customer_counts'), limit=5)}",
                f"- Sectors: {_format_lift_items(row.get('top_sectors'), row.get('top_sector_lifts'), limit=4)}",
                "",
            ]
        )
        if tribe_subsegments.height:
            lines.append(_markdown_table(tribe_subsegments.select(_atlas_subsegment_display_cols(tribe_subsegments))))
            lines.append("")
        else:
            lines.extend(["No subtribes passed the evidence thresholds for this tribe.", ""])
    return "\n".join(lines) + "\n"


def _clustering_atlas_html(
    profiles: pl.DataFrame,
    atlas_index: pl.DataFrame,
    subsegments: pl.DataFrame,
    *,
    profile_path: Path,
    figure_paths: dict[str, Any],
    output_path: Path,
    cfg: PipelineConfig = CONFIG,
) -> str:
    total_customers = int(profiles["profile_population_customers"][0]) if profiles.height else 0
    assigned_customers = int(profiles["n_customers"].sum()) if profiles.height else 0
    subtribe_count = int(atlas_index["subtribe_count"].sum()) if atlas_index.height else 0
    index_rows = "\n".join(_atlas_index_html_row(row) for row in atlas_index.iter_rows(named=True))
    tribe_cards = "\n".join(
        _atlas_tribe_card_html(row, _tribe_subsegments(subsegments, int(row.get("tribe_id")), limit=12), cfg=cfg)
        for row in profiles.iter_rows(named=True)
    )
    visual_cards = _atlas_visual_cards_html(figure_paths, output_path.parent)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Clustering Atlas</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #17202a; background: #f7f8fa; }}
header {{ background: #ffffff; border-bottom: 1px solid #d0d5dd; padding: 24px 32px; }}
main {{ padding: 24px 32px 40px; }}
h1, h2, h3 {{ margin: 0; }}
h2 {{ margin-top: 26px; margin-bottom: 12px; }}
h3 {{ margin-bottom: 8px; }}
.muted {{ color: #667085; font-size: 13px; }}
.snapshot {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin-top: 18px; }}
.metric {{ background: #ffffff; border: 1px solid #d0d5dd; border-radius: 8px; padding: 14px; }}
.metric strong {{ display: block; font-size: 22px; margin-top: 4px; }}
.visual-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
.visual {{ background: #ffffff; border: 1px solid #d0d5dd; border-radius: 8px; padding: 12px; }}
.visual img {{ width: 100%; max-height: 320px; object-fit: contain; display: block; margin-top: 8px; background: #ffffff; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; background: #ffffff; }}
th, td {{ border: 1px solid #d0d5dd; padding: 9px; vertical-align: top; font-size: 13px; }}
th {{ background: #eef2f6; text-align: left; }}
.tribe-card {{ background: #ffffff; border: 1px solid #d0d5dd; border-radius: 8px; padding: 16px; margin-bottom: 14px; }}
.tribe-meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px; }}
.pill {{ background: #eef2f6; border: 1px solid #d0d5dd; border-radius: 999px; padding: 4px 8px; font-size: 12px; }}
.evidence {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 10px 0 14px; }}
.evidence div {{ background: #fafafa; border: 1px solid #e4e7ec; border-radius: 6px; padding: 10px; font-size: 13px; }}
.subtribes {{ margin-top: 8px; }}
a {{ color: #175cd3; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
<h1>Clustering Atlas</h1>
<p class="muted">Run mode: {escape(str(cfg.mode))}. Profile artifact: {escape(str(profile_path))}</p>
<p class="muted">Labels and subtribes are purchase-evidence summaries, not demographic, religious, household, or identity claims.</p>
<div class="snapshot">
<div class="metric">Core tribes<strong>{profiles.height}</strong></div>
<div class="metric">Profiled customers<strong>{total_customers:,}</strong></div>
<div class="metric">Assigned customers<strong>{assigned_customers:,}</strong></div>
<div class="metric">Subtribes surfaced<strong>{subtribe_count:,}</strong></div>
</div>
</header>
<main>
<h2>Visual Outputs</h2>
<div class="visual-grid">{visual_cards or "<div class='visual muted'>No visualization links were provided.</div>"}</div>
<h2>Discovery Progression</h2>
<table>
<thead>
<tr>
<th>Layer</th>
<th>What it means</th>
<th>How to read it</th>
</tr>
</thead>
<tbody>
<tr>
<td>1. Organic core tribes</td>
<td>Dense product-purchase groups found by the clustering model.</td>
<td>Highest-confidence evidence that the tribe exists.</td>
</tr>
<tr>
<td>2. Confidence-scored soft assignment</td>
<td>Nearby non-core customers attached for operational coverage.</td>
<td>Useful for campaigns, but always review assignment source and confidence.</td>
</tr>
<tr>
<td>3. Evidence-backed subtribes</td>
<td>Within-tribe overlays from product, theme, sector, and term lift.</td>
<td>Campaign and loyalty-program hypotheses, not demographic identity claims.</td>
</tr>
</tbody>
</table>
<h2>Core Tribe Index</h2>
<table>
<thead>
<tr>
<th>Tribe</th>
<th>Tier</th>
<th>Working Label</th>
<th>Customers</th>
<th>Subtribes</th>
<th>Evidence Read</th>
</tr>
</thead>
<tbody>
{index_rows or "<tr><td colspan='6'>No atlas rows were produced.</td></tr>"}
</tbody>
</table>
<h2>Tribe Pages</h2>
{tribe_cards or "<p class='muted'>No tribe pages were produced.</p>"}
</main>
</body>
</html>
"""


def _atlas_visual_links_markdown(figure_paths: dict[str, Any], base_dir: Path) -> list[str]:
    links = []
    for label, path in _flatten_figure_paths(figure_paths):
        links.append(f"- {label}: [{Path(path).name}]({_artifact_href(path, base_dir)})")
    return links


def _atlas_visual_cards_html(figure_paths: dict[str, Any], base_dir: Path) -> str:
    cards = []
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    for label, path in _flatten_figure_paths(figure_paths):
        href = _artifact_href(path, base_dir)
        title = escape(label)
        suffix = Path(path).suffix.lower()
        image = f"<img src='{escape(href)}' alt='{title}'>" if suffix in image_suffixes else ""
        cards.append(
            "<div class='visual'>"
            f"<strong><a href='{escape(href)}'>{title}</a></strong>"
            f"<div class='muted'>{escape(Path(path).name)}</div>"
            f"{image}"
            "</div>"
        )
    return "\n".join(cards)


def _flatten_figure_paths(figure_paths: Any, prefix: str = "") -> list[tuple[str, Path]]:
    if not figure_paths:
        return []
    if isinstance(figure_paths, dict):
        flattened: list[tuple[str, Path]] = []
        for key, value in figure_paths.items():
            label = f"{prefix} {key}".strip().replace("_", " ").title()
            flattened.extend(_flatten_figure_paths(value, label))
        return flattened
    if isinstance(figure_paths, (list, tuple, set)):
        flattened = []
        for idx, value in enumerate(figure_paths, start=1):
            label = f"{prefix} {idx}".strip()
            flattened.extend(_flatten_figure_paths(value, label))
        return flattened
    if isinstance(figure_paths, (str, Path)):
        return [(prefix or Path(figure_paths).stem.replace("_", " ").title(), Path(figure_paths))]
    return []


def _artifact_href(path: str | Path, base_dir: Path) -> str:
    try:
        return Path(os.path.relpath(Path(path), base_dir)).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _atlas_index_html_row(row: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>T{escape(str(row.get('tribe_id')))}</td>"
        f"<td>{escape(str(row.get('size_tier') or ''))}</td>"
        f"<td><strong>{escape(str(row.get('working_tribe_name') or ''))}</strong><br>"
        f"<span class='muted'>{escape(str(row.get('label_confidence') or ''))} confidence</span></td>"
        f"<td>{int(row.get('n_customers') or 0):,}<br>"
        f"<span class='muted'>{float(row.get('population_share_pct') or 0):.2f}% of population</span></td>"
        f"<td>{int(row.get('subtribe_count') or 0)}<br>"
        f"<span class='muted'>{escape(str(row.get('largest_subtribe') or 'n/a'))}</span></td>"
        f"<td>{escape(str(row.get('interpretation_note') or ''))}</td>"
        "</tr>"
    )


def _atlas_tribe_card_html(row: dict[str, Any], subsegments: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    tribe_id = int(row.get("tribe_id"))
    name = fallback_tribe_name(row)
    subsegment_rows = "\n".join(_atlas_subsegment_html_row(item) for item in subsegments.iter_rows(named=True))
    if not subsegment_rows:
        subsegment_rows = '<tr><td colspan="5">No subtribes passed the evidence thresholds.</td></tr>'
    subsegment_table = (
        "<table class='subtribes'><thead><tr>"
        "<th>Subtribe</th><th>Type</th><th>Customers</th><th>Lift</th><th>Avg Matching Products</th>"
        "</tr></thead><tbody>"
        f"{subsegment_rows}"
        "</tbody></table>"
    )
    return f"""<section class="tribe-card" id="tribe-{tribe_id}">
<h3>Tribe {tribe_id}: {escape(name)}</h3>
<div class="tribe-meta">
<span class="pill">{int(row.get('n_customers') or 0):,} customers</span>
<span class="pill">{_fmt_pct(row.get('population_share'))} of population</span>
<span class="pill">{escape(_label_confidence(row, cfg=cfg))} label confidence</span>
<span class="pill">{int(row.get('core_customers') or 0):,} core / {int(row.get('soft_assigned_customers') or 0):,} soft-assigned</span>
</div>
<div class="evidence">
<div><strong>Themes</strong><br>{escape(_format_lift_items(row.get('top_themes'), row.get('top_theme_lifts'), row.get('top_theme_customer_counts')))}</div>
<div><strong>Data-driven terms</strong><br>{escape(_format_term_evidence(row, cfg=cfg))}</div>
<div><strong>Top products</strong><br>{escape(_format_lift_items(row.get('top_products'), row.get('top_product_lifts'), row.get('top_product_customer_counts'), limit=5))}</div>
<div><strong>Interpretation</strong><br>{escape(_interpretation_note(row, cfg=cfg))}</div>
</div>
{subsegment_table}
</section>"""


def _atlas_subsegment_display_cols(subsegments: pl.DataFrame) -> list[str]:
    preferred = [
        "subsegment_type",
        "subsegment_label",
        "subsegment_customers",
        "subsegment_share_pct",
        "subsegment_lift",
        "avg_matching_product_count",
    ]
    return [col for col in preferred if col in subsegments.columns]


def _atlas_subsegment_html_row(row: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><strong>{escape(str(row.get('subsegment_label') or ''))}</strong><br>"
        f"<span class='muted'>{escape(str(row.get('subsegment_key') or ''))}</span></td>"
        f"<td>{escape(str(row.get('subsegment_type') or ''))}</td>"
        f"<td>{int(row.get('subsegment_customers') or 0):,}<br>"
        f"<span class='muted'>{float(row.get('subsegment_share_pct') or 0):.1f}% of tribe</span></td>"
        f"<td>{_fmt_number(row.get('subsegment_lift'))}x</td>"
        f"<td>{_fmt_number(row.get('avg_matching_product_count'))}</td>"
        "</tr>"
    )


def _cluster_summary_markdown(summary: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    lines = [
        "# Cluster Summary",
        "",
        f"Run mode: `{cfg.mode}`",
        "",
    ]
    if summary.is_empty():
        lines.append("No cluster summary rows were produced.")
        return "\n".join(lines) + "\n"

    display_cols = [
        "tribe_id",
        "suggested_tribe_name",
        "n_customers",
        "population_share_pct",
        "soft_assigned_share_pct",
        "top_themes",
        "top_product_terms",
        "top_products",
        "profile_evidence_note",
    ]
    cols = [col for col in display_cols if col in summary.columns]
    lines.append(_markdown_table(summary.select(cols)))
    return "\n".join(lines) + "\n"


def _campaign_signal_markdown(campaign: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    lines = [
        "# Tribe Campaign Signal Overlays",
        "",
        f"Run mode: `{cfg.mode}`",
        "",
        "These are purchase-evidence overlays for activation planning. "
        "They should not be interpreted as demographic, religious, or household-status truth about a customer.",
        "",
    ]
    if campaign.is_empty():
        lines.append("No campaign signal overlays were found in the stored top themes.")
        return "\n".join(lines) + "\n"
    display_cols = [
        "campaign_signal",
        "tribe_id",
        "working_tribe_name",
        "signal_lift",
        "signal_coverage_pct",
        "signal_customers",
        "activation_read",
    ]
    lines.append(_markdown_table(campaign.select(display_cols)))
    return "\n".join(lines) + "\n"


def _campaign_signal_html(campaign: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    rows = []
    if not campaign.is_empty():
        for row in campaign.iter_rows(named=True):
            rows.append(
                "<tr>"
                f"<td><strong>{escape(str(row.get('campaign_signal') or ''))}</strong><br>"
                f"<span class='muted'>{escape(str(row.get('theme_key') or ''))}</span></td>"
                f"<td>T{escape(str(row.get('tribe_id')))}</td>"
                f"<td>{escape(str(row.get('working_tribe_name') or ''))}</td>"
                f"<td>{_fmt_number(row.get('signal_lift'))}x</td>"
                f"<td>{float(row.get('signal_coverage_pct') or 0):.1f}%<br>"
                f"<span class='muted'>{int(row.get('signal_customers') or 0):,} customers</span></td>"
                f"<td>{escape(str(row.get('activation_read') or ''))}</td>"
                "</tr>"
            )
    body = "\n".join(rows) or "<tr><td colspan='6'>No campaign signal overlays were found.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tribe Campaign Signal Overlays</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
.muted {{ color: #667085; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
th, td {{ border: 1px solid #d0d5dd; padding: 10px; vertical-align: top; font-size: 13px; }}
th {{ background: #f2f4f7; position: sticky; top: 0; z-index: 1; }}
tr:nth-child(even) {{ background: #fbfcfd; }}
</style>
</head>
<body>
<h1>Tribe Campaign Signal Overlays</h1>
<p class="muted">Run mode: {escape(str(cfg.mode))}. Signals are based on product purchase evidence and should be used as activation overlays, not demographic labels.</p>
<table>
<thead>
<tr>
<th>Signal</th>
<th>Tribe</th>
<th>Working Label</th>
<th>Lift</th>
<th>Coverage</th>
<th>Activation Read</th>
</tr>
</thead>
<tbody>
{body}
</tbody>
</table>
</body>
</html>
"""


def _campaign_activation_read(lift: float | None, coverage: float) -> str:
    if lift is None:
        return "Evidence available but lift is missing; inspect product examples before activation."
    if lift >= 1.5 and coverage >= 0.15:
        return "Core activation opportunity; strong enough to inform tribe narrative and campaign planning."
    if lift >= 1.2 and coverage >= 0.05:
        return "Useful overlay; target as a subsegment within the tribe rather than renaming the tribe."
    if lift >= 1.2:
        return "Niche overlay; interesting but small, best used with product-level targeting."
    return "Weak overlay; keep as supporting context only."


def _discovered_terms_markdown(discovered: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    lines = [
        "# Data-Driven Product-Term Evidence",
        "",
        f"Run mode: `{cfg.mode}`",
        "",
        "These phrases are discovered from product descriptions and ranked by customer-level lift inside each tribe. "
        "They are raw evidence for interpretation, not final tribe names by themselves.",
        "",
    ]
    if discovered.is_empty():
        lines.append("No data-driven product terms were found with the configured evidence thresholds.")
        return "\n".join(lines) + "\n"
    display_cols = [
        "tribe_id",
        "working_tribe_name",
        "discovered_product_term",
        "term_lift",
        "term_coverage_pct",
        "term_customers",
        "evidence_read",
    ]
    lines.append(_markdown_table(discovered.select(display_cols)))
    return "\n".join(lines) + "\n"


def _discovered_terms_html(discovered: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    rows = []
    if not discovered.is_empty():
        for row in discovered.iter_rows(named=True):
            rows.append(
                "<tr>"
                f"<td>T{escape(str(row.get('tribe_id')))}</td>"
                f"<td>{escape(str(row.get('working_tribe_name') or ''))}</td>"
                f"<td><strong>{escape(str(row.get('discovered_product_term') or ''))}</strong></td>"
                f"<td>{_fmt_number(row.get('term_lift'))}x</td>"
                f"<td>{float(row.get('term_coverage_pct') or 0):.1f}%<br>"
                f"<span class='muted'>{int(row.get('term_customers') or 0):,} customers</span></td>"
                f"<td>{escape(str(row.get('evidence_read') or ''))}</td>"
                "</tr>"
            )
    body = "\n".join(rows) or "<tr><td colspan='6'>No data-driven product terms were found.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Data-Driven Product-Term Evidence</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
.muted {{ color: #667085; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
th, td {{ border: 1px solid #d0d5dd; padding: 10px; vertical-align: top; font-size: 13px; }}
th {{ background: #f2f4f7; position: sticky; top: 0; z-index: 1; }}
tr:nth-child(even) {{ background: #fbfcfd; }}
</style>
</head>
<body>
<h1>Data-Driven Product-Term Evidence</h1>
<p class="muted">Run mode: {escape(str(cfg.mode))}. Terms are discovered from product descriptions and customer-level lift.</p>
<table>
<thead>
<tr>
<th>Tribe</th>
<th>Working Label</th>
<th>Discovered Term</th>
<th>Lift</th>
<th>Coverage</th>
<th>Evidence Read</th>
</tr>
</thead>
<tbody>
{body}
</tbody>
</table>
</body>
</html>
"""


def _discovered_term_evidence_read(lift: float | None, coverage: float) -> str:
    if lift is None:
        return "Evidence available but lift is missing; inspect product examples before using externally."
    if lift >= 1.5 and coverage >= 0.15:
        return "Strong raw phrase evidence; useful for naming or presentation support."
    if lift >= 1.25 and coverage >= 0.08:
        return "Moderate raw phrase evidence; useful as supporting proof."
    if lift >= 1.25:
        return "Niche raw phrase evidence; validate with top products before using in a campaign."
    return "Weak raw phrase evidence; keep as audit context only."


def _subsegment_definitions(profiles: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    min_coverage = float(cfg.get("profiling.min_subsegment_coverage", 0.01))
    theme_threshold = float(cfg.get("profiling.strong_theme_lift_threshold", 1.2))
    term_threshold = float(cfg.get("profiling.strong_term_lift_threshold", 1.25))
    rows = []
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row.get("tribe_id"))
        n_customers = max(int(row.get("n_customers") or 0), 1)
        name = fallback_tribe_name(row)
        _append_subsegment_definition_rows(
            rows,
            tribe_id=tribe_id,
            working_tribe_name=name,
            n_customers=n_customers,
            subsegment_type="strategic_theme",
            keys=row.get("top_themes") or [],
            labels=[THEME_LABELS.get(str(theme), str(theme).replace("_", " ").title()) for theme in (row.get("top_themes") or [])],
            lifts=row.get("top_theme_lifts") or [],
            counts=row.get("top_theme_customer_counts") or [],
            min_lift=theme_threshold,
            min_coverage=min_coverage,
        )
        _append_subsegment_definition_rows(
            rows,
            tribe_id=tribe_id,
            working_tribe_name=name,
            n_customers=n_customers,
            subsegment_type="data_driven_product_term",
            keys=row.get("top_product_terms") or [],
            labels=[_repair_display_text(str(term)) for term in (row.get("top_product_terms") or [])],
            lifts=row.get("top_product_term_lifts") or [],
            counts=row.get("top_product_term_customer_counts") or [],
            min_lift=term_threshold,
            min_coverage=min_coverage,
        )
    return pl.DataFrame(rows) if rows else _empty_subsegment_definitions()


def _append_subsegment_definition_rows(
    rows: list[dict[str, Any]],
    *,
    tribe_id: int,
    working_tribe_name: str,
    n_customers: int,
    subsegment_type: str,
    keys: list[Any],
    labels: list[str],
    lifts: list[Any],
    counts: list[Any],
    min_lift: float,
    min_coverage: float,
) -> None:
    for idx, key in enumerate(keys):
        lift = float(lifts[idx]) if idx < len(lifts) and lifts[idx] is not None else None
        count = int(counts[idx]) if idx < len(counts) and counts[idx] is not None else 0
        coverage = count / max(n_customers, 1)
        if lift is None or lift < min_lift or coverage < min_coverage:
            continue
        rows.append(
            {
                "tribe_id": tribe_id,
                "working_tribe_name": working_tribe_name,
                "subsegment_type": subsegment_type,
                "subsegment_key": str(key),
                "subsegment_label": labels[idx] if idx < len(labels) else str(key),
                "subsegment_lift": round(lift, 3),
                "profile_subsegment_customers": count,
                "profile_subsegment_coverage_pct": round(coverage * 100.0, 1),
                "subsegment_rank_in_tribe": idx + 1,
            }
        )


def _customer_subsegment_facts(
    customer_product: pl.LazyFrame,
    product_subsegment_map: pl.LazyFrame,
    subsegment_type: str,
) -> pl.LazyFrame:
    return (
        customer_product.join(product_subsegment_map, on="idarticu", how="inner")
        .group_by(["cliente", "subsegment_key"])
        .agg(pl.col("idarticu").n_unique().alias("matching_product_count"))
        .with_columns(pl.lit(subsegment_type).alias("subsegment_type"))
    )


def _empty_subsegment_definitions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "tribe_id": pl.Int64,
            "working_tribe_name": pl.Utf8,
            "subsegment_type": pl.Utf8,
            "subsegment_key": pl.Utf8,
            "subsegment_label": pl.Utf8,
            "subsegment_lift": pl.Float64,
            "profile_subsegment_customers": pl.Int64,
            "profile_subsegment_coverage_pct": pl.Float64,
            "subsegment_rank_in_tribe": pl.Int64,
        }
    )


def _empty_subsegment_tags() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "cliente": pl.Utf8,
            "tribe_id": pl.Int64,
            "working_tribe_name": pl.Utf8,
            "subsegment_type": pl.Utf8,
            "subsegment_key": pl.Utf8,
            "subsegment_label": pl.Utf8,
            "subsegment_lift": pl.Float64,
            "subsegment_rank_in_tribe": pl.Int64,
            "matching_product_count": pl.Int64,
        }
    )


def _subsegment_summary(customer_tags: pl.DataFrame, profiles: pl.DataFrame) -> pl.DataFrame:
    if customer_tags.is_empty():
        return pl.DataFrame(
            schema={
                "tribe_id": pl.Int64,
                "working_tribe_name": pl.Utf8,
                "subsegment_type": pl.Utf8,
                "subsegment_label": pl.Utf8,
                "subsegment_key": pl.Utf8,
                "subsegment_customers": pl.Int64,
                "tribe_customers": pl.Int64,
                "subsegment_share_pct": pl.Float64,
                "subsegment_lift": pl.Float64,
                "avg_matching_product_count": pl.Float64,
            }
        )
    tribe_sizes = profiles.select(["tribe_id", "n_customers"]).rename({"n_customers": "tribe_customers"})
    return (
        customer_tags.group_by(
            [
                "tribe_id",
                "working_tribe_name",
                "subsegment_type",
                "subsegment_label",
                "subsegment_key",
                "subsegment_lift",
            ]
        )
        .agg(
            [
                pl.col("cliente").n_unique().alias("subsegment_customers"),
                pl.col("matching_product_count").mean().round(2).alias("avg_matching_product_count"),
            ]
        )
        .join(tribe_sizes, on="tribe_id", how="left")
        .with_columns(
            (pl.col("subsegment_customers") / pl.col("tribe_customers") * 100.0)
            .round(1)
            .alias("subsegment_share_pct")
        )
        .sort(["tribe_id", "subsegment_type", "subsegment_lift", "subsegment_customers"], descending=[False, False, True, True])
    )


def _subsegment_summary_markdown(summary: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    lines = [
        "# Customer Subsegment Summary",
        "",
        f"Run mode: `{cfg.mode}`",
        "",
        "Subsegments are customer-level tags inside a tribe. They are generated only when a lifted strategic theme or discovered product term is both distinctive and present for that customer.",
        "Treat them as purchase evidence, not as demographic, religious, household, or identity truth.",
        "",
    ]
    if summary.is_empty():
        lines.append("No customer subsegment tags were produced with the configured evidence thresholds.")
        return "\n".join(lines) + "\n"
    display_cols = [
        "tribe_id",
        "working_tribe_name",
        "subsegment_type",
        "subsegment_label",
        "subsegment_customers",
        "subsegment_share_pct",
        "subsegment_lift",
        "avg_matching_product_count",
    ]
    lines.append(_markdown_table(summary.select(display_cols)))
    return "\n".join(lines) + "\n"


def _subsegment_summary_html(summary: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    rows = []
    if not summary.is_empty():
        for row in summary.iter_rows(named=True):
            rows.append(
                "<tr>"
                f"<td>T{escape(str(row.get('tribe_id')))}</td>"
                f"<td>{escape(str(row.get('working_tribe_name') or ''))}</td>"
                f"<td>{escape(str(row.get('subsegment_type') or ''))}</td>"
                f"<td><strong>{escape(str(row.get('subsegment_label') or ''))}</strong><br>"
                f"<span class='muted'>{escape(str(row.get('subsegment_key') or ''))}</span></td>"
                f"<td>{int(row.get('subsegment_customers') or 0):,}<br>"
                f"<span class='muted'>{float(row.get('subsegment_share_pct') or 0):.1f}% of tribe</span></td>"
                f"<td>{_fmt_number(row.get('subsegment_lift'))}x</td>"
                f"<td>{_fmt_number(row.get('avg_matching_product_count'))}</td>"
                "</tr>"
            )
    body = "\n".join(rows) or "<tr><td colspan='7'>No customer subsegment tags were produced.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Customer Subsegment Summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
.muted {{ color: #667085; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
th, td {{ border: 1px solid #d0d5dd; padding: 10px; vertical-align: top; font-size: 13px; }}
th {{ background: #f2f4f7; position: sticky; top: 0; z-index: 1; }}
tr:nth-child(even) {{ background: #fbfcfd; }}
</style>
</head>
<body>
<h1>Customer Subsegment Summary</h1>
<p class="muted">Run mode: {escape(str(cfg.mode))}. Customer-level parquet tags can be used for activation after business review. They indicate purchase evidence, not demographic, religious, household, or identity truth.</p>
<table>
<thead>
<tr>
<th>Tribe</th>
<th>Working Label</th>
<th>Type</th>
<th>Subsegment</th>
<th>Customers</th>
<th>Lift</th>
<th>Avg Matching Products</th>
</tr>
</thead>
<tbody>
{body}
</tbody>
</table>
</body>
</html>
"""


def _comparison_row(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    name = fallback_tribe_name(row)
    return {
        "tribe_id": row.get("tribe_id"),
        "working_tribe_name": name,
        "label_confidence": _label_confidence(row, cfg=cfg),
        "n_customers": row.get("n_customers"),
        "population_share_pct": round(float(row.get("population_share") or 0.0) * 100.0, 2),
        "core_customers": row.get("core_customers"),
        "soft_assigned_customers": row.get("soft_assigned_customers"),
        "soft_assigned_share_pct": round(float(row.get("soft_assigned_share") or 0.0) * 100.0, 2),
        "soft_assignment_confidence_mean": _round_optional(row.get("soft_assignment_confidence_mean")),
        "top_theme_evidence": _format_theme_evidence(row),
        "top_data_driven_term_evidence": _format_term_evidence(row, cfg=cfg),
        "top_sector_evidence": _format_lift_items(row.get("top_sectors"), row.get("top_sector_lifts"), limit=3),
        "top_product_evidence": _format_lift_items(row.get("top_products"), row.get("top_product_lifts"), row.get("top_product_customer_counts"), limit=4),
        "behavior_context": _format_behavior_context(row),
        "interpretation_note": _interpretation_note(row, cfg=cfg),
    }


def _tribe_comparison_markdown(comparison: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    lines = [
        "# Side-by-Side Tribe Comparison",
        "",
        f"Run mode: `{cfg.mode}`",
        "",
        "Use `working_tribe_name` as a review label, not as an immutable truth. "
        "The confidence column flags whether the label is strongly supported by product-theme/sector evidence and assignment provenance.",
        "",
    ]
    if comparison.is_empty():
        lines.append("No tribe comparison rows were produced.")
        return "\n".join(lines) + "\n"
    display_cols = [
        "tribe_id",
        "working_tribe_name",
        "label_confidence",
        "n_customers",
        "population_share_pct",
        "soft_assigned_share_pct",
        "top_theme_evidence",
        "top_data_driven_term_evidence",
        "top_sector_evidence",
        "top_product_evidence",
        "interpretation_note",
    ]
    lines.append(_markdown_table(comparison.select([col for col in display_cols if col in comparison.columns])))
    return "\n".join(lines) + "\n"


def _tribe_comparison_html(comparison: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    rows = []
    if not comparison.is_empty():
        for row in comparison.iter_rows(named=True):
            rows.append(
                "<tr>"
                f"<td class='tribe'>T{escape(str(row.get('tribe_id')))}</td>"
                f"<td><strong>{escape(str(row.get('working_tribe_name') or ''))}</strong><br>"
                f"<span class='muted'>{escape(str(row.get('label_confidence') or ''))} label confidence</span></td>"
                f"<td>{int(row.get('n_customers') or 0):,}<br><span class='muted'>{float(row.get('population_share_pct') or 0):.2f}%</span></td>"
                f"<td>{float(row.get('soft_assigned_share_pct') or 0):.1f}%<br><span class='muted'>mean confidence {_fmt_number(row.get('soft_assignment_confidence_mean'))}</span></td>"
                f"<td>{escape(str(row.get('top_theme_evidence') or ''))}</td>"
                f"<td>{escape(str(row.get('top_data_driven_term_evidence') or ''))}</td>"
                f"<td>{escape(str(row.get('top_sector_evidence') or ''))}</td>"
                f"<td>{escape(str(row.get('top_product_evidence') or ''))}</td>"
                f"<td>{escape(str(row.get('interpretation_note') or ''))}</td>"
                "</tr>"
            )
    body = "\n".join(rows) or "<tr><td colspan='9'>No tribe comparison rows were produced.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Side-by-Side Tribe Comparison</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
h1 {{ margin-bottom: 4px; }}
.muted {{ color: #667085; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
th, td {{ border: 1px solid #d0d5dd; padding: 10px; vertical-align: top; font-size: 13px; }}
th {{ background: #f2f4f7; position: sticky; top: 0; z-index: 1; }}
tr:nth-child(even) {{ background: #fbfcfd; }}
.tribe {{ font-weight: 700; width: 48px; }}
</style>
</head>
<body>
<h1>Side-by-Side Tribe Comparison</h1>
<p class="muted">Run mode: {escape(str(cfg.mode))}. Working names are deterministic labels derived from product, theme, sector, and assignment evidence.</p>
<table>
<thead>
<tr>
<th>Tribe</th>
<th>Working Label</th>
<th>Customers</th>
<th>Soft Assignment</th>
<th>Theme Evidence</th>
<th>Data-Driven Terms</th>
<th>Sector Evidence</th>
<th>Product Evidence</th>
<th>Interpretation</th>
</tr>
</thead>
<tbody>
{body}
</tbody>
</table>
</body>
</html>
"""


def _markdown_table(df: pl.DataFrame) -> str:
    columns = df.columns
    header = "| " + " | ".join(_escape_markdown(col) for col in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for row in df.iter_rows(named=True):
        rows.append("| " + " | ".join(_escape_markdown(_format_markdown_value(row.get(col))) for col in columns) + " |")
    return "\n".join(rows)


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    text = _repair_display_text(str(value)).replace("\n", " ").replace("\r", " ")
    return text if len(text) <= 160 else text[:157] + "..."


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")


def profile_quality_summary(profile_path: str | Path, cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    profiles = pl.read_parquet(profile_path)
    strong_product_lift = float(cfg.get("profiling.strong_product_lift_threshold", 1.5))
    strong_sector_lift = float(cfg.get("profiling.strong_sector_lift_threshold", 1.2))
    strong_theme_lift = float(cfg.get("profiling.strong_theme_lift_threshold", 1.2))
    product_lift_counts = []
    sector_lift_counts = []
    theme_lift_counts = []
    term_lift_counts = []
    max_product_lifts = []
    soft_assigned_shares = []
    for row in profiles.iter_rows(named=True):
        product_lifts = row.get("top_product_lifts") or []
        sector_lifts = row.get("top_sector_lifts") or []
        theme_lifts = row.get("top_theme_lifts") or []
        term_lifts = row.get("top_product_term_lifts") or []
        product_lift_counts.append(sum(1 for value in product_lifts if value and value >= strong_product_lift))
        sector_lift_counts.append(sum(1 for value in sector_lifts if value and value >= strong_sector_lift))
        theme_lift_counts.append(sum(1 for value in theme_lifts if value and value >= strong_theme_lift))
        term_lift_counts.append(
            sum(
                1
                for value in term_lifts
                if value and value >= float(cfg.get("profiling.strong_term_lift_threshold", 1.25))
            )
        )
        max_product_lifts.append(max(product_lifts) if product_lifts else 0.0)
        soft_assigned_shares.append(float(row.get("soft_assigned_share") or 0.0))
    return {
        "profiled_clusters": profiles.height,
        "clusters_with_product_lift": sum(1 for count in product_lift_counts if count > 0),
        "clusters_with_sector_lift": sum(1 for count in sector_lift_counts if count > 0),
        "clusters_with_theme_lift": sum(1 for count in theme_lift_counts if count > 0),
        "clusters_with_data_driven_term_lift": sum(1 for count in term_lift_counts if count > 0),
        "avg_strong_product_lifts_per_cluster": float(sum(product_lift_counts) / max(profiles.height, 1)),
        "avg_strong_theme_lifts_per_cluster": float(sum(theme_lift_counts) / max(profiles.height, 1)),
        "avg_strong_data_driven_term_lifts_per_cluster": float(sum(term_lift_counts) / max(profiles.height, 1)),
        "avg_max_product_lift": float(sum(max_product_lifts) / max(profiles.height, 1)),
        "avg_soft_assigned_share": float(sum(soft_assigned_shares) / max(profiles.height, 1)),
    }


def _fmt_pct(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _fmt_lift(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.2f}x"


def _format_lift_items(
    names: list[Any] | None,
    lifts: list[Any] | None,
    counts: list[Any] | None = None,
    limit: int = 5,
) -> str:
    names = names or []
    lifts = lifts or []
    counts = counts or []
    parts = []
    for idx, name in enumerate(names[:limit]):
        lift = lifts[idx] if idx < len(lifts) else None
        count_suffix = f", {int(counts[idx]):,} customers" if idx < len(counts) and counts[idx] is not None else ""
        parts.append(f"{_repair_display_text(str(name))} ({_fmt_lift(lift)}{count_suffix})")
    return "; ".join(parts) if parts else "n/a"


def _format_theme_evidence(row: dict[str, Any], limit: int = 4) -> str:
    names = row.get("top_themes") or []
    lifts = row.get("top_theme_lifts") or []
    counts = row.get("top_theme_customer_counts") or []
    n_customers = max(int(row.get("n_customers") or 0), 1)
    parts = []
    for idx, name in enumerate(names[:limit]):
        lift = lifts[idx] if idx < len(lifts) else None
        count = counts[idx] if idx < len(counts) else None
        coverage = f", {int(count) / n_customers:.0%} coverage" if count is not None else ""
        parts.append(f"{str(name).replace('_', ' ')} ({_fmt_lift(lift)}{coverage})")
    return "; ".join(parts) if parts else "n/a"


def _format_term_evidence(row: dict[str, Any], limit: int = 5, cfg: PipelineConfig = CONFIG) -> str:
    names = row.get("top_product_terms") or []
    lifts = row.get("top_product_term_lifts") or []
    counts = row.get("top_product_term_customer_counts") or []
    n_customers = max(int(row.get("n_customers") or 0), 1)
    min_lift = float(cfg.get("profiling.strong_term_lift_threshold", 1.25))
    min_coverage = float(cfg.get("profiling.min_subsegment_coverage", 0.01))
    parts = []
    for idx, name in enumerate(names):
        lift = lifts[idx] if idx < len(lifts) else None
        count = counts[idx] if idx < len(counts) else None
        coverage_value = int(count) / n_customers if count is not None else 0.0
        if lift is None or float(lift) < min_lift or coverage_value < min_coverage:
            continue
        coverage = f", {int(count) / n_customers:.0%} coverage" if count is not None else ""
        parts.append(f"{_repair_display_text(str(name))} ({_fmt_lift(lift)}{coverage})")
        if len(parts) >= limit:
            break
    return "; ".join(parts) if parts else "n/a"


def _label_confidence(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> str:
    theme_lifts = [float(value) for value in (row.get("top_theme_lifts") or []) if value is not None]
    sector_lifts = [float(value) for value in (row.get("top_sector_lifts") or []) if value is not None]
    term_lifts = [float(value) for value in (row.get("top_product_term_lifts") or []) if value is not None]
    strong_themes = sum(1 for value in theme_lifts if value >= float(cfg.get("profiling.strong_theme_lift_threshold", 1.2)))
    strong_sectors = sum(1 for value in sector_lifts if value >= float(cfg.get("profiling.strong_sector_lift_threshold", 1.2)))
    strong_terms = sum(1 for value in term_lifts if value >= float(cfg.get("profiling.strong_term_lift_threshold", 1.25)))
    soft_share = float(row.get("soft_assigned_share") or 0.0)
    if soft_share >= 0.80:
        return "low"
    if strong_themes >= 2 and soft_share < 0.65:
        return "high"
    if strong_terms >= 3 and soft_share < 0.65:
        return "medium"
    if strong_themes >= 1 or strong_sectors >= 1:
        return "medium"
    return "low"


def _interpretation_note(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> str:
    confidence = _label_confidence(row, cfg=cfg)
    if confidence == "low":
        return "Treat as a tentative/mixed tribe; review product evidence before naming externally."
    if float(row.get("soft_assigned_share") or 0.0) >= 0.60:
        return "Good thematic signal, but many customers are soft-assigned; use confidence/provenance when presenting."
    return "Clearer product-theme signal; suitable as a stronger working tribe label."


def _round_optional(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _fmt_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _repair_display_text(text: str) -> str:
    repaired = text
    for _ in range(3):
        if "Ã" not in repaired and "Â" not in repaired:
            break
        try:
            next_text = repaired.encode("latin1").decode("utf-8")
        except UnicodeError:
            break
        if next_text == repaired:
            break
        repaired = next_text
    return repaired


def _format_behavior_context(row: dict[str, Any]) -> str:
    fields = [
        ("avg_ticket_count", "tickets"),
        ("avg_total_units", "units"),
        ("avg_avg_basket_value", "avg basket value"),
        ("avg_promo_share", "promo line share"),
        ("avg_frequency_per_30d", "frequency / 30d"),
        ("avg_unique_products", "unique products"),
    ]
    parts = []
    for key, label in fields:
        value = row.get(key)
        if value is None or value == "":
            continue
        if "share" in key:
            parts.append(f"{label} {_fmt_pct(value)}")
        elif "value" in key or "spend" in key:
            parts.append(f"{label} {float(value):.2f}")
        else:
            parts.append(f"{label} {float(value):.2f}")
    return "; ".join(parts) if parts else "n/a"


def _format_soft_assignment_confidence(row: dict[str, Any]) -> str:
    if not row.get("soft_assigned_customers"):
        return "n/a"
    mean_value = row.get("soft_assignment_confidence_mean")
    p10_value = row.get("soft_assignment_confidence_p10")
    min_value = row.get("soft_assignment_confidence_min")
    if mean_value is None:
        return "not available"
    parts = [f"mean {float(mean_value):.2f}"]
    if p10_value is not None:
        parts.append(f"p10 {float(p10_value):.2f}")
    if min_value is not None:
        parts.append(f"min {float(min_value):.2f}")
    return "; ".join(parts)
