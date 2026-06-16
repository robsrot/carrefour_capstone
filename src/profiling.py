"""Commercial tribe profiling with product and sector lift evidence."""

from __future__ import annotations

import json
import hashlib
import math
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
from src.tribe_namer import THEME_LABELS, fallback_tribe_name, tribe_name_evidence
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
        "tribe_namer_logic": file_fingerprint(Path(__file__).with_name("tribe_namer.py")),
        "population_definition": "all_assigned_customers_including_noise",
    }
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata):
        log_event("Stage 7 profiling", "cache hit", cfg=cfg, assignments=assignments_file, path=output)
        return output

    with stage_timer("Stage 7 profiling", "building tribe profile", cfg=cfg, assignments=assignments_file):
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
            "Stage 7 profiling",
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
            col for col in ["desc_larga_articulo", "idsector", "desc_sector"] if col in product_cols
        ]
        assigned_line_items = lf.select(profile_line_cols).join(assignment_customers, on="cliente", how="inner")
        clustered_keys = clustered_assignments.select(["cliente", "tribe_id"])
        customer_product = assigned_line_items.select(["cliente", "idarticu"]).unique(subset=["cliente", "idarticu"])

        log_event(
            "Stage 7 profiling",
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
            if "idsector" in product_cols:
                metadata_cols.append("idsector")
                metadata_exprs.append(pl.col("idsector").first().cast(pl.Utf8).alias("sector_id"))
            log_event("Stage 7 profiling", "aggregating product metadata", cfg=cfg)
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
        if "sector_id" not in population_product.columns:
            population_product = population_product.with_columns(pl.lit(None).cast(pl.Utf8).alias("sector_id"))
        log_event(
            "Stage 7 profiling",
            "population product counts ready",
            cfg=cfg,
            products=population_product.height,
        )

        log_event("Stage 7 profiling", "aggregating tribe product lifts", cfg=cfg)
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
        product_lifts = _add_overindex_diagnostics(
            product_lifts,
            cluster_count_col="cluster_customers",
            population_count_col="population_customers",
            cluster_total_col="n_customers",
            population_total=total_customers,
            cfg=cfg,
        )
        log_event("Stage 7 profiling", "tribe product lifts ready", cfg=cfg, rows=product_lifts.height)

        theme_lifts = pl.DataFrame()
        term_lifts = pl.DataFrame()
        if "desc_larga_articulo" in product_cols:
            log_event("Stage 7 profiling", "building product theme map", cfg=cfg)
            product_theme_map = _build_product_theme_map(population_product.lazy())
            log_event("Stage 7 profiling", "product theme map ready", cfg=cfg, rows=product_theme_map.height)
            if product_theme_map.height:
                population_theme_events = customer_product.join(
                    product_theme_map.lazy(), on="idarticu", how="inner"
                ).select(["cliente", "strategic_theme"])
                log_event("Stage 7 profiling", "aggregating population theme counts", cfg=cfg)
                population_theme_counts = collect_streaming(
                    population_theme_events.group_by("strategic_theme").agg(
                        pl.col("cliente").n_unique().alias("population_customers")
                    )
                )
                log_event(
                    "Stage 7 profiling",
                    "population theme counts ready",
                    cfg=cfg,
                    rows=population_theme_counts.height,
                )
                log_event("Stage 7 profiling", "aggregating tribe theme lifts", cfg=cfg)
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
                theme_lifts = _add_overindex_diagnostics(
                    theme_lifts,
                    cluster_count_col="cluster_customers",
                    population_count_col="population_customers",
                    cluster_total_col="n_customers",
                    population_total=total_customers,
                    cfg=cfg,
                )
                log_event("Stage 7 profiling", "tribe theme lifts ready", cfg=cfg, rows=theme_lifts.height)

            log_event("Stage 7 profiling", "building product term map", cfg=cfg)
            product_term_map = _build_product_term_map(population_product.lazy(), cfg=cfg)
            log_event("Stage 7 profiling", "product term map ready", cfg=cfg, rows=product_term_map.height)
            if product_term_map.height:
                population_term_events = customer_product.join(product_term_map.lazy(), on="idarticu", how="inner").select(
                    ["cliente", "product_term"]
                )
                log_event(
                    "Stage 7 profiling",
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
                    "Stage 7 profiling",
                    "population term counts ready",
                    cfg=cfg,
                    rows=population_term_counts.height,
                )
                log_event(
                    "Stage 7 profiling",
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
                term_lifts = _add_overindex_diagnostics(
                    term_lifts,
                    cluster_count_col="cluster_customers",
                    population_count_col="population_customers",
                    cluster_total_col="n_customers",
                    population_total=total_customers,
                    cfg=cfg,
                )
                log_event("Stage 7 profiling", "tribe term lifts ready", cfg=cfg, rows=term_lifts.height)

        if "desc_sector" in product_cols:
            log_event("Stage 7 profiling", "aggregating population sector line counts", cfg=cfg)
            population_sector = collect_streaming(
                assigned_line_items.select("desc_sector").group_by("desc_sector").agg(pl.len().alias("population_lines"))
            )
            population_total = int(population_sector["population_lines"].sum()) if population_sector.height else 0
            log_event(
                "Stage 7 profiling",
                "population sector counts ready",
                cfg=cfg,
                rows=population_sector.height,
                population_lines=population_total,
            )
            log_event(
                "Stage 7 profiling",
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
                sector_lifts = _add_overindex_diagnostics(
                    sector_lifts,
                    cluster_count_col="cluster_lines",
                    population_count_col="population_lines",
                    cluster_total_col="cluster_lines_total",
                    population_total=population_total,
                    cfg=cfg,
                )
            else:
                sector_lifts = pl.DataFrame({"tribe_id": [], "desc_sector": [], "lift": []})
            log_event("Stage 7 profiling", "tribe sector lifts ready", cfg=cfg, rows=sector_lifts.height)
        else:
            sector_lifts = pl.DataFrame({"tribe_id": [], "desc_sector": [], "lift": []})

        behavior_summary = None
        if candidate_behavior_path.exists():
            behavior = pl.scan_parquet(candidate_behavior_path)
            behavior_cols = [col for col in schema_names(behavior) if col != "cliente"]
            aggregations = [pl.col(col).mean().alias(f"avg_{col}") for col in behavior_cols if col != "tribe_id"]
            log_event("Stage 7 profiling", "aggregating tribe KPI profiles", cfg=cfg, fields=len(aggregations))
            behavior_summary = collect_streaming(
                clustered_assignments.join(behavior, on="cliente", how="left").group_by("tribe_id").agg(aggregations)
            )
            log_event("Stage 7 profiling", "tribe KPI profiles ready", cfg=cfg, rows=behavior_summary.height)

        log_event("Stage 7 profiling", "assembling tribe profile rows", cfg=cfg, clusters=cluster_sizes.height)
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
                "top_product_sectors": products["sector_description"].to_list()
                if "sector_description" in products.columns
                else [],
                "top_product_sector_ids": products["sector_id"].to_list() if "sector_id" in products.columns else [],
                "top_product_lifts": products["lift"].round(3).to_list() if products.height else [],
                "top_product_lifts_vs_rest": products["lift_vs_rest"].round(3).to_list()
                if "lift_vs_rest" in products.columns
                else [],
                "top_product_q_values": products["lift_q_value"].round(6).to_list()
                if "lift_q_value" in products.columns
                else [],
                "top_product_customer_counts": products["cluster_customers"].to_list() if products.height else [],
                "top_sectors": sectors["desc_sector"].to_list() if "desc_sector" in sectors.columns else [],
                "top_sector_lifts": sectors["lift"].round(3).to_list() if sectors.height else [],
                "top_sector_lifts_vs_rest": sectors["lift_vs_rest"].round(3).to_list()
                if "lift_vs_rest" in sectors.columns
                else [],
                "top_sector_q_values": sectors["lift_q_value"].round(6).to_list()
                if "lift_q_value" in sectors.columns
                else [],
                "top_themes": themes["strategic_theme"].to_list() if "strategic_theme" in themes.columns else [],
                "top_theme_lifts": themes["lift"].round(3).to_list() if themes.height else [],
                "top_theme_lifts_vs_rest": themes["lift_vs_rest"].round(3).to_list()
                if "lift_vs_rest" in themes.columns
                else [],
                "top_theme_q_values": themes["lift_q_value"].round(6).to_list()
                if "lift_q_value" in themes.columns
                else [],
                "top_theme_customer_counts": themes["cluster_customers"].to_list() if themes.height else [],
                "top_product_terms": terms["product_term"].to_list() if "product_term" in terms.columns else [],
                "top_product_term_lifts": terms["lift"].round(3).to_list() if terms.height else [],
                "top_product_term_lifts_vs_rest": terms["lift_vs_rest"].round(3).to_list()
                if "lift_vs_rest" in terms.columns
                else [],
                "top_product_term_q_values": terms["lift_q_value"].round(6).to_list()
                if "lift_q_value" in terms.columns
                else [],
                "top_product_term_customer_counts": terms["cluster_customers"].to_list() if terms.height else [],
            }
            if behavior_summary is not None:
                behavior_match = behavior_summary.filter(pl.col("tribe_id") == tribe_id)
                if behavior_match.height:
                    for col in behavior_match.columns:
                        if col != "tribe_id":
                            row[col] = behavior_match[0, col]
            name_evidence = tribe_name_evidence(row)
            row["suggested_tribe_name"] = name_evidence["working_tribe_name"]
            row["suggested_tribe_name_source"] = name_evidence["working_tribe_name_source"]
            row["suggested_tribe_name_evidence"] = name_evidence["working_tribe_name_evidence"]
            row["profile_evidence_note"] = _profile_evidence_note(row, cfg=cfg)
            rows.append(row)

        if rows:
            profiles = pl.DataFrame(rows).sort("tribe_id")
            profiles = _add_name_collision_flags(profiles)
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
                    "top_product_sectors": pl.List(pl.Utf8),
                    "top_product_sector_ids": pl.List(pl.Utf8),
                    "top_product_lifts": pl.List(pl.Float64),
                    "top_product_lifts_vs_rest": pl.List(pl.Float64),
                    "top_product_q_values": pl.List(pl.Float64),
                    "top_product_customer_counts": pl.List(pl.Int64),
                    "top_sectors": pl.List(pl.Utf8),
                    "top_sector_lifts": pl.List(pl.Float64),
                    "top_sector_lifts_vs_rest": pl.List(pl.Float64),
                    "top_sector_q_values": pl.List(pl.Float64),
                    "top_themes": pl.List(pl.Utf8),
                    "top_theme_lifts": pl.List(pl.Float64),
                    "top_theme_lifts_vs_rest": pl.List(pl.Float64),
                    "top_theme_q_values": pl.List(pl.Float64),
                    "top_theme_customer_counts": pl.List(pl.Int64),
                    "top_product_terms": pl.List(pl.Utf8),
                    "top_product_term_lifts": pl.List(pl.Float64),
                    "top_product_term_lifts_vs_rest": pl.List(pl.Float64),
                    "top_product_term_q_values": pl.List(pl.Float64),
                    "top_product_term_customer_counts": pl.List(pl.Int64),
                    "suggested_tribe_name": pl.Utf8,
                    "suggested_tribe_name_source": pl.Utf8,
                    "suggested_tribe_name_evidence": pl.Utf8,
                    "suggested_tribe_name_duplicate_count": pl.Int64,
                    "suggested_tribe_name_status": pl.Utf8,
                    "profile_evidence_note": pl.Utf8,
                }
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        profiles.write_parquet(output)
        write_artifact_metadata(output, cache_metadata)
        log_event("Stage 7 profiling", "wrote profile", cfg=cfg, tribes=profiles.height, path=output)
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


def _add_overindex_diagnostics(
    frame: pl.DataFrame,
    *,
    cluster_count_col: str,
    population_count_col: str,
    cluster_total_col: str,
    population_total: int,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    """Add rest-of-population lift and FDR-adjusted significance evidence."""

    if frame.is_empty():
        return frame

    rows: list[dict[str, Any]] = []
    p_values: list[float | None] = []
    for row in frame.iter_rows(named=True):
        cluster_count = _numeric_or_none(row.get(cluster_count_col)) or 0.0
        population_count = _numeric_or_none(row.get(population_count_col)) or 0.0
        cluster_total = _numeric_or_none(row.get(cluster_total_col)) or 0.0
        rest_total = max(float(population_total) - cluster_total, 0.0)
        rest_count = max(population_count - cluster_count, 0.0)
        cluster_rate = cluster_count / cluster_total if cluster_total > 0 else None
        population_rate = population_count / population_total if population_total > 0 else None
        rest_rate = rest_count / rest_total if rest_total > 0 else None
        lift = cluster_rate / population_rate if cluster_rate is not None and population_rate and population_rate > 0 else None
        lift_vs_rest = cluster_rate / rest_rate if cluster_rate is not None and rest_rate and rest_rate > 0 else None
        p_value = _two_proportion_p_value(cluster_count, cluster_total, rest_count, rest_total)
        enriched = dict(row)
        enriched.update(
            {
                "cluster_rate": cluster_rate,
                "population_rate": population_rate,
                "rest_observations": int(round(rest_total)),
                "rest_positive_observations": int(round(rest_count)),
                "rest_rate": rest_rate,
                "lift": lift,
                "lift_vs_rest": lift_vs_rest,
                "lift_p_value": p_value,
            }
        )
        rows.append(enriched)
        p_values.append(p_value)

    q_values = _benjamini_hochberg(p_values)
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    for row, q_value in zip(rows, q_values):
        row["lift_q_value"] = q_value
        row["overindex_evidence"] = _overindex_evidence_label(row.get("lift_vs_rest"), q_value, q_threshold)

    return pl.DataFrame(rows)


def _noise_behavior_rows(
    assignments: pl.LazyFrame,
    *,
    behavior_path: str | Path | None,
    noise_total: int,
    core_total: int,
    cfg: PipelineConfig = CONFIG,
) -> list[dict[str, Any]]:
    candidate_behavior_path: Path | None
    if behavior_path is not None:
        candidate_behavior_path = Path(behavior_path)
    else:
        try:
            candidate_behavior_path = cfg.artifact_path(
                "behavioral_features",
                "output",
                directory=cfg.outputs / "features",
            )
        except Exception:
            candidate_behavior_path = None
    if candidate_behavior_path is None or not candidate_behavior_path.exists() or noise_total <= 0 or core_total <= 0:
        return []

    behavior = pl.scan_parquet(candidate_behavior_path)
    numeric_cols = _numeric_behavior_columns(behavior)
    if not numeric_cols:
        return []
    grouped = collect_streaming(
        assignments.select(["cliente", "noise_group"])
        .join(behavior.select(["cliente", *numeric_cols]), on="cliente", how="inner")
        .group_by("noise_group")
        .agg(
            [
                pl.col("cliente").n_unique().alias("customers"),
                *[pl.col(col).mean().alias(col) for col in numeric_cols],
            ]
        )
    )
    by_group = {str(row["noise_group"]): row for row in grouped.iter_rows(named=True)}
    noise_row = by_group.get("noise", {})
    core_row = by_group.get("core", {})
    rows = []
    for idx, metric in enumerate(numeric_cols, start=1):
        noise_value = _numeric_or_none(noise_row.get(metric))
        core_value = _numeric_or_none(core_row.get(metric))
        ratio = noise_value / core_value if noise_value is not None and core_value and core_value > 0 else None
        strength = _noise_behavior_strength(ratio)
        rows.append(
            _noise_audit_row(
                section="behavior evidence",
                rank=idx,
                evidence_type="behavior_metric",
                evidence_key=metric,
                evidence_label=metric,
                noise_observations=int(noise_row.get("customers") or 0),
                core_observations=int(core_row.get("customers") or 0),
                noise_rate_pct=None,
                core_rate_pct=None,
                lift_vs_core=ratio,
                q_value=None,
                evidence_strength=strength,
                interpretation=_noise_behavior_interpretation(metric, noise_value, core_value, ratio),
                recommended_action=_noise_behavior_action(metric, ratio),
                noise_customers_total=noise_total,
                core_customers_total=core_total,
                total_customers=noise_total + core_total,
                noise_share_pct=100.0 * noise_total / max(noise_total + core_total, 1),
            )
        )
    return rows


def _collect_noise_event_counts(
    events: pl.LazyFrame,
    key_col: str,
    *,
    min_observations: int,
) -> pl.DataFrame:
    return collect_streaming(
        events.filter(pl.col(key_col).is_not_null())
        .group_by(key_col)
        .agg(
            [
                pl.col("cliente")
                .filter(pl.col("noise_group") == "noise")
                .n_unique()
                .alias("noise_observations"),
                pl.col("cliente")
                .filter(pl.col("noise_group") == "core")
                .n_unique()
                .alias("core_observations"),
            ]
        )
        .filter(pl.col("noise_observations") >= min_observations)
    )


def _noise_lift_rows(
    frame: pl.DataFrame,
    *,
    evidence_type: str,
    key_col: str,
    label_col: str | None,
    section: str,
    noise_total: int,
    core_total: int,
    top_n: int,
    cfg: PipelineConfig = CONFIG,
) -> list[dict[str, Any]]:
    if frame.is_empty() or noise_total <= 0 or core_total <= 0:
        return []
    enriched = _add_noise_vs_core_diagnostics(frame, noise_total=noise_total, core_total=core_total, cfg=cfg)
    if enriched.is_empty():
        return []
    sort_cols = ["noise_lift_sort", "noise_observations"]
    enriched = enriched.sort(sort_cols, descending=[True, True]).head(top_n)
    rows = []
    for idx, row in enumerate(enriched.iter_rows(named=True), start=1):
        key = row.get(key_col)
        label = row.get(label_col) if label_col and label_col in enriched.columns else key
        rows.append(
            _noise_audit_row(
                section=section,
                rank=idx,
                evidence_type=evidence_type,
                evidence_key=str(key),
                evidence_label=str(label or key),
                noise_observations=int(row.get("noise_observations") or 0),
                core_observations=int(row.get("core_observations") or 0),
                noise_rate_pct=_rate_pct(row.get("noise_observations"), noise_total),
                core_rate_pct=_rate_pct(row.get("core_observations"), core_total),
                lift_vs_core=row.get("lift_vs_core"),
                q_value=row.get("lift_q_value"),
                evidence_strength=row.get("overindex_evidence"),
                interpretation=_noise_lift_interpretation(evidence_type, label, row, noise_total, core_total),
                recommended_action=_noise_lift_action(row, cfg=cfg),
                noise_customers_total=noise_total,
                core_customers_total=core_total,
                total_customers=noise_total + core_total,
                noise_share_pct=100.0 * noise_total / max(noise_total + core_total, 1),
            )
        )
    return rows


def _add_noise_vs_core_diagnostics(
    frame: pl.DataFrame,
    *,
    noise_total: int,
    core_total: int,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    p_values: list[float | None] = []
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    strong_lift = float(cfg.get("profiling.noise_audit_strong_lift_threshold", 1.25))
    for row in frame.iter_rows(named=True):
        noise_count = int(row.get("noise_observations") or 0)
        core_count = int(row.get("core_observations") or 0)
        noise_rate = noise_count / max(noise_total, 1)
        core_rate = core_count / max(core_total, 1)
        lift = noise_rate / core_rate if core_rate > 0 else None
        p_value = _two_proportion_p_value(noise_count, noise_total, core_count, core_total)
        sort_score = lift if lift is not None else (999999.0 if noise_count > 0 and core_count == 0 else -1.0)
        enriched = dict(row)
        enriched.update(
            {
                "noise_rate": noise_rate,
                "core_rate": core_rate,
                "lift_vs_core": lift,
                "noise_lift_sort": sort_score,
                "lift_p_value": p_value,
            }
        )
        rows.append(enriched)
        p_values.append(p_value)

    q_values = _benjamini_hochberg(p_values)
    for row, q_value in zip(rows, q_values):
        row["lift_q_value"] = q_value
        lift = _numeric_or_none(row.get("lift_vs_core"))
        if lift is None and int(row.get("noise_observations") or 0) > 0 and int(row.get("core_observations") or 0) == 0:
            row["overindex_evidence"] = "noise_exclusive_signal"
        elif lift is None:
            row["overindex_evidence"] = "insufficient_core_baseline"
        elif lift < 1.0:
            row["overindex_evidence"] = "under_indexed_in_noise"
        elif q_value is not None and q_value <= q_threshold and lift >= strong_lift:
            row["overindex_evidence"] = "strong_significant_noise_overindex"
        elif q_value is not None and q_value <= q_threshold:
            row["overindex_evidence"] = "significant_noise_overindex"
        elif lift >= strong_lift:
            row["overindex_evidence"] = "directional_noise_overindex"
        else:
            row["overindex_evidence"] = "weak_noise_difference"
    return pl.DataFrame(rows)


def _noise_audit_row(
    *,
    section: str,
    rank: int,
    evidence_type: str,
    evidence_key: str,
    evidence_label: str,
    noise_observations: int,
    core_observations: int,
    noise_rate_pct: float | None,
    core_rate_pct: float | None,
    lift_vs_core: Any,
    q_value: Any,
    evidence_strength: str | None,
    interpretation: str,
    recommended_action: str,
    noise_customers_total: int,
    core_customers_total: int,
    total_customers: int,
    noise_share_pct: float,
) -> dict[str, Any]:
    return {
        "section": section,
        "rank": int(rank),
        "evidence_type": evidence_type,
        "evidence_key": evidence_key,
        "evidence_label": _repair_display_text(str(evidence_label)),
        "noise_observations": int(noise_observations),
        "core_observations": int(core_observations),
        "noise_rate_pct": _round_optional(noise_rate_pct, 3),
        "core_rate_pct": _round_optional(core_rate_pct, 3),
        "lift_vs_core": _round_optional(lift_vs_core, 3),
        "q_value": _round_optional(q_value, 6),
        "evidence_strength": str(evidence_strength or "not_checked"),
        "interpretation": interpretation,
        "recommended_action": recommended_action,
        "noise_customers_total": int(noise_customers_total),
        "core_customers_total": int(core_customers_total),
        "total_customers": int(total_customers),
        "noise_share_pct": _round_optional(noise_share_pct, 3),
    }


def _empty_noise_audit_table() -> pl.DataFrame:
    return pl.DataFrame(schema=_noise_audit_schema())


def _noise_audit_schema() -> dict[str, Any]:
    return {
        "section": pl.Utf8,
        "rank": pl.Int64,
        "evidence_type": pl.Utf8,
        "evidence_key": pl.Utf8,
        "evidence_label": pl.Utf8,
        "noise_observations": pl.Int64,
        "core_observations": pl.Int64,
        "noise_rate_pct": pl.Float64,
        "core_rate_pct": pl.Float64,
        "lift_vs_core": pl.Float64,
        "q_value": pl.Float64,
        "evidence_strength": pl.Utf8,
        "interpretation": pl.Utf8,
        "recommended_action": pl.Utf8,
        "noise_customers_total": pl.Int64,
        "core_customers_total": pl.Int64,
        "total_customers": pl.Int64,
        "noise_share_pct": pl.Float64,
    }


def _rate_pct(count: Any, total: int) -> float | None:
    numeric = _numeric_or_none(count)
    if numeric is None or total <= 0:
        return None
    return 100.0 * numeric / total


def _noise_behavior_strength(ratio: float | None) -> str:
    if ratio is None:
        return "insufficient_behavior_baseline"
    distance = abs(ratio - 1.0)
    if distance >= 0.50:
        return "large_behavior_difference"
    if distance >= 0.25:
        return "moderate_behavior_difference"
    if distance >= 0.10:
        return "small_behavior_difference"
    return "similar_to_core"


def _noise_behavior_interpretation(metric: str, noise_value: float | None, core_value: float | None, ratio: float | None) -> str:
    if ratio is None:
        return f"{metric}: insufficient baseline to compare noise with core customers."
    direction = "higher" if ratio > 1.0 else "lower"
    if _metric_looks_like_recency(metric) and ratio > 1.25:
        read = "noise may contain less recent or dormant customers"
    elif _metric_looks_like_activity(metric) and ratio < 0.75:
        read = "noise may contain low-activity customers with too little repeated signal"
    elif _metric_looks_like_diversity(metric) and ratio > 1.25:
        read = "noise may contain broad generalists or customers spanning too many categories"
    elif ratio >= 1.25 or ratio <= 0.75:
        read = "noise differs materially from the retained core tribes on this profiling metric"
    else:
        read = "noise looks broadly similar to core tribes on this metric"
    return (
        f"{metric}: noise mean {_fmt_number(noise_value)} vs core mean {_fmt_number(core_value)} "
        f"({direction}, {ratio:.2f}x); {read}."
    )


def _noise_behavior_action(metric: str, ratio: float | None) -> str:
    if ratio is None:
        return "keep_as_caveat"
    if _metric_looks_like_activity(metric) and ratio < 0.75:
        return "inspect_sparse_history_noise"
    if _metric_looks_like_recency(metric) and ratio > 1.25:
        return "inspect_dormant_or_lapsed_noise"
    if _metric_looks_like_diversity(metric) and ratio > 1.25:
        return "inspect_generalist_or_mixed_basket_noise"
    if ratio >= 1.50 or ratio <= 0.50:
        return "review_metric_before_second_pass"
    return "keep_as_context"


def _noise_lift_interpretation(
    evidence_type: str,
    label: Any,
    row: dict[str, Any],
    noise_total: int,
    core_total: int,
) -> str:
    lift = _numeric_or_none(row.get("lift_vs_core"))
    noise_rate = _rate_pct(row.get("noise_observations"), noise_total)
    core_rate = _rate_pct(row.get("core_observations"), core_total)
    if lift is None:
        if int(row.get("noise_observations") or 0) > 0 and int(row.get("core_observations") or 0) == 0:
            return (
                f"{label}: appears in {_fmt_number(noise_rate)}% of noise customers and was not observed "
                "among core customers at the configured audit threshold."
            )
        return f"{label}: no comparable core baseline."
    return (
        f"{label}: {evidence_type.replace('_', ' ')} appears in "
        f"{_fmt_number(noise_rate)}% of noise customers vs {_fmt_number(core_rate)}% of core customers "
        f"({_fmt_number(lift)}x)."
    )


def _noise_lift_action(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> str:
    strength = str(row.get("overindex_evidence") or "")
    lift = _numeric_or_none(row.get("lift_vs_core"))
    if strength == "noise_exclusive_signal":
        return "candidate_noise_microstructure"
    if strength == "strong_significant_noise_overindex":
        return "candidate_noise_microstructure"
    if strength == "significant_noise_overindex":
        return "review_as_noise_theme"
    if lift is not None and lift >= float(cfg.get("profiling.noise_audit_strong_lift_threshold", 1.25)):
        return "directional_noise_theme"
    return "keep_as_context"


def _noise_audit_recommendation(
    noise_share_pct: float,
    *,
    signal_count: int,
    behavior_signal_count: int,
    cfg: PipelineConfig = CONFIG,
) -> tuple[str, str]:
    high_noise_pct = float(cfg.get("profiling.noise_audit_high_noise_pct", cfg.get("quality_gates.max_noise_pct", 60.0)))
    second_pass_signals = int(cfg.get("profiling.noise_audit_second_pass_signal_count", 5))
    if noise_share_pct <= 0:
        return "no_noise_action_needed", "No HDBSCAN noise customers are present in the selected assignment."
    if noise_share_pct >= high_noise_pct and signal_count >= second_pass_signals:
        return (
            "run_second_pass_noise_clustering_audit",
            (
                f"Noise is high ({noise_share_pct:.1f}%) and has {signal_count} audit signals. "
                "Keep the official core tribes intact, but investigate a separate noise-only clustering before handoff."
            ),
        )
    if noise_share_pct >= high_noise_pct:
        return (
            "revisit_stage6_density_before_soft_assignment",
            (
                f"Noise is high ({noise_share_pct:.1f}%) but the audit found only {signal_count} strong signals. "
                "Revisit UMAP/HDBSCAN density settings before forcing assignments."
            ),
        )
    if signal_count >= second_pass_signals:
        return (
            "profile_noise_as_secondary_opportunity",
            (
                f"Noise is manageable ({noise_share_pct:.1f}%) but contains {signal_count} audit signals. "
                "Treat it as a secondary opportunity pool rather than renaming the core tribes."
            ),
        )
    if behavior_signal_count:
        return (
            "keep_unassigned_with_behavior_caveat",
            (
                f"Noise is manageable ({noise_share_pct:.1f}%) and mainly differs on customer-behavior metrics. "
                "Keep it unassigned and describe the behavioral caveat."
            ),
        )
    return (
        "keep_noise_unassigned",
        (
            f"Noise is manageable ({noise_share_pct:.1f}%) and no strong hidden product structure was detected. "
            "Keep it outside the official tribe story."
        ),
    )


def _is_noise_audit_signal(strength: Any) -> bool:
    return str(strength or "") in {
        "noise_exclusive_signal",
        "strong_significant_noise_overindex",
        "significant_noise_overindex",
        "directional_noise_overindex",
        "large_behavior_difference",
        "moderate_behavior_difference",
    }


def _metric_looks_like_activity(metric: str) -> bool:
    text = metric.lower()
    return any(token in text for token in ["ticket", "frequency", "visit", "line", "unit", "basket_count", "purchase"])


def _metric_looks_like_recency(metric: str) -> bool:
    text = metric.lower()
    return any(token in text for token in ["recency", "days_since", "inactive", "lapsed"])


def _metric_looks_like_diversity(metric: str) -> bool:
    text = metric.lower()
    return any(token in text for token in ["unique", "divers", "breadth", "sector"])


def _noise_audit_markdown(audit: pl.DataFrame, *, assignments_path: Path, cfg: PipelineConfig = CONFIG) -> str:
    lines = [
        "# Stage 7 Noise Population Audit",
        "",
        f"Run mode: `{cfg.mode}`",
        f"Assignment artifact: `{assignments_path}`",
        "",
        "Noise customers remain outside the official core-tribe story. This audit checks whether that population is merely weak signal or whether it hides product/behavior structure worth follow-up.",
        "",
    ]
    if audit.is_empty():
        lines.append("No noise audit rows were produced.")
        return "\n".join(lines) + "\n"
    summary = audit.filter(pl.col("evidence_type") == "overall_recommendation").head(1)
    if summary.height:
        row = summary.row(0, named=True)
        lines.extend(
            [
                "## Recommendation",
                "",
                f"- Noise customers: {int(row.get('noise_observations') or 0):,} ({float(row.get('noise_share_pct') or 0):.1f}%)",
                f"- Core customers: {int(row.get('core_observations') or 0):,}",
                f"- Audit signals: {row.get('evidence_strength')}",
                f"- Recommended action: `{row.get('recommended_action')}`",
                f"- Read: {row.get('interpretation')}",
                "",
            ]
        )
    display_cols = [
        "section",
        "rank",
        "evidence_type",
        "evidence_label",
        "noise_observations",
        "core_observations",
        "noise_rate_pct",
        "core_rate_pct",
        "lift_vs_core",
        "q_value",
        "evidence_strength",
        "recommended_action",
    ]
    lines.extend(
        [
            "## Evidence",
            "",
            _markdown_table(audit.select([col for col in display_cols if col in audit.columns])),
        ]
    )
    return "\n".join(lines) + "\n"


def _noise_audit_html(audit: pl.DataFrame, *, assignments_path: Path, cfg: PipelineConfig = CONFIG) -> str:
    rows = []
    if not audit.is_empty():
        for row in audit.iter_rows(named=True):
            rows.append(
                "<tr>"
                f"<td>{escape(str(row.get('section') or ''))}</td>"
                f"<td>{int(row.get('rank') or 0)}</td>"
                f"<td><strong>{escape(str(row.get('evidence_label') or ''))}</strong><br>"
                f"<span class='muted'>{escape(str(row.get('evidence_type') or ''))}</span></td>"
                f"<td>{int(row.get('noise_observations') or 0):,}<br><span class='muted'>{_fmt_number(row.get('noise_rate_pct'))}%</span></td>"
                f"<td>{int(row.get('core_observations') or 0):,}<br><span class='muted'>{_fmt_number(row.get('core_rate_pct'))}%</span></td>"
                f"<td>{_fmt_number(row.get('lift_vs_core'))}x<br><span class='muted'>q {_fmt_number(row.get('q_value'))}</span></td>"
                f"<td>{escape(str(row.get('evidence_strength') or ''))}</td>"
                f"<td>{escape(str(row.get('recommended_action') or ''))}</td>"
                f"<td>{escape(str(row.get('interpretation') or ''))}</td>"
                "</tr>"
            )
    body = "\n".join(rows) or "<tr><td colspan='9'>No noise audit rows were produced.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 7 Noise Population Audit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
h1 {{ margin-bottom: 4px; }}
.muted {{ color: #667085; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
th, td {{ border: 1px solid #d0d5dd; padding: 10px; vertical-align: top; font-size: 13px; }}
th {{ background: #f2f4f7; position: sticky; top: 0; z-index: 1; }}
tr:nth-child(even) {{ background: #fbfcfd; }}
</style>
</head>
<body>
<h1>Stage 7 Noise Population Audit</h1>
<p class="muted">Run mode: {escape(str(cfg.mode))}. Assignment artifact: {escape(str(assignments_path))}. Noise remains outside the official core-tribe story.</p>
<table>
<thead>
<tr>
<th>Section</th>
<th>Rank</th>
<th>Evidence</th>
<th>Noise</th>
<th>Core</th>
<th>Lift</th>
<th>Strength</th>
<th>Action</th>
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


def _two_proportion_p_value(
    cluster_count: float,
    cluster_total: float,
    rest_count: float,
    rest_total: float,
) -> float | None:
    if cluster_total <= 0 or rest_total <= 0:
        return None
    p1 = cluster_count / cluster_total
    p2 = rest_count / rest_total
    pooled = (cluster_count + rest_count) / (cluster_total + rest_total)
    variance = pooled * (1.0 - pooled) * ((1.0 / cluster_total) + (1.0 / rest_total))
    if variance <= 0:
        return 1.0 if abs(p1 - p2) < 1e-12 else None
    z_score = (p1 - p2) / math.sqrt(variance)
    return float(math.erfc(abs(z_score) / math.sqrt(2.0)))


def _benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    valid = [
        (idx, float(value))
        for idx, value in enumerate(p_values)
        if value is not None and math.isfinite(float(value))
    ]
    if not valid:
        return [None for _ in p_values]

    valid.sort(key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running_min = 1.0
    m = float(len(valid))
    for rank, (idx, p_value) in enumerate(reversed(valid), start=1):
        original_rank = len(valid) - rank + 1
        running_min = min(running_min, p_value * m / max(original_rank, 1))
        adjusted[idx] = min(max(running_min, 0.0), 1.0)
    return [adjusted.get(idx) for idx in range(len(p_values))]


def _overindex_evidence_label(lift_vs_rest: Any, q_value: Any, q_threshold: float) -> str:
    lift = _numeric_or_none(lift_vs_rest)
    q = _numeric_or_none(q_value)
    if lift is None:
        return "insufficient_rest_baseline"
    if lift < 1.0:
        return "under_indexed"
    if q is not None and q <= q_threshold and lift >= 1.5:
        return "strong_significant_overindex"
    if q is not None and q <= q_threshold:
        return "significant_overindex"
    return "directional_overindex"


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _numeric_at(values: list[Any], idx: int) -> float | None:
    return _numeric_or_none(values[idx]) if idx < len(values) else None


def _int_at(values: list[Any], idx: int) -> int:
    numeric = _numeric_at(values, idx)
    return int(numeric) if numeric is not None else 0


def _text_at(values: list[Any], idx: int) -> str | None:
    if idx >= len(values) or values[idx] is None:
        return None
    return _repair_display_text(str(values[idx]))


def _max_numeric(values: list[Any]) -> float | None:
    numeric = [_numeric_or_none(value) for value in values]
    finite = [value for value in numeric if value is not None]
    return max(finite) if finite else None


def _format_q_value(value: Any) -> str:
    q_value = _numeric_or_none(value)
    if q_value is None:
        return "q n/a"
    if q_value < 0.001:
        return "q<0.001"
    return f"q={q_value:.3f}"


def _profile_evidence_note(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> str:
    product_lifts = row.get("top_product_lifts") or []
    sector_lifts = row.get("top_sector_lifts") or []
    theme_lifts = row.get("top_theme_lifts") or []
    term_lifts = row.get("top_product_term_lifts") or []
    product_q_values = row.get("top_product_q_values") or []
    theme_q_values = row.get("top_theme_q_values") or []
    term_q_values = row.get("top_product_term_q_values") or []
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
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
    significant_profile_signals = (
        _significant_strong_count(
            product_lifts,
            product_q_values,
            threshold=float(cfg.get("profiling.strong_product_lift_threshold", 1.5)),
            q_threshold=q_threshold,
        )
        + _significant_strong_count(
            theme_lifts,
            theme_q_values,
            threshold=float(cfg.get("profiling.strong_theme_lift_threshold", 1.2)),
            q_threshold=q_threshold,
        )
        + _significant_strong_count(
            term_lifts,
            term_q_values,
            threshold=float(cfg.get("profiling.strong_term_lift_threshold", 1.25)),
            q_threshold=q_threshold,
        )
    )
    return (
        f"{strong_product} strong product lifts; {strong_sector} strong sector lifts; "
        f"{strong_theme} strong theme lifts; {strong_term} strong data-driven term lifts; "
        f"{significant_profile_signals} strong signals pass FDR q<={q_threshold:.2f}; "
        f"{row.get('soft_assigned_share', 0.0):.1%} soft-assigned customers."
    )


def _significant_strong_count(
    lifts: list[Any],
    q_values: list[Any],
    *,
    threshold: float,
    q_threshold: float,
) -> int:
    total = 0
    for idx, raw_lift in enumerate(lifts):
        lift = _numeric_or_none(raw_lift)
        q_value = _numeric_or_none(q_values[idx]) if idx < len(q_values) else None
        if lift is not None and q_value is not None and lift >= threshold and q_value <= q_threshold:
            total += 1
    return total


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


def _add_name_collision_flags(profiles: pl.DataFrame) -> pl.DataFrame:
    if profiles.is_empty() or "suggested_tribe_name" not in profiles.columns:
        return profiles
    return profiles.with_columns(
        pl.len().over("suggested_tribe_name").alias("suggested_tribe_name_duplicate_count")
    ).with_columns(
        pl.when(pl.col("suggested_tribe_name_duplicate_count") > 1)
        .then(pl.lit("duplicate_working_name_review"))
        .otherwise(pl.lit("unique_working_name"))
        .alias("suggested_tribe_name_status")
    )


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
                "suggested_tribe_name": _working_tribe_name(row),
                "suggested_tribe_name_source": row.get("suggested_tribe_name_source"),
                "suggested_tribe_name_evidence": row.get("suggested_tribe_name_evidence"),
                "suggested_tribe_name_status": row.get("suggested_tribe_name_status"),
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
    log_event("Stage 7 exports", "wrote cluster summary artifacts", cfg=cfg, **log_kwargs)
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
        "working_tribe_name_source",
        "working_tribe_name_status",
        "label_confidence",
        "n_customers",
        "population_share_pct",
        "soft_assigned_share_pct",
        "working_tribe_name_evidence",
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
        product_lifts_vs_rest = [
            float(value) for value in (row.get("top_product_lifts_vs_rest") or []) if value is not None
        ]
        sector_lifts = [float(value) for value in (row.get("top_sector_lifts") or []) if value is not None]
        theme_lifts = [float(value) for value in (row.get("top_theme_lifts") or []) if value is not None]
        theme_lifts_vs_rest = [
            float(value) for value in (row.get("top_theme_lifts_vs_rest") or []) if value is not None
        ]
        term_lifts = [float(value) for value in (row.get("top_product_term_lifts") or []) if value is not None]
        term_lifts_vs_rest = [
            float(value) for value in (row.get("top_product_term_lifts_vs_rest") or []) if value is not None
        ]
        q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
        rows.append(
            {
                "tribe_id": row.get("tribe_id"),
                "working_tribe_name": _working_tribe_name(row),
                "working_tribe_name_source": row.get("suggested_tribe_name_source"),
                "working_tribe_name_status": row.get("suggested_tribe_name_status"),
                "n_customers": int(row.get("n_customers") or 0),
                "population_share_pct": round(float(row.get("population_share") or 0.0) * 100.0, 2),
                "soft_assigned_share_pct": round(float(row.get("soft_assigned_share") or 0.0) * 100.0, 2),
                "strong_product_lift_count": sum(1 for value in product_lifts if value >= product_threshold),
                "strong_sector_lift_count": sum(1 for value in sector_lifts if value >= sector_threshold),
                "strong_theme_lift_count": sum(1 for value in theme_lifts if value >= theme_threshold),
                "strong_term_lift_count": sum(1 for value in term_lifts if value >= term_threshold),
                "significant_strong_product_lift_count": _significant_strong_count(
                    row.get("top_product_lifts") or [],
                    row.get("top_product_q_values") or [],
                    threshold=product_threshold,
                    q_threshold=q_threshold,
                ),
                "significant_strong_theme_lift_count": _significant_strong_count(
                    row.get("top_theme_lifts") or [],
                    row.get("top_theme_q_values") or [],
                    threshold=theme_threshold,
                    q_threshold=q_threshold,
                ),
                "significant_strong_term_lift_count": _significant_strong_count(
                    row.get("top_product_term_lifts") or [],
                    row.get("top_product_term_q_values") or [],
                    threshold=term_threshold,
                    q_threshold=q_threshold,
                ),
                "max_product_lift": round(max(product_lifts), 3) if product_lifts else None,
                "max_product_lift_vs_rest": round(max(product_lifts_vs_rest), 3) if product_lifts_vs_rest else None,
                "min_product_q_value": _round_optional(_min_numeric(row.get("top_product_q_values") or []), 6),
                "max_theme_lift": round(max(theme_lifts), 3) if theme_lifts else None,
                "max_theme_lift_vs_rest": round(max(theme_lifts_vs_rest), 3) if theme_lifts_vs_rest else None,
                "min_theme_q_value": _round_optional(_min_numeric(row.get("top_theme_q_values") or []), 6),
                "max_term_lift": round(max(term_lifts), 3) if term_lifts else None,
                "max_term_lift_vs_rest": round(max(term_lifts_vs_rest), 3) if term_lifts_vs_rest else None,
                "min_term_q_value": _round_optional(_min_numeric(row.get("top_product_term_q_values") or []), 6),
                "label_confidence": _label_confidence(row, cfg=cfg),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def profile_readiness_evidence_table(
    profile_path: str | Path,
    cluster_readiness_path: str | Path | None = None,
    *,
    output_csv: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    """Combine Stage 6.4 readiness with Stage 7 profile interpretability evidence."""

    evidence = profile_evidence_metrics_table(profile_path, cfg=cfg)
    if evidence.is_empty():
        result = pl.DataFrame()
    else:
        readiness = _read_optional_table(cluster_readiness_path)
        if not readiness.is_empty() and "tribe_id" in readiness.columns:
            readiness_cols = [
                col
                for col in [
                    "tribe_id",
                    "profile_readiness",
                    "readiness_issues",
                    "customers",
                    "mean_assignment_confidence",
                    "p10_assignment_confidence",
                    "jitter_label_recovery_accuracy_mean",
                ]
                if col in readiness.columns
            ]
            evidence = evidence.join(readiness.select(readiness_cols), on="tribe_id", how="left")
        rows = [_profile_readiness_evidence_row(row, cfg=cfg) for row in evidence.iter_rows(named=True)]
        result = pl.DataFrame(rows)

    if output_csv is not None:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_csv(output)
    return result


def _profile_readiness_evidence_row(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    min_signals = int(cfg.get("profiling.min_ready_evidence_signals", 1))
    significant_signals = int(row.get("significant_strong_product_lift_count") or 0) + int(
        row.get("significant_strong_theme_lift_count") or 0
    ) + int(row.get("significant_strong_term_lift_count") or 0)
    strong_signals = int(row.get("strong_product_lift_count") or 0) + int(
        row.get("strong_theme_lift_count") or 0
    ) + int(row.get("strong_term_lift_count") or 0)
    label_confidence = str(row.get("label_confidence") or "low")
    stage6_readiness = str(row.get("profile_readiness") or "missing")
    issues: list[str] = []
    if stage6_readiness in {"review", "missing"}:
        issues.append(f"stage6_readiness={stage6_readiness}")
    if label_confidence == "low":
        issues.append("low_label_confidence")
    if strong_signals < min_signals:
        issues.append(f"strong_lift_signals<{min_signals}")
    elif significant_signals < min_signals:
        issues.append(f"significant_strong_lift_signals<{min_signals}")

    if not issues and label_confidence == "high" and significant_signals >= max(min_signals + 1, 2):
        profiling_readiness = "ready_strong"
    elif not issues:
        profiling_readiness = "ready"
    else:
        profiling_readiness = "review"

    enriched = dict(row)
    enriched.update(
        {
            "stage6_profile_readiness": stage6_readiness,
            "profile_interpretability_signals": strong_signals,
            "profile_significant_interpretability_signals": significant_signals,
            "profiling_readiness": profiling_readiness,
            "profiling_readiness_issues": "; ".join(issues) if issues else "pass",
        }
    )
    return enriched


def _read_optional_table(path: str | Path | None) -> pl.DataFrame:
    if path is None:
        return pl.DataFrame()
    table_path = Path(path)
    if not table_path.exists():
        return pl.DataFrame()
    if table_path.suffix.lower() == ".parquet":
        return pl.read_parquet(table_path)
    if table_path.suffix.lower() == ".csv":
        return pl.read_csv(table_path)
    return pl.DataFrame()


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


def tribe_product_summary_tables(
    profile_path: str | Path,
    *,
    max_products: int | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[int, pl.DataFrame]:
    """Return one product evidence summary table per tribe."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    limit = int(max_products or cfg.get("profiling.top_n_products", 15))
    tables: dict[int, pl.DataFrame] = {}
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row.get("tribe_id"))
        tables[tribe_id] = _tribe_product_summary_frame(row, max_products=limit, cfg=cfg)
    return tables


def write_tribe_product_summary_artifacts(
    profile_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    combined_output_csv: str | Path | None = None,
    max_products: int | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write per-tribe top-product profile tables plus one combined long table."""

    out_dir = Path(output_dir) if output_dir else cfg.artifacts / "stage7" / "tribe_product_summaries"
    combined_output = (
        Path(combined_output_csv)
        if combined_output_csv
        else cfg.artifacts / "stage7" / f"tribe_product_summary_long_{cfg.mode}.csv"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_output.parent.mkdir(parents=True, exist_ok=True)

    tables = tribe_product_summary_tables(profile_path, max_products=max_products, cfg=cfg)
    combined_frames = []
    paths: dict[str, Path] = {"directory": out_dir, "combined_csv": combined_output}
    for tribe_id, table in tables.items():
        path = out_dir / f"tribe_{tribe_id:02d}_product_summary.csv"
        table.write_csv(path)
        paths[f"tribe_{tribe_id:02d}_csv"] = path
        if not table.is_empty():
            combined_frames.append(table)

    combined = pl.concat(combined_frames, how="vertical") if combined_frames else _empty_tribe_product_summary()
    combined.write_csv(combined_output)
    log_event(
        "Stage 7 exports",
        "wrote per-tribe product summary tables",
        cfg=cfg,
        directory=out_dir,
        combined_csv=combined_output,
    )
    return paths


def tribe_comparison_table(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    """Return one side-by-side comparison row per tribe."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    if profiles.is_empty():
        return pl.DataFrame()

    rows = [_tribe_comparison_row(row, profiles, cfg=cfg) for row in profiles.iter_rows(named=True)]
    comparison = pl.DataFrame(rows)
    readiness = _read_optional_table(readiness_path)
    if not readiness.is_empty() and "tribe_id" in readiness.columns:
        readiness_cols = [
            col
            for col in [
                "tribe_id",
                "stage6_profile_readiness",
                "profiling_readiness",
                "profiling_readiness_issues",
                "profile_significant_interpretability_signals",
                "mean_assignment_confidence",
                "p10_assignment_confidence",
                "jitter_label_recovery_accuracy_mean",
            ]
            if col in readiness.columns
        ]
        comparison = comparison.join(readiness.select(readiness_cols), on="tribe_id", how="left")
    return comparison.sort("tribe_id")


def customer_metric_anova_table(
    assignments_path: str | Path,
    behavior_path: str | Path | None = None,
    *,
    output_csv: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    """Test whether customer-level profiling metrics differ across tribes."""

    candidate_behavior_path = (
        Path(behavior_path)
        if behavior_path
        else cfg.artifact_path("behavioral_features", "output", directory=cfg.outputs / "features")
    )
    if not candidate_behavior_path.exists():
        result = _empty_customer_metric_tests()
        if output_csv is not None:
            output = Path(output_csv)
            output.parent.mkdir(parents=True, exist_ok=True)
            result.write_csv(output)
        return result

    assignments_file = Path(assignments_path)
    assignments = (
        pl.scan_parquet(assignments_file)
        .select(["cliente", "tribe_id"])
        .filter(pl.col("tribe_id") >= 0)
        .unique(subset=["cliente"], keep="first")
    )
    behavior = pl.scan_parquet(candidate_behavior_path)
    numeric_cols = _numeric_behavior_columns(behavior)
    rows: list[dict[str, Any]] = []
    p_values: list[float | None] = []

    for metric in numeric_cols:
        grouped = collect_streaming(
            assignments.join(behavior.select(["cliente", metric]), on="cliente", how="inner")
            .filter(pl.col(metric).is_not_null())
            .group_by("tribe_id")
            .agg(
                [
                    pl.len().alias("n_customers"),
                    pl.col(metric).mean().alias("mean"),
                    pl.col(metric).var().alias("variance"),
                ]
            )
            .sort("tribe_id")
        )
        row = _anova_result_row(metric, grouped, cfg=cfg)
        rows.append(row)
        p_values.append(row.get("anova_p_value"))

    q_values = _benjamini_hochberg(p_values)
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    for row, q_value in zip(rows, q_values):
        row["anova_q_value"] = q_value
        row["statistical_result"] = _anova_result_label(row.get("anova_p_value"), q_value, row.get("anova_effect_eta_squared"), q_threshold)

    result = pl.DataFrame(rows) if rows else _empty_customer_metric_tests()
    if output_csv is not None:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_csv(output)
    return result


def noise_audit_table(
    assignments_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    behavior_path: str | Path | None = None,
    *,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    """Audit the HDBSCAN noise population without forcing it into core tribes."""

    assignments_file = Path(assignments_path)
    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    columns = set(schema_names(lf))
    if "cliente" not in columns:
        return _empty_noise_audit_table()

    assignment_schema = set(schema_names(pl.scan_parquet(assignments_file)))
    if "cliente" not in assignment_schema or "tribe_id" not in assignment_schema:
        return _empty_noise_audit_table()

    assignments = (
        pl.scan_parquet(assignments_file)
        .select(["cliente", "tribe_id"])
        .unique(subset=["cliente"], keep="first")
        .with_columns(
            pl.when(pl.col("tribe_id") < 0)
            .then(pl.lit("noise"))
            .otherwise(pl.lit("core"))
            .alias("noise_group")
        )
    )
    group_counts = collect_streaming(
        assignments.group_by("noise_group").agg(pl.col("cliente").n_unique().alias("customers"))
    )
    counts = {str(row["noise_group"]): int(row["customers"]) for row in group_counts.iter_rows(named=True)}
    noise_total = counts.get("noise", 0)
    core_total = counts.get("core", 0)
    total_customers = noise_total + core_total
    noise_share_pct = 100.0 * noise_total / max(total_customers, 1)
    min_observations = int(cfg.get("profiling.noise_audit_min_customers", cfg.get("profiling.min_product_customers", 10)))
    top_n = int(cfg.get("profiling.noise_audit_top_n", 12))

    detail_rows: list[dict[str, Any]] = []
    behavior_rows = _noise_behavior_rows(
        assignments,
        behavior_path=behavior_path,
        noise_total=noise_total,
        core_total=core_total,
        cfg=cfg,
    )
    detail_rows.extend(behavior_rows)

    if noise_total > 0 and core_total > 0 and "idarticu" in columns:
        product_cols = ["cliente", "idarticu"]
        for col in ["desc_larga_articulo", "idsector", "desc_sector"]:
            if col in columns:
                product_cols.append(col)
        assigned_line_items = lf.select(product_cols).join(
            assignments.select(["cliente", "noise_group"]),
            on="cliente",
            how="inner",
        )
        customer_product = assigned_line_items.select(["cliente", "idarticu", "noise_group"]).unique(
            subset=["cliente", "idarticu"]
        )

        product_counts = _collect_noise_event_counts(
            customer_product,
            "idarticu",
            min_observations=min_observations,
        )
        if not product_counts.is_empty():
            metadata_exprs = []
            metadata_cols = []
            if "desc_larga_articulo" in product_cols:
                metadata_cols.append("desc_larga_articulo")
                metadata_exprs.append(pl.col("desc_larga_articulo").drop_nulls().first().alias("evidence_label"))
            if "desc_sector" in product_cols:
                metadata_cols.append("desc_sector")
                metadata_exprs.append(pl.col("desc_sector").drop_nulls().first().alias("sector_description"))
            if "idsector" in product_cols:
                metadata_cols.append("idsector")
                metadata_exprs.append(pl.col("idsector").drop_nulls().first().cast(pl.Utf8).alias("sector_id"))
            if metadata_exprs:
                product_metadata = collect_streaming(
                    assigned_line_items.select(["idarticu", *metadata_cols]).group_by("idarticu").agg(metadata_exprs)
                )
                product_counts = product_counts.join(product_metadata, on="idarticu", how="left")
            detail_rows.extend(
                _noise_lift_rows(
                    product_counts,
                    evidence_type="product_lift",
                    key_col="idarticu",
                    label_col="evidence_label" if "evidence_label" in product_counts.columns else None,
                    section="product evidence",
                    noise_total=noise_total,
                    core_total=core_total,
                    top_n=top_n,
                    cfg=cfg,
                )
            )

        if "desc_sector" in product_cols:
            customer_sector = assigned_line_items.select(["cliente", "desc_sector", "noise_group"]).unique(
                subset=["cliente", "desc_sector"]
            )
            sector_counts = _collect_noise_event_counts(
                customer_sector,
                "desc_sector",
                min_observations=min_observations,
            )
            detail_rows.extend(
                _noise_lift_rows(
                    sector_counts,
                    evidence_type="sector_lift",
                    key_col="desc_sector",
                    label_col="desc_sector",
                    section="sector evidence",
                    noise_total=noise_total,
                    core_total=core_total,
                    top_n=top_n,
                    cfg=cfg,
                )
            )

        if "desc_larga_articulo" in product_cols:
            population_product = collect_streaming(
                assigned_line_items.select(["idarticu", "desc_larga_articulo"])
                .group_by("idarticu")
                .agg(pl.col("desc_larga_articulo").drop_nulls().first().alias("product_description"))
            )
            product_theme_map = _build_product_theme_map(population_product.lazy())
            if product_theme_map.height:
                customer_theme = (
                    customer_product.join(product_theme_map.lazy(), on="idarticu", how="inner")
                    .select(["cliente", "strategic_theme", "noise_group"])
                    .unique(subset=["cliente", "strategic_theme"])
                )
                theme_counts = _collect_noise_event_counts(
                    customer_theme,
                    "strategic_theme",
                    min_observations=min_observations,
                )
                detail_rows.extend(
                    _noise_lift_rows(
                        theme_counts,
                        evidence_type="theme_lift",
                        key_col="strategic_theme",
                        label_col="strategic_theme",
                        section="theme evidence",
                        noise_total=noise_total,
                        core_total=core_total,
                        top_n=top_n,
                        cfg=cfg,
                    )
                )

            product_term_map = _build_product_term_map(population_product.lazy(), cfg=cfg)
            if product_term_map.height:
                customer_term = (
                    customer_product.join(product_term_map.lazy(), on="idarticu", how="inner")
                    .select(["cliente", "product_term", "noise_group"])
                    .unique(subset=["cliente", "product_term"])
                )
                term_counts = _collect_noise_event_counts(
                    customer_term,
                    "product_term",
                    min_observations=min_observations,
                )
                detail_rows.extend(
                    _noise_lift_rows(
                        term_counts,
                        evidence_type="term_lift",
                        key_col="product_term",
                        label_col="product_term",
                        section="term evidence",
                        noise_total=noise_total,
                        core_total=core_total,
                        top_n=top_n,
                        cfg=cfg,
                    )
                )

    signal_count = sum(1 for row in detail_rows if _is_noise_audit_signal(row.get("evidence_strength")))
    behavior_signal_count = sum(
        1
        for row in detail_rows
        if row.get("evidence_type") == "behavior_metric" and _is_noise_audit_signal(row.get("evidence_strength"))
    )
    recommendation, recommendation_read = _noise_audit_recommendation(
        noise_share_pct,
        signal_count=signal_count,
        behavior_signal_count=behavior_signal_count,
        cfg=cfg,
    )
    summary_row = _noise_audit_row(
        section="summary",
        rank=1,
        evidence_type="overall_recommendation",
        evidence_key="noise_population",
        evidence_label="Noise population",
        noise_observations=noise_total,
        core_observations=core_total,
        noise_rate_pct=noise_share_pct,
        core_rate_pct=100.0 * core_total / max(total_customers, 1),
        lift_vs_core=None,
        q_value=None,
        evidence_strength=f"{signal_count} audit signals",
        interpretation=recommendation_read,
        recommended_action=recommendation,
        noise_customers_total=noise_total,
        core_customers_total=core_total,
        total_customers=total_customers,
        noise_share_pct=noise_share_pct,
    )
    rows = [summary_row, *detail_rows]
    return pl.DataFrame(rows, schema=_noise_audit_schema()) if rows else _empty_noise_audit_table()


def write_noise_audit_artifacts(
    assignments_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    behavior_path: str | Path | None = None,
    *,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write aggregate-only evidence about the HDBSCAN noise population."""

    base_output = (
        Path(output_csv)
        if output_csv
        else cfg.artifacts / "stage7" / f"stage7_noise_audit_{cfg.mode}.csv"
    )
    md_output = Path(output_md) if output_md else base_output.with_suffix(".md")
    html_output = Path(output_html) if output_html else base_output.with_suffix(".html")
    audit = noise_audit_table(assignments_path, transactions=transactions, behavior_path=behavior_path, cfg=cfg)

    base_output.parent.mkdir(parents=True, exist_ok=True)
    audit.write_csv(base_output)
    md_output.write_text(_noise_audit_markdown(audit, assignments_path=Path(assignments_path), cfg=cfg), encoding="utf-8")
    html_output.write_text(_noise_audit_html(audit, assignments_path=Path(assignments_path), cfg=cfg), encoding="utf-8")
    log_event(
        "Stage 7 noise audit",
        "wrote noise-population evidence",
        cfg=cfg,
        csv=base_output,
        markdown=md_output,
        html=html_output,
    )
    return {"csv": base_output, "markdown": md_output, "html": html_output}


def write_tribe_comparison_artifacts(
    profile_path: str | Path,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    readiness_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write side-by-side tribe comparison artifacts for review and presentation."""

    base_output = (
        Path(output_csv)
        if output_csv
        else cfg.reports / str(cfg.get("exports.tribe_comparison_template", "tribe_side_by_side_{mode}.csv")).format(mode=cfg.mode)
    )
    md_output = Path(output_md) if output_md else base_output.with_suffix(".md")
    html_output = Path(output_html) if output_html else base_output.with_suffix(".html")
    comparison = tribe_comparison_table(profile_path, readiness_path=readiness_path, cfg=cfg)

    base_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(base_output)
    md_output.write_text(_tribe_comparison_markdown(comparison, cfg=cfg), encoding="utf-8")
    html_output.write_text(_tribe_comparison_html(comparison, cfg=cfg), encoding="utf-8")
    log_event(
        "Stage 7 exports",
        "wrote side-by-side tribe comparison",
        cfg=cfg,
        csv=base_output,
        markdown=md_output,
        html=html_output,
    )
    return {"csv": base_output, "markdown": md_output, "html": html_output}


def stage7_storyline_table(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    subsegment_summary_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    """Return a compact Stage 7 storyline index, one row per tribe.

    This is the notebook-facing layer: it keeps the detailed evidence available
    in files, but gives reviewers a single ordered way to read the tribes.
    """

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    if profiles.is_empty():
        return pl.DataFrame()

    readiness = _read_optional_table(readiness_path)
    comparison = _read_optional_table(comparison_path)
    if comparison.is_empty():
        comparison = tribe_comparison_table(profile_path, readiness_path=readiness_path, cfg=cfg)
    subsegments = _read_subsegment_summary(subsegment_summary_path)
    size_rank_by_tribe = {
        int(row["tribe_id"]): idx
        for idx, row in enumerate(profiles.sort("n_customers", descending=True).iter_rows(named=True), start=1)
    }

    rows = []
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row.get("tribe_id"))
        readiness_row = _row_by_tribe(readiness, tribe_id)
        comparison_row = _row_by_tribe(comparison, tribe_id)
        tribe_subsegments = _tribe_subsegments(subsegments, tribe_id, limit=3)
        rows.append(
            {
                "story_order": size_rank_by_tribe.get(tribe_id),
                "tribe_id": tribe_id,
                "working_tribe_name": _working_tribe_name(row),
                "story_role": _stage7_story_role(row, size_rank_by_tribe.get(tribe_id)),
                "evidence_strength": _stage7_evidence_strength(row, readiness_row, cfg=cfg),
                "readiness": readiness_row.get("profiling_readiness")
                or readiness_row.get("stage6_profile_readiness")
                or "not checked",
                "audience_size": f"{int(row.get('n_customers') or 0):,} customers ({_fmt_pct(row.get('population_share'))})",
                "core_assignment_read": _stage7_assignment_read(row),
                "distinctive_product_evidence": comparison_row.get("top_product_and_category_evidence")
                or _format_lift_items(
                    row.get("top_products"),
                    row.get("top_product_lifts"),
                    row.get("top_product_customer_counts"),
                    limit=4,
                ),
                "supporting_theme_term_evidence": _stage7_supporting_evidence(row, cfg=cfg),
                "customer_context": comparison_row.get("customer_behavior_over_under_index")
                or _format_behavior_context(row),
                "subsegment_overlay": _stage7_subsegment_read(tribe_subsegments),
                "caveat": _stage7_caveat(row, readiness_row, cfg=cfg),
            }
        )
    return pl.DataFrame(rows).sort("story_order")


def write_stage7_storyline_artifacts(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    subsegment_summary_path: str | Path | None = None,
    customer_metric_tests_path: str | Path | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write the Stage 7 evidence storyline for handoff review."""

    base_output = (
        Path(output_csv)
        if output_csv
        else cfg.artifacts / "stage7" / f"stage7_evidence_storyline_{cfg.mode}.csv"
    )
    md_output = Path(output_md) if output_md else base_output.with_suffix(".md")
    html_output = Path(output_html) if output_html else base_output.with_suffix(".html")

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    readiness = _read_optional_table(readiness_path)
    subsegments = _read_subsegment_summary(subsegment_summary_path)
    customer_metric_tests = _read_optional_table(customer_metric_tests_path)
    storyline = stage7_storyline_table(
        profile_path,
        readiness_path=readiness_path,
        comparison_path=comparison_path,
        subsegment_summary_path=subsegment_summary_path,
        cfg=cfg,
    )

    base_output.parent.mkdir(parents=True, exist_ok=True)
    storyline.write_csv(base_output)
    md_output.write_text(
        _stage7_storyline_markdown(
            storyline,
            profiles,
            readiness,
            subsegments,
            customer_metric_tests,
            profile_path=Path(profile_path),
            cfg=cfg,
        ),
        encoding="utf-8",
    )
    html_output.write_text(
        _stage7_storyline_html(
            storyline,
            profiles,
            readiness,
            subsegments,
            customer_metric_tests,
            profile_path=Path(profile_path),
            cfg=cfg,
        ),
        encoding="utf-8",
    )
    log_event(
        "Stage 7 exports",
        "wrote evidence storyline",
        cfg=cfg,
        csv=base_output,
        markdown=md_output,
        html=html_output,
    )
    return {"csv": base_output, "markdown": md_output, "html": html_output}


def _stage7_story_role(row: dict[str, Any], size_rank: int | None) -> str:
    share = float(row.get("population_share") or 0.0)
    if size_rank == 1:
        return "largest anchor tribe"
    if size_rank is not None and size_rank <= 3:
        return "major comparison tribe"
    if share >= 0.05:
        return "mid-sized distinct tribe"
    return "niche distinct tribe"


def _stage7_evidence_strength(
    row: dict[str, Any],
    readiness_row: dict[str, Any],
    *,
    cfg: PipelineConfig = CONFIG,
) -> str:
    readiness = str(readiness_row.get("profiling_readiness") or "")
    confidence = _label_confidence(row, cfg=cfg)
    signals = int(readiness_row.get("profile_significant_interpretability_signals") or 0)
    if readiness == "ready_strong" and confidence == "high":
        return f"strong story ({signals} significant lift signals)"
    if readiness in {"ready", "ready_strong"} and confidence in {"high", "medium"}:
        return f"usable story ({signals} significant lift signals)"
    if signals > 0:
        return f"review story ({signals} significant lift signals)"
    return "weak or mixed story"


def _stage7_assignment_read(row: dict[str, Any]) -> str:
    core = int(row.get("core_customers") or 0)
    soft = int(row.get("soft_assigned_customers") or 0)
    soft_share = float(row.get("soft_assigned_share") or 0.0)
    if soft == 0:
        return f"{core:,} core customers; no soft assignment used"
    confidence = _format_soft_assignment_confidence(row)
    return f"{core:,} core / {soft:,} soft-assigned ({soft_share:.1%}); confidence {confidence}"


def _stage7_supporting_evidence(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> str:
    parts = []
    terms = _format_term_evidence(row, cfg=cfg)
    themes = _format_theme_evidence(row)
    sectors = _format_lift_items(row.get("top_sectors"), row.get("top_sector_lifts"), limit=3)
    if terms != "n/a":
        parts.append(f"terms: {terms}")
    if themes != "n/a":
        parts.append(f"themes: {themes}")
    if sectors != "n/a":
        parts.append(f"sectors: {sectors}")
    return " | ".join(parts) if parts else "n/a"


def _stage7_subsegment_read(subsegments: pl.DataFrame) -> str:
    if subsegments.is_empty():
        return "No subsegment overlay passed the evidence thresholds."
    parts = []
    for row in subsegments.iter_rows(named=True):
        label = row.get("subsegment_label") or row.get("subsegment_key") or "subsegment"
        customers = int(row.get("subsegment_customers") or 0)
        share = float(row.get("subsegment_share_pct") or 0.0)
        lift = row.get("subsegment_lift")
        parts.append(f"{label} ({customers:,} customers, {share:.1f}% of tribe, {_fmt_lift(lift)})")
    return "; ".join(parts)


def _stage7_caveat(row: dict[str, Any], readiness_row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> str:
    issues = str(readiness_row.get("profiling_readiness_issues") or "").strip()
    interpretation = _interpretation_note(row, cfg=cfg)
    status = str(readiness_row.get("profiling_readiness") or "")
    caveats = []
    if issues and issues != "pass":
        caveats.append(f"readiness issue: {issues}")
    if status == "review":
        caveats.append("review before external naming")
    caveats.append(interpretation)
    caveats.append("working label is purchase evidence, not a demographic persona")
    return "; ".join(dict.fromkeys(caveats))


def _stage7_storyline_snapshot(
    profiles: pl.DataFrame,
    readiness: pl.DataFrame,
    subsegments: pl.DataFrame,
    customer_metric_tests: pl.DataFrame,
    *,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    total_customers = int(profiles["profile_population_customers"][0]) if profiles.height else 0
    assigned_customers = int(profiles["n_customers"].sum()) if profiles.height else 0
    noise_customers = int(profiles["profile_noise_customers"][0]) if profiles.height else 0
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    significant_metric_tests = (
        customer_metric_tests.filter(pl.col("anova_q_value") <= q_threshold).height
        if not customer_metric_tests.is_empty() and "anova_q_value" in customer_metric_tests.columns
        else 0
    )
    ready_count = (
        readiness.filter(pl.col("profiling_readiness").is_in(["ready", "ready_strong"])).height
        if not readiness.is_empty() and "profiling_readiness" in readiness.columns
        else None
    )
    review_count = (
        readiness.filter(pl.col("profiling_readiness") == "review").height
        if not readiness.is_empty() and "profiling_readiness" in readiness.columns
        else None
    )
    return pl.DataFrame(
        [
            {"checkpoint": "core tribes explained", "value": profiles.height, "read": "one storyline row per retained tribe"},
            {"checkpoint": "profiled customers", "value": total_customers, "read": "customers in the selected profile population"},
            {"checkpoint": "assigned customers", "value": assigned_customers, "read": "customers inside retained core tribes"},
            {"checkpoint": "unassigned noise customers", "value": noise_customers, "read": "left outside the tribe story"},
            {"checkpoint": "ready tribes", "value": ready_count, "read": "passed Stage 6.4 plus Stage 7 interpretability checks"},
            {"checkpoint": "review tribes", "value": review_count, "read": "keep evidence visible before external naming"},
            {
                "checkpoint": "significant customer-context tests",
                "value": significant_metric_tests,
                "read": "post-clustering profiling only, not modeling signal",
            },
            {
                "checkpoint": "subsegment overlays",
                "value": subsegments.height,
                "read": "within-tribe activation overlays, not replacement tribe definitions",
            },
        ]
    )


def _stage7_metric_test_read(customer_metric_tests: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    if customer_metric_tests.is_empty() or "anova_q_value" not in customer_metric_tests.columns:
        return "No customer-context metric test table was provided."
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    significant = customer_metric_tests.filter(pl.col("anova_q_value") <= q_threshold)
    if significant.is_empty():
        return "No customer-context metric passed the FDR threshold; use metrics as descriptive context only."
    display_cols = [
        col
        for col in [
            "metric",
            "anova_q_value",
            "anova_effect_eta_squared",
            "highest_mean_tribe_id",
            "lowest_mean_tribe_id",
            "statistical_result",
        ]
        if col in significant.columns
    ]
    preview = significant.sort("anova_q_value").head(6).select(display_cols)
    return _markdown_table(preview)


def _stage7_storyline_markdown(
    storyline: pl.DataFrame,
    profiles: pl.DataFrame,
    readiness: pl.DataFrame,
    subsegments: pl.DataFrame,
    customer_metric_tests: pl.DataFrame,
    *,
    profile_path: Path,
    cfg: PipelineConfig = CONFIG,
) -> str:
    snapshot = _stage7_storyline_snapshot(profiles, readiness, subsegments, customer_metric_tests, cfg=cfg)
    display_cols = [
        "story_order",
        "tribe_id",
        "working_tribe_name",
        "story_role",
        "evidence_strength",
        "readiness",
        "audience_size",
        "distinctive_product_evidence",
        "supporting_theme_term_evidence",
        "subsegment_overlay",
        "caveat",
    ]
    lines = [
        "# Stage 7 Evidence Storyline",
        "",
        f"Run mode: `{cfg.mode}`",
        f"Profile artifact: `{profile_path}`",
        "",
        "This report is the reading layer for Stage 7. It turns the clustering output into a structured evidence story so the model is not treated as a black box.",
        "",
        "Labels are working purchase-behavior summaries. They are not demographic, religious, household, health, income, age, gender, nationality, or identity claims.",
        "",
        "## Evidence Ladder",
        "",
        "| Step | Question | Evidence Used | What It Allows Us To Say |",
        "|---|---|---|---|",
        "| 1 | Which model are we explaining? | Stage 6 selected hard UMAP-HDBSCAN assignment | The story is tied to one promoted solution, not cherry-picked tribes. |",
        "| 2 | Are the tribes stable enough to interpret? | Stage 6.4 readiness and assignment confidence | Weak or unstable groups stay marked for review. |",
        "| 3 | What makes each tribe different? | Product/category lift vs the rest, q-values, and customer coverage | Tribe labels must be backed by distinctive products, not raw popularity. |",
        "| 4 | What supports or refines the read? | Theme, sector, product-term, and customer-context contrasts | Supporting evidence enriches the story without becoming the clustering signal. |",
        "| 5 | Where can action happen? | Subsegment overlays from lifted themes and terms | Subsegments become activation hypotheses inside tribes, not new personas. |",
        "| 6 | What must stay caveated? | Readiness issues, label confidence, noise customers, and guardrails | The handoff keeps uncertainty visible. |",
        "",
        "## Run Snapshot",
        "",
        _markdown_table(snapshot),
        "",
        "## Tribe Storyline Index",
        "",
    ]
    if storyline.is_empty():
        lines.append("No storyline rows were produced.")
    else:
        lines.append(_markdown_table(storyline.select([col for col in display_cols if col in storyline.columns])))
    lines.extend(
        [
            "",
            "## Customer-Context Checks",
            "",
            _stage7_metric_test_read(customer_metric_tests, cfg=cfg),
            "",
            "## Handoff Rule",
            "",
            "Lead with the storyline table. Use the detailed product summaries, comparison tables, atlas, and lift plots only to support a specific claim or investigate a caveat.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _stage7_storyline_html(
    storyline: pl.DataFrame,
    profiles: pl.DataFrame,
    readiness: pl.DataFrame,
    subsegments: pl.DataFrame,
    customer_metric_tests: pl.DataFrame,
    *,
    profile_path: Path,
    cfg: PipelineConfig = CONFIG,
) -> str:
    snapshot = _stage7_storyline_snapshot(profiles, readiness, subsegments, customer_metric_tests, cfg=cfg)
    snapshot_cards = "\n".join(
        "<div class='metric'>"
        f"<span>{escape(str(row.get('checkpoint')))}</span>"
        f"<strong>{'' if row.get('value') is None else escape(str(row.get('value')))}</strong>"
        f"<small>{escape(str(row.get('read') or ''))}</small>"
        "</div>"
        for row in snapshot.iter_rows(named=True)
    )
    storyline_rows = "\n".join(_stage7_storyline_html_row(row) for row in storyline.iter_rows(named=True))
    if not storyline_rows:
        storyline_rows = "<tr><td colspan='9'>No storyline rows were produced.</td></tr>"
    metric_read = _stage7_metric_test_read(customer_metric_tests, cfg=cfg)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 7 Evidence Storyline</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #17202a; background: #f7f8fa; }}
header {{ background: #ffffff; border-bottom: 1px solid #d0d5dd; padding: 24px 32px; }}
main {{ padding: 24px 32px 40px; }}
h1, h2 {{ margin: 0 0 8px; }}
h2 {{ margin-top: 26px; }}
p {{ line-height: 1.45; }}
.muted {{ color: #667085; font-size: 13px; }}
.snapshot {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 18px; }}
.metric {{ background: #ffffff; border: 1px solid #d0d5dd; border-radius: 8px; padding: 14px; }}
.metric span {{ display: block; color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
.metric strong {{ display: block; font-size: 22px; margin: 4px 0; }}
.metric small {{ color: #667085; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; background: #ffffff; }}
th, td {{ border: 1px solid #d0d5dd; padding: 9px; vertical-align: top; font-size: 13px; }}
th {{ background: #eef2f6; text-align: left; }}
tr:nth-child(even) {{ background: #fbfcfd; }}
.ladder td:first-child {{ font-weight: 700; width: 50px; }}
.tribe {{ font-weight: 700; }}
.caveat {{ color: #7a2e0e; }}
</style>
</head>
<body>
<header>
<h1>Stage 7 Evidence Storyline</h1>
<p class="muted">Run mode: {escape(str(cfg.mode))}. Profile artifact: {escape(str(profile_path))}</p>
<p>This is the reading layer for Stage 7. It turns the clustering output into a structured evidence story so the model is not treated as a black box.</p>
<p class="muted">Labels are working purchase-behavior summaries, not demographic, religious, household, health, income, age, gender, nationality, or identity claims.</p>
<div class="snapshot">{snapshot_cards}</div>
</header>
<main>
<h2>Evidence Ladder</h2>
<table class="ladder">
<thead><tr><th>Step</th><th>Question</th><th>Evidence Used</th><th>What It Allows Us To Say</th></tr></thead>
<tbody>
<tr><td>1</td><td>Which model are we explaining?</td><td>Stage 6 selected hard UMAP-HDBSCAN assignment</td><td>The story is tied to one promoted solution, not cherry-picked tribes.</td></tr>
<tr><td>2</td><td>Are the tribes stable enough to interpret?</td><td>Stage 6.4 readiness and assignment confidence</td><td>Weak or unstable groups stay marked for review.</td></tr>
<tr><td>3</td><td>What makes each tribe different?</td><td>Product/category lift vs the rest, q-values, and customer coverage</td><td>Tribe labels must be backed by distinctive products, not raw popularity.</td></tr>
<tr><td>4</td><td>What supports or refines the read?</td><td>Theme, sector, product-term, and customer-context contrasts</td><td>Supporting evidence enriches the story without becoming the clustering signal.</td></tr>
<tr><td>5</td><td>Where can action happen?</td><td>Subsegment overlays from lifted themes and terms</td><td>Subsegments become activation hypotheses inside tribes, not new personas.</td></tr>
<tr><td>6</td><td>What must stay caveated?</td><td>Readiness issues, label confidence, noise customers, and guardrails</td><td>The handoff keeps uncertainty visible.</td></tr>
</tbody>
</table>
<h2>Tribe Storyline Index</h2>
<table>
<thead>
<tr>
<th>Order</th>
<th>Tribe</th>
<th>Role</th>
<th>Evidence Strength</th>
<th>Audience</th>
<th>Distinctive Product Evidence</th>
<th>Supporting Evidence</th>
<th>Subsegment Overlay</th>
<th>Caveat</th>
</tr>
</thead>
<tbody>
{storyline_rows}
</tbody>
</table>
<h2>Customer-Context Checks</h2>
<p class="muted">Customer-context metrics are interpretation evidence only. They are not part of the clustering signal.</p>
<pre>{escape(metric_read)}</pre>
<h2>Handoff Rule</h2>
<p>Lead with the storyline table. Use the detailed product summaries, comparison tables, atlas, and lift plots only to support a specific claim or investigate a caveat.</p>
</main>
</body>
</html>
"""


def _stage7_storyline_html_row(row: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{escape(str(row.get('story_order') or ''))}</td>"
        f"<td class='tribe'>T{escape(str(row.get('tribe_id')))}<br>{escape(str(row.get('working_tribe_name') or ''))}</td>"
        f"<td>{escape(str(row.get('story_role') or ''))}</td>"
        f"<td>{escape(str(row.get('evidence_strength') or ''))}<br><span class='muted'>{escape(str(row.get('readiness') or ''))}</span></td>"
        f"<td>{escape(str(row.get('audience_size') or ''))}<br><span class='muted'>{escape(str(row.get('core_assignment_read') or ''))}</span></td>"
        f"<td>{escape(str(row.get('distinctive_product_evidence') or ''))}</td>"
        f"<td>{escape(str(row.get('supporting_theme_term_evidence') or ''))}<br><span class='muted'>{escape(str(row.get('customer_context') or ''))}</span></td>"
        f"<td>{escape(str(row.get('subsegment_overlay') or ''))}</td>"
        f"<td class='caveat'>{escape(str(row.get('caveat') or ''))}</td>"
        "</tr>"
    )


def write_campaign_signal_artifacts(
    profile_path: str | Path,
    campaign_themes: set[str] | list[str] | tuple[str, ...] | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write opt-in activation-oriented overlays from product-theme evidence."""

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    campaign_set = _campaign_signal_theme_set(campaign_themes, cfg=cfg)
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
                    "working_tribe_name": _working_tribe_name(row),
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
        "Stage 7 exports",
        "wrote campaign signal overlays",
        cfg=cfg,
        csv=base_output,
        markdown=md_output,
        html=html_output,
    )
    return {"csv": base_output, "markdown": md_output, "html": html_output}


def _campaign_signal_theme_set(
    campaign_themes: set[str] | list[str] | tuple[str, ...] | None,
    *,
    cfg: PipelineConfig = CONFIG,
) -> set[str]:
    if campaign_themes is None:
        campaign_themes = cfg.get("profiling.campaign_signal_themes", []) or []
    return {str(theme) for theme in campaign_themes}


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
                    "working_tribe_name": _working_tribe_name(row),
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
        "Stage 7 exports",
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
        customer_tags_lf = None
    elif definitions.is_empty():
        customer_tags_lf = None
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
            customer_tags_lf = (
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
            )
        else:
            customer_tags_lf = None

    parquet_output.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if customer_tags_lf is None:
        customer_tags = _empty_subsegment_tags()
        summary = _subsegment_summary(customer_tags, profiles)
        customer_tags.write_parquet(parquet_output)
    else:
        summary = _subsegment_summary_lazy(customer_tags_lf, profiles)
        _sink_lazy_parquet(customer_tags_lf, parquet_output)
    summary.write_csv(summary_csv)
    summary_md.write_text(_subsegment_summary_markdown(summary, cfg=cfg), encoding="utf-8")
    summary_html.write_text(_subsegment_summary_html(summary, cfg=cfg), encoding="utf-8")
    log_event(
        "Stage 7 exports",
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
        name = _working_tribe_name(row)
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
                f"- Name source: {row.get('suggested_tribe_name_source') or 'n/a'}",
                f"- Name evidence: {row.get('suggested_tribe_name_evidence') or 'n/a'}",
                f"- Name status: {row.get('suggested_tribe_name_status') or 'n/a'}",
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
    log_event("Stage 7 exports", "wrote tribe profile evidence report", cfg=cfg, path=output)
    return output


def write_llm_profile_interpretation_pack(
    profile_path: str | Path,
    readiness_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write sanitized prompts for optional LLM-assisted Stage 7 interpretation.

    The pack contains aggregate evidence only. It intentionally excludes customer ids
    and raw transaction rows, and it instructs the LLM not to infer demographics or
    unsupported personas.
    """

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    readiness = _read_optional_table(readiness_path)
    comparison = _read_optional_table(comparison_path)
    base_output = (
        Path(output_json)
        if output_json
        else cfg.artifacts / "stage7" / f"stage7_llm_interpretation_pack_{cfg.mode}.json"
    )
    md_output = Path(output_md) if output_md else base_output.with_suffix(".md")
    pack = _llm_interpretation_payload(
        profiles,
        readiness,
        comparison,
        profile_path=Path(profile_path),
        readiness_path=Path(readiness_path) if readiness_path else None,
        comparison_path=Path(comparison_path) if comparison_path else None,
        cfg=cfg,
    )

    base_output.parent.mkdir(parents=True, exist_ok=True)
    base_output.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    md_output.write_text(_llm_interpretation_markdown(pack), encoding="utf-8")
    log_event(
        "Stage 7 exports",
        "wrote LLM interpretation prompt pack",
        cfg=cfg,
        json=base_output,
        markdown=md_output,
    )
    return {"json": base_output, "markdown": md_output}


def _llm_interpretation_payload(
    profiles: pl.DataFrame,
    readiness: pl.DataFrame,
    comparison: pl.DataFrame,
    *,
    profile_path: Path,
    readiness_path: Path | None,
    comparison_path: Path | None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    total_customers = int(profiles["profile_population_customers"][0]) if profiles.height else 0
    assigned_customers = int(profiles["n_customers"].sum()) if profiles.height else 0
    noise_customers = int(profiles["profile_noise_customers"][0]) if profiles.height else 0
    tribes = [
        _llm_tribe_payload(row, profiles, readiness, comparison, cfg=cfg)
        for row in profiles.iter_rows(named=True)
    ]
    return {
        "stage": "7_deep_tribe_profiling",
        "mode": cfg.mode,
        "source_artifacts": {
            "profile_path": str(profile_path),
            "readiness_path": str(readiness_path) if readiness_path else None,
            "comparison_path": str(comparison_path) if comparison_path else None,
        },
        "global_context": {
            "profiled_tribes": profiles.height,
            "profiled_customers": total_customers,
            "assigned_core_customers": assigned_customers,
            "remaining_noise_customers": noise_customers,
            "assignment_policy": "hard organic HDBSCAN core tribes; noise remains unassigned",
            "modeling_signal": (
                "Product identity plus unidades with configured recency and product-frequency weighting. "
                "No importe, spend, revenue tier, demographics, or customer identity fields are used for clustering."
            ),
        },
        "llm_guardrails": _llm_guardrails(),
        "recommended_output_schema": {
            "tribe_id": "integer",
            "working_name": "short evidence-backed label",
            "plain_language_summary": "2-4 sentences grounded only in supplied evidence",
            "distinctive_purchase_signal": "specific products/terms/themes that separate the tribe",
            "what_not_to_claim": "unsupported demographic/persona/identity claims to avoid",
            "activation_hypotheses": "evidence-backed CRM/category-test ideas, phrased as hypotheses",
            "confidence_read": "high/medium/low with caveats from readiness and lift evidence",
            "expert_review_questions": "questions for Carrefour category experts",
        },
        "tribes": tribes,
    }


def _llm_guardrails() -> list[str]:
    return [
        "Use only the aggregate evidence supplied in this pack.",
        "Do not infer age, gender, income, family status, nationality, religion, health status, or other demographic/identity traits.",
        "Do not claim that a lifted product proves a lifestyle or identity; phrase commercial reads as hypotheses.",
        "Preserve uncertainty from Stage 6.4 readiness, assignment confidence, noise coverage, and profile-readiness issues.",
        "Name tribes from product, product-term, category, sector, or theme evidence only.",
        "Mention spend or KPIs only as post-clustering context, never as clustering drivers.",
        "Flag sibling or overlapping tribes instead of forcing artificial differentiation.",
    ]


def _llm_tribe_payload(
    row: dict[str, Any],
    profiles: pl.DataFrame,
    readiness: pl.DataFrame,
    comparison: pl.DataFrame,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    tribe_id = int(row.get("tribe_id"))
    readiness_row = _row_by_tribe(readiness, tribe_id)
    comparison_row = _row_by_tribe(comparison, tribe_id)
    evidence = {
        "top_products": _llm_lift_items(
            row.get("top_products"),
            row.get("top_product_lifts_vs_rest") or row.get("top_product_lifts"),
            row.get("top_product_customer_counts"),
            row.get("top_product_q_values"),
            limit=8,
        ),
        "top_product_terms": _llm_lift_items(
            row.get("top_product_terms"),
            row.get("top_product_term_lifts_vs_rest") or row.get("top_product_term_lifts"),
            row.get("top_product_term_customer_counts"),
            row.get("top_product_term_q_values"),
            limit=8,
        ),
        "top_themes": _llm_lift_items(
            row.get("top_themes"),
            row.get("top_theme_lifts_vs_rest") or row.get("top_theme_lifts"),
            row.get("top_theme_customer_counts"),
            row.get("top_theme_q_values"),
            limit=8,
            label_transform=lambda value: str(value).replace("_", " "),
        ),
        "top_sectors": _llm_lift_items(
            row.get("top_sectors"),
            row.get("top_sector_lifts_vs_rest") or row.get("top_sector_lifts"),
            None,
            row.get("top_sector_q_values"),
            limit=6,
        ),
    }
    prompt = _llm_tribe_prompt(tribe_id, row, readiness_row, comparison_row, evidence, cfg=cfg)
    return {
        "tribe_id": tribe_id,
        "working_name": _working_tribe_name(row),
        "working_name_source": row.get("suggested_tribe_name_source"),
        "working_name_evidence": row.get("suggested_tribe_name_evidence"),
        "working_name_status": row.get("suggested_tribe_name_status"),
        "label_confidence": _label_confidence(row, cfg=cfg),
        "size": {
            "customers": int(row.get("n_customers") or 0),
            "population_share_pct": round(float(row.get("population_share") or 0.0) * 100.0, 2),
            "core_customers": int(row.get("core_customers") or 0),
            "soft_assigned_customers": int(row.get("soft_assigned_customers") or 0),
            "soft_assigned_share_pct": round(float(row.get("soft_assigned_share") or 0.0) * 100.0, 2),
        },
        "stage6_readiness": {
            "profile_readiness": readiness_row.get("stage6_profile_readiness")
            or readiness_row.get("profile_readiness"),
            "profiling_readiness": readiness_row.get("profiling_readiness"),
            "readiness_issues": readiness_row.get("profiling_readiness_issues")
            or readiness_row.get("readiness_issues"),
            "mean_assignment_confidence": _round_optional(readiness_row.get("mean_assignment_confidence")),
            "p10_assignment_confidence": _round_optional(readiness_row.get("p10_assignment_confidence")),
            "jitter_label_recovery_accuracy_mean": _round_optional(
                readiness_row.get("jitter_label_recovery_accuracy_mean")
            ),
        },
        "evidence": evidence,
        "behavior_context": {
            "summary": _behavior_ratio_summary(row, profiles, limit=5),
            "profile_statistical_read": comparison_row.get("profile_statistical_read")
            or _profile_statistical_read(row, cfg=cfg),
        },
        "interpretation_note": _interpretation_note(row, cfg=cfg),
        "prompt": prompt,
    }


def _row_by_tribe(frame: pl.DataFrame, tribe_id: int) -> dict[str, Any]:
    if frame.is_empty() or "tribe_id" not in frame.columns:
        return {}
    rows = frame.filter(pl.col("tribe_id") == tribe_id)
    return rows.row(0, named=True) if rows.height else {}


def _llm_lift_items(
    names: list[Any] | None,
    lifts: list[Any] | None,
    counts: list[Any] | None,
    q_values: list[Any] | None,
    *,
    limit: int,
    label_transform: Any | None = None,
) -> list[dict[str, Any]]:
    result = []
    names = names or []
    lifts = lifts or []
    counts = counts or []
    q_values = q_values or []
    for idx, name in enumerate(names[:limit]):
        label = label_transform(name) if label_transform else _repair_display_text(str(name))
        result.append(
            {
                "label": label,
                "lift": _round_optional(lifts[idx]) if idx < len(lifts) else None,
                "customers": int(counts[idx]) if idx < len(counts) and counts[idx] is not None else None,
                "q_value": _round_optional(q_values[idx], digits=6) if idx < len(q_values) else None,
            }
        )
    return result


def _llm_tribe_prompt(
    tribe_id: int,
    row: dict[str, Any],
    readiness_row: dict[str, Any],
    comparison_row: dict[str, Any],
    evidence: dict[str, Any],
    cfg: PipelineConfig = CONFIG,
) -> str:
    payload = {
        "tribe_id": tribe_id,
        "working_name": _working_tribe_name(row),
        "size": {
            "customers": int(row.get("n_customers") or 0),
            "population_share_pct": round(float(row.get("population_share") or 0.0) * 100.0, 2),
        },
        "readiness": {
            "stage6_profile_readiness": readiness_row.get("stage6_profile_readiness")
            or readiness_row.get("profile_readiness"),
            "profiling_readiness": readiness_row.get("profiling_readiness"),
            "issues": readiness_row.get("profiling_readiness_issues") or readiness_row.get("readiness_issues"),
        },
        "evidence": evidence,
        "behavior_context": comparison_row.get("customer_behavior_over_under_index") or _format_behavior_context(row),
    }
    return (
        "You are helping interpret Carrefour product-first customer tribes.\n"
        "Use only the aggregate evidence below. Do not infer demographics, household status, identity, religion, "
        "health status, nationality, age, gender, or income. Do not invent unsupported motivations.\n"
        "Return JSON with: working_name, sharper_name_options, plain_language_summary, distinctive_purchase_signal, "
        "what_not_to_claim, activation_hypotheses, confidence_read, expert_review_questions.\n\n"
        f"Evidence:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )

def _llm_interpretation_markdown(pack: dict[str, Any]) -> str:
    lines = [
        "# Stage 7 LLM Interpretation Prompt Pack",
        "",
        f"Run mode: `{pack.get('mode')}`",
        "",
        "This artifact is for optional LLM-assisted interpretation of aggregate tribe evidence. "
        "It contains no customer IDs or raw transaction rows.",
        "",
        "## Guardrails",
        "",
    ]
    lines.extend(f"- {item}" for item in pack.get("llm_guardrails", []))
    context = pack.get("global_context", {})
    lines.extend(
        [
            "",
            "## Global Context",
            "",
            f"- Profiled tribes: {context.get('profiled_tribes')}",
            f"- Profiled customers: {context.get('profiled_customers')}",
            f"- Assigned core customers: {context.get('assigned_core_customers')}",
            f"- Remaining noise customers: {context.get('remaining_noise_customers')}",
            f"- Assignment policy: {context.get('assignment_policy')}",
            "",
            "## Tribe Prompts",
            "",
        ]
    )
    for tribe in pack.get("tribes", []):
        lines.extend(
            [
                f"### Tribe {tribe.get('tribe_id')}: {tribe.get('working_name')}",
                "",
                "```text",
                tribe.get("prompt", ""),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def write_clustering_atlas_artifacts(
    profile_path: str | Path,
    subsegment_summary_path: str | Path | None = None,
    figure_paths: dict[str, Any] | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    write_markdown: bool = True,
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
    if write_markdown:
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
        "Stage 7 exports",
        "wrote clustering atlas",
        cfg=cfg,
        csv=csv_output,
        markdown=md_output if write_markdown else None,
        html=html_output,
    )
    result = {"csv": csv_output, "html": html_output}
    if write_markdown:
        result["markdown"] = md_output
    return result


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
                "working_tribe_name": _working_tribe_name(row),
                "working_tribe_name_source": row.get("suggested_tribe_name_source"),
                "working_tribe_name_status": row.get("suggested_tribe_name_status"),
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
        "working_tribe_name_source",
        "working_tribe_name_status",
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
        name = _working_tribe_name(row)
        tribe_subsegments = _tribe_subsegments(subsegments, tribe_id, limit=12)
        lines.extend(
            [
                f"### Tribe {tribe_id}: {name}",
                "",
                f"- Customers: {int(row.get('n_customers') or 0):,} ({_fmt_pct(row.get('population_share'))})",
                f"- Assignment: {int(row.get('core_customers') or 0):,} core / {int(row.get('soft_assigned_customers') or 0):,} soft-assigned ({_fmt_pct(row.get('soft_assigned_share'))})",
                f"- Label confidence: {_label_confidence(row, cfg=cfg)}",
                f"- Name source: {row.get('suggested_tribe_name_source') or 'n/a'}",
                f"- Name evidence: {row.get('suggested_tribe_name_evidence') or 'n/a'}",
                f"- Name status: {row.get('suggested_tribe_name_status') or 'n/a'}",
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
    name = _working_tribe_name(row)
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
<span class="pill">name source: {escape(str(row.get('suggested_tribe_name_source') or 'n/a'))}</span>
<span class="pill">{escape(str(row.get('suggested_tribe_name_status') or 'n/a'))}</span>
<span class="pill">{int(row.get('core_customers') or 0):,} core / {int(row.get('soft_assigned_customers') or 0):,} soft-assigned</span>
</div>
<div class="evidence">
<div><strong>Name evidence</strong><br>{escape(str(row.get('suggested_tribe_name_evidence') or 'n/a'))}</div>
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
        "suggested_tribe_name_source",
        "suggested_tribe_name_status",
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
        name = _working_tribe_name(row)
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


def _subsegment_summary_lazy(customer_tags: pl.LazyFrame, profiles: pl.DataFrame) -> pl.DataFrame:
    tribe_sizes = profiles.select(["tribe_id", "n_customers"]).rename({"n_customers": "tribe_customers"}).lazy()
    summary = collect_streaming(
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
        .sort(
            ["tribe_id", "subsegment_type", "subsegment_lift", "subsegment_customers"],
            descending=[False, False, True, True],
        )
    )
    return summary if not summary.is_empty() else _subsegment_summary(_empty_subsegment_tags(), profiles)


def _sink_lazy_parquet(lf: pl.LazyFrame, output: Path) -> None:
    try:
        lf.sink_parquet(str(output))
    except TypeError:
        collect_streaming(lf).write_parquet(output)


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


def _tribe_product_summary_frame(
    row: dict[str, Any],
    *,
    max_products: int,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    tribe_id = int(row.get("tribe_id"))
    n_customers = max(int(row.get("n_customers") or 0), 1)
    product_ids = row.get("top_product_ids") or []
    products = row.get("top_products") or []
    sectors = row.get("top_product_sectors") or []
    sector_ids = row.get("top_product_sector_ids") or []
    lifts = row.get("top_product_lifts") or []
    lifts_vs_rest = row.get("top_product_lifts_vs_rest") or []
    q_values = row.get("top_product_q_values") or []
    counts = row.get("top_product_customer_counts") or []
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))

    rows = []
    for idx, product_id in enumerate(product_ids[:max_products]):
        count = _int_at(counts, idx)
        lift_vs_rest = _numeric_at(lifts_vs_rest, idx)
        q_value = _numeric_at(q_values, idx)
        rows.append(
            {
                "tribe_id": tribe_id,
                "working_tribe_name": _working_tribe_name(row),
                "rank": idx + 1,
                "product_id": str(product_id),
                "product_description": _text_at(products, idx),
                "category": _text_at(sectors, idx),
                "category_id": _text_at(sector_ids, idx),
                "tribe_customers": n_customers,
                "product_customers": count,
                "product_coverage_pct": round(count / n_customers * 100.0, 1),
                "lift_vs_population": _round_optional(_numeric_at(lifts, idx)),
                "lift_vs_rest": _round_optional(lift_vs_rest),
                "q_value": _round_optional(q_value, 6),
                "statistical_result": _overindex_evidence_label(lift_vs_rest, q_value, q_threshold),
            }
        )
    return pl.DataFrame(rows) if rows else _empty_tribe_product_summary()


def _empty_tribe_product_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "tribe_id": pl.Int64,
            "working_tribe_name": pl.Utf8,
            "rank": pl.Int64,
            "product_id": pl.Utf8,
            "product_description": pl.Utf8,
            "category": pl.Utf8,
            "category_id": pl.Utf8,
            "tribe_customers": pl.Int64,
            "product_customers": pl.Int64,
            "product_coverage_pct": pl.Float64,
            "lift_vs_population": pl.Float64,
            "lift_vs_rest": pl.Float64,
            "q_value": pl.Float64,
            "statistical_result": pl.Utf8,
        }
    )


def _tribe_comparison_row(row: dict[str, Any], profiles: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    product_lifts = row.get("top_product_lifts") or []
    product_lifts_vs_rest = row.get("top_product_lifts_vs_rest") or []
    product_q_values = row.get("top_product_q_values") or []
    sector_lifts_vs_rest = row.get("top_sector_lifts_vs_rest") or []
    theme_lifts_vs_rest = row.get("top_theme_lifts_vs_rest") or []
    term_lifts_vs_rest = row.get("top_product_term_lifts_vs_rest") or []
    return {
        "tribe_id": row.get("tribe_id"),
        "working_tribe_name": _working_tribe_name(row),
        "working_tribe_name_source": row.get("suggested_tribe_name_source"),
        "working_tribe_name_evidence": row.get("suggested_tribe_name_evidence"),
        "working_tribe_name_status": row.get("suggested_tribe_name_status"),
        "label_confidence": _label_confidence(row, cfg=cfg),
        "n_customers": row.get("n_customers"),
        "population_share_pct": round(float(row.get("population_share") or 0.0) * 100.0, 2),
        "core_customers": row.get("core_customers"),
        "soft_assigned_customers": row.get("soft_assigned_customers"),
        "soft_assigned_share_pct": round(float(row.get("soft_assigned_share") or 0.0) * 100.0, 2),
        "soft_assignment_confidence_mean": _round_optional(row.get("soft_assignment_confidence_mean")),
        "top_product_and_category_evidence": _top_product_category_evidence(row, limit=4),
        "top_sector_evidence": _format_lift_items(row.get("top_sectors"), row.get("top_sector_lifts"), limit=3),
        "top_data_driven_term_evidence": _format_term_evidence(row, cfg=cfg),
        "top_theme_evidence": _format_theme_evidence(row),
        "max_product_lift_vs_rest": _round_optional(_max_numeric(product_lifts_vs_rest)),
        "max_sector_lift_vs_rest": _round_optional(_max_numeric(sector_lifts_vs_rest)),
        "max_theme_lift_vs_rest": _round_optional(_max_numeric(theme_lifts_vs_rest)),
        "max_term_lift_vs_rest": _round_optional(_max_numeric(term_lifts_vs_rest)),
        "min_product_q_value": _round_optional(_min_numeric(product_q_values), 6),
        "significant_product_overindex_count": _significant_strong_count(
            product_lifts,
            product_q_values,
            threshold=float(cfg.get("profiling.strong_product_lift_threshold", 1.5)),
            q_threshold=q_threshold,
        ),
        "customer_behavior_over_under_index": _behavior_ratio_summary(row, profiles, limit=4),
        "profile_statistical_read": _profile_statistical_read(row, cfg=cfg),
        "interpretation_note": _interpretation_note(row, cfg=cfg),
    }


def _top_product_category_evidence(row: dict[str, Any], limit: int = 4) -> str:
    products = row.get("top_products") or []
    sectors = row.get("top_product_sectors") or []
    lifts_vs_rest = row.get("top_product_lifts_vs_rest") or []
    q_values = row.get("top_product_q_values") or []
    counts = row.get("top_product_customer_counts") or []
    parts = []
    for idx, product in enumerate(products[:limit]):
        sector = _text_at(sectors, idx) or "unknown category"
        lift = _numeric_at(lifts_vs_rest, idx)
        q_value = _numeric_at(q_values, idx)
        count = _int_at(counts, idx)
        stat = _format_q_value(q_value)
        lift_text = f"{lift:.1f}x vs rest" if lift is not None else "lift n/a"
        count_text = f"{count:,} customers" if count else "customers n/a"
        parts.append(f"{_repair_display_text(str(product))} ({sector}; {lift_text}; {stat}; {count_text})")
    return "; ".join(parts)


def _profile_statistical_read(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> str:
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    product_count = _significant_strong_count(
        row.get("top_product_lifts") or [],
        row.get("top_product_q_values") or [],
        threshold=float(cfg.get("profiling.strong_product_lift_threshold", 1.5)),
        q_threshold=q_threshold,
    )
    theme_count = _significant_strong_count(
        row.get("top_theme_lifts") or [],
        row.get("top_theme_q_values") or [],
        threshold=float(cfg.get("profiling.strong_theme_lift_threshold", 1.2)),
        q_threshold=q_threshold,
    )
    term_count = _significant_strong_count(
        row.get("top_product_term_lifts") or [],
        row.get("top_product_term_q_values") or [],
        threshold=float(cfg.get("profiling.strong_term_lift_threshold", 1.25)),
        q_threshold=q_threshold,
    )
    return f"{product_count} product, {theme_count} theme, {term_count} product-term signals pass lift and q<={q_threshold:g}."


def _behavior_ratio_summary(row: dict[str, Any], profiles: pl.DataFrame, limit: int = 4) -> str:
    items = []
    for column, label in _profile_behavior_fields(profiles):
        value = _numeric_or_none(row.get(column))
        rest_mean = _weighted_rest_mean(profiles, column, int(row.get("tribe_id")))
        if value is None or rest_mean is None or rest_mean <= 0:
            continue
        ratio = value / rest_mean
        if ratio >= 1.10 or ratio <= 0.90:
            items.append((abs(math.log(max(ratio, 1e-9))), f"{label} {ratio:.2f}x rest"))
    items.sort(key=lambda item: item[0], reverse=True)
    return "; ".join(item[1] for item in items[:limit]) or "No large behavior/KPI ratio vs rest."


def _profile_behavior_fields(profiles: pl.DataFrame) -> list[tuple[str, str]]:
    candidates = [
        ("avg_ticket_count", "Tickets"),
        ("avg_total_units", "Units"),
        ("avg_unique_products", "Unique products"),
        ("avg_unique_sectors", "Unique sectors"),
        ("avg_frequency_per_30d", "Frequency"),
        ("avg_promo_share", "Promo share"),
        ("avg_avg_basket_value", "Basket value"),
    ]
    return [(column, label) for column, label in candidates if column in profiles.columns]


def _weighted_rest_mean(profiles: pl.DataFrame, column: str, tribe_id: int) -> float | None:
    if column not in profiles.columns:
        return None
    total_weight = 0.0
    total_value = 0.0
    own_weight = 0.0
    own_value = 0.0
    for row in profiles.iter_rows(named=True):
        value = _numeric_or_none(row.get(column))
        weight = _numeric_or_none(row.get("n_customers")) or 0.0
        if value is None or weight <= 0:
            continue
        total_weight += weight
        total_value += value * weight
        if int(row.get("tribe_id")) == tribe_id:
            own_weight += weight
            own_value += value * weight
    rest_weight = total_weight - own_weight
    if rest_weight <= 0:
        return None
    return (total_value - own_value) / rest_weight


def _numeric_behavior_columns(behavior: pl.LazyFrame) -> list[str]:
    schema = behavior.collect_schema()
    columns = []
    for name, dtype in schema.items():
        if name in {"cliente", "tribe_id"}:
            continue
        if _is_numeric_dtype(dtype):
            columns.append(name)
    return columns


def _is_numeric_dtype(dtype: Any) -> bool:
    checker = getattr(dtype, "is_numeric", None)
    if callable(checker):
        return bool(checker())
    return dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }


def _anova_result_row(metric: str, grouped: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    valid = [
        row
        for row in grouped.iter_rows(named=True)
        if int(row.get("n_customers") or 0) > 1 and _numeric_or_none(row.get("mean")) is not None
    ]
    if len(valid) < 2:
        return {
            "metric": metric,
            "included_tribes": len(valid),
            "included_customers": sum(int(row.get("n_customers") or 0) for row in valid),
            "global_mean": None,
            "anova_f_statistic": None,
            "anova_p_value": None,
            "anova_effect_eta_squared": None,
            "highest_mean_tribe_id": None,
            "highest_mean": None,
            "highest_mean_ratio_vs_rest": None,
            "lowest_mean_tribe_id": None,
            "lowest_mean": None,
            "lowest_mean_ratio_vs_rest": None,
        }

    total_n = sum(int(row.get("n_customers") or 0) for row in valid)
    total_sum = sum(int(row.get("n_customers") or 0) * float(row.get("mean")) for row in valid)
    global_mean = total_sum / max(total_n, 1)
    ss_between = sum(int(row.get("n_customers") or 0) * (float(row.get("mean")) - global_mean) ** 2 for row in valid)
    ss_within = sum(
        max(int(row.get("n_customers") or 0) - 1, 0) * max(_numeric_or_none(row.get("variance")) or 0.0, 0.0)
        for row in valid
    )
    df_between = len(valid) - 1
    df_within = total_n - len(valid)
    f_statistic = None
    p_value = None
    if df_between > 0 and df_within > 0 and ss_within > 0:
        f_statistic = (ss_between / df_between) / (ss_within / df_within)
        p_value = _f_distribution_sf(f_statistic, df_between, df_within)
    eta_squared = ss_between / (ss_between + ss_within) if ss_between + ss_within > 0 else None

    ratio_rows = []
    for row in valid:
        n = int(row.get("n_customers") or 0)
        mean = float(row.get("mean"))
        rest_n = total_n - n
        rest_mean = (total_sum - n * mean) / rest_n if rest_n > 0 else None
        ratio = mean / rest_mean if rest_mean and rest_mean > 0 else None
        ratio_rows.append({**row, "ratio_vs_rest": ratio})

    highest = max(ratio_rows, key=lambda item: _numeric_or_none(item.get("mean")) or float("-inf"))
    lowest = min(ratio_rows, key=lambda item: _numeric_or_none(item.get("mean")) or float("inf"))
    return {
        "metric": metric,
        "included_tribes": len(valid),
        "included_customers": total_n,
        "global_mean": _round_optional(global_mean, 4),
        "anova_f_statistic": _round_optional(f_statistic, 4),
        "anova_p_value": _round_optional(p_value, 8),
        "anova_effect_eta_squared": _round_optional(eta_squared, 4),
        "highest_mean_tribe_id": highest.get("tribe_id"),
        "highest_mean": _round_optional(highest.get("mean"), 4),
        "highest_mean_ratio_vs_rest": _round_optional(highest.get("ratio_vs_rest"), 3),
        "lowest_mean_tribe_id": lowest.get("tribe_id"),
        "lowest_mean": _round_optional(lowest.get("mean"), 4),
        "lowest_mean_ratio_vs_rest": _round_optional(lowest.get("ratio_vs_rest"), 3),
    }


def _f_distribution_sf(f_statistic: float, df_between: int, df_within: int) -> float | None:
    try:
        from scipy.stats import f as f_distribution

        return float(f_distribution.sf(f_statistic, df_between, df_within))
    except Exception:
        return None


def _anova_result_label(
    p_value: float | None,
    q_value: float | None,
    eta_squared: float | None,
    q_threshold: float,
) -> str:
    if p_value is None:
        return "effect_size_only_no_p_value_backend"
    if q_value is not None and q_value <= q_threshold:
        if eta_squared is not None and eta_squared >= 0.06:
            return "significant_between_tribe_difference_medium_effect"
        return "significant_between_tribe_difference_small_effect"
    return "not_significant_after_fdr"


def _empty_customer_metric_tests() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "metric": pl.Utf8,
            "included_tribes": pl.Int64,
            "included_customers": pl.Int64,
            "global_mean": pl.Float64,
            "anova_f_statistic": pl.Float64,
            "anova_p_value": pl.Float64,
            "anova_q_value": pl.Float64,
            "anova_effect_eta_squared": pl.Float64,
            "highest_mean_tribe_id": pl.Int64,
            "highest_mean": pl.Float64,
            "highest_mean_ratio_vs_rest": pl.Float64,
            "lowest_mean_tribe_id": pl.Int64,
            "lowest_mean": pl.Float64,
            "lowest_mean_ratio_vs_rest": pl.Float64,
            "statistical_result": pl.Utf8,
        }
    )


def _comparison_row(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    name = _working_tribe_name(row)
    return {
        "tribe_id": row.get("tribe_id"),
        "working_tribe_name": name,
        "working_tribe_name_source": row.get("suggested_tribe_name_source"),
        "working_tribe_name_evidence": row.get("suggested_tribe_name_evidence"),
        "working_tribe_name_status": row.get("suggested_tribe_name_status"),
        "working_tribe_name_duplicate_count": row.get("suggested_tribe_name_duplicate_count"),
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


def _working_tribe_name(row: dict[str, Any]) -> str:
    return str(row.get("suggested_tribe_name") or fallback_tribe_name(row))


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
        "profiling_readiness",
        "n_customers",
        "population_share_pct",
        "soft_assigned_share_pct",
        "top_product_and_category_evidence",
        "top_sector_evidence",
        "customer_behavior_over_under_index",
        "profile_statistical_read",
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
                f"<span class='muted'>{escape(str(row.get('label_confidence') or ''))} label confidence</span><br>"
                f"<span class='muted'>{escape(str(row.get('profiling_readiness') or 'readiness n/a'))}</span><br>"
                f"<span class='muted'>source: {escape(str(row.get('working_tribe_name_source') or 'n/a'))}</span><br>"
                f"<span class='muted'>{escape(str(row.get('working_tribe_name_status') or 'n/a'))}</span></td>"
                f"<td>{int(row.get('n_customers') or 0):,}<br><span class='muted'>{float(row.get('population_share_pct') or 0):.2f}%</span></td>"
                f"<td>{float(row.get('soft_assigned_share_pct') or 0):.1f}%<br><span class='muted'>mean confidence {_fmt_number(row.get('soft_assignment_confidence_mean'))}</span></td>"
                f"<td>{escape(str(row.get('top_product_and_category_evidence') or ''))}</td>"
                f"<td>{escape(str(row.get('top_sector_evidence') or ''))}</td>"
                f"<td>{escape(str(row.get('customer_behavior_over_under_index') or ''))}</td>"
                f"<td>{escape(str(row.get('profile_statistical_read') or ''))}</td>"
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
<th>Top Product/Category Evidence</th>
<th>Sector Evidence</th>
<th>Customer Behavior Contrast</th>
<th>Statistical Read</th>
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
    significant_product_lift_counts = []
    significant_theme_lift_counts = []
    significant_term_lift_counts = []
    max_product_lifts = []
    max_product_lifts_vs_rest = []
    soft_assigned_shares = []
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
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
        significant_product_lift_counts.append(
            _significant_strong_count(
                product_lifts,
                row.get("top_product_q_values") or [],
                threshold=strong_product_lift,
                q_threshold=q_threshold,
            )
        )
        significant_theme_lift_counts.append(
            _significant_strong_count(
                theme_lifts,
                row.get("top_theme_q_values") or [],
                threshold=strong_theme_lift,
                q_threshold=q_threshold,
            )
        )
        significant_term_lift_counts.append(
            _significant_strong_count(
                term_lifts,
                row.get("top_product_term_q_values") or [],
                threshold=float(cfg.get("profiling.strong_term_lift_threshold", 1.25)),
                q_threshold=q_threshold,
            )
        )
        max_product_lifts.append(max(product_lifts) if product_lifts else 0.0)
        rest_lifts = [value for value in (row.get("top_product_lifts_vs_rest") or []) if value is not None]
        max_product_lifts_vs_rest.append(max(rest_lifts) if rest_lifts else 0.0)
        soft_assigned_shares.append(float(row.get("soft_assigned_share") or 0.0))
    return {
        "profiled_clusters": profiles.height,
        "clusters_with_product_lift": sum(1 for count in product_lift_counts if count > 0),
        "clusters_with_sector_lift": sum(1 for count in sector_lift_counts if count > 0),
        "clusters_with_theme_lift": sum(1 for count in theme_lift_counts if count > 0),
        "clusters_with_data_driven_term_lift": sum(1 for count in term_lift_counts if count > 0),
        "clusters_with_significant_product_lift": sum(1 for count in significant_product_lift_counts if count > 0),
        "clusters_with_significant_theme_lift": sum(1 for count in significant_theme_lift_counts if count > 0),
        "clusters_with_significant_data_driven_term_lift": sum(
            1 for count in significant_term_lift_counts if count > 0
        ),
        "avg_strong_product_lifts_per_cluster": float(sum(product_lift_counts) / max(profiles.height, 1)),
        "avg_strong_theme_lifts_per_cluster": float(sum(theme_lift_counts) / max(profiles.height, 1)),
        "avg_strong_data_driven_term_lifts_per_cluster": float(sum(term_lift_counts) / max(profiles.height, 1)),
        "avg_significant_product_lifts_per_cluster": float(
            sum(significant_product_lift_counts) / max(profiles.height, 1)
        ),
        "avg_significant_theme_lifts_per_cluster": float(
            sum(significant_theme_lift_counts) / max(profiles.height, 1)
        ),
        "avg_max_product_lift": float(sum(max_product_lifts) / max(profiles.height, 1)),
        "avg_max_product_lift_vs_rest": float(sum(max_product_lifts_vs_rest) / max(profiles.height, 1)),
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


def _min_numeric(values: list[Any]) -> float | None:
    numeric = [_numeric_or_none(value) for value in values]
    finite = [value for value in numeric if value is not None]
    return min(finite) if finite else None


def _fmt_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _repair_display_text(text: str) -> str:
    repaired = text
    for _ in range(3):
        if "Ãƒ" not in repaired and "Ã‚" not in repaired:
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

