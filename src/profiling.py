"""Organic Stage 7 tribe profiling from transaction-derived evidence only."""

from __future__ import annotations

import json
import math
import re
import shutil
import warnings
from html import escape
from pathlib import Path
from typing import Any, Mapping

import polars as pl
from dotenv import load_dotenv

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.organic_profiler import (
    _compute_copurchase_missions,
    _compute_loyalty_profile,
    _compute_temporal_patterns,
    synthesize_tribe_with_claude,
    synthesize_tribe_with_gemini,
)
from src.progress import log_event, stage_timer
from src.product_themes import detect_product_themes
from src.tribe_namer import THEME_LABELS
from src.utils import collect_streaming, file_fingerprint, schema_names, should_use_cache, write_artifact_metadata


BEHAVIOR_OUTPUT_MAP = {
    "ticket_count": "avg_ticket_count",
    "total_spend": "avg_total_spend",
    "avg_basket_value": "avg_basket_value",
    "promo_share": "avg_promo_share",
    "unique_products": "avg_unique_products",
    "unique_sectors": "avg_unique_sectors",
    "recency_days": "avg_recency_days",
    "frequency_per_30d": "avg_frequency_per_30d",
}

PRODUCT_RANKING_FORMULA = "lift_vs_rest x log(customer_count + 1)"
PRODUCT_RANKING_BASIS = (
    "Product rows are ranked by lift_vs_rest x log(customer_count + 1), favoring distinctive products "
    "with enough tribe reach rather than the most-bought products."
)
PRODUCT_RANKING_CARD_NOTE = "Ranked by lift_vs_rest x log(customer_count + 1); distinctive, not most-bought."
STAGE7_STAGE68_REQUIRED_MESSAGE = "Stage 6.8 must run before Stage 7. Run run_stage68() first."


def profile_tribes(
    assignments_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    behavior_path: str | Path | None = None,
    output_path: str | Path | None = None,
    force: bool | None = None,
    enable_copurchase: bool = True,
    enable_temporal: bool = True,
    enable_loyalty: bool = True,
    enable_llm: bool = False,
    llm_api_key: str | None = None,
    llm_model: str = "claude-sonnet-4-6",
    llm_provider: str | None = None,
    llm_api_key_env: str | None = None,
    min_copurchase_baskets: int = 5,
    top_n_products: int = 15,
    top_n_sectors: int = 5,
    top_n_copurchase_pairs: int = 10,
    product_lifts_output_path: str | Path | None = None,
    sector_lifts_output_path: str | Path | None = None,
    stage_label: str = "Stage 7 profiling",
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create one row per hard organic tribe using only transaction-derived evidence."""

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
    effective_llm_provider = _resolve_llm_provider(enable_llm=enable_llm, llm_provider=llm_provider, cfg=cfg)
    effective_llm_model = _resolve_llm_model(effective_llm_provider, llm_model=llm_model, cfg=cfg)
    effective_llm_api_key_env = _resolve_llm_api_key_env(effective_llm_provider, llm_api_key_env=llm_api_key_env, cfg=cfg)
    effective_llm_api_key = llm_api_key
    if enable_llm and not effective_llm_api_key:
        load_dotenv(cfg.root / ".env", override=False)
        if effective_llm_api_key_env:
            import os

            effective_llm_api_key = os.getenv(effective_llm_api_key_env)
    cache_metadata = {
        "stage": "tribe_profiles_v2",
        "mode": cfg.mode,
        "assignments": file_fingerprint(assignments_file),
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "behavior": file_fingerprint(candidate_behavior_path),
        "profiling": cfg.get("profiling", {}),
        "enable_copurchase": enable_copurchase,
        "enable_temporal": enable_temporal,
        "enable_loyalty": enable_loyalty,
        "enable_llm": enable_llm,
        "llm_provider": effective_llm_provider if enable_llm else None,
        "llm_model": effective_llm_model if enable_llm else None,
    }
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata):
        log_event(stage_label, "cache hit", cfg=cfg, assignments=assignments_file, path=output)
        return output

    with stage_timer(stage_label, "building organic tribe profile", cfg=cfg, assignments=assignments_file):
        lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
        txn_columns = set(schema_names(lf))
        assignment_scan = _scan_assignments(assignments_file, stem)
        assignments_all = assignment_scan.unique(subset=["cliente"], keep="first")
        assigned = assignments_all.filter(pl.col("tribe_id") >= 0)
        assigned_customers = assigned.select("cliente").unique()
        assigned_keys = assigned.select(["cliente", "tribe_id"])

        cluster_sizes = collect_streaming(
            assigned.group_by("tribe_id").agg(pl.col("cliente").n_unique().alias("n_customers")).sort("tribe_id")
        )
        total_assigned_customers = int(cluster_sizes["n_customers"].sum()) if cluster_sizes.height else 0
        total_customers = int(
            collect_streaming(assignments_all.select(pl.col("cliente").n_unique().alias("n_customers")))[0, "n_customers"]
        )
        noise_customers = max(total_customers - total_assigned_customers, 0)
        assignment_meta = collect_streaming(
            assignments_all.select(
                [
                    pl.col("model_name").drop_nulls().first().alias("model_name"),
                    pl.col("model_variant").drop_nulls().first().alias("model_variant"),
                ]
            )
        ).row(0, named=True)
        confidence_summary = _assignment_confidence_summary(assigned)
        assigned_customer_frame = collect_streaming(assigned_keys)

        log_event(
            stage_label,
            "assignment population",
            cfg=cfg,
            clusters=cluster_sizes.height,
            assigned_customers=total_assigned_customers,
            noise_customers=noise_customers,
            model=assignment_meta.get("model_name"),
            variant=assignment_meta.get("model_variant"),
        )

        product_cols = ["cliente", "idarticu"]
        for col in ["ticket", "fecha", "hora", "desc_larga_articulo", "idsector", "desc_sector"]:
            if col in txn_columns:
                product_cols.append(col)
        line_items = (
            lf.select(product_cols)
            .join(assigned_customers, on="cliente", how="inner")
            .filter(pl.col("idarticu").is_not_null())
        )
        customer_product = line_items.select(["cliente", "idarticu"]).unique(subset=["cliente", "idarticu"])

        min_product_customers = int(cfg.get("profiling.min_product_customers", 10))
        log_event(
            stage_label,
            "aggregating product lift evidence",
            cfg=cfg,
            min_product_customers=min_product_customers,
        )
        population_product = _population_product_counts(line_items, customer_product, txn_columns)
        product_lifts = _product_lift_table(
            customer_product,
            population_product,
            assigned_keys,
            cluster_sizes,
            total_assigned_customers,
            min_product_customers=min_product_customers,
            cfg=cfg,
        )
        if product_lifts_output_path is not None:
            product_lifts_output = Path(product_lifts_output_path)
            product_lifts_output.parent.mkdir(parents=True, exist_ok=True)
            product_lifts.write_parquet(product_lifts_output)
            write_artifact_metadata(product_lifts_output, {**cache_metadata, "table": "product_lifts"})
        log_event(stage_label, "product lift evidence ready", cfg=cfg, rows=product_lifts.height)

        sector_lifts = _sector_lift_table(line_items, assigned_keys, total_assigned_customers, cfg=cfg)
        if sector_lifts_output_path is not None:
            sector_lifts_output = Path(sector_lifts_output_path)
            sector_lifts_output.parent.mkdir(parents=True, exist_ok=True)
            sector_lifts.write_parquet(sector_lifts_output)
            write_artifact_metadata(sector_lifts_output, {**cache_metadata, "table": "sector_lifts"})
        log_event(stage_label, "sector concentration evidence ready", cfg=cfg, rows=sector_lifts.height)

        behavior_frame = _behavior_frame(candidate_behavior_path, assigned_keys)
        behavior_summary = _behavior_summary(behavior_frame)
        behavior_ratios = _behavior_ratios_by_tribe(behavior_frame)

        reference_date = None
        if "fecha" in txn_columns:
            reference_date = collect_streaming(lf.select(pl.col("fecha").max().alias("reference_date")))[0, "reference_date"]

        log_event(stage_label, "assembling profile rows", cfg=cfg, clusters=cluster_sizes.height)
        rows: list[dict[str, Any]] = []
        llm_stop_note: str | None = None
        for cluster_row in cluster_sizes.iter_rows(named=True):
            tribe_id = int(cluster_row["tribe_id"])
            tribe_customers = assigned_customer_frame.filter(pl.col("tribe_id") == tribe_id).select("cliente")
            rest_customers = assigned_customer_frame.filter(pl.col("tribe_id") != tribe_id).select("cliente")
            products = product_lifts.filter(pl.col("tribe_id") == tribe_id).head(top_n_products)
            sectors = sector_lifts.filter(pl.col("tribe_id") == tribe_id).head(top_n_sectors)
            behavior_row = _row_by_tribe(behavior_summary, tribe_id)
            confidence_row = _row_by_tribe(confidence_summary, tribe_id)

            copurchase_pairs: list[dict[str, Any]] = []
            if enable_copurchase:
                candidate_product_ids = _copurchase_candidate_ids(product_lifts, tribe_id, top_n_products, top_n_copurchase_pairs)
                copurchase_pairs = _compute_copurchase_missions(
                    tribe_id,
                    tribe_customers,
                    lf,
                    min_baskets=min_copurchase_baskets,
                    top_n_pairs=top_n_copurchase_pairs,
                    rest_customers=rest_customers,
                    candidate_product_ids=candidate_product_ids,
                )

            temporal_pattern: dict[str, Any] = {}
            if enable_temporal:
                temporal_pattern = _compute_temporal_patterns(
                    tribe_id,
                    tribe_customers,
                    lf,
                    total_assigned_customers,
                    rest_customers=rest_customers,
                )

            loyalty_profile: dict[str, Any] = {}
            if enable_loyalty:
                loyalty_profile = _compute_loyalty_profile(
                    tribe_id,
                    tribe_customers,
                    lf,
                    reference_date,
                    rest_customers=rest_customers,
                )

            top_product_dicts = _products_for_llm(products)
            top_sector_dicts = _sectors_for_llm(sectors)
            behavior_ratio_dict = behavior_ratios.get(tribe_id, {})
            llm_payload = _empty_llm_payload(
                status="skipped",
                model=effective_llm_model if enable_llm else None,
                provider=effective_llm_provider if enable_llm else None,
            )
            if enable_llm and llm_stop_note:
                llm_payload["llm_confidence_note"] = llm_stop_note
            elif enable_llm and effective_llm_api_key:
                llm_payload = _synthesize_llm_payload(
                    tribe_id=tribe_id,
                    n_customers=int(cluster_row["n_customers"]),
                    population_share=float(cluster_row["n_customers"] / max(total_assigned_customers, 1)) * 100.0,
                    top_products=top_product_dicts,
                    top_sectors=top_sector_dicts,
                    behavior_ratios=behavior_ratio_dict,
                    copurchase_pairs=copurchase_pairs,
                    temporal_pattern=temporal_pattern,
                    loyalty_profile=loyalty_profile,
                    api_key=effective_llm_api_key,
                    model=effective_llm_model,
                    provider=effective_llm_provider,
                )
                if _is_llm_rate_limit_error(llm_payload):
                    llm_stop_note = (
                        llm_payload.get("llm_confidence_note")
                        or "LLM synthesis stopped after provider rate/quota limit."
                    )
            elif enable_llm:
                llm_payload["llm_confidence_note"] = f"Missing {effective_llm_api_key_env or 'LLM API key'}."

            rows.append(
                {
                    "tribe_id": tribe_id,
                    "n_customers": int(cluster_row["n_customers"]),
                    "population_share": float(cluster_row["n_customers"] / max(total_assigned_customers, 1)),
                    "profile_population_customers": total_assigned_customers,
                    "unassigned_noise_customers_global": noise_customers,
                    "core_customers": int(cluster_row["n_customers"]),
                    "top_products": _list_from_column(products, "product_description", repair=True),
                    "top_product_ids": [_to_int_or_none(value) for value in _list_from_column(products, "idarticu")],
                    "top_product_lifts_vs_rest": _rounded_list_with_fallback(products, "lift_vs_rest", "lift", 3),
                    "top_product_lifts": _rounded_list(products, "lift", 3),
                    "top_product_q_values": _rounded_list(products, "lift_q_value", 6),
                    "top_product_customer_counts": [_to_int_or_zero(value) for value in _list_from_column(products, "cluster_customers")],
                    "top_product_reach_pct": _rounded_list(products, "reach_pct", 3),
                    "top_product_sectors": _list_from_column(products, "sector_description", repair=True),
                    "product_ranking_basis": PRODUCT_RANKING_BASIS,
                    "top_sectors": _list_from_column(sectors, "desc_sector", repair=True),
                    "top_sector_lifts": _rounded_list(sectors, "lift", 3),
                    "top_sector_line_counts": [_to_int_or_zero(value) for value in _list_from_column(sectors, "cluster_lines")],
                    **_behavior_values(behavior_row),
                    "behavior_ratio_vs_rest": _json_dumps(behavior_ratio_dict),
                    "copurchase_pairs": _json_dumps(copurchase_pairs),
                    "temporal_pattern": _json_dumps(temporal_pattern),
                    "dominant_shopping_day": temporal_pattern.get("dominant_shopping_day"),
                    "dominant_shopping_time": temporal_pattern.get("dominant_shopping_time"),
                    "loyalty_profile": _json_dumps(loyalty_profile),
                    "mean_tenure_days": _safe_float(loyalty_profile.get("mean_tenure_days")),
                    "mean_recency_days": _safe_float(loyalty_profile.get("mean_recency_days")),
                    "loyalty_cohort": _loyalty_cohort(loyalty_profile),
                    **llm_payload,
                    "mean_assignment_confidence": _safe_float(confidence_row.get("mean_assignment_confidence")),
                    "p10_assignment_confidence": _safe_float(confidence_row.get("p10_assignment_confidence")),
                    "jitter_label_recovery_accuracy": _safe_float(confidence_row.get("jitter_label_recovery_accuracy")),
                    "stage6_profile_readiness": confidence_row.get("stage6_profile_readiness"),
                }
            )

        profiles = _profiles_from_rows(rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        profiles.write_parquet(output)
        write_artifact_metadata(output, cache_metadata)
        log_event(stage_label, "wrote profile", cfg=cfg, tribes=profiles.height, path=output)
    return output


def _scan_assignments(assignments_file: Path, stem: str) -> pl.LazyFrame:
    scan = pl.scan_parquet(assignments_file)
    columns = set(schema_names(scan))
    exprs: list[pl.Expr] = [pl.col("cliente"), pl.col("tribe_id").cast(pl.Int64)]
    exprs.append(pl.col("model_name").cast(pl.Utf8) if "model_name" in columns else pl.lit(stem).alias("model_name"))
    exprs.append(
        pl.col("model_variant").cast(pl.Utf8)
        if "model_variant" in columns
        else pl.lit(None).cast(pl.Utf8).alias("model_variant")
    )
    exprs.append(
        pl.col("assignment_confidence_score").cast(pl.Float64)
        if "assignment_confidence_score" in columns
        else pl.lit(None).cast(pl.Float64).alias("assignment_confidence_score")
    )
    if "jitter_label_recovery_accuracy" in columns:
        exprs.append(pl.col("jitter_label_recovery_accuracy").cast(pl.Float64))
    elif "jitter_label_recovery_accuracy_mean" in columns:
        exprs.append(pl.col("jitter_label_recovery_accuracy_mean").cast(pl.Float64).alias("jitter_label_recovery_accuracy"))
    else:
        exprs.append(pl.lit(None).cast(pl.Float64).alias("jitter_label_recovery_accuracy"))
    if "stage6_profile_readiness" in columns:
        exprs.append(pl.col("stage6_profile_readiness").cast(pl.Utf8))
    elif "profile_readiness" in columns:
        exprs.append(pl.col("profile_readiness").cast(pl.Utf8).alias("stage6_profile_readiness"))
    else:
        exprs.append(pl.lit(None).cast(pl.Utf8).alias("stage6_profile_readiness"))
    return scan.select(exprs)


def _assignment_export_frame(assignments_path: str | Path) -> pl.LazyFrame:
    assignments_file = Path(assignments_path)
    scan = pl.scan_parquet(assignments_file)
    columns = set(schema_names(scan))
    stem = assignments_file.stem.replace("cluster_assignments_", "")
    exprs: list[pl.Expr] = [pl.col("cliente"), pl.col("tribe_id").cast(pl.Int64)]
    specs = [
        ("model_name", pl.Utf8, stem),
        ("model_variant", pl.Utf8, None),
        ("assignment_probability", pl.Float64, None),
        ("assignment_confidence_score", pl.Float64, None),
        ("assignment_confidence_type", pl.Utf8, None),
        ("assignment_source", pl.Utf8, None),
        ("jitter_label_recovery_accuracy", pl.Float64, None),
        ("stage6_profile_readiness", pl.Utf8, None),
    ]
    for column, dtype, default in specs:
        if column in columns:
            exprs.append(pl.col(column).cast(dtype, strict=False))
        elif column == "jitter_label_recovery_accuracy" and "jitter_label_recovery_accuracy_mean" in columns:
            exprs.append(pl.col("jitter_label_recovery_accuracy_mean").cast(dtype, strict=False).alias(column))
        elif column == "stage6_profile_readiness" and "profile_readiness" in columns:
            exprs.append(pl.col("profile_readiness").cast(dtype, strict=False).alias(column))
        else:
            exprs.append(pl.lit(default).cast(dtype).alias(column))
    return scan.select(exprs).unique(subset=["cliente"], keep="first")


def _non_overlapping_source_columns(source: pl.LazyFrame, assignment_columns: list[str]) -> list[str]:
    assignment_set = set(assignment_columns)
    return [column for column in schema_names(source) if column == "cliente" or column not in assignment_set]


def _write_partitioned_tribe_export(
    frame: pl.LazyFrame,
    *,
    assignments: pl.LazyFrame,
    output_dir: Path,
    filename_suffix: str,
    manifest_filename: str,
    export_type: str,
) -> dict[str, Path]:
    manifest_path = output_dir / manifest_filename
    paths: dict[str, Path] = {"directory": output_dir, "manifest_csv": manifest_path}
    assignment_fields = [column for column in schema_names(assignments) if column != "cliente"]
    tribe_ids = (
        collect_streaming(assignments.select("tribe_id").drop_nulls().unique().sort("tribe_id"))["tribe_id"].to_list()
    )
    manifest_rows: list[dict[str, Any]] = []
    for raw_tribe_id in tribe_ids:
        tribe_id = int(raw_tribe_id)
        tribe_frame = collect_streaming(frame.filter(pl.col("tribe_id") == tribe_id))
        tribe_path = output_dir / f"tribe_{tribe_id:02d}_{filename_suffix}.parquet"
        tribe_frame.write_parquet(tribe_path)
        paths[f"tribe_{tribe_id:02d}_parquet"] = tribe_path
        manifest_rows.append(
            {
                "tribe_id": tribe_id,
                "export_type": export_type,
                "file_format": "parquet",
                "rows": tribe_frame.height,
                "customers": tribe_frame["cliente"].n_unique() if "cliente" in tribe_frame.columns else 0,
                "path": str(tribe_path),
                "schema_columns": _json_dumps(tribe_frame.columns),
                "assignment_fields": _json_dumps([column for column in assignment_fields if column in tribe_frame.columns]),
            }
        )
    manifest = pl.DataFrame(manifest_rows) if manifest_rows else _empty_partitioned_tribe_manifest()
    manifest.write_csv(manifest_path)
    return paths


def _empty_partitioned_tribe_manifest() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "tribe_id": pl.Int64,
            "export_type": pl.Utf8,
            "file_format": pl.Utf8,
            "rows": pl.Int64,
            "customers": pl.Int64,
            "path": pl.Utf8,
            "schema_columns": pl.Utf8,
            "assignment_fields": pl.Utf8,
        }
    )


def _resolve_llm_provider(*, enable_llm: bool, llm_provider: str | None, cfg: PipelineConfig) -> str | None:
    if not enable_llm:
        return None
    provider = (llm_provider or cfg.get("llm_analysis.provider") or "claude").strip().lower()
    if provider in {"anthropic", "claude"}:
        return "claude"
    if provider in {"google", "gemini"}:
        return "gemini"
    return provider


def _resolve_llm_model(provider: str | None, *, llm_model: str, cfg: PipelineConfig) -> str:
    if provider == "gemini":
        configured = cfg.get("llm_analysis.model")
        if configured and llm_model == "claude-sonnet-4-6":
            return str(configured)
        return llm_model if llm_model != "claude-sonnet-4-6" else "gemini-3.5-flash"
    return llm_model or "claude-sonnet-4-6"


def _resolve_llm_api_key_env(provider: str | None, *, llm_api_key_env: str | None, cfg: PipelineConfig) -> str | None:
    if llm_api_key_env:
        return llm_api_key_env
    if provider == "gemini":
        return str(cfg.get("llm_analysis.api_key_env", "GEMINI_API_KEY"))
    if provider == "claude":
        return "ANTHROPIC_API_KEY"
    return None


def _assignment_confidence_summary(assigned: pl.LazyFrame) -> pl.DataFrame:
    return collect_streaming(
        assigned.group_by("tribe_id").agg(
            [
                pl.col("assignment_confidence_score").mean().alias("mean_assignment_confidence"),
                pl.col("assignment_confidence_score").quantile(0.10).alias("p10_assignment_confidence"),
                pl.col("jitter_label_recovery_accuracy").mean().alias("jitter_label_recovery_accuracy"),
                pl.col("stage6_profile_readiness").drop_nulls().first().alias("stage6_profile_readiness"),
            ]
        )
    )


def _population_product_counts(
    line_items: pl.LazyFrame,
    customer_product: pl.LazyFrame,
    txn_columns: set[str],
) -> pl.DataFrame:
    population_product = collect_streaming(
        customer_product.group_by("idarticu").agg(pl.len().alias("population_customers"))
    )
    metadata_exprs: list[pl.Expr] = []
    metadata_cols = ["idarticu"]
    if "desc_larga_articulo" in txn_columns:
        metadata_cols.append("desc_larga_articulo")
        metadata_exprs.append(pl.col("desc_larga_articulo").drop_nulls().first().cast(pl.Utf8).alias("product_description"))
    if "desc_sector" in txn_columns:
        metadata_cols.append("desc_sector")
        metadata_exprs.append(pl.col("desc_sector").drop_nulls().first().cast(pl.Utf8).alias("sector_description"))
    if metadata_exprs:
        product_metadata = collect_streaming(line_items.select(metadata_cols).group_by("idarticu").agg(metadata_exprs))
        population_product = population_product.join(product_metadata, on="idarticu", how="left")
    if "product_description" not in population_product.columns:
        population_product = population_product.with_columns(pl.col("idarticu").cast(pl.Utf8).alias("product_description"))
    if "sector_description" not in population_product.columns:
        population_product = population_product.with_columns(pl.lit(None).cast(pl.Utf8).alias("sector_description"))
    return population_product


def _product_lift_table(
    customer_product: pl.LazyFrame,
    population_product: pl.DataFrame,
    assigned_keys: pl.LazyFrame,
    cluster_sizes: pl.DataFrame,
    total_assigned_customers: int,
    *,
    min_product_customers: int,
    cfg: PipelineConfig,
) -> pl.DataFrame:
    cluster_product = (
        customer_product.join(assigned_keys, on="cliente", how="inner")
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
                (pl.col("population_customers") / max(total_assigned_customers, 1)).alias("population_rate"),
            ]
        )
        .with_columns((pl.col("cluster_rate") / pl.col("population_rate")).alias("lift"))
    )
    product_lifts = _add_overindex_diagnostics(
        product_lifts,
        cluster_count_col="cluster_customers",
        population_count_col="population_customers",
        cluster_total_col="n_customers",
        population_total=total_assigned_customers,
        cfg=cfg,
    )
    if product_lifts.is_empty():
        return product_lifts
    return (
        product_lifts.with_columns(
            [
                (pl.col("cluster_rate") * 100.0).alias("reach_pct"),
                (
                    pl.col("lift_vs_rest").fill_null(0.0)
                    * (pl.col("cluster_customers").cast(pl.Float64) + 1).log()
                ).alias("_sort_score"),
            ]
        )
        .sort(["tribe_id", "_sort_score"], descending=[False, True])
    )


def _sector_lift_table(line_items: pl.LazyFrame, assigned_keys: pl.LazyFrame, total_assigned_customers: int, cfg: PipelineConfig) -> pl.DataFrame:
    del total_assigned_customers
    if "desc_sector" not in schema_names(line_items):
        return pl.DataFrame(schema={"tribe_id": pl.Int64, "desc_sector": pl.Utf8, "lift": pl.Float64, "cluster_lines": pl.Int64})
    population_sector = collect_streaming(
        line_items.select("desc_sector").group_by("desc_sector").agg(pl.len().alias("population_lines"))
    )
    population_total = int(population_sector["population_lines"].sum()) if population_sector.height else 0
    cluster_sector = collect_streaming(
        line_items.select(["cliente", "desc_sector"])
        .join(assigned_keys, on="cliente", how="inner")
        .group_by(["tribe_id", "desc_sector"])
        .agg(pl.len().alias("cluster_lines"))
    )
    if cluster_sector.is_empty():
        return pl.DataFrame(schema={"tribe_id": pl.Int64, "desc_sector": pl.Utf8, "lift": pl.Float64, "cluster_lines": pl.Int64})
    cluster_totals = cluster_sector.group_by("tribe_id").agg(pl.col("cluster_lines").sum().alias("cluster_lines_total"))
    sector_lifts = (
        cluster_sector.join(population_sector, on="desc_sector", how="left")
        .join(cluster_totals, on="tribe_id", how="left")
        .with_columns(
            [
                (pl.col("cluster_lines") / pl.col("cluster_lines_total")).alias("cluster_sector_share"),
                (pl.col("population_lines") / max(population_total, 1)).alias("population_sector_share"),
            ]
        )
        .with_columns((pl.col("cluster_sector_share") / pl.col("population_sector_share")).alias("lift"))
        .sort(["tribe_id", "lift", "cluster_lines"], descending=[False, True, True])
    )
    return _add_overindex_diagnostics(
        sector_lifts,
        cluster_count_col="cluster_lines",
        population_count_col="population_lines",
        cluster_total_col="cluster_lines_total",
        population_total=population_total,
        cfg=cfg,
    )


def _behavior_frame(candidate_behavior_path: Path, assigned_keys: pl.LazyFrame) -> pl.DataFrame:
    if not candidate_behavior_path.exists():
        return pl.DataFrame()
    behavior = pl.scan_parquet(candidate_behavior_path)
    behavior_columns = set(schema_names(behavior))
    selected = ["cliente", *[col for col in BEHAVIOR_OUTPUT_MAP if col in behavior_columns]]
    if len(selected) == 1:
        return pl.DataFrame()
    return collect_streaming(assigned_keys.join(behavior.select(selected), on="cliente", how="left"))


def _behavior_summary(behavior_frame: pl.DataFrame) -> pl.DataFrame:
    if behavior_frame.is_empty():
        return pl.DataFrame()
    aggregations = [
        pl.col(source).mean().alias(target)
        for source, target in BEHAVIOR_OUTPUT_MAP.items()
        if source in behavior_frame.columns
    ]
    return behavior_frame.group_by("tribe_id").agg(aggregations) if aggregations else pl.DataFrame()


def _behavior_ratios_by_tribe(behavior_frame: pl.DataFrame) -> dict[int, dict[str, float]]:
    if behavior_frame.is_empty():
        return {}
    rows: dict[int, dict[str, float]] = {}
    for tribe_id in sorted(int(value) for value in behavior_frame["tribe_id"].drop_nulls().unique().to_list()):
        tribe = behavior_frame.filter(pl.col("tribe_id") == tribe_id)
        rest = behavior_frame.filter(pl.col("tribe_id") != tribe_id)
        ratios: dict[str, float] = {}
        for source, target in BEHAVIOR_OUTPUT_MAP.items():
            if source not in behavior_frame.columns:
                continue
            tribe_mean = _safe_float(tribe.select(pl.col(source).mean()).item())
            rest_mean = _safe_float(rest.select(pl.col(source).mean()).item()) if not rest.is_empty() else None
            ratio = _safe_ratio(tribe_mean, rest_mean)
            if ratio is not None:
                ratios[target] = round(ratio, 4)
        rows[tribe_id] = ratios
    return rows


def _behavior_values(row: dict[str, Any]) -> dict[str, float | None]:
    return {target: _safe_float(row.get(target)) for target in BEHAVIOR_OUTPUT_MAP.values()}


def _copurchase_candidate_ids(product_lifts: pl.DataFrame, tribe_id: int, top_n_products: int, top_n_pairs: int) -> list[Any] | None:
    if product_lifts.is_empty() or "idarticu" not in product_lifts.columns:
        return None
    limit = max(int(top_n_products) * 4, int(top_n_pairs) * 4, 20)
    ids = product_lifts.filter(pl.col("tribe_id") == tribe_id).head(limit)["idarticu"].to_list()
    return ids or None


def _products_for_llm(products: pl.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in products.iter_rows(named=True):
        rows.append(
            {
                "desc": _repair_display_text(str(row.get("product_description") or row.get("idarticu") or "")),
                "lift_vs_rest": _safe_float(row.get("lift_vs_rest")) or _safe_float(row.get("lift")) or 0.0,
                "reach_pct": _safe_float(row.get("reach_pct")) or 0.0,
                "q_value": _safe_float(row.get("lift_q_value")) or 1.0,
                "customers": int(row.get("cluster_customers") or 0),
            }
        )
    return rows


def _sectors_for_llm(sectors: pl.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sectors.iter_rows(named=True):
        rows.append(
            {
                "desc_sector": _repair_display_text(str(row.get("desc_sector") or "")),
                "lift": _safe_float(row.get("lift")) or 0.0,
                "line_count": int(row.get("cluster_lines") or 0),
            }
        )
    return rows


def _synthesize_llm_payload(
    *,
    tribe_id: int,
    n_customers: int,
    population_share: float,
    top_products: list[dict[str, Any]],
    top_sectors: list[dict[str, Any]],
    behavior_ratios: dict[str, float],
    copurchase_pairs: list[dict[str, Any]],
    temporal_pattern: dict[str, Any],
    loyalty_profile: dict[str, Any],
    api_key: str,
    model: str,
    provider: str | None,
) -> dict[str, Any]:
    try:
        if provider == "gemini":
            synthesis = synthesize_tribe_with_gemini(
                tribe_id=tribe_id,
                n_customers=n_customers,
                population_share_pct=population_share,
                top_products=top_products,
                top_sectors=top_sectors,
                behavior_ratios=behavior_ratios,
                copurchase_pairs=copurchase_pairs,
                temporal_pattern=temporal_pattern,
                loyalty_profile=loyalty_profile,
                api_key=api_key,
                model=model,
            )
        elif provider == "claude":
            synthesis = synthesize_tribe_with_claude(
                tribe_id=tribe_id,
                n_customers=n_customers,
                population_share_pct=population_share,
                top_products=top_products,
                top_sectors=top_sectors,
                behavior_ratios=behavior_ratios,
                copurchase_pairs=copurchase_pairs,
                temporal_pattern=temporal_pattern,
                loyalty_profile=loyalty_profile,
                api_key=api_key,
                model=model,
            )
        else:
            raise RuntimeError(f"Unsupported llm_provider={provider!r}.")
    except Exception as exc:
        payload = _empty_llm_payload(status="error", model=model, provider=provider)
        payload["llm_confidence_note"] = f"{provider or 'LLM'} synthesis failed: {exc}"
        return payload
    return {
        "llm_working_label": synthesis.get("working_label"),
        "llm_shopping_mission": synthesis.get("shopping_mission"),
        "llm_customer_description": synthesis.get("customer_description"),
        "llm_commercial_opportunities": _json_dumps(synthesis.get("commercial_opportunities") or []),
        "llm_confidence_note": synthesis.get("confidence_note"),
        "llm_model": f"{provider}:{model}" if provider and model else model,
        "llm_status": "success",
    }


def _empty_llm_payload(*, status: str, model: str | None, provider: str | None = None) -> dict[str, Any]:
    return {
        "llm_working_label": None,
        "llm_shopping_mission": None,
        "llm_customer_description": None,
        "llm_commercial_opportunities": _json_dumps([]),
        "llm_confidence_note": None,
        "llm_model": f"{provider}:{model}" if provider and model else model,
        "llm_status": status,
    }


def _is_llm_rate_limit_error(payload: dict[str, Any]) -> bool:
    if payload.get("llm_status") != "error":
        return False
    note = str(payload.get("llm_confidence_note") or "").lower()
    return any(token in note for token in [" 429", "too many request", "quota", "rate limit", "resource_exhausted"])


def _profile_schema() -> dict[str, Any]:
    return {
        "tribe_id": pl.Int64,
        "n_customers": pl.Int64,
        "population_share": pl.Float64,
        "profile_population_customers": pl.Int64,
        "unassigned_noise_customers_global": pl.Int64,
        "core_customers": pl.Int64,
        "top_products": pl.List(pl.Utf8),
        "top_product_ids": pl.List(pl.Int64),
        "top_product_lifts_vs_rest": pl.List(pl.Float64),
        "top_product_lifts": pl.List(pl.Float64),
        "top_product_q_values": pl.List(pl.Float64),
        "top_product_customer_counts": pl.List(pl.Int64),
        "top_product_reach_pct": pl.List(pl.Float64),
        "top_product_sectors": pl.List(pl.Utf8),
        "product_ranking_basis": pl.Utf8,
        "top_sectors": pl.List(pl.Utf8),
        "top_sector_lifts": pl.List(pl.Float64),
        "top_sector_line_counts": pl.List(pl.Int64),
        "avg_ticket_count": pl.Float64,
        "avg_total_spend": pl.Float64,
        "avg_basket_value": pl.Float64,
        "avg_promo_share": pl.Float64,
        "avg_unique_products": pl.Float64,
        "avg_unique_sectors": pl.Float64,
        "avg_recency_days": pl.Float64,
        "avg_frequency_per_30d": pl.Float64,
        "behavior_ratio_vs_rest": pl.Utf8,
        "copurchase_pairs": pl.Utf8,
        "temporal_pattern": pl.Utf8,
        "dominant_shopping_day": pl.Utf8,
        "dominant_shopping_time": pl.Utf8,
        "loyalty_profile": pl.Utf8,
        "mean_tenure_days": pl.Float64,
        "mean_recency_days": pl.Float64,
        "loyalty_cohort": pl.Utf8,
        "llm_working_label": pl.Utf8,
        "llm_shopping_mission": pl.Utf8,
        "llm_customer_description": pl.Utf8,
        "llm_commercial_opportunities": pl.Utf8,
        "llm_confidence_note": pl.Utf8,
        "llm_model": pl.Utf8,
        "llm_status": pl.Utf8,
        "mean_assignment_confidence": pl.Float64,
        "p10_assignment_confidence": pl.Float64,
        "jitter_label_recovery_accuracy": pl.Float64,
        "stage6_profile_readiness": pl.Utf8,
    }


def _profiles_from_rows(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = _profile_schema()
    if not rows:
        return pl.DataFrame(schema=schema)
    frame = pl.from_dicts(rows, infer_schema_length=None)
    for column, dtype in schema.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(column))
        frame = frame.with_columns(pl.col(column).cast(dtype, strict=False))
    return frame.select(list(schema)).sort("tribe_id")


def _add_overindex_diagnostics(
    frame: pl.DataFrame,
    *,
    cluster_count_col: str,
    population_count_col: str,
    cluster_total_col: str,
    population_total: int,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    """Add rest-of-population lift, p-value, q-value, and evidence label."""

    if frame.is_empty():
        return frame

    rows: list[dict[str, Any]] = []
    p_values: list[float | None] = []
    for row in frame.iter_rows(named=True):
        cluster_count = _safe_float(row.get(cluster_count_col)) or 0.0
        population_count = _safe_float(row.get(population_count_col)) or 0.0
        cluster_total = _safe_float(row.get(cluster_total_col)) or 0.0
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
    lift = _safe_float(lift_vs_rest)
    q = _safe_float(q_value)
    if lift is None:
        return "insufficient_rest_baseline"
    if lift < 1.0:
        return "under_indexed"
    if q is not None and q <= q_threshold and lift >= 1.5:
        return "strong_significant_overindex"
    if q is not None and q <= q_threshold:
        return "significant_overindex"
    return "directional_overindex"


def cluster_size_summary(labels: Any) -> dict[str, Any]:
    values = [int(value) for value in list(labels)]
    total = len(values)
    assigned = [value for value in values if value >= 0]
    noise = total - len(assigned)
    counts: dict[int, int] = {}
    for value in assigned:
        counts[value] = counts.get(value, 0) + 1
    cluster_sizes = list(counts.values())
    return {
        "cluster_count": len(counts),
        "customers": total,
        "assigned_customers": len(assigned),
        "noise_customers": noise,
        "noise_pct": 100.0 * noise / max(total, 1),
        "core_coverage_pct": 100.0 * len(assigned) / max(total, 1),
        "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
    }


def flatten_profiles_for_csv(profile_path: str | Path, output_csv: str | Path) -> Path:
    profiles = pl.read_parquet(profile_path)
    if "top_products" in profiles.columns and "product_ranking_basis" not in profiles.columns:
        profiles = profiles.with_columns(pl.lit(PRODUCT_RANKING_BASIS).alias("product_ranking_basis"))
    rows = [{key: _csv_safe_value(value) for key, value in row.items()} for row in profiles.iter_rows(named=True)]
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    (pl.DataFrame(rows) if rows else pl.DataFrame()).write_csv(output)
    return output


def profile_quality_summary(profile_path: str | Path, cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    profiles = pl.read_parquet(profile_path)
    threshold = float(cfg.get("profiling.strong_product_lift_threshold", 1.5))
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    strong_counts: list[int] = []
    significant_counts: list[int] = []
    max_lifts: list[float] = []
    max_rest_lifts: list[float] = []
    sector_counts: list[int] = []
    for row in profiles.iter_rows(named=True):
        lifts = [_safe_float(value) for value in (row.get("top_product_lifts") or [])]
        rest_lifts = [_safe_float(value) for value in (row.get("top_product_lifts_vs_rest") or [])]
        q_values = row.get("top_product_q_values") or []
        finite_lifts = [value for value in lifts if value is not None]
        finite_rest = [value for value in rest_lifts if value is not None]
        strong_indices = [idx for idx, value in enumerate(lifts) if value is not None and value >= threshold]
        significant = 0
        for idx in strong_indices:
            q_value = _safe_float(q_values[idx]) if idx < len(q_values) else None
            if q_value is not None and q_value <= q_threshold:
                significant += 1
        strong_counts.append(len(strong_indices))
        significant_counts.append(significant)
        max_lifts.append(max(finite_lifts) if finite_lifts else 0.0)
        max_rest_lifts.append(max(finite_rest) if finite_rest else 0.0)
        sector_counts.append(
            sum(
                1
                for value in (row.get("top_sector_lifts") or [])
                if _safe_float(value) is not None and float(value) >= float(cfg.get("profiling.strong_sector_lift_threshold", 1.2))
            )
        )
    return {
        "profiled_clusters": profiles.height,
        "clusters_with_product_lift": sum(1 for count in strong_counts if count > 0),
        "clusters_with_sector_lift": sum(1 for count in sector_counts if count > 0),
        "clusters_with_significant_product_lift": sum(1 for count in significant_counts if count > 0),
        "avg_strong_product_lifts_per_cluster": float(sum(strong_counts) / max(profiles.height, 1)),
        "avg_significant_product_lifts_per_cluster": float(sum(significant_counts) / max(profiles.height, 1)),
        "avg_max_product_lift": float(sum(max_lifts) / max(profiles.height, 1)),
        "avg_max_product_lift_vs_rest": float(sum(max_rest_lifts) / max(profiles.height, 1)),
    }


def profile_readiness_evidence_table(
    profile_path: str | Path,
    *,
    cluster_readiness_path: str | Path | None = None,
    output_csv: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    cluster_readiness = _read_optional_table(cluster_readiness_path)
    threshold = float(cfg.get("profiling.strong_product_lift_threshold", 1.5))
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    min_signals = int(cfg.get("profiling.min_ready_evidence_signals", 1))
    rows: list[dict[str, Any]] = []
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row["tribe_id"])
        readiness_row = _row_by_tribe(cluster_readiness, tribe_id)
        stage6_status = str(
            readiness_row.get("stage6_profile_readiness")
            or readiness_row.get("profile_readiness")
            or row.get("stage6_profile_readiness")
            or "missing"
        ).strip()
        signal_count = _significant_product_signal_count(row, threshold=threshold, q_threshold=q_threshold)
        signals_ok = signal_count >= min_signals
        mean_confidence = _safe_float(
            readiness_row.get("mean_assignment_confidence") or row.get("mean_assignment_confidence")
        )
        p10_confidence = _safe_float(
            readiness_row.get("p10_assignment_confidence") or row.get("p10_assignment_confidence")
        )
        jitter_recovery = _safe_float(
            readiness_row.get("jitter_label_recovery_accuracy")
            or readiness_row.get("jitter_label_recovery_accuracy_mean")
            or row.get("jitter_label_recovery_accuracy")
        )
        issues: list[str] = []
        strong_stage6_statuses = {"strong", "ready_strong"}
        profileable_stage6_statuses = strong_stage6_statuses | {"usable", "ready", "pass"}
        stage6_ok = stage6_status in profileable_stage6_statuses
        if not stage6_ok:
            issues.append(f"stage6_readiness={stage6_status}")
        if not signals_ok:
            issues.append(f"product_signals<{min_signals}")
        if not issues and stage6_status in strong_stage6_statuses:
            profiling_status = "ready_strong"
        elif not issues:
            profiling_status = "ready"
        else:
            profiling_status = "review"
        rows.append(
            {
                "tribe_id": tribe_id,
                "n_customers": int(row.get("n_customers") or 0),
                "stage6_profile_readiness": stage6_status,
                "profiling_readiness": profiling_status,
                "profiling_readiness_issues": "; ".join(issues) if issues else "pass",
                "profile_significant_interpretability_signals": signal_count,
                "mean_assignment_confidence": mean_confidence,
                "p10_assignment_confidence": p10_confidence,
                "jitter_label_recovery_accuracy": jitter_recovery,
            }
        )
    result = pl.DataFrame(rows) if rows else pl.DataFrame()
    if output_csv is not None:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_csv(output)
    return result


def tribe_product_summary_tables(
    profile_path: str | Path,
    *,
    max_products: int | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[int, pl.DataFrame]:
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    limit = int(max_products or cfg.get("profiling.top_n_products", 15))
    return {int(row["tribe_id"]): _tribe_product_summary_frame(row, limit, cfg=cfg) for row in profiles.iter_rows(named=True)}


def write_tribe_product_summary_artifacts(
    profile_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    combined_output_csv: str | Path | None = None,
    max_products: int | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    out_dir = Path(output_dir) if output_dir else cfg.artifacts / "stage7" / "tribe_product_summaries"
    combined_output = (
        Path(combined_output_csv)
        if combined_output_csv
        else cfg.artifacts / "stage7" / f"tribe_product_summary_long_{cfg.mode}.csv"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    tables = tribe_product_summary_tables(profile_path, max_products=max_products, cfg=cfg)
    frames: list[pl.DataFrame] = []
    paths: dict[str, Path] = {"directory": out_dir, "combined_csv": combined_output}
    for tribe_id, table in tables.items():
        path = out_dir / f"tribe_{tribe_id:02d}_product_summary.csv"
        table.write_csv(path)
        paths[f"tribe_{tribe_id:02d}_csv"] = path
        if not table.is_empty():
            frames.append(table)
    combined = pl.concat(frames, how="vertical") if frames else _empty_tribe_product_summary()
    combined.write_csv(combined_output)
    log_event("Stage 7 exports", "wrote per-tribe product summary tables", cfg=cfg, directory=out_dir, combined_csv=combined_output)
    return paths


def tribe_comparison_table(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    readiness = _read_optional_table(readiness_path)
    rows = []
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row["tribe_id"])
        readiness_row = _row_by_tribe(readiness, tribe_id)
        rows.append(
            {
                "tribe_id": tribe_id,
                "working_label": _working_label(row),
                "n_customers": int(row.get("n_customers") or 0),
                "population_share_pct": round(float(row.get("population_share") or 0.0) * 100.0, 2),
                "readiness": readiness_row.get("profiling_readiness") or row.get("stage6_profile_readiness") or "not checked",
                "top_product_and_category_evidence": _product_evidence_text(row),
                "product_ranking_basis": PRODUCT_RANKING_BASIS,
                "top_sector_evidence": _format_lift_items(row.get("top_sectors"), row.get("top_sector_lifts"), row.get("top_sector_line_counts")),
                "customer_behavior_over_under_index": _behavior_ratio_summary(row, profiles),
                "dominant_shopping_day": row.get("dominant_shopping_day"),
                "dominant_shopping_time": row.get("dominant_shopping_time"),
                "loyalty_context": _loyalty_text(row),
                "llm_confidence_note": row.get("llm_confidence_note"),
            }
        )
    return pl.DataFrame(rows).sort("tribe_id") if rows else pl.DataFrame()


def write_tribe_comparison_artifacts(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    table = tribe_comparison_table(profile_path, readiness_path=readiness_path, cfg=cfg)
    csv_path = Path(output_csv) if output_csv else cfg.artifacts / "stage7" / f"tribe_comparison_{cfg.mode}.csv"
    md_path = Path(output_md) if output_md else csv_path.with_suffix(".md")
    html_path = Path(output_html) if output_html else csv_path.with_suffix(".html")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(csv_path)
    md_path.write_text(_markdown_table(table), encoding="utf-8")
    html_path.write_text(_html_table(table, title="Stage 7 Tribe Comparison"), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "html": html_path}


def customer_metric_anova_table(
    assignments_path: str | Path,
    behavior_path: str | Path | None = None,
    *,
    output_csv: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    candidate_behavior_path = (
        Path(behavior_path)
        if behavior_path
        else cfg.artifact_path("behavioral_features", "output", directory=cfg.outputs / "features")
    )
    if not candidate_behavior_path.exists():
        result = _empty_customer_metric_tests()
        if output_csv is not None:
            _write_csv(result, output_csv)
        return result
    assignments = (
        pl.scan_parquet(assignments_path)
        .select(["cliente", "tribe_id"])
        .filter(pl.col("tribe_id") >= 0)
        .unique(subset=["cliente"], keep="first")
    )
    behavior = pl.scan_parquet(candidate_behavior_path)
    numeric_cols = [col for col in schema_names(behavior) if col != "cliente" and behavior.collect_schema()[col].is_numeric()]
    frame = collect_streaming(assignments.join(behavior.select(["cliente", *numeric_cols]), on="cliente", how="inner"))
    rows: list[dict[str, Any]] = []
    for metric in numeric_cols:
        rows.append(_anova_metric_row(frame, metric))
    q_values = _benjamini_hochberg([row.get("anova_p_value") for row in rows])
    for row, q_value in zip(rows, q_values):
        row["anova_q_value"] = q_value
        row["statistical_result"] = _anova_statistical_result(row.get("anova_effect_eta_squared"), q_value)
    result = pl.DataFrame(rows) if rows else _empty_customer_metric_tests()
    if output_csv is not None:
        _write_csv(result, output_csv)
    return result


def noise_vs_core_customer_metric_table(
    assignments_path: str | Path,
    behavior_path: str | Path | None = None,
    *,
    output_csv: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    """Compare behavioral KPIs for unassigned/noise customers against assigned core customers."""

    candidate_behavior_path = (
        Path(behavior_path)
        if behavior_path
        else cfg.artifact_path("behavioral_features", "output", directory=cfg.outputs / "features")
    )
    if not candidate_behavior_path.exists():
        result = _empty_noise_vs_core_metric_tests()
        if output_csv is not None:
            _write_csv(result, output_csv)
        return result

    assignments = (
        pl.scan_parquet(assignments_path)
        .select(
            [
                pl.col("cliente"),
                pl.when(pl.col("tribe_id") < 0)
                .then(pl.lit("noise"))
                .otherwise(pl.lit("core"))
                .alias("assignment_group"),
            ]
        )
        .unique(subset=["cliente"], keep="first")
    )
    behavior = pl.scan_parquet(candidate_behavior_path)
    numeric_cols = [col for col in schema_names(behavior) if col != "cliente" and behavior.collect_schema()[col].is_numeric()]
    frame = collect_streaming(assignments.join(behavior.select(["cliente", *numeric_cols]), on="cliente", how="inner"))
    rows: list[dict[str, Any]] = []
    anova_frame = frame.with_columns(
        pl.when(pl.col("assignment_group") == "noise").then(pl.lit(1)).otherwise(pl.lit(0)).alias("tribe_id")
    )
    for metric in numeric_cols:
        rows.append(_noise_vs_core_metric_row(frame, anova_frame, metric))
    q_values = _benjamini_hochberg([row.get("anova_p_value") for row in rows])
    for row, q_value in zip(rows, q_values):
        row["anova_q_value"] = q_value
        row["statistical_result"] = _anova_statistical_result(row.get("anova_effect_eta_squared"), q_value)
        row["interpretation"] = _noise_vs_core_interpretation(row)
    result = pl.DataFrame(rows) if rows else _empty_noise_vs_core_metric_tests()
    if output_csv is not None:
        _write_csv(result, output_csv)
    return result


def noise_audit_table(
    assignments_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    behavior_path: str | Path | None = None,
    *,
    output_csv: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    del transactions, behavior_path
    assignments = collect_streaming(pl.scan_parquet(assignments_path).select(["cliente", "tribe_id"]))
    total = assignments["cliente"].n_unique() if assignments.height else 0
    noise = assignments.filter(pl.col("tribe_id") < 0)["cliente"].n_unique() if assignments.height else 0
    core = max(total - noise, 0)
    rows = [
        {
            "section": "assignment coverage",
            "rank": 1,
            "evidence_type": "noise_share",
            "evidence_label": "Unassigned customers",
            "noise_observations": int(noise),
            "core_observations": int(core),
            "noise_share_pct": round(100.0 * noise / max(total, 1), 3),
            "recommended_action": "keep_unassigned_for_core_profile",
            "interpretation": "Noise customers are reported separately and are not projected into the hard organic tribe profiles.",
        }
    ]
    result = pl.DataFrame(rows)
    if output_csv is not None:
        _write_csv(result, output_csv)
    return result


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
    table = noise_audit_table(assignments_path, transactions, behavior_path, cfg=cfg)
    csv_path = Path(output_csv) if output_csv else cfg.artifacts / "stage7" / f"noise_audit_{cfg.mode}.csv"
    md_path = Path(output_md) if output_md else csv_path.with_suffix(".md")
    html_path = Path(output_html) if output_html else csv_path.with_suffix(".html")
    _write_csv(table, csv_path)
    md_path.write_text(_markdown_table(table), encoding="utf-8")
    html_path.write_text(_html_table(table, title="Stage 7 Noise Audit"), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "html": html_path}


def write_tribe_transaction_exports(
    assignments_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    *,
    output_dir: str | Path | None = None,
    filename_suffix: str = "transactions",
    manifest_filename: str | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write one prepared transaction-line parquet per tribe with assignment fields attached."""

    out_dir = Path(output_dir) if output_dir else cfg.artifacts / "stage7" / "tribe_raw_transactions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"tribe_*_{filename_suffix}.parquet"):
        stale.unlink()

    assignments = _assignment_export_frame(assignments_path).filter(pl.col("tribe_id") >= 0)
    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    selected_transaction_cols = _non_overlapping_source_columns(lf, schema_names(assignments))
    joined = lf.select(selected_transaction_cols).join(assignments, on="cliente", how="inner")
    return _write_partitioned_tribe_export(
        joined,
        assignments=assignments,
        output_dir=out_dir,
        filename_suffix=filename_suffix,
        manifest_filename=manifest_filename or f"tribe_raw_transactions_manifest_{cfg.mode}.csv",
        export_type="prepared_transaction_lines_by_tribe",
    )


def write_tribe_customer_summary_exports(
    assignments_path: str | Path,
    behavior_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    filename_suffix: str = "customer_summary",
    manifest_filename: str | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write one customer-level behavior/KPI parquet per tribe with assignment fields attached."""

    out_dir = Path(output_dir) if output_dir else cfg.artifacts / "stage7" / "tribe_customer_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"tribe_*_{filename_suffix}.parquet"):
        stale.unlink()

    candidate_behavior_path = Path(behavior_path)
    manifest_path = out_dir / (manifest_filename or f"tribe_customer_summaries_manifest_{cfg.mode}.csv")
    if not candidate_behavior_path.exists():
        _empty_partitioned_tribe_manifest().write_csv(manifest_path)
        return {"directory": out_dir, "manifest_csv": manifest_path}

    assignments = _assignment_export_frame(assignments_path).filter(pl.col("tribe_id") >= 0)
    behavior = pl.scan_parquet(candidate_behavior_path)
    selected_behavior_cols = _non_overlapping_source_columns(behavior, schema_names(assignments))
    joined = (
        behavior.select(selected_behavior_cols)
        .unique(subset=["cliente"], keep="first")
        .join(assignments, on="cliente", how="inner")
    )
    return _write_partitioned_tribe_export(
        joined,
        assignments=assignments,
        output_dir=out_dir,
        filename_suffix=filename_suffix,
        manifest_filename=manifest_filename or f"tribe_customer_summaries_manifest_{cfg.mode}.csv",
        export_type="customer_behavior_summaries_by_tribe",
    )


def stage68_output_dir(cfg: PipelineConfig = CONFIG) -> Path:
    """Return the Stage 6.8 evidence root under the normal mode-scoped output tree."""

    return cfg.artifacts / "stage6" / "stage6_8_evidence"


def stage68_artifact_paths(cfg: PipelineConfig = CONFIG) -> dict[str, Path]:
    root = stage68_output_dir(cfg)
    return {
        "directory": root,
        "tribe_evidence_path": root / f"tribe_evidence_{cfg.mode}.parquet",
        "product_lifts_path": root / f"product_lifts_{cfg.mode}.parquet",
        "sector_lifts_path": root / f"sector_lifts_{cfg.mode}.parquet",
        "customer_metric_tests_csv": root / f"customer_metric_tests_{cfg.mode}.csv",
        "noise_vs_core_customer_metrics_csv": root / f"noise_vs_core_customer_metrics_{cfg.mode}.csv",
        "transaction_export_dir": root / "tribe_transactions",
        "customer_export_dir": root / "tribe_customers",
        "manifest_json": root / f"stage68_manifest_{cfg.mode}.json",
    }


def build_stage68_tribe_evidence(
    assignments_path: str | Path,
    *,
    cluster_readiness_path: str | Path | None = None,
    behavior_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Stage 6.8: precompute all raw-data tribe evidence for Stage 7 interpretation."""

    cfg.ensure_directories()
    paths = stage68_artifact_paths(cfg)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    force = cfg.get("cache.force", False) if force is None else force
    assignments_file = Path(assignments_path)
    candidate_behavior_path = (
        Path(behavior_path)
        if behavior_path
        else cfg.artifact_path("behavioral_features", "output", directory=cfg.outputs / "features")
    )
    readiness_file = Path(cluster_readiness_path) if cluster_readiness_path else None
    cache_metadata = {
        "stage": "6.8_tribe_evidence_assembly",
        "mode": cfg.mode,
        "assignments": file_fingerprint(assignments_file),
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "behavioral_features": file_fingerprint(candidate_behavior_path),
        "cluster_readiness": file_fingerprint(readiness_file) if readiness_file else None,
        "profiling": {
            "min_product_customers": cfg.get("profiling.min_product_customers", 10),
            "significance_q_threshold": cfg.get("profiling.significance_q_threshold", 0.05),
            "top_n_products": 15,
            "top_n_sectors": 5,
            "top_n_copurchase_pairs": 10,
            "min_copurchase_baskets": 5,
        },
        "stage68_schema_version": 2,
    }
    required_outputs = [
        paths["tribe_evidence_path"],
        paths["product_lifts_path"],
        paths["sector_lifts_path"],
        paths["customer_metric_tests_csv"],
        paths["noise_vs_core_customer_metrics_csv"],
        paths["transaction_export_dir"] / f"tribe_transactions_manifest_{cfg.mode}.csv",
        paths["customer_export_dir"] / f"tribe_customers_manifest_{cfg.mode}.csv",
        paths["manifest_json"],
    ]
    if (
        all(path.exists() for path in required_outputs)
        and should_use_cache(
            paths["manifest_json"],
            force=force,
            use_cached=cfg.get("cache.use_cached", True),
            metadata=cache_metadata,
        )
    ):
        log_event("Stage 6.8 evidence", "cache hit", cfg=cfg, manifest=paths["manifest_json"])
        return _stage68_result_from_manifest(paths["manifest_json"], cfg=cfg)

    transactions = pl.scan_parquet(cfg.prepared_transactions_path)
    force_profile = (
        force
        or not paths["tribe_evidence_path"].exists()
        or not paths["product_lifts_path"].exists()
        or not paths["sector_lifts_path"].exists()
    )
    with stage_timer("Stage 6.8 evidence", "assembling tribe evidence from raw inputs", cfg=cfg):
        tribe_evidence_path = profile_tribes(
            assignments_file,
            transactions=transactions,
            behavior_path=candidate_behavior_path,
            output_path=paths["tribe_evidence_path"],
            force=force_profile,
            enable_llm=False,
            product_lifts_output_path=paths["product_lifts_path"],
            sector_lifts_output_path=paths["sector_lifts_path"],
            stage_label="Stage 6.8 evidence",
            cfg=cfg,
        )
        tribe_evidence_path = _attach_stage6_readiness_to_profile(
            tribe_evidence_path,
            readiness_file,
            cfg=cfg,
        )
        if candidate_behavior_path.exists():
            customer_metric_anova_table(
                assignments_file,
                behavior_path=candidate_behavior_path,
                output_csv=paths["customer_metric_tests_csv"],
                cfg=cfg,
            )
            noise_vs_core_customer_metric_table(
                assignments_file,
                behavior_path=candidate_behavior_path,
                output_csv=paths["noise_vs_core_customer_metrics_csv"],
                cfg=cfg,
            )
        else:
            _empty_customer_metric_tests().write_csv(paths["customer_metric_tests_csv"])
            _empty_noise_vs_core_metric_tests().write_csv(paths["noise_vs_core_customer_metrics_csv"])

        transaction_exports = write_tribe_transaction_exports(
            assignments_file,
            transactions=transactions,
            output_dir=paths["transaction_export_dir"],
            filename_suffix=f"transactions_{cfg.mode}",
            manifest_filename=f"tribe_transactions_manifest_{cfg.mode}.csv",
            cfg=cfg,
        )
        customer_exports = write_tribe_customer_summary_exports(
            assignments_file,
            candidate_behavior_path,
            output_dir=paths["customer_export_dir"],
            filename_suffix=f"customers_{cfg.mode}",
            manifest_filename=f"tribe_customers_manifest_{cfg.mode}.csv",
            cfg=cfg,
        )

        manifest = _stage68_manifest(
            paths,
            cfg=cfg,
            assignments_file=assignments_file,
            behavior_path=candidate_behavior_path,
            cluster_readiness_path=readiness_file,
            cache_metadata=cache_metadata,
            transaction_exports=transaction_exports,
            customer_exports=customer_exports,
            tribe_evidence_path=tribe_evidence_path,
        )
        paths["manifest_json"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        write_artifact_metadata(paths["manifest_json"], cache_metadata)
        write_artifact_metadata(paths["tribe_evidence_path"], cache_metadata)
        log_event("Stage 6.8 evidence", "wrote evidence bundle", cfg=cfg, manifest=paths["manifest_json"])
    return _stage68_result_from_manifest(paths["manifest_json"], cfg=cfg)


def _stage68_manifest(
    paths: dict[str, Path],
    *,
    cfg: PipelineConfig,
    assignments_file: Path,
    behavior_path: Path,
    cluster_readiness_path: Path | None,
    cache_metadata: dict[str, Any],
    transaction_exports: dict[str, Path],
    customer_exports: dict[str, Path],
    tribe_evidence_path: Path,
) -> dict[str, Any]:
    outputs = {
        "tribe_evidence_path": str(tribe_evidence_path),
        "product_lifts_path": str(paths["product_lifts_path"]),
        "sector_lifts_path": str(paths["sector_lifts_path"]),
        "customer_metric_tests_csv": str(paths["customer_metric_tests_csv"]),
        "noise_vs_core_customer_metrics_csv": str(paths["noise_vs_core_customer_metrics_csv"]),
        "transaction_export_dir": str(transaction_exports.get("directory", paths["transaction_export_dir"])),
        "transaction_export_manifest_csv": str(transaction_exports.get("manifest_csv")) if transaction_exports.get("manifest_csv") else None,
        "customer_export_dir": str(customer_exports.get("directory", paths["customer_export_dir"])),
        "customer_export_manifest_csv": str(customer_exports.get("manifest_csv")) if customer_exports.get("manifest_csv") else None,
    }
    return {
        "stage": "6.8_tribe_evidence_assembly",
        "mode": cfg.mode,
        "memory_policy": (
            "Stage 6.8 owns all prepared transaction and behavioral feature scans. Stage 7 should read "
            "tribe_evidence_path plus this manifest and must not reopen raw evidence inputs."
        ),
        "inputs": {
            "assignments_path": str(assignments_file),
            "prepared_transactions_path": str(cfg.prepared_transactions_path),
            "behavior_path": str(behavior_path),
            "cluster_readiness_path": str(cluster_readiness_path) if cluster_readiness_path else None,
        },
        "cache_fingerprint": cache_metadata,
        "outputs": outputs,
        "transaction_exports": _partition_manifest_entries(transaction_exports.get("manifest_csv")),
        "customer_exports": _partition_manifest_entries(customer_exports.get("manifest_csv")),
        "readiness_summary": _stage68_readiness_summary(tribe_evidence_path, cfg=cfg),
        "row_counts": {
            "tribe_evidence": _parquet_row_count(tribe_evidence_path),
            "product_lifts": _parquet_row_count(paths["product_lifts_path"]),
            "sector_lifts": _parquet_row_count(paths["sector_lifts_path"]),
            "customer_metric_tests": _csv_row_count(paths["customer_metric_tests_csv"]),
            "noise_vs_core_customer_metrics": _csv_row_count(paths["noise_vs_core_customer_metrics_csv"]),
        },
    }


def _attach_stage6_readiness_to_profile(
    profile_path: str | Path,
    cluster_readiness_path: str | Path | None,
    *,
    cfg: PipelineConfig,
) -> Path:
    """Copy the final Stage 6.6 readiness status into the Stage 6.8 evidence parquet."""

    output = Path(profile_path)
    readiness = _stage6_readiness_table(cluster_readiness_path)
    if readiness.is_empty():
        return output
    profiles = pl.read_parquet(output)
    if profiles.is_empty() or "tribe_id" not in profiles.columns:
        return output
    readiness = readiness.rename({"stage6_profile_readiness": "stage6_profile_readiness_from_stage6"})
    enriched = profiles.join(readiness, on="tribe_id", how="left")
    if "stage6_profile_readiness" in enriched.columns:
        readiness_expr = pl.coalesce(
            [
                pl.col("stage6_profile_readiness_from_stage6"),
                pl.col("stage6_profile_readiness"),
            ]
        )
    else:
        readiness_expr = pl.col("stage6_profile_readiness_from_stage6")
    enriched = (
        enriched.with_columns(readiness_expr.cast(pl.Utf8).alias("stage6_profile_readiness"))
        .drop("stage6_profile_readiness_from_stage6")
        .select(profiles.columns)
    )
    enriched.write_parquet(output)
    missing = int(enriched.filter(pl.col("stage6_profile_readiness").is_null()).height)
    log_event(
        "Stage 6.8 evidence",
        "attached Stage 6.6 readiness to evidence parquet",
        cfg=cfg,
        profile_path=output,
        readiness_path=cluster_readiness_path,
        missing_readiness_rows=missing,
    )
    return output


def _stage6_readiness_table(cluster_readiness_path: str | Path | None) -> pl.DataFrame:
    if cluster_readiness_path is None:
        return pl.DataFrame(schema={"tribe_id": pl.Int64, "stage6_profile_readiness": pl.Utf8})
    readiness = _read_optional_table(cluster_readiness_path)
    if readiness.is_empty() or "tribe_id" not in readiness.columns:
        return pl.DataFrame(schema={"tribe_id": pl.Int64, "stage6_profile_readiness": pl.Utf8})
    status_columns = [
        column
        for column in ["stage6_profile_readiness", "profile_readiness", "profiling_readiness", "readiness"]
        if column in readiness.columns
    ]
    if not status_columns:
        return pl.DataFrame(schema={"tribe_id": pl.Int64, "stage6_profile_readiness": pl.Utf8})
    return (
        readiness.select(
            [
                pl.col("tribe_id").cast(pl.Int64),
                pl.coalesce([pl.col(column).cast(pl.Utf8) for column in status_columns]).alias("stage6_profile_readiness"),
            ]
        )
        .group_by("tribe_id")
        .agg(pl.col("stage6_profile_readiness").drop_nulls().first().alias("stage6_profile_readiness"))
        .sort("tribe_id")
    )


def _stage68_readiness_summary(profile_path: str | Path, *, cfg: PipelineConfig) -> dict[str, Any]:
    try:
        profiles = pl.read_parquet(profile_path, columns=["tribe_id", "stage6_profile_readiness"])
    except Exception:
        return {
            "retained_tribes": 0,
            "promoted_tribes": 0,
            "review_tribes": 0,
            "missing_readiness_tribes": 0,
            "promoted_statuses": sorted(_final_readiness_statuses(None, cfg)),
            "status_counts": {},
            "promoted_tribe_ids": [],
            "review_tribe_ids": [],
            "missing_readiness_tribe_ids": [],
        }
    allowed_statuses = _final_readiness_statuses(None, cfg)
    status_counts: dict[str, int] = {}
    promoted_ids: list[int] = []
    review_ids: list[int] = []
    missing_ids: list[int] = []
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row["tribe_id"])
        status = str(row.get("stage6_profile_readiness") or "").strip()
        if not status:
            status = "missing"
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in allowed_statuses:
            promoted_ids.append(tribe_id)
        else:
            review_ids.append(tribe_id)
            if status == "missing":
                missing_ids.append(tribe_id)
    return {
        "retained_tribes": profiles.height,
        "promoted_tribes": len(promoted_ids),
        "review_tribes": len(review_ids),
        "missing_readiness_tribes": len(missing_ids),
        "promoted_statuses": sorted(allowed_statuses),
        "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "promoted_tribe_ids": promoted_ids,
        "review_tribe_ids": review_ids,
        "missing_readiness_tribe_ids": missing_ids,
    }


def _partition_manifest_entries(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    return pl.read_csv(path).to_dicts()


def _parquet_row_count(path: str | Path) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    return int(collect_streaming(pl.scan_parquet(file_path).select(pl.len().alias("rows")))[0, "rows"])


def _csv_row_count(path: str | Path) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    return int(pl.read_csv(file_path).height)


def _stage68_rebased_path(path: str | Path, *, cfg: PipelineConfig) -> Path:
    """Return the active checkout's Stage 6.8 artifact when a manifest path is stale."""

    original = Path(path)
    if original.exists():
        return original

    parts = [part for part in str(path).replace("\\", "/").split("/") if part and part != "."]
    try:
        anchor_index = parts.index("stage6_8_evidence")
    except ValueError:
        return original

    suffix = parts[anchor_index + 1 :]
    candidate = stage68_output_dir(cfg).joinpath(*suffix)
    return candidate if candidate.exists() else original


def _stage68_manifest_with_local_paths(manifest: dict[str, Any], *, cfg: PipelineConfig) -> dict[str, Any]:
    """Rebase portable Stage 6.8 paths from a manifest onto the current repo if present."""

    normalized = dict(manifest)
    outputs = {
        key: str(_stage68_rebased_path(value, cfg=cfg))
        for key, value in (manifest.get("outputs") or {}).items()
        if value
    }
    normalized["outputs"] = outputs

    for export_key in ["transaction_exports", "customer_exports"]:
        entries: list[dict[str, Any]] = []
        for entry in manifest.get(export_key) or []:
            normalized_entry = dict(entry)
            if normalized_entry.get("path"):
                normalized_entry["path"] = str(_stage68_rebased_path(normalized_entry["path"], cfg=cfg))
            entries.append(normalized_entry)
        normalized[export_key] = entries

    return normalized


def _stage68_result_from_manifest(manifest_path: str | Path, *, cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = _stage68_manifest_with_local_paths(json.loads(manifest_file.read_text(encoding="utf-8")), cfg=cfg)
    outputs = {key: Path(value) for key, value in (manifest.get("outputs") or {}).items() if value}
    return {
        "manifest_json": manifest_file,
        "manifest": manifest,
        **outputs,
    }


def _stage68_manifest_paths(stage68_manifest_path: str | Path | None, *, cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    if not stage68_manifest_path:
        return {}
    manifest_file = Path(stage68_manifest_path)
    if not manifest_file.exists():
        return {}
    return _stage68_result_from_manifest(manifest_file, cfg=cfg)


def _require_stage68_manifest_paths(stage68_manifest_path: str | Path | None, *, cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    if not stage68_manifest_path or not Path(stage68_manifest_path).exists():
        raise RuntimeError(STAGE7_STAGE68_REQUIRED_MESSAGE)
    paths = _stage68_manifest_paths(stage68_manifest_path, cfg=cfg)
    if not paths.get("manifest"):
        raise RuntimeError(STAGE7_STAGE68_REQUIRED_MESSAGE)
    return paths


def _require_stage68_output(paths: dict[str, Any], key: str) -> Path:
    output = paths.get(key)
    if output is None or not Path(output).exists():
        raise RuntimeError(STAGE7_STAGE68_REQUIRED_MESSAGE)
    return Path(output)


def _stage68_partition_export_paths(manifest: dict[str, Any], key: str) -> dict[str, Path]:
    entries = list(manifest.get(key) or [])
    output_key = "transaction_export" if key == "transaction_exports" else "customer_export"
    outputs = manifest.get("outputs") or {}
    paths: dict[str, Path] = {}
    directory = outputs.get(f"{output_key}_dir")
    manifest_csv = outputs.get(f"{output_key}_manifest_csv")
    if directory:
        paths["directory"] = Path(directory)
    if manifest_csv:
        paths["manifest_csv"] = Path(manifest_csv)
    for entry in entries:
        tribe_id = entry.get("tribe_id")
        entry_path = entry.get("path")
        if tribe_id is None or not entry_path:
            continue
        paths[f"tribe_{int(tribe_id):02d}_parquet"] = Path(entry_path)
    return paths


def _stage7_distribution_metrics_by_tribe(
    profiles: pl.DataFrame,
    transaction_export_paths: dict[str, Path],
    customer_export_paths: dict[str, Path],
    *,
    cfg: PipelineConfig,
) -> dict[int, dict[str, Any]]:
    metrics: dict[int, dict[str, Any]] = {}
    for row, _, _ in _final_profile_records(profiles, pl.DataFrame(), cfg=cfg):
        tribe_id = int(row["tribe_id"])
        transaction_path = transaction_export_paths.get(f"tribe_{tribe_id:02d}_parquet")
        customer_path = customer_export_paths.get(f"tribe_{tribe_id:02d}_parquet")
        if transaction_path is None or not transaction_path.exists():
            raise FileNotFoundError(
                f"Stage 6.8 transaction export is missing for tribe {tribe_id}: {transaction_path}"
            )
        if customer_path is None or not customer_path.exists():
            raise FileNotFoundError(f"Stage 6.8 customer export is missing for tribe {tribe_id}: {customer_path}")
        metrics[tribe_id] = _stage7_distribution_metrics(row, transaction_path, customer_path, cfg=cfg)
    return metrics


def _stage7_distribution_metrics(
    row: dict[str, Any],
    transaction_path: Path,
    customer_path: Path,
    *,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    transaction_metrics = _stage7_transaction_distribution_metrics(row, transaction_path, cfg=cfg)
    customer_metrics = _stage7_customer_distribution_metrics(customer_path)
    return {**_empty_stage7_distribution_metrics(), **transaction_metrics, **customer_metrics}


def _empty_stage7_distribution_metrics() -> dict[str, Any]:
    return {
        "widely_purchased_products": [],
        "sector_spend_shares": [],
        "top_reach_product": None,
        "top_reach_product_customers": None,
        "top_reach_product_reach_pct": None,
        "stage7_name_theme_candidates": [],
        "spend_p25_eur": None,
        "spend_p50_eur": None,
        "spend_p75_eur": None,
        "spend_p90_eur": None,
        "active_customer_pct": None,
        "at_risk_customer_pct": None,
        "lapsed_customer_pct": None,
    }


def _stage7_transaction_distribution_metrics(
    row: dict[str, Any],
    transaction_path: Path,
    *,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    scan = pl.scan_parquet(transaction_path)
    columns = set(schema_names(scan))
    if "cliente" not in columns:
        return {}

    product_col = _first_existing_column(columns, ["desc_larga_articulo", "idarticu"])
    sector_col = _first_existing_column(columns, ["desc_sector", "idsector"])
    amount_col = "importe" if "importe" in columns else None
    total_customers = _lazy_scalar_int(scan.select(pl.col("cliente").n_unique().alias("n_customers")), "n_customers")

    widely_purchased: list[dict[str, Any]] = []
    if product_col:
        product_frame = collect_streaming(
            scan.select(
                [
                    pl.col("cliente"),
                    pl.col(product_col).cast(pl.Utf8, strict=False).fill_null("Unknown product").alias("product"),
                ]
            )
            .filter(pl.col("product").is_not_null())
            .group_by("product")
            .agg(pl.col("cliente").n_unique().alias("customers"))
            .with_columns((pl.col("customers") * 100.0 / max(total_customers, 1)).alias("reach_pct"))
            .sort(["customers", "product"], descending=[True, False])
            .head(5)
        )
        widely_purchased = [
            {
                "product": _repair_display_text(str(item["product"])),
                "customers": int(item["customers"] or 0),
                "reach_pct": _safe_float(item["reach_pct"]),
            }
            for item in product_frame.iter_rows(named=True)
        ]

    sector_shares: list[dict[str, Any]] = []
    if sector_col and amount_col:
        total_spend = _lazy_scalar_float(
            scan.select(pl.col(amount_col).cast(pl.Float64, strict=False).sum().alias("total_spend")),
            "total_spend",
        )
        if total_spend and total_spend > 0:
            sector_frame = collect_streaming(
                scan.select(
                    [
                        pl.col(sector_col).cast(pl.Utf8, strict=False).fill_null("Unknown department").alias("sector"),
                        pl.col(amount_col).cast(pl.Float64, strict=False).fill_null(0.0).alias("importe"),
                    ]
                )
                .group_by("sector")
                .agg(pl.col("importe").sum().alias("spend_eur"))
                .with_columns((pl.col("spend_eur") * 100.0 / total_spend).alias("spend_share_pct"))
                .sort(["spend_eur", "sector"], descending=[True, False])
                .head(6)
            )
            sector_lifts = _sector_lift_lookup(row)
            sector_shares = [
                {
                    "sector": _repair_display_text(str(item["sector"])),
                    "spend_share_pct": _safe_float(item["spend_share_pct"]),
                    "spend_eur": _safe_float(item["spend_eur"]),
                    "lift_vs_rest": sector_lifts.get(str(item["sector"])),
                }
                for item in sector_frame.iter_rows(named=True)
            ]

    top_reach = widely_purchased[0] if widely_purchased else {}
    return {
        "widely_purchased_products": widely_purchased,
        "sector_spend_shares": sector_shares,
        "stage7_name_theme_candidates": _stage7_name_theme_candidates(row, scan, columns, total_customers, cfg=cfg),
        "top_reach_product": top_reach.get("product"),
        "top_reach_product_customers": top_reach.get("customers"),
        "top_reach_product_reach_pct": top_reach.get("reach_pct"),
    }


def _stage7_name_theme_candidates(
    row: dict[str, Any],
    scan: pl.LazyFrame,
    columns: set[str],
    total_customers: int,
    *,
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    if "cliente" not in columns:
        return []
    product_col = _first_existing_column(columns, ["desc_larga_articulo", "idarticu"])
    if not product_col:
        return []
    theme_map = _stage7_product_theme_map(scan, product_col)
    if theme_map.is_empty():
        return []
    customer_theme = collect_streaming(
        scan.select(
            [
                pl.col("cliente"),
                pl.col(product_col).cast(pl.Utf8, strict=False).fill_null("Unknown product").alias("product"),
            ]
        )
        .join(theme_map.lazy(), on="product", how="inner")
        .select(["cliente", "theme_key"])
        .unique(subset=["cliente", "theme_key"])
        .group_by("theme_key")
        .agg(pl.col("cliente").n_unique().alias("customers"))
        .with_columns((pl.col("customers") * 100.0 / max(total_customers, 1)).alias("reach_pct"))
    )
    if customer_theme.is_empty():
        return []

    lifted_support = _stage7_lifted_theme_support(row, cfg=cfg)
    candidates: list[dict[str, Any]] = []
    for item in customer_theme.iter_rows(named=True):
        theme_key = str(item["theme_key"])
        support = lifted_support.get(theme_key)
        if not support:
            continue
        customers = int(item.get("customers") or 0)
        reach_pct = _safe_float(item.get("reach_pct")) or 0.0
        if not _stage7_theme_candidate_passes(support, customers, reach_pct, cfg=cfg):
            continue
        max_lift = _safe_float(support.get("max_lift_vs_rest"))
        min_q = _safe_float(support.get("min_q_value"))
        candidates.append(
            {
                "theme_key": theme_key,
                "theme_label": THEME_LABELS.get(theme_key, _theme_label_from_key(theme_key)),
                "customers": customers,
                "reach_pct": round(reach_pct, 3),
                "supporting_product_count": int(support.get("supporting_product_count") or 0),
                "tagged_product_count": int(support.get("tagged_product_count") or 0),
                "max_lift_vs_rest": max_lift,
                "min_q_value": min_q,
                "product_evidence": support.get("product_evidence"),
                "score": _stage7_theme_candidate_score(reach_pct, max_lift, support.get("supporting_product_count")),
            }
        )
    return sorted(
        candidates,
        key=lambda item: (
            -float(item.get("score") or 0.0),
            -float(item.get("reach_pct") or 0.0),
            str(item.get("theme_label") or ""),
        ),
    )


def _stage7_product_theme_map(scan: pl.LazyFrame, product_col: str) -> pl.DataFrame:
    products = collect_streaming(
        scan.select(pl.col(product_col).cast(pl.Utf8, strict=False).fill_null("Unknown product").alias("product"))
        .filter(pl.col("product").is_not_null())
        .unique()
    )
    rows: list[dict[str, str]] = []
    for item in products.iter_rows(named=True):
        product = str(item.get("product") or "")
        for theme_key in detect_product_themes(product):
            rows.append({"product": product, "theme_key": theme_key})
    return (
        pl.from_dicts(rows, infer_schema_length=None).unique(subset=["product", "theme_key"])
        if rows
        else pl.DataFrame(schema={"product": pl.Utf8, "theme_key": pl.Utf8})
    )


def _stage7_lifted_theme_support(row: dict[str, Any], *, cfg: PipelineConfig) -> dict[str, dict[str, Any]]:
    min_lift = float(cfg.get("profiling.theme_label_min_lift", 1.5))
    q_threshold = float(cfg.get("profiling.theme_label_q_threshold", cfg.get("profiling.significance_q_threshold", 0.05)))
    require_significant = bool(cfg.get("profiling.theme_label_require_significant", True))
    max_examples = int(cfg.get("profiling.theme_product_example_count", 4))
    support: dict[str, dict[str, Any]] = {}
    products = row.get("top_products") or []
    lifts = row.get("top_product_lifts_vs_rest") or row.get("top_product_lifts") or []
    q_values = row.get("top_product_q_values") or []
    counts = row.get("top_product_customer_counts") or []
    reaches = row.get("top_product_reach_pct") or []
    for idx, product in enumerate(products):
        product_text = _repair_display_text(str(product))
        theme_keys = detect_product_themes(product_text)
        if not theme_keys:
            continue
        lift = _safe_float(_value_at(lifts, idx))
        q_value = _safe_float(_value_at(q_values, idx))
        customers = int(_safe_float(_value_at(counts, idx)) or 0)
        reach_pct = _safe_float(_value_at(reaches, idx)) or 0.0
        significant = q_value is not None and q_value <= q_threshold
        lifted = lift is not None and lift >= min_lift and (significant or not require_significant)
        evidence = _proof_signal_text(
            {
                "label": product_text,
                "lift_vs_rest": lift or 0.0,
                "reach_pct": reach_pct,
                "customers": customers,
                "q_value": q_value,
            }
        )
        for theme_key in theme_keys:
            item = support.setdefault(
                theme_key,
                {
                    "tagged_product_count": 0,
                    "supporting_product_count": 0,
                    "max_lift_vs_rest": None,
                    "min_q_value": None,
                    "examples": [],
                },
            )
            item["tagged_product_count"] = int(item["tagged_product_count"]) + 1
            if lifted:
                item["supporting_product_count"] = int(item["supporting_product_count"]) + 1
                item["max_lift_vs_rest"] = max(
                    [value for value in [_safe_float(item.get("max_lift_vs_rest")), lift] if value is not None]
                    or [0.0]
                )
                if q_value is not None:
                    existing_q = _safe_float(item.get("min_q_value"))
                    item["min_q_value"] = q_value if existing_q is None else min(existing_q, q_value)
                if len(item["examples"]) < max_examples:
                    item["examples"].append(evidence)
    for item in support.values():
        item["product_evidence"] = "; ".join(item.pop("examples", []))
    return support


def _stage7_theme_candidate_passes(
    support: dict[str, Any],
    customers: int,
    reach_pct: float,
    *,
    cfg: PipelineConfig,
) -> bool:
    min_coverage = float(cfg.get("profiling.theme_label_min_coverage", 0.15))
    min_customers = int(cfg.get("profiling.theme_label_min_customers", 100))
    min_tagged_products = int(cfg.get("profiling.theme_label_min_tagged_products", 2))
    return (
        reach_pct / 100.0 >= min_coverage
        and customers >= min_customers
        and int(support.get("supporting_product_count") or 0) >= min_tagged_products
    )


def _stage7_theme_candidate_score(reach_pct: float, lift: Any, supporting_product_count: Any) -> float:
    lift_value = _safe_float(lift) or 1.0
    count = max(int(_safe_float(supporting_product_count) or 0), 1)
    return (reach_pct / 100.0) * lift_value * math.log(count + 1.0)


def _stage7_customer_distribution_metrics(customer_path: Path) -> dict[str, Any]:
    scan = pl.scan_parquet(customer_path)
    columns = set(schema_names(scan))
    customer_count_expr = pl.col("cliente").n_unique().alias("customer_count") if "cliente" in columns else pl.len().alias("customer_count")
    exprs = [customer_count_expr]
    if "total_spend" in columns:
        spend = pl.col("total_spend").cast(pl.Float64, strict=False)
        exprs.extend(
            [
                spend.quantile(0.25).alias("spend_p25_eur"),
                spend.quantile(0.50).alias("spend_p50_eur"),
                spend.quantile(0.75).alias("spend_p75_eur"),
                spend.quantile(0.90).alias("spend_p90_eur"),
            ]
        )
    if "recency_days" in columns:
        recency = pl.col("recency_days").cast(pl.Float64, strict=False)
        exprs.extend(
            [
                pl.when(recency < 30).then(1).otherwise(0).sum().alias("active_customers"),
                pl.when((recency >= 30) & (recency <= 90)).then(1).otherwise(0).sum().alias("at_risk_customers"),
                pl.when(recency > 90).then(1).otherwise(0).sum().alias("lapsed_customers"),
            ]
        )
    stats = collect_streaming(scan.select(exprs)).row(0, named=True)
    customer_count = int(stats.get("customer_count") or 0)
    return {
        "spend_p25_eur": _safe_float(stats.get("spend_p25_eur")),
        "spend_p50_eur": _safe_float(stats.get("spend_p50_eur")),
        "spend_p75_eur": _safe_float(stats.get("spend_p75_eur")),
        "spend_p90_eur": _safe_float(stats.get("spend_p90_eur")),
        "active_customer_pct": _pct(stats.get("active_customers"), customer_count),
        "at_risk_customer_pct": _pct(stats.get("at_risk_customers"), customer_count),
        "lapsed_customer_pct": _pct(stats.get("lapsed_customers"), customer_count),
    }


def _first_existing_column(columns: set[str], candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in columns), None)


def _lazy_scalar_int(lazy: pl.LazyFrame, column: str) -> int:
    return int(collect_streaming(lazy)[0, column] or 0)


def _lazy_scalar_float(lazy: pl.LazyFrame, column: str) -> float | None:
    return _safe_float(collect_streaming(lazy)[0, column])


def _pct(numerator: Any, denominator: int) -> float | None:
    value = _safe_float(numerator)
    if value is None or denominator <= 0:
        return None
    return round(100.0 * value / denominator, 2)


def _sector_lift_lookup(row: dict[str, Any]) -> dict[str, float | None]:
    sectors = row.get("top_sectors") or []
    lifts = row.get("top_sector_lifts_vs_rest") or row.get("top_sector_lifts") or []
    return {str(sector): _safe_float(_value_at(lifts, idx)) for idx, sector in enumerate(sectors)}


def stage7_storyline_table(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    subsegment_summary_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    del subsegment_summary_path
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    readiness = _read_optional_table(readiness_path)
    comparison = _read_optional_table(comparison_path)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(profiles.iter_rows(named=True), start=1):
        tribe_id = int(row["tribe_id"])
        readiness_row = _row_by_tribe(readiness, tribe_id)
        comparison_row = _row_by_tribe(comparison, tribe_id)
        reach_warning = _top_product_reach_warning(row)
        base_caveat = row.get("llm_confidence_note") or "This is an organic product-purchase profile, not a demographic persona."
        rows.append(
            {
                "story_order": idx,
                "tribe_id": tribe_id,
                "working_label": _working_label(row),
                "readiness": readiness_row.get("profiling_readiness") or "not checked",
                "size_read": f"{int(row.get('n_customers') or 0):,} customers ({float(row.get('population_share') or 0.0) * 100:.1f}% of assigned)",
                "distinctive_product_evidence": comparison_row.get("top_product_and_category_evidence") or _product_evidence_text(row),
                "product_ranking_basis": PRODUCT_RANKING_BASIS,
                "sector_evidence": _format_lift_items(row.get("top_sectors"), row.get("top_sector_lifts"), row.get("top_sector_line_counts")),
                "mission_evidence": _copurchase_text(row),
                "behavior_evidence": _behavior_ratio_summary(row, profiles),
                "temporal_evidence": _temporal_text(row),
                "loyalty_evidence": _loyalty_text(row),
                "top_product_reach_warning": reach_warning,
                "caveat": f"{base_caveat} {reach_warning}" if reach_warning else base_caveat,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


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
    del customer_metric_tests_path
    storyline = stage7_storyline_table(
        profile_path,
        readiness_path=readiness_path,
        comparison_path=comparison_path,
        subsegment_summary_path=subsegment_summary_path,
        cfg=cfg,
    )
    csv_path = Path(output_csv) if output_csv else cfg.artifacts / "stage7" / f"stage7_storyline_{cfg.mode}.csv"
    md_path = Path(output_md) if output_md else csv_path.with_suffix(".md")
    html_path = Path(output_html) if output_html else csv_path.with_suffix(".html")
    _write_csv(storyline, csv_path)
    md_path.write_text("# Stage 7 Evidence Storyline\n\n" + _markdown_table(storyline), encoding="utf-8")
    html_path.write_text(_html_table(storyline, title="Stage 7 Evidence Storyline"), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "html": html_path}


def write_campaign_signal_artifacts(
    profile_path: str | Path,
    *,
    campaign_signals: set[str] | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
    **_: Any,
) -> dict[str, Path]:
    del profile_path, campaign_signals
    table = pl.DataFrame(schema={"tribe_id": pl.Int64, "signal": pl.Utf8, "note": pl.Utf8})
    csv_path = Path(output_csv) if output_csv else cfg.artifacts / "stage7" / f"campaign_signals_{cfg.mode}.csv"
    md_path = Path(output_md) if output_md else csv_path.with_suffix(".md")
    html_path = Path(output_html) if output_html else csv_path.with_suffix(".html")
    _write_csv(table, csv_path)
    md_path.write_text("# Campaign Signals\n\nNo opt-in campaign signals were derived in the organic Stage 7 profile.\n", encoding="utf-8")
    html_path.write_text(_html_table(table, title="Campaign Signals"), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "html": html_path}


def write_llm_profile_interpretation_pack(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    readiness = _read_optional_table(readiness_path)
    payload = {
        "instruction": "Interpret these organic Carrefour Spain tribes from transaction evidence only. Do not infer age, gender, income, household structure, or other demographics.",
        "tribes": [
            _llm_pack_row(row, _row_by_tribe(readiness, int(row["tribe_id"])), cfg=cfg)
            for row in profiles.iter_rows(named=True)
        ],
    }
    json_path = Path(output_json) if output_json else cfg.artifacts / "stage7" / f"llm_profile_pack_{cfg.mode}.json"
    md_path = Path(output_md) if output_md else json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md_path.write_text("# Stage 7 LLM Evidence Pack\n\n```text\n" + json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n```\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def stage7_llm_evidence_table(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    output_csv: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    readiness = _read_optional_table(readiness_path)
    rows: list[dict[str, Any]] = []
    for row, readiness_row, actionability in _final_profile_records(profiles, readiness, cfg=cfg):
        tribe_id = int(row["tribe_id"])
        rows.extend(_stage7_llm_evidence_rows(tribe_id, row, readiness_row, actionability, cfg=cfg))
    table = _stage7_llm_evidence_frame(rows)
    if output_csv is not None:
        _write_csv(table, output_csv)
    return table


def _stage7_llm_evidence_schema() -> dict[str, Any]:
    return {
        "tribe_id": pl.Int64,
        "evidence_rank": pl.Int64,
        "evidence_type": pl.Utf8,
        "proof_role": pl.Utf8,
        "label": pl.Utf8,
        "customers": pl.Int64,
        "tribe_reach_pct": pl.Float64,
        "lift_vs_rest": pl.Float64,
        "q_value": pl.Float64,
        "profiling_readiness": pl.Utf8,
        "actionability_proof_source": pl.Utf8,
        "actionability_proof": pl.Utf8,
        "product_ranking_basis": pl.Utf8,
        "evidence": pl.Utf8,
    }


def _stage7_llm_evidence_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = _stage7_llm_evidence_schema()
    if not rows:
        return pl.DataFrame(schema=schema)
    frame = pl.from_dicts(rows, infer_schema_length=None)
    for column, dtype in schema.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(column))
        else:
            frame = frame.with_columns(pl.col(column).cast(dtype, strict=False).alias(column))
    return frame.select(list(schema))


def _stage7_final_index_schema() -> dict[str, Any]:
    return {
        "story_order": pl.Int64,
        "tribe_id": pl.Int64,
        "tribe_name": pl.Utf8,
        "name_source": pl.Utf8,
        "stage6_profile_readiness": pl.Utf8,
        "customers_count": pl.Int64,
        "customers": pl.Int64,
        "population_share_pct": pl.Float64,
        "population_share_basis": pl.Utf8,
        "promoted_population_customers": pl.Int64,
        "assigned_population_share_pct": pl.Float64,
        "assigned_population_customers": pl.Int64,
        "review_excluded_customers": pl.Int64,
        "core_customers": pl.Int64,
        "primary_theme": pl.Utf8,
        "primary_theme_confidence": pl.Utf8,
        "theme_read": pl.Utf8,
        "primary_theme_product_evidence": pl.Utf8,
        "actionability_proof_source": pl.Utf8,
        "actionability_proof": pl.Utf8,
        "top_product": pl.Utf8,
        "top_product_reach_pct": pl.Float64,
        "top_product_lift": pl.Float64,
        "top_reach_product": pl.Utf8,
        "top_reach_product_customers": pl.Int64,
        "top_reach_product_reach_pct": pl.Float64,
        "top_product_reach_warning": pl.Utf8,
        "distinctive_products": pl.Utf8,
        "product_ranking_basis": pl.Utf8,
        "shopping_mission": pl.Utf8,
        "total_spend_ratio_vs_rest": pl.Float64,
        "visit_frequency_ratio_vs_rest": pl.Float64,
        "basket_value_ratio_vs_rest": pl.Float64,
        "promo_sensitivity_ratio_vs_rest": pl.Float64,
        "spend_and_visit_context": pl.Utf8,
        "spend_p25_eur": pl.Float64,
        "spend_p50_eur": pl.Float64,
        "spend_p75_eur": pl.Float64,
        "spend_p90_eur": pl.Float64,
        "active_customer_pct": pl.Float64,
        "at_risk_customer_pct": pl.Float64,
        "lapsed_customer_pct": pl.Float64,
        "dominant_shopping_day": pl.Utf8,
        "dominant_shopping_time": pl.Utf8,
        "mean_tenure_days": pl.Float64,
        "tenure_vs_rest_ratio": pl.Float64,
        "mean_recency_days": pl.Float64,
        "visit_trend": pl.Utf8,
        "loyalty_context": pl.Utf8,
        "card_png": pl.Utf8,
        "caveat": pl.Utf8,
    }


def _stage7_final_index_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = _stage7_final_index_schema()
    if not rows:
        return pl.DataFrame(schema=schema)
    frame = pl.from_dicts(rows, infer_schema_length=None)
    for column, dtype in schema.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(column))
        else:
            frame = frame.with_columns(pl.col(column).cast(dtype, strict=False).alias(column))
    return frame.select(list(schema)).sort("story_order")


def write_stage7_final_handoff_pack(
    profile_path: str | Path,
    *,
    assignments_path: str | Path | None = None,
    cluster_readiness_path: str | Path | None = None,
    readiness_path: str | Path | None = None,
    behavior_path: str | Path | None = None,
    transactions: pl.LazyFrame | None = None,
    stage68_manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_cards: bool = True,
    max_card_products: int = 8,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    del assignments_path, cluster_readiness_path, readiness_path, behavior_path, transactions
    out_dir = Path(output_dir) if output_dir else cfg.artifacts / "stage7" / "final_handoff"
    support_dir = out_dir / "supporting_tables"
    card_dir = out_dir / "tribe_cards"
    stage68_paths = _require_stage68_manifest_paths(stage68_manifest_path, cfg=cfg)
    stage68_manifest = stage68_paths.get("manifest") or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    if write_cards:
        card_dir.mkdir(parents=True, exist_ok=True)
    for stale in [
        out_dir / f"stage7_review_candidates_{cfg.mode}.csv",
        out_dir / f"stage7_final_readiness_{cfg.mode}.csv",
        out_dir.parent / f"stage7_profile_readiness_evidence_{cfg.mode}.csv",
    ]:
        if stale.exists():
            stale.unlink()

    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    raw_transaction_export_paths = _stage68_partition_export_paths(stage68_manifest, "transaction_exports")
    customer_summary_export_paths = _stage68_partition_export_paths(stage68_manifest, "customer_exports")
    distribution_metrics = _stage7_distribution_metrics_by_tribe(
        profiles,
        raw_transaction_export_paths,
        customer_summary_export_paths,
        cfg=cfg,
    )

    product_summary_paths = write_tribe_product_summary_artifacts(
        profile_path,
        output_dir=support_dir / "tribe_product_summaries",
        combined_output_csv=support_dir / f"stage7_final_product_summary_long_{cfg.mode}.csv",
        cfg=cfg,
    )
    comparison_paths = write_tribe_comparison_artifacts(
        profile_path,
        output_csv=support_dir / f"stage7_final_tribe_comparison_{cfg.mode}.csv",
        output_md=support_dir / f"stage7_final_tribe_comparison_{cfg.mode}.md",
        output_html=support_dir / f"stage7_final_tribe_comparison_{cfg.mode}.html",
        cfg=cfg,
    )
    llm_evidence_csv = support_dir / f"stage7_llm_evidence_long_{cfg.mode}.csv"
    stage7_llm_evidence_table(profile_path, output_csv=llm_evidence_csv, cfg=cfg)

    metric_tests_path = support_dir / f"stage7_final_customer_metric_tests_{cfg.mode}.csv"
    shutil.copyfile(_require_stage68_output(stage68_paths, "customer_metric_tests_csv"), metric_tests_path)

    noise_vs_core_metrics_path = support_dir / f"stage7_final_noise_vs_core_customer_metrics_{cfg.mode}.csv"
    shutil.copyfile(_require_stage68_output(stage68_paths, "noise_vs_core_customer_metrics_csv"), noise_vs_core_metrics_path)

    storyline_paths = write_stage7_storyline_artifacts(
        profile_path,
        comparison_path=comparison_paths["csv"],
        output_csv=support_dir / f"stage7_final_evidence_storyline_{cfg.mode}.csv",
        output_md=support_dir / f"stage7_final_evidence_storyline_{cfg.mode}.md",
        output_html=support_dir / f"stage7_final_evidence_storyline_{cfg.mode}.html",
        cfg=cfg,
    )
    card_paths = (
        write_stage7_tribe_card_pngs(
            profile_path,
            output_dir=card_dir,
            max_products=max_card_products,
            distribution_metrics=distribution_metrics,
            cfg=cfg,
        )
        if write_cards
        else {}
    )
    final_index = stage7_final_index_table(
        profile_path,
        comparison_path=comparison_paths["csv"],
        customer_metric_tests_path=metric_tests_path,
        card_paths=card_paths,
        distribution_metrics=distribution_metrics,
        cfg=cfg,
    )
    review_candidates = _review_candidates_from_profiles(profiles, final_index, cfg=cfg)
    review_tribes = _review_tribe_manifest(review_candidates)
    promoted_tribes = (
        [int(item) for item in final_index["tribe_id"].to_list()]
        if not final_index.is_empty() and "tribe_id" in final_index.columns
        else []
    )

    index_csv = out_dir / f"stage7_final_index_{cfg.mode}.csv"
    story_md = out_dir / f"stage7_final_story_{cfg.mode}.md"
    story_html = out_dir / f"stage7_final_story_{cfg.mode}.html"
    manifest_json = out_dir / f"stage7_final_manifest_{cfg.mode}.json"
    final_index.write_csv(index_csv)
    story = _final_story_markdown(final_index, review_candidates)
    story_md.write_text(story, encoding="utf-8")
    story_html.write_text(_simple_html("Stage 7 Tribe Profiles", story), encoding="utf-8")
    population_share_denominator = _promoted_population_denominator(final_index)
    manifest = {
        "stage": "7_tribe_profiles",
        "mode": cfg.mode,
        "memory_policy": (
            "Stage 7 is a read-only profiling and communication layer. It reads the Stage 6.8 master evidence "
            "parquet plus Stage 6.8 per-tribe exports and never reopens global transactions, behavioral features, "
            "or assignments."
        ),
        "stage68_fingerprint": stage68_manifest.get("cache_fingerprint"),
        "promoted_tribes": promoted_tribes,
        "review_tribes": review_tribes,
        "name_sources_used": _name_sources_used(final_index),
        "population_share_denominator": population_share_denominator,
        "global_noise_customers": _profile_noise_customer_count(profiles),
        "llm_enabled": False,
        "promotion_policy": {
            "readiness_statuses": sorted(_final_readiness_statuses(None, cfg)),
            "readiness_source": "stage6_profile_readiness",
            "additional_stage7_gates": [],
        },
        "primary_outputs": {
            "final_index_csv": str(index_csv),
            "final_story_markdown": str(story_md),
            "tribe_card_directory": str(card_dir) if write_cards else None,
        },
        "supporting_outputs": {
            "stage68_manifest_json": str(stage68_manifest_path) if stage68_manifest_path else None,
            "stage68_product_lifts_parquet": (
                str(stage68_paths.get("product_lifts_path")) if stage68_paths.get("product_lifts_path") else None
            ),
            "stage68_sector_lifts_parquet": (
                str(stage68_paths.get("sector_lifts_path")) if stage68_paths.get("sector_lifts_path") else None
            ),
            "product_summary_long_csv": str(product_summary_paths["combined_csv"]),
            "comparison_csv": str(comparison_paths["csv"]),
            "customer_metric_tests_csv": str(metric_tests_path),
            "noise_vs_core_customer_metrics_csv": str(noise_vs_core_metrics_path),
            "evidence_storyline_markdown": str(storyline_paths["markdown"]),
            "llm_evidence_long_csv": str(llm_evidence_csv),
            "tribe_raw_transaction_export_manifest_csv": (
                str(raw_transaction_export_paths.get("manifest_csv")) if raw_transaction_export_paths else None
            ),
            "tribe_raw_transaction_export_directory": (
                str(raw_transaction_export_paths.get("directory")) if raw_transaction_export_paths else None
            ),
            "tribe_customer_summary_export_manifest_csv": (
                str(customer_summary_export_paths.get("manifest_csv")) if customer_summary_export_paths else None
            ),
            "tribe_customer_summary_export_directory": (
                str(customer_summary_export_paths.get("directory")) if customer_summary_export_paths else None
            ),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {
        "directory": out_dir,
        "index_csv": index_csv,
        "story_markdown": story_md,
        "story_html": story_html,
        "manifest_json": manifest_json,
        "card_directory": card_dir if write_cards else None,
        "card_paths": card_paths,
        "final_index": final_index,
        "review_candidates": review_candidates,
        "review_tribes": review_tribes,
        "product_summary_paths": product_summary_paths,
        "comparison_paths": comparison_paths,
        "customer_metric_tests_csv": metric_tests_path,
        "noise_vs_core_customer_metrics_csv": noise_vs_core_metrics_path,
        "raw_transaction_export_paths": raw_transaction_export_paths,
        "customer_summary_export_paths": customer_summary_export_paths,
        "llm_evidence_csv": llm_evidence_csv,
        "storyline_paths": storyline_paths,
        "distribution_metrics": distribution_metrics,
    }


def stage7_final_index_table(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    customer_metric_tests_path: str | Path | None = None,
    card_paths: dict[int, Path] | None = None,
    distribution_metrics: dict[int, dict[str, Any]] | None = None,
    readiness_statuses: list[str] | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    del customer_metric_tests_path, readiness_path
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    comparison = _read_optional_table(comparison_path)
    if comparison.is_empty():
        comparison = tribe_comparison_table(profile_path)
    rows: list[dict[str, Any]] = []
    final_records = _final_profile_records(
        profiles,
        pl.DataFrame(),
        readiness_statuses=readiness_statuses,
        cfg=cfg,
    )
    promoted_population_customers = sum(int(item[0].get("n_customers") or 0) for item in final_records)
    assigned_population_customers = _profile_population_customer_count(profiles)
    review_excluded_customers = max(assigned_population_customers - promoted_population_customers, 0)
    final_records = sorted(final_records, key=lambda item: int(item[0].get("n_customers") or 0), reverse=True)
    for order, (row, readiness_row, actionability) in enumerate(final_records, start=1):
        tribe_id = int(row["tribe_id"])
        customers = int(row.get("n_customers") or 0)
        comparison_row = _row_by_tribe(comparison, tribe_id)
        products = _top_card_products(row, limit=1)
        metrics = _stage7_distribution_for_tribe(distribution_metrics, tribe_id)
        enriched_row = _stage7_row_with_distribution_metrics(row, metrics)
        loyalty_profile = _parse_json(enriched_row.get("loyalty_profile"), {})
        theme = _primary_theme_context(enriched_row, cfg=cfg)
        name_info = _stage7_name_info(enriched_row, cfg=cfg)
        ratios = _fixed_behavior_ratios(row)
        rows.append(
            {
                "story_order": order,
                "tribe_id": tribe_id,
                "tribe_name": name_info["tribe_name"],
                "name_source": name_info["name_source"],
                "stage6_profile_readiness": readiness_row.get("stage6_profile_readiness") or "not checked",
                "customers_count": customers,
                "customers": customers,
                # Population share is normalized over promoted Stage 7 tribes only, not every Stage 6 candidate.
                "population_share_pct": round(100.0 * customers / max(promoted_population_customers, 1), 2),
                "population_share_basis": "promoted_final_tribes",
                "promoted_population_customers": promoted_population_customers,
                "assigned_population_share_pct": round(100.0 * customers / max(assigned_population_customers, 1), 2),
                "assigned_population_customers": assigned_population_customers,
                "review_excluded_customers": review_excluded_customers,
                "core_customers": int(row.get("core_customers") or 0),
                "primary_theme": theme["primary_theme"],
                "primary_theme_confidence": theme["primary_theme_confidence"],
                "theme_read": theme["theme_read"],
                "primary_theme_product_evidence": theme["primary_theme_product_evidence"],
                "actionability_proof_source": actionability.get("actionability_proof_source"),
                "actionability_proof": actionability.get("actionability_proof"),
                "top_product": products[0]["product"] if products else None,
                "top_product_reach_pct": products[0]["reach_pct"] if products else None,
                "top_product_lift": products[0]["lift_vs_rest"] if products else None,
                "top_reach_product": metrics.get("top_reach_product"),
                "top_reach_product_customers": metrics.get("top_reach_product_customers"),
                "top_reach_product_reach_pct": metrics.get("top_reach_product_reach_pct"),
                "top_product_reach_warning": _top_product_reach_warning(row),
                "distinctive_products": comparison_row.get("top_product_and_category_evidence") or _product_evidence_text(row),
                "product_ranking_basis": PRODUCT_RANKING_BASIS,
                "shopping_mission": enriched_row.get("llm_shopping_mission") or _copurchase_text(enriched_row),
                "total_spend_ratio_vs_rest": ratios["avg_total_spend"],
                "visit_frequency_ratio_vs_rest": ratios["avg_frequency_per_30d"],
                "basket_value_ratio_vs_rest": ratios["avg_basket_value"],
                "promo_sensitivity_ratio_vs_rest": ratios["avg_promo_share"],
                "spend_and_visit_context": _spend_and_visit_context(row, metrics=metrics),
                "spend_p25_eur": metrics.get("spend_p25_eur"),
                "spend_p50_eur": metrics.get("spend_p50_eur"),
                "spend_p75_eur": metrics.get("spend_p75_eur"),
                "spend_p90_eur": metrics.get("spend_p90_eur"),
                "active_customer_pct": metrics.get("active_customer_pct"),
                "at_risk_customer_pct": metrics.get("at_risk_customer_pct"),
                "lapsed_customer_pct": metrics.get("lapsed_customer_pct"),
                "dominant_shopping_day": enriched_row.get("dominant_shopping_day"),
                "dominant_shopping_time": enriched_row.get("dominant_shopping_time"),
                "mean_tenure_days": _safe_float(enriched_row.get("mean_tenure_days") or loyalty_profile.get("mean_tenure_days")),
                "tenure_vs_rest_ratio": _safe_float(loyalty_profile.get("tenure_vs_rest_ratio")),
                "mean_recency_days": _safe_float(enriched_row.get("mean_recency_days") or loyalty_profile.get("mean_recency_days")),
                "visit_trend": loyalty_profile.get("visit_trend"),
                "loyalty_context": _loyalty_text(enriched_row),
                "card_png": str(card_paths.get(tribe_id)) if card_paths and tribe_id in card_paths else None,
                "caveat": enriched_row.get("llm_confidence_note") or "Evidence is based on purchase patterns only.",
            }
        )
    return _stage7_final_index_frame(rows)


def _profile_population_customer_count(profiles: pl.DataFrame) -> int:
    for column in ["profile_population_customers"]:
        if column not in profiles.columns:
            continue
        values = [
            int(value)
            for value in (_safe_float(item) for item in profiles[column].drop_nulls().to_list())
            if value is not None and value > 0
        ]
        if values:
            return values[0]
    if "n_customers" not in profiles.columns or profiles.is_empty():
        return 0
    return int(sum(int(_safe_float(value) or 0) for value in profiles["n_customers"].to_list()))


def _profile_noise_customer_count(profiles: pl.DataFrame) -> int:
    for column in ["unassigned_noise_customers_global", "profile_noise_customers", "noise_customers"]:
        if column not in profiles.columns:
            continue
        values = [
            int(value)
            for value in (_safe_float(item) for item in profiles[column].drop_nulls().to_list())
            if value is not None and value >= 0
        ]
        if values:
            return values[0]
    return 0


def _stage7_distribution_for_tribe(
    distribution_metrics: dict[int, dict[str, Any]] | None,
    tribe_id: int,
) -> dict[str, Any]:
    if not distribution_metrics:
        return _empty_stage7_distribution_metrics()
    return {**_empty_stage7_distribution_metrics(), **(distribution_metrics.get(tribe_id) or {})}


def _stage7_row_with_distribution_metrics(row: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["stage7_name_theme_candidates"] = metrics.get("stage7_name_theme_candidates") or []
    return enriched


def write_stage7_tribe_card_pngs(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_products: int = 8,
    distribution_metrics: dict[int, dict[str, Any]] | None = None,
    readiness_statuses: list[str] | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[int, Path]:
    del readiness_path
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    out_dir = Path(output_dir) if output_dir else cfg.figures / "stage7_tribe_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_card in out_dir.glob("tribe_*_card.png"):
        stale_card.unlink()
    paths: dict[int, Path] = {}
    final_records = _final_profile_records(
        profiles,
        pl.DataFrame(),
        readiness_statuses=readiness_statuses,
        cfg=cfg,
    )
    promoted_population_customers = sum(int(item[0].get("n_customers") or 0) for item in final_records)
    for row, _, _ in final_records:
        tribe_id = int(row["tribe_id"])
        output = out_dir / f"tribe_{tribe_id:02d}_card.png"
        card_row = dict(row)
        metrics = _stage7_distribution_for_tribe(distribution_metrics, tribe_id)
        card_row = _stage7_row_with_distribution_metrics(card_row, metrics)
        card_row.update(_stage7_name_info(card_row, cfg=cfg))
        card_row["population_share_promoted_pct"] = round(
            100.0 * int(row.get("n_customers") or 0) / max(promoted_population_customers, 1),
            2,
        )
        card_row["distribution_metrics"] = metrics
        _write_stage7_tribe_card_png(card_row, output, max_products=max_products)
        paths[tribe_id] = output
    return paths


def _tribe_product_summary_frame(row: dict[str, Any], max_products: int, cfg: PipelineConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    for idx, product in enumerate((row.get("top_products") or [])[:max_products]):
        q_value = _value_at(row.get("top_product_q_values"), idx)
        lift_vs_rest = _value_at(row.get("top_product_lifts_vs_rest"), idx)
        customers = _value_at(row.get("top_product_customer_counts"), idx)
        rows.append(
            {
                "tribe_id": int(row.get("tribe_id") or 0),
                "rank": idx + 1,
                "product_id": _value_at(row.get("top_product_ids"), idx),
                "product_description": _repair_display_text(str(product)),
                "category": _value_at(row.get("top_product_sectors"), idx),
                "lift_vs_rest": lift_vs_rest,
                "lift_vs_population": _value_at(row.get("top_product_lifts"), idx),
                "reach_pct": _value_at(row.get("top_product_reach_pct"), idx),
                "q_value": q_value,
                "customers": customers,
                "product_rank_score": _product_rank_score(lift_vs_rest, customers),
                "product_ranking_formula": PRODUCT_RANKING_FORMULA,
                "product_ranking_basis": PRODUCT_RANKING_BASIS,
                "statistical_result": _overindex_evidence_label(lift_vs_rest, q_value, q_threshold),
            }
        )
    return pl.DataFrame(rows) if rows else _empty_tribe_product_summary()


def _empty_tribe_product_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "tribe_id": pl.Int64,
            "rank": pl.Int64,
            "product_id": pl.Int64,
            "product_description": pl.Utf8,
            "category": pl.Utf8,
            "lift_vs_rest": pl.Float64,
            "lift_vs_population": pl.Float64,
            "reach_pct": pl.Float64,
            "q_value": pl.Float64,
            "customers": pl.Int64,
            "product_rank_score": pl.Float64,
            "product_ranking_formula": pl.Utf8,
            "product_ranking_basis": pl.Utf8,
            "statistical_result": pl.Utf8,
        }
    )


def _empty_customer_metric_tests() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "metric": pl.Utf8,
            "included_tribes": pl.Int64,
            "anova_f_statistic": pl.Float64,
            "anova_p_value": pl.Float64,
            "anova_q_value": pl.Float64,
            "anova_effect_eta_squared": pl.Float64,
            "anova_effect_interpretation": pl.Utf8,
            "highest_mean_tribe_id": pl.Int64,
            "lowest_mean_tribe_id": pl.Int64,
            "statistical_result": pl.Utf8,
        }
    )


def _empty_noise_vs_core_metric_tests() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "metric": pl.Utf8,
            "core_customers": pl.Int64,
            "noise_customers": pl.Int64,
            "core_mean": pl.Float64,
            "noise_mean": pl.Float64,
            "noise_minus_core": pl.Float64,
            "noise_vs_core_ratio": pl.Float64,
            "anova_f_statistic": pl.Float64,
            "anova_p_value": pl.Float64,
            "anova_q_value": pl.Float64,
            "anova_effect_eta_squared": pl.Float64,
            "anova_effect_interpretation": pl.Utf8,
            "statistical_result": pl.Utf8,
            "interpretation": pl.Utf8,
        }
    )


def _anova_metric_row(frame: pl.DataFrame, metric: str) -> dict[str, Any]:
    grouped = frame.group_by("tribe_id").agg([pl.col(metric).mean().alias("mean"), pl.col(metric).count().alias("n")])
    grand_mean = _safe_float(frame.select(pl.col(metric).mean()).item()) or 0.0
    ss_between = 0.0
    ss_total = 0.0
    for group in grouped.iter_rows(named=True):
        ss_between += float(group["n"] or 0) * (float(group["mean"] or 0.0) - grand_mean) ** 2
    for value in frame[metric].drop_nulls().to_list():
        ss_total += (float(value) - grand_mean) ** 2
    eta = ss_between / ss_total if ss_total > 0 else 0.0
    f_statistic, p_value = _customer_metric_anova(frame, metric)
    sorted_groups = grouped.sort("mean")
    return {
        "metric": metric,
        "included_tribes": grouped.height,
        "anova_f_statistic": f_statistic,
        "anova_p_value": p_value,
        "anova_q_value": None,
        "anova_effect_eta_squared": eta,
        "anova_effect_interpretation": _anova_effect_interpretation(eta),
        "highest_mean_tribe_id": int(sorted_groups[-1, "tribe_id"]) if grouped.height else None,
        "lowest_mean_tribe_id": int(sorted_groups[0, "tribe_id"]) if grouped.height else None,
        "statistical_result": _anova_statistical_result(eta, p_value),
    }


def _noise_vs_core_metric_row(frame: pl.DataFrame, anova_frame: pl.DataFrame, metric: str) -> dict[str, Any]:
    core = frame.filter(pl.col("assignment_group") == "core")
    noise = frame.filter(pl.col("assignment_group") == "noise")
    core_mean = _safe_float(core.select(pl.col(metric).mean()).item()) if core.height else None
    noise_mean = _safe_float(noise.select(pl.col(metric).mean()).item()) if noise.height else None
    anova = _anova_metric_row(anova_frame, metric)
    return {
        "metric": metric,
        "core_customers": core["cliente"].n_unique() if "cliente" in core.columns else core.height,
        "noise_customers": noise["cliente"].n_unique() if "cliente" in noise.columns else noise.height,
        "core_mean": core_mean,
        "noise_mean": noise_mean,
        "noise_minus_core": (noise_mean - core_mean) if noise_mean is not None and core_mean is not None else None,
        "noise_vs_core_ratio": _safe_ratio(noise_mean, core_mean),
        "anova_f_statistic": anova.get("anova_f_statistic"),
        "anova_p_value": anova.get("anova_p_value"),
        "anova_q_value": None,
        "anova_effect_eta_squared": anova.get("anova_effect_eta_squared"),
        "anova_effect_interpretation": anova.get("anova_effect_interpretation"),
        "statistical_result": anova.get("statistical_result"),
        "interpretation": None,
    }


def _noise_vs_core_interpretation(row: dict[str, Any]) -> str:
    ratio = _safe_float(row.get("noise_vs_core_ratio"))
    metric = str(row.get("metric") or "metric")
    effect = str(row.get("anova_effect_interpretation") or "unknown_effect")
    significance = str(row.get("statistical_result") or "unknown")
    if ratio is None:
        direction = "cannot compare"
    elif ratio >= 1.05:
        direction = f"noise higher than core ({ratio:.2f}x)"
    elif ratio <= 0.95:
        direction = f"noise lower than core ({ratio:.2f}x)"
    else:
        direction = f"noise broadly similar to core ({ratio:.2f}x)"
    return f"{metric}: {direction}; {effect}; {significance}."


def _customer_metric_anova(frame: pl.DataFrame, metric: str) -> tuple[float | None, float | None]:
    values_by_tribe = (
        frame.select(["tribe_id", metric])
        .drop_nulls()
        .group_by("tribe_id")
        .agg(pl.col(metric).alias("values"))
        .sort("tribe_id")
    )
    arrays = [
        [float(value) for value in row["values"] if _safe_float(value) is not None]
        for row in values_by_tribe.iter_rows(named=True)
    ]
    arrays = [values for values in arrays if values]
    if len(arrays) < 2 or sum(len(values) for values in arrays) <= len(arrays):
        return None, None
    try:
        from scipy.stats import f_oneway
    except ImportError:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = f_oneway(*arrays)
    f_statistic = _safe_float(getattr(result, "statistic", None))
    p_value = _safe_float(getattr(result, "pvalue", None))
    return f_statistic, p_value


def _anova_effect_interpretation(eta: Any) -> str:
    value = _safe_float(eta)
    if value is None:
        return "unknown_effect"
    if value >= 0.14:
        return "large_effect"
    if value >= 0.06:
        return "medium_effect"
    if value >= 0.01:
        return "small_effect"
    return "negligible_effect"


def _anova_statistical_result(eta: Any, q_value: Any) -> str:
    effect = _anova_effect_interpretation(eta)
    q = _safe_float(q_value)
    if q is None:
        return f"{effect}_insufficient_significance_test"
    significance = "significant" if q <= 0.05 else "directional"
    return f"{effect}_{significance}"


def _llm_pack_row(row: dict[str, Any], readiness_row: dict[str, Any], *, cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    theme = _primary_theme_context(row, cfg=cfg)
    actionability = _actionability_proof(row, cfg=cfg)
    return {
        "tribe_id": row.get("tribe_id"),
        "working_label": _final_tribe_name(row),
        "readiness": readiness_row.get("profiling_readiness"),
        "size": {"customers": row.get("n_customers"), "population_share": row.get("population_share")},
        "actionability_proof_source": actionability.get("actionability_proof_source"),
        "actionability_proof": actionability.get("actionability_proof"),
        "primary_theme": theme.get("primary_theme"),
        "supporting_curated_themes": _supporting_curated_themes(row),
        "products": _product_evidence_text(row),
        "product_ranking_basis": PRODUCT_RANKING_BASIS,
        "sectors": _format_lift_items(row.get("top_sectors"), row.get("top_sector_lifts"), row.get("top_sector_line_counts")),
        "copurchase": _parse_json(row.get("copurchase_pairs"), []),
        "behavior_ratios": _parse_json(row.get("behavior_ratio_vs_rest"), {}),
        "temporal_pattern": _parse_json(row.get("temporal_pattern"), {}),
        "loyalty_profile": _parse_json(row.get("loyalty_profile"), {}),
    }


def _review_candidates_from_profiles(profiles: pl.DataFrame, final_index: pl.DataFrame, *, cfg: PipelineConfig) -> pl.DataFrame:
    if profiles.is_empty() or "tribe_id" not in profiles.columns:
        return pl.DataFrame(
            schema={"tribe_id": pl.Int64, "stage6_profile_readiness": pl.Utf8, "review_reason": pl.Utf8}
        )
    final_ids = set(final_index["tribe_id"].to_list()) if not final_index.is_empty() and "tribe_id" in final_index.columns else set()
    allowed_statuses = _final_readiness_statuses(None, cfg)
    rows = []
    for row in profiles.iter_rows(named=True):
        if int(row["tribe_id"]) in final_ids:
            continue
        status = str(row.get("stage6_profile_readiness") or "missing").strip()
        if status in allowed_statuses:
            reason = "promoted_status_missing_from_final_index"
        else:
            reason = f"stage6_profile_readiness={status}"
        rows.append(
            {
                "tribe_id": int(row["tribe_id"]),
                "stage6_profile_readiness": status,
                "review_reason": reason,
            }
        )
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(schema={"tribe_id": pl.Int64, "stage6_profile_readiness": pl.Utf8, "review_reason": pl.Utf8})
    )


def _review_tribe_manifest(review_candidates: pl.DataFrame) -> dict[str, str]:
    if review_candidates.is_empty():
        return {}
    return {
        str(row["tribe_id"]): str(row.get("review_reason") or row.get("stage6_profile_readiness") or "review")
        for row in review_candidates.iter_rows(named=True)
    }


def _name_sources_used(final_index: pl.DataFrame) -> dict[str, list[int]]:
    if final_index.is_empty() or not {"tribe_id", "name_source"}.issubset(set(final_index.columns)):
        return {}
    result: dict[str, list[int]] = {}
    for row in final_index.select(["tribe_id", "name_source"]).iter_rows(named=True):
        source = str(row.get("name_source") or "unknown")
        result.setdefault(source, []).append(int(row["tribe_id"]))
    return {source: sorted(ids) for source, ids in sorted(result.items())}


def _promoted_population_denominator(final_index: pl.DataFrame) -> int:
    if final_index.is_empty() or "promoted_population_customers" not in final_index.columns:
        return 0
    value = _safe_float(final_index[0, "promoted_population_customers"])
    return int(value or 0)


def _write_stage7_tribe_card_png(row: dict[str, Any], output_path: Path, *, max_products: int) -> Path:
    import textwrap

    import matplotlib.pyplot as plt

    metrics = row.get("distribution_metrics") or _empty_stage7_distribution_metrics()
    products = _top_card_products(row, limit=max_products)
    fig = plt.figure(figsize=(14, 10), facecolor="#f8fafc")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    title = f"T{int(row.get('tribe_id') or 0)}: {row.get('tribe_name') or _stage7_name_info(row, cfg=CONFIG)['tribe_name']}"
    ax.text(0.04, 0.955, title, fontsize=21, weight="bold", color="#111827")
    ax.text(
        0.04,
        0.918,
        (
            f"{int(row.get('n_customers') or 0):,} customers | "
            f"{float(row.get('population_share_promoted_pct') or 0.0):.1f}% of promoted tribe population"
        ),
        fontsize=12,
        color="#475467",
    )
    left_x = 0.04
    right_x = 0.53
    col_width = 62

    y = 0.858
    ax.text(left_x, y, "MOST DISTINCTIVE PRODUCTS", fontsize=13, weight="bold", color="#111827")
    ax.text(left_x, y - 0.024, PRODUCT_RANKING_CARD_NOTE, fontsize=9.2, color="#667085")
    y -= 0.058
    distinctive_footnotes: set[str] = set()
    for product in products[:max_products]:
        markers = ""
        if product["reach_pct"] < 1.0:
            markers += " ⚠"
            distinctive_footnotes.add("⚠ reach below 1% of tribe")
        if product["customers"] < 50:
            markers += " †"
            distinctive_footnotes.add("† fewer than 50 customers")
        line = (
            f"{product['rank']}. {product['product']}{markers} "
            f"({product['lift_vs_rest']:.2f}x vs rest, {product['reach_pct']:.1f}% reach, n={product['customers']})"
        )
        wrapped = textwrap.wrap(line, col_width) or [line]
        if y - (len(wrapped) * 0.022) < 0.56:
            break
        ax.text(left_x + 0.015, y, "\n".join(wrapped), fontsize=9.6, color="#344054", va="top")
        y -= len(wrapped) * 0.022 + 0.008
    if distinctive_footnotes:
        ax.text(left_x, 0.548, "; ".join(sorted(distinctive_footnotes)), fontsize=8.8, color="#92400e", va="top")

    y = 0.858
    ax.text(right_x, y, "MOST WIDELY PURCHASED - use for campaign sizing", fontsize=13, weight="bold", color="#111827")
    y -= 0.04
    for idx, product in enumerate(metrics.get("widely_purchased_products") or [], start=1):
        line = f"{idx}. {product['product']} ({product['customers']:,} customers, {product['reach_pct']:.1f}% of tribe)"
        wrapped = textwrap.wrap(line, col_width) or [line]
        if y - (len(wrapped) * 0.022) < 0.56:
            break
        ax.text(right_x + 0.015, y, "\n".join(wrapped), fontsize=9.8, color="#344054", va="top")
        y -= len(wrapped) * 0.022 + 0.010

    y = 0.492
    ax.text(left_x, y, "SPEND AND VISITS", fontsize=13, weight="bold", color="#111827")
    y -= 0.034
    for line in _card_spend_visit_lines(row, metrics):
        ax.text(left_x + 0.015, y, line, fontsize=9.7, color="#344054", va="top")
        y -= 0.030

    y = 0.492
    ax.text(right_x, y, "SPEND BY DEPARTMENT", fontsize=13, weight="bold", color="#111827")
    y -= 0.036
    sector_rows = metrics.get("sector_spend_shares") or []
    max_share = max([_safe_float(item.get("spend_share_pct")) or 0.0 for item in sector_rows] or [1.0])
    for item in sector_rows[:6]:
        share = _safe_float(item.get("spend_share_pct")) or 0.0
        lift = _safe_float(item.get("lift_vs_rest"))
        label = _short_text(str(item.get("sector") or "Unknown department"), 30)
        bar_width = 0.23 * (share / max(max_share, 0.01))
        ax.text(right_x + 0.015, y, label, fontsize=9.3, color="#344054", va="center")
        ax.add_patch(plt.Rectangle((right_x + 0.205, y - 0.010), bar_width, 0.018, color="#2563eb", alpha=0.82))
        lift_text = f", {lift:.2f}x lift" if lift is not None else ""
        ax.text(right_x + 0.445, y, f"{share:.1f}%{lift_text}", fontsize=9.1, color="#475467", va="center")
        y -= 0.032

    mission = row.get("llm_shopping_mission") or _copurchase_text(row)
    ax.text(left_x, 0.238, "SHOPPING MISSIONS / CO-PURCHASE", fontsize=13, weight="bold", color="#111827")
    ax.text(left_x + 0.015, 0.205, "\n".join(textwrap.wrap(str(mission), 82)), fontsize=9.7, color="#344054", va="top")

    ax.text(right_x, 0.238, "TEMPORAL AND LOYALTY", fontsize=13, weight="bold", color="#111827")
    loyalty_lines = [
        _temporal_text(row),
        _loyalty_text(row),
        _recency_segment_text(metrics),
    ]
    y = 0.205
    for line in loyalty_lines:
        for wrapped in textwrap.wrap(str(line), 70) or [str(line)]:
            ax.text(right_x + 0.015, y, wrapped, fontsize=9.6, color="#344054", va="top")
            y -= 0.024
        y -= 0.004

    footer = (
        f"name_source={row.get('name_source')}; "
        f"stage6_profile_readiness={row.get('stage6_profile_readiness') or 'not checked'} | "
        "Profiles based on purchase behaviour only. Not a demographic profile."
    )
    ax.text(0.04, 0.035, footer, fontsize=9.0, color="#475467")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _top_card_products(row: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for idx, product in enumerate((row.get("top_products") or [])[:limit]):
        reach_pct = _safe_float(_value_at(row.get("top_product_reach_pct"), idx))
        rows.append(
            {
                "rank": idx + 1,
                "product": _repair_display_text(str(product)),
                "sector": _value_at(row.get("top_product_sectors"), idx),
                "lift_vs_rest": _safe_float(_value_at(row.get("top_product_lifts_vs_rest"), idx)) or 0.0,
                "reach_pct": reach_pct or 0.0,
                "has_reach_pct": reach_pct is not None,
                "customers": int(_safe_float(_value_at(row.get("top_product_customer_counts"), idx)) or 0),
            }
        )
    return rows


def _card_spend_visit_lines(row: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    ratios = _fixed_behavior_ratios(row)
    spend = _safe_float(row.get("avg_total_spend"))
    frequency = _safe_float(row.get("avg_frequency_per_30d"))
    basket_value = _safe_float(row.get("avg_basket_value") or row.get("avg_avg_basket_value"))
    promo = _safe_float(row.get("avg_promo_share"))
    lines = [
        _ratio_context("Total spend", ratios["avg_total_spend"], f"{_eur(spend)} average" if spend is not None else None),
        _ratio_context(
            "Visit frequency",
            ratios["avg_frequency_per_30d"],
            f"{frequency:.1f} visits per 30 days" if frequency is not None else None,
        ),
        _ratio_context("Basket value", ratios["avg_basket_value"], f"{_eur(basket_value)} average" if basket_value is not None else None),
        _ratio_context("Promo sensitivity", ratios["avg_promo_share"], f"{promo:.1%} promo share" if promo is not None else None),
    ]
    if metrics.get("spend_p50_eur") is not None:
        lines.append(
            "Spend percentiles: "
            f"p25 {_eur(metrics.get('spend_p25_eur'))}, "
            f"p50 {_eur(metrics.get('spend_p50_eur'))}, "
            f"p75 {_eur(metrics.get('spend_p75_eur'))}"
        )
    return lines


def _recency_segment_text(metrics: dict[str, Any]) -> str:
    active = _safe_float(metrics.get("active_customer_pct"))
    at_risk = _safe_float(metrics.get("at_risk_customer_pct"))
    lapsed = _safe_float(metrics.get("lapsed_customer_pct"))
    if active is None and at_risk is None and lapsed is None:
        return "Recency segments unavailable."
    return (
        f"Recency segments: active {active or 0.0:.1f}%, "
        f"at risk {at_risk or 0.0:.1f}%, lapsed {lapsed or 0.0:.1f}%."
    )


def _short_text(value: str, limit: int) -> str:
    text = _repair_display_text(value)
    return text if len(text) <= limit else text[: max(limit - 3, 0)].rstrip() + "..."


def _highest_reach_product(row: dict[str, Any]) -> dict[str, Any] | None:
    products = _top_card_products(row, limit=len(row.get("top_products") or []))
    products = [product for product in products if product.get("has_reach_pct")]
    if not products:
        return None
    return max(products, key=lambda product: product["reach_pct"])


def _top_product_reach_warning(row: dict[str, Any]) -> str | None:
    products = _top_card_products(row, limit=len(row.get("top_products") or []))
    if not products or not any(product.get("has_reach_pct") for product in products):
        return None
    top_product = products[0]
    if not top_product.get("has_reach_pct"):
        return None
    highest_reach_product = max(products, key=lambda product: product["reach_pct"])
    if top_product["reach_pct"] < 10.0:
        return (
            f"Reach caveat: top lifted product reaches {top_product['reach_pct']:.1f}% of the tribe; "
            "read it as a niche over-index signal."
        )
    reach_gap = highest_reach_product["reach_pct"] - top_product["reach_pct"]
    if highest_reach_product["rank"] != top_product["rank"] and reach_gap >= 5.0:
        return (
            f"Reach caveat: top product is ranked by lift; broadest listed product is "
            f"{highest_reach_product['product']} at {highest_reach_product['reach_pct']:.1f}% reach."
        )
    return None


def _final_readiness_statuses(readiness_statuses: list[str] | None, cfg: PipelineConfig) -> set[str]:
    configured = readiness_statuses
    if configured is None:
        configured = cfg.get("profiling.final_handoff_readiness_statuses", ["strong", "usable"])
    values = _configured_status_set(configured)
    return values or {"strong", "usable"}


def _configured_status_set(configured: Any) -> set[str]:
    if isinstance(configured, str):
        configured = [configured]
    return {str(value).strip() for value in (configured or []) if str(value).strip()}


def _final_profile_records(
    profiles: pl.DataFrame,
    readiness: pl.DataFrame,
    *,
    readiness_statuses: list[str] | None = None,
    cfg: PipelineConfig = CONFIG,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    del readiness
    allowed_statuses = _final_readiness_statuses(readiness_statuses, cfg)
    records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row["tribe_id"])
        status = str(row.get("stage6_profile_readiness") or "").strip()
        if status not in allowed_statuses:
            continue
        readiness_row = {
            "tribe_id": tribe_id,
            "stage6_profile_readiness": status,
            "profiling_readiness": status,
            "profiling_readiness_issues": "pass",
        }
        actionability = _actionability_proof(row, cfg=cfg)
        records.append((row, readiness_row, actionability))
    return records


def _actionability_proof(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    term_signals = _actionable_product_term_signals(row, cfg=cfg)
    if term_signals:
        labels = "; ".join(_proof_signal_text(signal) for signal in term_signals)
        primary = term_signals[0]
        return {
            "actionability_proof_source": "data_driven_product_terms",
            "actionability_proof": f"Product-term proof: {labels}",
            "evidence_type": "product_term",
            "label": primary["label"],
            "customers": primary["customers"],
            "tribe_reach_pct": primary["reach_pct"],
            "lift_vs_rest": primary["lift_vs_rest"],
            "q_value": primary["q_value"],
        }

    product_signals = _actionable_lifted_product_signals(row, cfg=cfg)
    if product_signals:
        labels = "; ".join(_proof_signal_text(signal) for signal in product_signals)
        primary = product_signals[0]
        return {
            "actionability_proof_source": "lifted_product_evidence",
            "actionability_proof": f"Lifted-product proof: {labels}",
            "evidence_type": "lifted_product",
            "label": primary["label"],
            "customers": primary["customers"],
            "tribe_reach_pct": primary["reach_pct"],
            "lift_vs_rest": primary["lift_vs_rest"],
            "q_value": primary["q_value"],
        }

    theme = _primary_theme_context(row, cfg=cfg)
    if bool(cfg.get("profiling.final_actionability_allow_theme_proof", False)) and theme.get("theme_key"):
        return {
            "actionability_proof_source": "curated_theme",
            "actionability_proof": theme["theme_read"],
            "evidence_type": "curated_theme",
            "label": theme["primary_theme"],
            "customers": theme.get("customers"),
            "tribe_reach_pct": theme.get("reach_pct"),
            "lift_vs_rest": theme.get("lift_vs_rest"),
            "q_value": theme.get("q_value"),
        }

    return {
        "actionability_proof_source": None,
        "actionability_proof": None,
        "evidence_type": None,
        "label": None,
        "customers": None,
        "tribe_reach_pct": None,
        "lift_vs_rest": None,
        "q_value": None,
    }


def _actionable_product_term_signals(row: dict[str, Any], cfg: PipelineConfig) -> list[dict[str, Any]]:
    min_signals = int(cfg.get("profiling.final_actionability_min_term_signals", 1))
    min_lift = float(cfg.get("profiling.final_actionability_min_term_lift", 1.5))
    min_coverage = float(cfg.get("profiling.final_actionability_min_term_coverage", 0.02))
    min_customers = int(cfg.get("profiling.final_actionability_min_term_customers", 25))
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    n_customers = max(int(row.get("n_customers") or 0), 1)
    terms = row.get("top_product_terms") or []
    lifts = row.get("top_product_term_lifts_vs_rest") or row.get("top_product_term_lifts") or []
    q_values = row.get("top_product_term_q_values") or []
    counts = row.get("top_product_term_customer_counts") or []
    signals = []
    for idx, term in enumerate(terms):
        signal = _actionable_signal(
            label=str(term),
            lift=_value_at(lifts, idx),
            q_value=_value_at(q_values, idx),
            customers=_value_at(counts, idx),
            n_customers=n_customers,
            min_lift=min_lift,
            min_coverage=min_coverage,
            min_customers=min_customers,
            q_threshold=q_threshold,
        )
        if signal:
            signals.append(signal)
    signals = sorted(signals, key=lambda item: (-float(item["lift_vs_rest"] or 0.0), -int(item["customers"] or 0), item["label"]))
    return signals[:max(min_signals, 1)] if len(signals) >= min_signals else []


def _actionable_lifted_product_signals(row: dict[str, Any], cfg: PipelineConfig) -> list[dict[str, Any]]:
    min_signals = int(cfg.get("profiling.final_actionability_min_product_signals", 2))
    min_lift = float(cfg.get("profiling.final_actionability_min_product_lift", 1.5))
    min_coverage = float(cfg.get("profiling.final_actionability_min_product_coverage", 0.005))
    min_customers = int(cfg.get("profiling.final_actionability_min_product_customers", 25))
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    n_customers = max(int(row.get("n_customers") or 0), 1)
    products = row.get("top_products") or []
    lifts = row.get("top_product_lifts_vs_rest") or row.get("top_product_lifts") or []
    q_values = row.get("top_product_q_values") or []
    counts = row.get("top_product_customer_counts") or []
    signals = []
    for idx, product in enumerate(products):
        signal = _actionable_signal(
            label=_repair_display_text(str(product)),
            lift=_value_at(lifts, idx),
            q_value=_value_at(q_values, idx),
            customers=_value_at(counts, idx),
            n_customers=n_customers,
            min_lift=min_lift,
            min_coverage=min_coverage,
            min_customers=min_customers,
            q_threshold=q_threshold,
        )
        if signal:
            signals.append(signal)
    signals = sorted(signals, key=lambda item: (-float(item["lift_vs_rest"] or 0.0), -int(item["customers"] or 0), item["label"]))
    return signals[:max(min_signals, 1)] if len(signals) >= min_signals else []


def _actionable_signal(
    *,
    label: str,
    lift: Any,
    q_value: Any,
    customers: Any,
    n_customers: int,
    min_lift: float,
    min_coverage: float,
    min_customers: int,
    q_threshold: float,
) -> dict[str, Any] | None:
    lift_value = _safe_float(lift)
    q = _safe_float(q_value)
    count = int(_safe_float(customers) or 0)
    reach = 100.0 * count / max(n_customers, 1)
    if lift_value is None or lift_value < min_lift:
        return None
    if q is None or q > q_threshold:
        return None
    if count < min_customers and reach / 100.0 < min_coverage:
        return None
    return {
        "label": _repair_display_text(str(label)),
        "lift_vs_rest": lift_value,
        "q_value": q,
        "customers": count,
        "reach_pct": reach,
    }


def _proof_signal_text(signal: dict[str, Any]) -> str:
    q_value = signal.get("q_value")
    q_text = "" if q_value is None else f", q={float(q_value):.3g}"
    return (
        f"{signal['label']} ({float(signal.get('lift_vs_rest') or 0.0):.2f}x vs rest, "
        f"{float(signal.get('reach_pct') or 0.0):.1f}% reach, n={int(signal.get('customers') or 0)}{q_text})"
    )


def _primary_theme_context(row: dict[str, Any], cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    stage7_candidates = row.get("stage7_name_theme_candidates") or []
    if stage7_candidates:
        candidate = stage7_candidates[0]
        theme_key = str(candidate.get("theme_key") or "")
        label = str(candidate.get("theme_label") or THEME_LABELS.get(theme_key, _theme_label_from_key(theme_key)))
        lift = _safe_float(candidate.get("max_lift_vs_rest"))
        q_value = _safe_float(candidate.get("min_q_value"))
        reach = _safe_float(candidate.get("reach_pct"))
        customers = int(_safe_float(candidate.get("customers")) or 0)
        support_count = int(_safe_float(candidate.get("supporting_product_count")) or 0)
        q_text = "n/a" if q_value is None else f"{q_value:.3g}"
        lift_text = "n/a" if lift is None else f"{lift:.2f}x"
        return {
            "theme_key": theme_key,
            "primary_theme": label,
            "primary_theme_confidence": "stage7_transaction_theme_supported",
            "theme_read": (
                f"{label}: {customers:,} customers ({reach or 0.0:.1f}% reach) bought products in this theme; "
                f"{support_count} lifted products support the label, max lift {lift_text} vs rest, min q={q_text}."
            ),
            "primary_theme_product_evidence": _repair_display_text(str(candidate.get("product_evidence") or "")),
            "customers": customers,
            "reach_pct": reach,
            "lift_vs_rest": lift,
            "q_value": q_value,
        }

    themes = row.get("top_themes") or []
    if not themes:
        return _product_led_theme_context()

    min_lift = float(cfg.get("profiling.theme_label_min_lift", 1.5))
    min_coverage = float(cfg.get("profiling.theme_label_min_coverage", 0.15))
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    n_customers = max(int(row.get("n_customers") or 0), 1)
    lifts = row.get("top_theme_lifts_vs_rest") or row.get("top_theme_lifts") or []
    q_values = row.get("top_theme_q_values") or []
    counts = row.get("top_theme_customer_counts") or []

    for idx, theme_key in enumerate(themes):
        key = str(theme_key)
        label = THEME_LABELS.get(key, _theme_label_from_key(key))
        lift = _safe_float(_value_at(lifts, idx))
        q_value = _safe_float(_value_at(q_values, idx))
        customers = int(_safe_float(_value_at(counts, idx)) or 0)
        reach = 100.0 * customers / max(n_customers, 1)
        if lift is None or lift < min_lift or reach / 100.0 < min_coverage or q_value is None or q_value > q_threshold:
            continue
        product_evidence = _value_at(row.get("top_theme_product_evidence"), idx) or _product_evidence_text(row)
        return {
            "theme_key": key,
            "primary_theme": label,
            "primary_theme_confidence": "curated_theme_supported",
            "theme_read": f"{label}: {lift:.2f}x vs rest, {reach:.1f}% reach, q={q_value:.3g}. Product evidence remains primary.",
            "primary_theme_product_evidence": _repair_display_text(str(product_evidence)),
            "customers": customers,
            "reach_pct": reach,
            "lift_vs_rest": lift,
            "q_value": q_value,
        }
    return _product_led_theme_context()


def _product_led_theme_context() -> dict[str, Any]:
    return {
        "theme_key": None,
        "primary_theme": "Product-led tribe; no broad theme evidence",
        "primary_theme_confidence": "product_led",
        "theme_read": "No curated product theme passed the reporting gate; use SKU and product-term evidence as the primary read.",
        "primary_theme_product_evidence": "n/a",
        "customers": None,
        "reach_pct": None,
        "lift_vs_rest": None,
        "q_value": None,
    }


def _supporting_curated_themes(row: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    themes = row.get("top_themes") or []
    lifts = row.get("top_theme_lifts_vs_rest") or row.get("top_theme_lifts") or []
    q_values = row.get("top_theme_q_values") or []
    counts = row.get("top_theme_customer_counts") or []
    n_customers = max(int(row.get("n_customers") or 0), 1)
    result = []
    for idx, theme_key in enumerate(themes[:limit]):
        customers = int(_safe_float(_value_at(counts, idx)) or 0)
        result.append(
            {
                "theme_key": str(theme_key),
                "theme_label": THEME_LABELS.get(str(theme_key), _theme_label_from_key(str(theme_key))),
                "lift_vs_rest": _safe_float(_value_at(lifts, idx)),
                "q_value": _safe_float(_value_at(q_values, idx)),
                "customers": customers,
                "reach_pct": 100.0 * customers / max(n_customers, 1),
                "product_evidence": _value_at(row.get("top_theme_product_evidence"), idx),
            }
        )
    return result


def _theme_label_from_key(key: str) -> str:
    return " ".join(part.capitalize() for part in key.replace("-", "_").split("_") if part) + " Buyers"


def _stage7_name_info(row: dict[str, Any], *, cfg: PipelineConfig) -> dict[str, str]:
    tribe_id = int(row.get("tribe_id") or 0)
    lookup = cfg.get("profiling.tribe_name_lookup", {}) or {}
    manual = lookup.get(str(tribe_id)) if isinstance(lookup, Mapping) else None
    if manual:
        return {"tribe_name": _normalise_tribe_name(str(manual)), "name_source": "manual"}

    editorial = _editorial_theme_name(row, cfg=cfg)
    if editorial:
        return {"tribe_name": _normalise_tribe_name(editorial), "name_source": "editorial_theme"}

    fallback = _normalise_tribe_name(_working_label(row))
    log_event("Stage 7 naming", "SKU fallback tribe name used", cfg=cfg, tribe_id=tribe_id, tribe_name=fallback)
    return {"tribe_name": fallback, "name_source": "sku_fallback"}


def _editorial_theme_name(row: dict[str, Any], *, cfg: PipelineConfig = CONFIG) -> str | None:
    theme = _primary_theme_context(row, cfg=cfg)
    if not theme.get("theme_key"):
        return None
    return str(theme.get("primary_theme") or "").strip() or None


def _normalise_tribe_name(raw: str) -> str:
    name = _repair_display_text(str(raw)).strip()
    for suffix in (" Product Evidence Cluster", " Evidence Cluster", " Purchase Cluster", " Cluster"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
            break
    if name.endswith(" Buyers"):
        return name
    return f"{name} Buyers" if name else "Unnamed Tribe Buyers"


def _final_tribe_name(row: dict[str, Any]) -> str:
    return _stage7_name_info(row, cfg=CONFIG)["tribe_name"]


def _fixed_behavior_ratios(row: dict[str, Any]) -> dict[str, float | None]:
    ratios = _parse_json(row.get("behavior_ratio_vs_rest"), {})
    return {
        "avg_total_spend": _safe_float(ratios.get("avg_total_spend")),
        "avg_frequency_per_30d": _safe_float(ratios.get("avg_frequency_per_30d")),
        "avg_basket_value": _safe_float(ratios.get("avg_basket_value") or ratios.get("avg_avg_basket_value")),
        "avg_promo_share": _safe_float(ratios.get("avg_promo_share")),
    }


def _spend_and_visit_context(row: dict[str, Any], *, metrics: dict[str, Any] | None = None) -> str:
    metrics = metrics or {}
    ratios = _fixed_behavior_ratios(row)
    parts = []
    spend = _safe_float(row.get("avg_total_spend"))
    frequency = _safe_float(row.get("avg_frequency_per_30d"))
    basket_value = _safe_float(row.get("avg_basket_value") or row.get("avg_avg_basket_value"))
    promo = _safe_float(row.get("avg_promo_share"))
    parts.append(_ratio_context("total spend", ratios["avg_total_spend"], f"{_eur(spend)} avg" if spend is not None else None))
    parts.append(_ratio_context("visit frequency", ratios["avg_frequency_per_30d"], f"{frequency:.1f}/30d" if frequency is not None else None))
    parts.append(_ratio_context("basket value", ratios["avg_basket_value"], f"{_eur(basket_value)} avg" if basket_value is not None else None))
    parts.append(_ratio_context("promo sensitivity", ratios["avg_promo_share"], f"{promo:.1%} promo share" if promo is not None else None))
    if metrics.get("spend_p50_eur") is not None:
        parts.append(
            "spend percentiles "
            f"p25 {_eur(metrics.get('spend_p25_eur'))}, "
            f"p50 {_eur(metrics.get('spend_p50_eur'))}, "
            f"p75 {_eur(metrics.get('spend_p75_eur'))}"
        )
    return "; ".join(parts) if parts else "Spend and visit context unavailable."


def _ratio_context(label: str, ratio: float | None, value_text: str | None = None) -> str:
    ratio_text = f"{ratio:.2f}x vs rest" if ratio is not None else "ratio n/a"
    return f"{label}: {ratio_text}" + (f", {value_text}" if value_text else "")


def _eur(value: Any) -> str:
    numeric = _safe_float(value)
    return "n/a" if numeric is None else f"€{numeric:,.2f}"


def _stage7_llm_evidence_rows(
    tribe_id: int,
    row: dict[str, Any],
    readiness_row: dict[str, Any],
    actionability: dict[str, Any],
    *,
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    rows = []
    if actionability.get("actionability_proof_source"):
        rows.append(
            _llm_evidence_row(
                tribe_id,
                evidence_rank=1,
                evidence_type=str(actionability.get("evidence_type") or "actionability"),
                proof_role="primary_actionability_proof",
                label=actionability.get("label"),
                customers=actionability.get("customers"),
                tribe_reach_pct=actionability.get("tribe_reach_pct"),
                lift_vs_rest=actionability.get("lift_vs_rest"),
                q_value=actionability.get("q_value"),
                evidence=actionability.get("actionability_proof"),
                readiness_row=readiness_row,
                actionability=actionability,
            )
        )

    for product in _top_card_products(row, limit=5):
        rows.append(
            _llm_evidence_row(
                tribe_id,
                evidence_rank=int(product["rank"]),
                evidence_type="lifted_product",
                proof_role="primary_candidate",
                label=product["product"],
                customers=product["customers"],
                tribe_reach_pct=product["reach_pct"],
                lift_vs_rest=product["lift_vs_rest"],
                q_value=_value_at(row.get("top_product_q_values"), int(product["rank"]) - 1),
                evidence=_proof_signal_text(
                    {
                        "label": product["product"],
                        "lift_vs_rest": product["lift_vs_rest"],
                        "reach_pct": product["reach_pct"],
                        "customers": product["customers"],
                        "q_value": _value_at(row.get("top_product_q_values"), int(product["rank"]) - 1),
                    }
                ),
                readiness_row=readiness_row,
                actionability=actionability,
            )
        )

    sector_lifts = row.get("top_sector_lifts_vs_rest") or row.get("top_sector_lifts") or []
    for idx, sector in enumerate((row.get("top_sectors") or [])[:3], start=1):
        rows.append(
            _llm_evidence_row(
                tribe_id,
                evidence_rank=idx,
                evidence_type="sector",
                proof_role="supporting_category_context",
                label=sector,
                customers=_value_at(row.get("top_sector_line_counts"), idx - 1),
                tribe_reach_pct=None,
                lift_vs_rest=_value_at(sector_lifts, idx - 1),
                q_value=_value_at(row.get("top_sector_q_values"), idx - 1),
                evidence=f"{sector}: {_fmt_number(_value_at(sector_lifts, idx - 1))}x vs rest",
                readiness_row=readiness_row,
                actionability=actionability,
            )
        )

    for idx, theme in enumerate(_supporting_curated_themes(row, limit=3), start=1):
        rows.append(
            _llm_evidence_row(
                tribe_id,
                evidence_rank=idx,
                evidence_type="curated_theme",
                proof_role="supplemental_curated_theme_context",
                label=theme["theme_label"],
                customers=theme["customers"],
                tribe_reach_pct=theme["reach_pct"],
                lift_vs_rest=theme["lift_vs_rest"],
                q_value=theme["q_value"],
                evidence=theme.get("product_evidence") or theme["theme_label"],
                readiness_row=readiness_row,
                actionability=actionability,
            )
        )
    return rows


def _llm_evidence_row(
    tribe_id: int,
    *,
    evidence_rank: int,
    evidence_type: str,
    proof_role: str,
    label: Any,
    customers: Any,
    tribe_reach_pct: Any,
    lift_vs_rest: Any,
    q_value: Any,
    evidence: Any,
    readiness_row: dict[str, Any],
    actionability: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tribe_id": tribe_id,
        "evidence_rank": evidence_rank,
        "evidence_type": evidence_type,
        "proof_role": proof_role,
        "label": _repair_display_text(str(label)) if label is not None else None,
        "customers": int(_safe_float(customers) or 0) if customers is not None else None,
        "tribe_reach_pct": _safe_float(tribe_reach_pct),
        "lift_vs_rest": _safe_float(lift_vs_rest),
        "q_value": _safe_float(q_value),
        "profiling_readiness": readiness_row.get("profiling_readiness"),
        "actionability_proof_source": actionability.get("actionability_proof_source"),
        "actionability_proof": actionability.get("actionability_proof"),
        "product_ranking_basis": PRODUCT_RANKING_BASIS if evidence_type == "lifted_product" else None,
        "evidence": _repair_display_text(str(evidence)) if evidence is not None else None,
    }


def _product_evidence_text(row: dict[str, Any], limit: int = 5) -> str:
    products = _top_card_products(row, limit=limit)
    if not products:
        return "n/a"
    return "; ".join(
        f"{item['product']} ({item['sector'] or 'n/a'}, {item['lift_vs_rest']:.2f}x vs rest, {item['reach_pct']:.1f}% reach)"
        for item in products
    )


def _copurchase_text(row: dict[str, Any], limit: int = 3) -> str:
    pairs = _parse_json(row.get("copurchase_pairs"), [])
    if not pairs:
        return "No strong co-purchase pair passed the reporting threshold."
    parts = []
    for pair in pairs[:limit]:
        parts.append(
            f"{pair.get('product_a')} + {pair.get('product_b')} ({_fmt_number(pair.get('lift_vs_rest'))}x vs rest)"
        )
    return "; ".join(parts)


def _temporal_text(row: dict[str, Any]) -> str:
    temporal = _parse_json(row.get("temporal_pattern"), {})
    if not temporal:
        return "Temporal pattern unavailable."
    day = temporal.get("dominant_shopping_day") or "n/a"
    time = temporal.get("dominant_shopping_time") or "n/a"
    weekend = _fmt_number(temporal.get("weekend_vs_weekday_ratio"))
    return f"Dominant day: {day}; dominant time: {time}; weekend/weekday ratio vs rest: {weekend}x."


def _loyalty_text(row: dict[str, Any]) -> str:
    loyalty = _parse_json(row.get("loyalty_profile"), {})
    if not loyalty:
        return "Loyalty profile unavailable."
    return (
        f"Tenure {float(loyalty.get('mean_tenure_days') or 0):.0f} days "
        f"({_fmt_number(loyalty.get('tenure_vs_rest_ratio'))}x vs rest); "
        f"recency {float(loyalty.get('mean_recency_days') or 0):.0f} days; "
        f"trend {loyalty.get('visit_trend', 'unknown')}."
    )


def _behavior_ratio_summary(row: dict[str, Any], profiles: pl.DataFrame, limit: int = 4) -> str:
    del profiles
    ratios = _parse_json(row.get("behavior_ratio_vs_rest"), {})
    if not ratios:
        return "n/a"
    labels = {
        "avg_ticket_count": "Tickets",
        "avg_total_spend": "Total spend",
        "avg_basket_value": "Basket value",
        "avg_promo_share": "Promo share",
        "avg_unique_products": "Product variety",
        "avg_recency_days": "Recency days",
        "avg_frequency_per_30d": "Visit frequency",
    }
    items = sorted(ratios.items(), key=lambda item: abs(float(item[1]) - 1.0), reverse=True)
    return "; ".join(f"{labels.get(key, key)} {float(value):.2f}x vs rest" for key, value in items[:limit])


def _working_label(row: dict[str, Any]) -> str:
    products = row.get("top_products") or []
    if products:
        first = _repair_display_text(str(products[0]))
        words = [token for token in re.findall(r"[A-Za-z0-9]+", first.title()) if len(token) > 2]
        return " ".join(words[:3]) + " Buyers" if words else f"Tribe {row.get('tribe_id')}"
    return f"Tribe {row.get('tribe_id')}"


def _final_story_markdown(final_index: pl.DataFrame, review_candidates: pl.DataFrame) -> str:
    lines = [
        "# Stage 7 Tribe Profiles",
        "",
        "Stage 7 turns Stage 6.8 evidence into business-readable tribe profiles. It accepts Stage 6.6 readiness as final and reads only Stage 6.8 aggregate evidence plus per-tribe exports.",
        "In the final index, population_share_pct is normalized over promoted final tribes; assigned_population_share_pct preserves the full assigned-profile denominator before review tribes are held out.",
        "",
        "## Final Tribes",
        "",
    ]
    if final_index.is_empty():
        lines.append("No tribes were promoted into the final index.")
    else:
        lines.append(_markdown_table(final_index))
    lines.extend(["", "## Review Candidates", "", "Review candidates held out of the final tribe story remain auditable below.", ""])
    lines.append(_markdown_table(review_candidates) if not review_candidates.is_empty() else "No review candidates.")
    return "\n".join(lines) + "\n"


def _read_optional_table(path: str | Path | None) -> pl.DataFrame:
    if path is None:
        return pl.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pl.DataFrame()
    if file_path.suffix.lower() == ".parquet":
        return pl.read_parquet(file_path)
    return pl.read_csv(file_path)


def _row_by_tribe(frame: pl.DataFrame, tribe_id: int) -> dict[str, Any]:
    if frame.is_empty() or "tribe_id" not in frame.columns:
        return {}
    match = frame.filter(pl.col("tribe_id") == tribe_id)
    return match.row(0, named=True) if match.height else {}


def _significant_product_signal_count(row: dict[str, Any], *, threshold: float, q_threshold: float) -> int:
    total = 0
    for idx, lift_raw in enumerate(row.get("top_product_lifts_vs_rest") or row.get("top_product_lifts") or []):
        lift = _safe_float(lift_raw)
        q_value = _safe_float(_value_at(row.get("top_product_q_values"), idx))
        if lift is not None and lift >= threshold and q_value is not None and q_value <= q_threshold:
            total += 1
    return total


def _loyalty_cohort(loyalty: dict[str, Any]) -> str | None:
    tenure = _safe_float(loyalty.get("mean_tenure_days"))
    if tenure is None:
        return None
    if tenure < 30:
        return "new"
    if tenure > 90:
        return "loyal"
    return "established"


def _list_from_column(frame: pl.DataFrame, column: str, *, repair: bool = False) -> list[Any]:
    if frame.is_empty() or column not in frame.columns:
        return []
    values = frame[column].to_list()
    if repair:
        return [_repair_display_text("" if value is None else str(value)) for value in values]
    return values


def _rounded_list(frame: pl.DataFrame, column: str, digits: int) -> list[float | None]:
    values = _list_from_column(frame, column)
    return [round(value, digits) if (value := _safe_float(item)) is not None else None for item in values]


def _rounded_list_with_fallback(frame: pl.DataFrame, column: str, fallback_column: str, digits: int) -> list[float | None]:
    values = _list_from_column(frame, column)
    fallback_values = _list_from_column(frame, fallback_column)
    rounded: list[float | None] = []
    for idx, item in enumerate(values):
        value = _safe_float(item)
        if value is None:
            value = _safe_float(_value_at(fallback_values, idx))
        rounded.append(round(value, digits) if value is not None else None)
    return rounded


def _value_at(values: Any, idx: int) -> Any:
    if values is None or idx >= len(values):
        return None
    return values[idx]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_ratio(value: Any, baseline: Any) -> float | None:
    numeric = _safe_float(value)
    base = _safe_float(baseline)
    if numeric is None or base is None or base <= 0:
        return None
    return numeric / base


def _product_rank_score(lift_vs_rest: Any, customers: Any) -> float | None:
    lift = _safe_float(lift_vs_rest)
    customer_count = _safe_float(customers)
    if lift is None or customer_count is None or customer_count < 0:
        return None
    return lift * math.log(customer_count + 1.0)


def _to_int_or_none(value: Any) -> int | None:
    numeric = _safe_float(value)
    return int(numeric) if numeric is not None else None


def _to_int_or_zero(value: Any) -> int:
    numeric = _safe_float(value)
    return int(numeric) if numeric is not None else 0


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _parse_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _csv_safe_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _format_lift_items(names: list[Any] | None, lifts: list[Any] | None, counts: list[Any] | None = None, limit: int = 5) -> str:
    names = names or []
    lifts = lifts or []
    counts = counts or []
    parts = []
    for idx, name in enumerate(names[:limit]):
        lift = _fmt_number(_value_at(lifts, idx))
        count = _value_at(counts, idx)
        count_suffix = f", n={int(count)}" if count is not None else ""
        parts.append(f"{_repair_display_text(str(name))} ({lift}x{count_suffix})")
    return "; ".join(parts) if parts else "n/a"


def _fmt_number(value: Any) -> str:
    numeric = _safe_float(value)
    return "n/a" if numeric is None else f"{numeric:.2f}"


def _repair_display_text(text: str) -> str:
    repaired = "" if text is None else str(text)
    for _ in range(2):
        try:
            next_text = repaired.encode("latin1").decode("utf-8")
        except UnicodeError:
            break
        if next_text == repaired:
            break
        repaired = next_text
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", repaired)


def _markdown_table(frame: pl.DataFrame, max_rows: int = 200) -> str:
    if frame.is_empty():
        return "No rows."
    rows = frame.head(max_rows).iter_rows(named=True)
    columns = frame.columns
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown(_stringify_cell(row.get(col))) for col in columns) + " |")
    return "\n".join(lines)


def _html_table(frame: pl.DataFrame, *, title: str) -> str:
    body = _markdown_table(frame)
    return _simple_html(title, body)


def _simple_html(title: str, markdown_text: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{escape(title)}</title></head>
<body><pre>{escape(markdown_text)}</pre></body>
</html>
"""


def _stringify_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return "" if value is None else str(value)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _write_csv(frame: pl.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(output)
