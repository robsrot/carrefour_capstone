"""Organic Stage 7 tribe profiling from transaction-derived evidence only."""

from __future__ import annotations

import json
import math
import re
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
from src.tribe_namer import THEME_LABELS
from src.utils import collect_streaming, file_fingerprint, schema_names, should_use_cache, write_artifact_metadata


BEHAVIOR_OUTPUT_MAP = {
    "ticket_count": "avg_ticket_count",
    "total_spend": "avg_total_spend",
    "avg_basket_value": "avg_basket_value",
    "promo_share": "avg_promo_share",
    "unique_products": "avg_unique_products",
    "recency_days": "avg_recency_days",
    "frequency_per_30d": "avg_frequency_per_30d",
}


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
        log_event("Stage 7 profiling", "cache hit", cfg=cfg, assignments=assignments_file, path=output)
        return output

    with stage_timer("Stage 7 profiling", "building organic tribe profile", cfg=cfg, assignments=assignments_file):
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
            "Stage 7 profiling",
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
            "Stage 7 profiling",
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
        log_event("Stage 7 profiling", "product lift evidence ready", cfg=cfg, rows=product_lifts.height)

        sector_lifts = _sector_lift_table(line_items, assigned_keys, total_assigned_customers, cfg=cfg)
        log_event("Stage 7 profiling", "sector concentration evidence ready", cfg=cfg, rows=sector_lifts.height)

        behavior_frame = _behavior_frame(candidate_behavior_path, assigned_keys)
        behavior_summary = _behavior_summary(behavior_frame)
        behavior_ratios = _behavior_ratios_by_tribe(behavior_frame)

        reference_date = None
        if "fecha" in txn_columns:
            reference_date = collect_streaming(lf.select(pl.col("fecha").max().alias("reference_date")))[0, "reference_date"]

        log_event("Stage 7 profiling", "assembling profile rows", cfg=cfg, clusters=cluster_sizes.height)
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
                    "core_customers": int(cluster_row["n_customers"]),
                    "noise_customers": noise_customers,
                    "top_products": _list_from_column(products, "product_description", repair=True),
                    "top_product_ids": [_to_int_or_none(value) for value in _list_from_column(products, "idarticu")],
                    "top_product_lifts_vs_rest": _rounded_list_with_fallback(products, "lift_vs_rest", "lift", 3),
                    "top_product_lifts": _rounded_list(products, "lift", 3),
                    "top_product_q_values": _rounded_list(products, "lift_q_value", 6),
                    "top_product_customer_counts": [_to_int_or_zero(value) for value in _list_from_column(products, "cluster_customers")],
                    "top_product_reach_pct": _rounded_list(products, "reach_pct", 3),
                    "top_product_sectors": _list_from_column(products, "sector_description", repair=True),
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
        log_event("Stage 7 profiling", "wrote profile", cfg=cfg, tribes=profiles.height, path=output)
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
        "core_customers": pl.Int64,
        "noise_customers": pl.Int64,
        "top_products": pl.List(pl.Utf8),
        "top_product_ids": pl.List(pl.Int64),
        "top_product_lifts_vs_rest": pl.List(pl.Float64),
        "top_product_lifts": pl.List(pl.Float64),
        "top_product_q_values": pl.List(pl.Float64),
        "top_product_customer_counts": pl.List(pl.Int64),
        "top_product_reach_pct": pl.List(pl.Float64),
        "top_product_sectors": pl.List(pl.Utf8),
        "top_sectors": pl.List(pl.Utf8),
        "top_sector_lifts": pl.List(pl.Float64),
        "top_sector_line_counts": pl.List(pl.Int64),
        "avg_ticket_count": pl.Float64,
        "avg_total_spend": pl.Float64,
        "avg_basket_value": pl.Float64,
        "avg_promo_share": pl.Float64,
        "avg_unique_products": pl.Float64,
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
        stage6_status = (
            readiness_row.get("stage6_profile_readiness")
            or readiness_row.get("profile_readiness")
            or row.get("stage6_profile_readiness")
            or "missing"
        )
        signal_count = _significant_product_signal_count(row, threshold=threshold, q_threshold=q_threshold)
        issues: list[str] = []
        strong_stage6_statuses = {"strong", "ready_strong"}
        profileable_stage6_statuses = strong_stage6_statuses | {"usable", "ready", "pass"}
        if stage6_status not in profileable_stage6_statuses:
            issues.append(f"stage6_readiness={stage6_status}")
        if signal_count < min_signals:
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
                "mean_assignment_confidence": _safe_float(
                    readiness_row.get("mean_assignment_confidence") or row.get("mean_assignment_confidence")
                ),
                "p10_assignment_confidence": _safe_float(
                    readiness_row.get("p10_assignment_confidence") or row.get("p10_assignment_confidence")
                ),
                "jitter_label_recovery_accuracy": _safe_float(
                    readiness_row.get("jitter_label_recovery_accuracy")
                    or readiness_row.get("jitter_label_recovery_accuracy_mean")
                    or row.get("jitter_label_recovery_accuracy")
                ),
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
                "top_sector_evidence": _format_lift_items(row.get("top_sectors"), row.get("top_sector_lifts"), row.get("top_sector_line_counts")),
                "customer_behavior_over_under_index": _behavior_ratio_summary(row, profiles),
                "dominant_shopping_day": row.get("dominant_shopping_day"),
                "dominant_shopping_time": row.get("dominant_shopping_time"),
                "loyalty_cohort": row.get("loyalty_cohort"),
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
    result = pl.DataFrame(rows) if rows else _empty_customer_metric_tests()
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
        rows.append(
            {
                "story_order": idx,
                "tribe_id": tribe_id,
                "working_label": _working_label(row),
                "readiness": readiness_row.get("profiling_readiness") or "not checked",
                "size_read": f"{int(row.get('n_customers') or 0):,} customers ({float(row.get('population_share') or 0.0) * 100:.1f}% of assigned)",
                "distinctive_product_evidence": comparison_row.get("top_product_and_category_evidence") or _product_evidence_text(row),
                "sector_evidence": _format_lift_items(row.get("top_sectors"), row.get("top_sector_lifts"), row.get("top_sector_line_counts")),
                "mission_evidence": _copurchase_text(row),
                "behavior_evidence": _behavior_ratio_summary(row, profiles),
                "temporal_evidence": _temporal_text(row),
                "loyalty_evidence": _loyalty_text(row),
                "caveat": row.get("llm_confidence_note") or "This is an organic product-purchase profile, not a demographic persona.",
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
    table = pl.DataFrame(rows) if rows else pl.DataFrame()
    if output_csv is not None:
        _write_csv(table, output_csv)
    return table


def write_stage7_final_handoff_pack(
    profile_path: str | Path,
    *,
    assignments_path: str | Path | None = None,
    cluster_readiness_path: str | Path | None = None,
    readiness_path: str | Path | None = None,
    behavior_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_cards: bool = True,
    max_card_products: int = 8,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    out_dir = Path(output_dir) if output_dir else cfg.artifacts / "stage7" / "final_handoff"
    support_dir = out_dir / "supporting_tables"
    card_dir = out_dir / "tribe_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    if write_cards:
        card_dir.mkdir(parents=True, exist_ok=True)

    readiness_output = out_dir / f"stage7_final_readiness_{cfg.mode}.csv"
    if readiness_path and Path(readiness_path).exists():
        _read_optional_table(readiness_path).write_csv(readiness_output)
    else:
        profile_readiness_evidence_table(
            profile_path,
            cluster_readiness_path=cluster_readiness_path,
            output_csv=readiness_output,
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
        readiness_path=readiness_output,
        output_csv=support_dir / f"stage7_final_tribe_comparison_{cfg.mode}.csv",
        output_md=support_dir / f"stage7_final_tribe_comparison_{cfg.mode}.md",
        output_html=support_dir / f"stage7_final_tribe_comparison_{cfg.mode}.html",
        cfg=cfg,
    )
    llm_evidence_csv = support_dir / f"stage7_llm_evidence_long_{cfg.mode}.csv"
    stage7_llm_evidence_table(profile_path, readiness_path=readiness_output, output_csv=llm_evidence_csv, cfg=cfg)

    metric_tests_path = support_dir / f"stage7_final_customer_metric_tests_{cfg.mode}.csv"
    if assignments_path is not None and behavior_path is not None and Path(behavior_path).exists():
        customer_metric_anova_table(assignments_path, behavior_path=behavior_path, output_csv=metric_tests_path, cfg=cfg)
    else:
        _empty_customer_metric_tests().write_csv(metric_tests_path)

    storyline_paths = write_stage7_storyline_artifacts(
        profile_path,
        readiness_path=readiness_output,
        comparison_path=comparison_paths["csv"],
        output_csv=support_dir / f"stage7_final_evidence_storyline_{cfg.mode}.csv",
        output_md=support_dir / f"stage7_final_evidence_storyline_{cfg.mode}.md",
        output_html=support_dir / f"stage7_final_evidence_storyline_{cfg.mode}.html",
        cfg=cfg,
    )
    card_paths = (
        write_stage7_tribe_card_pngs(
            profile_path,
            readiness_path=readiness_output,
            output_dir=card_dir,
            max_products=max_card_products,
            cfg=cfg,
        )
        if write_cards
        else {}
    )
    final_index = stage7_final_index_table(
        profile_path,
        readiness_path=readiness_output,
        comparison_path=comparison_paths["csv"],
        customer_metric_tests_path=metric_tests_path,
        card_paths=card_paths,
        cfg=cfg,
    )
    review_candidates = _review_candidates(readiness_output, final_index)

    index_csv = out_dir / f"stage7_final_index_{cfg.mode}.csv"
    review_candidates_csv = out_dir / f"stage7_review_candidates_{cfg.mode}.csv"
    story_md = out_dir / f"stage7_final_story_{cfg.mode}.md"
    story_html = out_dir / f"stage7_final_story_{cfg.mode}.html"
    manifest_json = out_dir / f"stage7_final_manifest_{cfg.mode}.json"
    final_index.write_csv(index_csv)
    review_candidates.write_csv(review_candidates_csv)
    story = _final_story_markdown(final_index, review_candidates)
    story_md.write_text(story, encoding="utf-8")
    story_html.write_text(_simple_html("Stage 7 Final Handoff", story), encoding="utf-8")
    manifest = {
        "stage": "7_final_handoff",
        "mode": cfg.mode,
        "memory_policy": "Final handoff uses aggregate organic profile evidence and does not rescan raw transaction lines.",
        "promotion_policy": {
            "readiness_statuses": sorted(_final_readiness_statuses(None, cfg)),
            "requires_actionability_proof": bool(cfg.get("profiling.final_handoff_require_actionability_proof", True)),
        },
        "primary_outputs": {
            "final_index_csv": str(index_csv),
            "final_story_markdown": str(story_md),
            "tribe_card_directory": str(card_dir) if write_cards else None,
        },
        "supporting_outputs": {
            "product_summary_long_csv": str(product_summary_paths["combined_csv"]),
            "comparison_csv": str(comparison_paths["csv"]),
            "customer_metric_tests_csv": str(metric_tests_path),
            "evidence_storyline_markdown": str(storyline_paths["markdown"]),
            "llm_evidence_long_csv": str(llm_evidence_csv),
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
        "readiness_csv": readiness_output,
        "review_candidates_csv": review_candidates_csv,
        "review_candidates": review_candidates,
        "product_summary_paths": product_summary_paths,
        "comparison_paths": comparison_paths,
        "customer_metric_tests_csv": metric_tests_path,
        "llm_evidence_csv": llm_evidence_csv,
        "storyline_paths": storyline_paths,
    }


def stage7_final_index_table(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    customer_metric_tests_path: str | Path | None = None,
    card_paths: dict[int, Path] | None = None,
    readiness_statuses: list[str] | None = None,
    cfg: PipelineConfig = CONFIG,
) -> pl.DataFrame:
    del customer_metric_tests_path
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    readiness = _read_optional_table(readiness_path)
    comparison = _read_optional_table(comparison_path)
    if comparison.is_empty():
        comparison = tribe_comparison_table(profile_path, readiness_path=readiness_path)
    rows: list[dict[str, Any]] = []
    final_records = _final_profile_records(
        profiles,
        readiness,
        readiness_statuses=readiness_statuses,
        cfg=cfg,
    )
    final_records = sorted(final_records, key=lambda item: int(item[0].get("n_customers") or 0), reverse=True)
    for order, (row, readiness_row, actionability) in enumerate(final_records, start=1):
        tribe_id = int(row["tribe_id"])
        comparison_row = _row_by_tribe(comparison, tribe_id)
        products = _top_card_products(row, limit=1)
        theme = _primary_theme_context(row, cfg=cfg)
        rows.append(
            {
                "story_order": order,
                "tribe_id": tribe_id,
                "tribe_name": _final_tribe_name(row),
                "name_source": row.get("suggested_tribe_name_source") or row.get("llm_working_label_source") or "product_evidence",
                "readiness": readiness_row.get("profiling_readiness") or "not checked",
                "customers": int(row.get("n_customers") or 0),
                "population_share_pct": round(float(row.get("population_share") or 0.0) * 100.0, 2),
                "core_customers": int(row.get("core_customers") or 0),
                "noise_customers": int(row.get("noise_customers") or 0),
                "primary_theme": theme["primary_theme"],
                "primary_theme_confidence": theme["primary_theme_confidence"],
                "theme_read": theme["theme_read"],
                "primary_theme_product_evidence": theme["primary_theme_product_evidence"],
                "actionability_proof_source": actionability.get("actionability_proof_source"),
                "actionability_proof": actionability.get("actionability_proof"),
                "top_product": products[0]["product"] if products else None,
                "top_product_reach_pct": products[0]["reach_pct"] if products else None,
                "top_product_lift": products[0]["lift_vs_rest"] if products else None,
                "distinctive_products": comparison_row.get("top_product_and_category_evidence") or _product_evidence_text(row),
                "shopping_mission": row.get("llm_shopping_mission") or _copurchase_text(row),
                "behavior_context": _behavior_ratio_summary(row, profiles),
                "spend_and_visit_context": _spend_and_visit_context(row),
                "dominant_shopping_day": row.get("dominant_shopping_day"),
                "dominant_shopping_time": row.get("dominant_shopping_time"),
                "loyalty_cohort": row.get("loyalty_cohort"),
                "card_png": str(card_paths.get(tribe_id)) if card_paths and tribe_id in card_paths else None,
                "caveat": row.get("llm_confidence_note") or "Evidence is based on purchase patterns only.",
            }
        )
    return pl.DataFrame(rows).sort("story_order") if rows else pl.DataFrame()


def write_stage7_tribe_card_pngs(
    profile_path: str | Path,
    *,
    readiness_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_products: int = 8,
    readiness_statuses: list[str] | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[int, Path]:
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    readiness = _read_optional_table(readiness_path)
    out_dir = Path(output_dir) if output_dir else cfg.figures / "stage7_tribe_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_card in out_dir.glob("tribe_*_card.png"):
        stale_card.unlink()
    paths: dict[int, Path] = {}
    for row, _, _ in _final_profile_records(
        profiles,
        readiness,
        readiness_statuses=readiness_statuses,
        cfg=cfg,
    ):
        tribe_id = int(row["tribe_id"])
        output = out_dir / f"tribe_{tribe_id:02d}_card.png"
        _write_stage7_tribe_card_png(row, output, max_products=max_products)
        paths[tribe_id] = output
    return paths


def _tribe_product_summary_frame(row: dict[str, Any], max_products: int, cfg: PipelineConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    q_threshold = float(cfg.get("profiling.significance_q_threshold", 0.05))
    for idx, product in enumerate((row.get("top_products") or [])[:max_products]):
        q_value = _value_at(row.get("top_product_q_values"), idx)
        lift_vs_rest = _value_at(row.get("top_product_lifts_vs_rest"), idx)
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
                "customers": _value_at(row.get("top_product_customer_counts"), idx),
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
            "statistical_result": pl.Utf8,
        }
    )


def _empty_customer_metric_tests() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "metric": pl.Utf8,
            "included_tribes": pl.Int64,
            "anova_p_value": pl.Float64,
            "anova_q_value": pl.Float64,
            "anova_effect_eta_squared": pl.Float64,
            "highest_mean_tribe_id": pl.Int64,
            "lowest_mean_tribe_id": pl.Int64,
            "statistical_result": pl.Utf8,
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
    p_value = max(0.0, min(1.0, 1.0 - eta))
    sorted_groups = grouped.sort("mean")
    return {
        "metric": metric,
        "included_tribes": grouped.height,
        "anova_p_value": p_value,
        "anova_q_value": p_value,
        "anova_effect_eta_squared": eta,
        "highest_mean_tribe_id": int(sorted_groups[-1, "tribe_id"]) if grouped.height else None,
        "lowest_mean_tribe_id": int(sorted_groups[0, "tribe_id"]) if grouped.height else None,
        "statistical_result": "significant" if p_value <= 0.05 else "directional",
    }


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
        "sectors": _format_lift_items(row.get("top_sectors"), row.get("top_sector_lifts"), row.get("top_sector_line_counts")),
        "copurchase": _parse_json(row.get("copurchase_pairs"), []),
        "behavior_ratios": _parse_json(row.get("behavior_ratio_vs_rest"), {}),
        "temporal_pattern": _parse_json(row.get("temporal_pattern"), {}),
        "loyalty_profile": _parse_json(row.get("loyalty_profile"), {}),
    }


def _review_candidates(readiness_path: Path, final_index: pl.DataFrame) -> pl.DataFrame:
    readiness = _read_optional_table(readiness_path)
    if readiness.is_empty() or "tribe_id" not in readiness.columns:
        return pl.DataFrame(schema={"tribe_id": pl.Int64, "profiling_readiness": pl.Utf8, "review_reason": pl.Utf8})
    final_ids = set(final_index["tribe_id"].to_list()) if not final_index.is_empty() and "tribe_id" in final_index.columns else set()
    rows = []
    for row in readiness.iter_rows(named=True):
        if int(row["tribe_id"]) in final_ids:
            continue
        rows.append(
            {
                "tribe_id": int(row["tribe_id"]),
                "profiling_readiness": row.get("profiling_readiness"),
                "review_reason": row.get("profiling_readiness_issues"),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"tribe_id": pl.Int64, "profiling_readiness": pl.Utf8, "review_reason": pl.Utf8})


def _write_stage7_tribe_card_png(row: dict[str, Any], output_path: Path, *, max_products: int) -> Path:
    import textwrap

    import matplotlib.pyplot as plt

    products = _top_card_products(row, limit=max_products)
    fig = plt.figure(figsize=(12, 7), facecolor="#f8fafc")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    title = f"T{int(row.get('tribe_id') or 0)}: {_working_label(row)}"
    ax.text(0.04, 0.92, title, fontsize=22, weight="bold", color="#111827")
    ax.text(
        0.04,
        0.86,
        f"{int(row.get('n_customers') or 0):,} customers | {float(row.get('population_share') or 0.0) * 100:.1f}% of assigned population",
        fontsize=12,
        color="#475467",
    )
    mission = row.get("llm_shopping_mission") or _copurchase_text(row)
    ax.text(0.04, 0.76, "\n".join(textwrap.wrap(str(mission), 105)), fontsize=12, color="#111827", va="top")
    y = 0.58
    ax.text(0.04, y, "Top product evidence", fontsize=14, weight="bold", color="#111827")
    for product in products:
        y -= 0.055
        line = (
            f"{product['rank']}. {product['product']} "
            f"({product['lift_vs_rest']:.2f}x vs rest, {product['reach_pct']:.1f}% reach, n={product['customers']})"
        )
        ax.text(0.06, y, "\n".join(textwrap.wrap(line, 100)), fontsize=10.5, color="#344054")
    ax.text(0.04, 0.14, _temporal_text(row), fontsize=10.5, color="#344054")
    ax.text(0.04, 0.09, _loyalty_text(row), fontsize=10.5, color="#344054")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _top_card_products(row: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for idx, product in enumerate((row.get("top_products") or [])[:limit]):
        rows.append(
            {
                "rank": idx + 1,
                "product": _repair_display_text(str(product)),
                "sector": _value_at(row.get("top_product_sectors"), idx),
                "lift_vs_rest": _safe_float(_value_at(row.get("top_product_lifts_vs_rest"), idx)) or 0.0,
                "reach_pct": _safe_float(_value_at(row.get("top_product_reach_pct"), idx)) or 0.0,
                "customers": int(_safe_float(_value_at(row.get("top_product_customer_counts"), idx)) or 0),
            }
        )
    return rows


def _final_readiness_statuses(readiness_statuses: list[str] | None, cfg: PipelineConfig) -> set[str]:
    configured = readiness_statuses
    if configured is None:
        configured = cfg.get("profiling.final_handoff_readiness_statuses", ["ready_strong"])
    if isinstance(configured, str):
        configured = [configured]
    values = [str(value).strip() for value in (configured or ["ready_strong"]) if str(value).strip()]
    return set(values or ["ready_strong"])


def _final_profile_records(
    profiles: pl.DataFrame,
    readiness: pl.DataFrame,
    *,
    readiness_statuses: list[str] | None = None,
    cfg: PipelineConfig = CONFIG,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    allowed_statuses = _final_readiness_statuses(readiness_statuses, cfg)
    require_actionability = bool(cfg.get("profiling.final_handoff_require_actionability_proof", True))
    records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row["tribe_id"])
        readiness_row = _row_by_tribe(readiness, tribe_id)
        status = str(readiness_row.get("profiling_readiness") or "").strip()
        if status not in allowed_statuses:
            continue
        actionability = _actionability_proof(row, cfg=cfg)
        if require_actionability and not actionability.get("actionability_proof_source"):
            continue
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


def _final_tribe_name(row: dict[str, Any]) -> str:
    raw = (
        row.get("llm_working_label")
        or row.get("suggested_tribe_name")
        or row.get("working_tribe_name")
        or _working_label(row)
    )
    name = _repair_display_text(str(raw)).strip()
    for suffix in (" Product Evidence Cluster", " Evidence Cluster", " Purchase Cluster", " Cluster"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
            break
    if name.endswith(" Buyers"):
        return name
    return f"{name} Buyers" if name else f"Tribe {row.get('tribe_id')} Buyers"


def _spend_and_visit_context(row: dict[str, Any]) -> str:
    parts = []
    visits = _safe_float(row.get("avg_ticket_count"))
    frequency = _safe_float(row.get("avg_frequency_per_30d"))
    basket_value = _safe_float(row.get("avg_basket_value") or row.get("avg_avg_basket_value"))
    spend = _safe_float(row.get("avg_total_spend"))
    promo = _safe_float(row.get("avg_promo_share"))
    if visits is not None:
        parts.append(f"{visits:.1f} avg visits")
    if frequency is not None:
        parts.append(f"{frequency:.1f} visits per 30d")
    if basket_value is not None:
        parts.append(f"{basket_value:.2f} avg basket value")
    if spend is not None:
        parts.append(f"{spend:.2f} avg total spend")
    if promo is not None:
        parts.append(f"{promo:.1%} promo share")
    return "; ".join(parts) if parts else "Spend and visit context unavailable."


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
    if row.get("llm_working_label"):
        return _repair_display_text(str(row["llm_working_label"]))
    products = row.get("top_products") or []
    if products:
        first = _repair_display_text(str(products[0]))
        words = [token for token in re.findall(r"[A-Za-z0-9]+", first.title()) if len(token) > 2]
        return " ".join(words[:3]) + " Buyers" if words else f"Tribe {row.get('tribe_id')}"
    return f"Tribe {row.get('tribe_id')}"


def _final_story_markdown(final_index: pl.DataFrame, review_candidates: pl.DataFrame) -> str:
    lines = [
        "# Stage 7 Final Handoff",
        "",
        "Stage 7 turns hard product-purchase tribes into an organic evidence story. Profiles are based on products, co-purchase missions, sector concentration, timing, and loyalty signals.",
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
