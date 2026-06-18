"""Qualitative validation for product embeddings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.product_themes import STRATEGIC_THEME_PATTERNS, normalize_product_text
from src.progress import log_event, stage_timer
from src.utils import (
    collect_streaming,
    deterministic_sample_indices,
    file_fingerprint,
    numeric_feature_columns,
    should_use_cache,
    write_artifact_metadata,
)


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_text(value: Any) -> str:
    return normalize_product_text(None if value is None else str(value))


def _category_patterns(cfg: PipelineConfig) -> dict[str, list[str]]:
    configured = cfg.get("embedding_validation.niche_categories", "auto")
    if configured is None or configured == "" or configured == "auto":
        return {category: list(patterns) for category, patterns in STRATEGIC_THEME_PATTERNS.items()}
    if not isinstance(configured, dict):
        return {}

    categories: dict[str, list[str]] = {}
    for category, patterns in configured.items():
        normalized_patterns = [str(pattern).strip() for pattern in patterns or []]
        normalized_patterns = [pattern for pattern in normalized_patterns if pattern]
        if normalized_patterns:
            categories[str(category)] = normalized_patterns
    return categories


def _category_pattern_metadata(cfg: PipelineConfig) -> dict[str, Any]:
    configured = cfg.get("embedding_validation.niche_categories", "auto")
    if configured is None or configured == "" or configured == "auto":
        return {
            "source": "auto_product_themes",
            "candidate_theme_count": len(STRATEGIC_THEME_PATTERNS),
            "auto_niche_categories": cfg.get("embedding_validation.auto_niche_categories", {}),
        }
    return {"source": "configured", "categories": _category_patterns(cfg)}


def _product_search_text(meta_lookup: dict[Any, dict[str, Any]], product_id: Any) -> str:
    return " ".join(
        part
        for part in [
            _normalized_text(_meta_value(meta_lookup, product_id, "product_description")),
            _normalized_text(_meta_value(meta_lookup, product_id, "sector_description")),
            _normalized_text(_meta_value(meta_lookup, product_id, "sector_id")),
        ]
        if part
    )


def _matches_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        try:
            if re.search(pattern, text):
                return True
        except re.error:
            if _normalized_text(pattern) in text:
                return True
    return False


def _has_group_expr(group_name: str) -> pl.Expr:
    return pl.col("sample_group").str.contains(rf"(^|;){re.escape(group_name)}(;|$)")


def _filter_group(report: pl.DataFrame, group_name: str) -> pl.DataFrame:
    if report.is_empty() or "sample_group" not in report.columns:
        return report
    return report.filter(_has_group_expr(group_name))


def _is_niche_focus_group(groups: set[str]) -> bool:
    return any(group in {"niche", "rare_frequency"} or group.startswith("category_") for group in groups)


def _selected_category_candidates(
    product_ids: Sequence[Any],
    meta_lookup: dict[Any, dict[str, Any]],
    cfg: PipelineConfig,
) -> dict[str, list[int]]:
    category_patterns = _category_patterns(cfg)
    if not category_patterns:
        return {}

    configured_categories = cfg.get("embedding_validation.niche_categories", "auto")
    auto_mode = configured_categories is None or configured_categories == "" or configured_categories == "auto"
    auto_cfg = cfg.get("embedding_validation.auto_niche_categories", {}) or {}
    min_products = int(auto_cfg.get("min_matched_products", 10 if auto_mode else 1))
    max_products = int(auto_cfg.get("max_matched_products", 5000))
    max_categories = int(auto_cfg.get("max_categories", 12))
    max_median_basket_penetration = float(auto_cfg.get("max_median_basket_penetration", 0.005))

    rows: list[dict[str, Any]] = []
    for category, patterns in category_patterns.items():
        candidates = [
            idx
            for idx, pid in enumerate(product_ids)
            if _matches_any_pattern(_product_search_text(meta_lookup, pid), patterns)
        ]
        if not candidates:
            continue
        basket_penetrations = [
            _safe_float(_meta_value(meta_lookup, product_ids[idx], "basket_penetration"), 1.0)
            for idx in candidates
        ]
        median_basket_penetration = float(np.median(basket_penetrations)) if basket_penetrations else 1.0
        rows.append(
            {
                "category": category,
                "candidates": candidates,
                "matched_products": len(candidates),
                "median_basket_penetration": median_basket_penetration,
            }
        )

    if auto_mode:
        rows = [
            row
            for row in rows
            if min_products <= int(row["matched_products"]) <= max_products
            and _safe_float(row["median_basket_penetration"], 1.0) <= max_median_basket_penetration
        ]
        rows = sorted(
            rows,
            key=lambda row: (
                _safe_float(row["median_basket_penetration"], 1.0),
                -int(row["matched_products"]),
                str(row["category"]),
            ),
        )[:max_categories]
    else:
        rows = [row for row in rows if int(row["matched_products"]) >= min_products]

    return {str(row["category"]): list(row["candidates"]) for row in rows}


def _product_metadata(transactions: pl.LazyFrame, cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    available = set(transactions.collect_schema().names())
    select_cols = ["idarticu", "ticket"]
    for col in ["cliente", "unidades", "desc_larga_articulo", "idsector", "desc_sector"]:
        if col in available:
            select_cols.append(col)

    base = transactions.select(select_cols)
    if "unidades" in available:
        base = base.with_columns(
            pl.when(pl.col("unidades").cast(pl.Float64) > 0)
            .then(pl.col("unidades").cast(pl.Float64))
            .otherwise(1.0)
            .alias("_units")
        )
    else:
        base = base.with_columns(pl.lit(1.0).alias("_units"))

    total_exprs = [
        pl.col("ticket").n_unique().alias("total_baskets"),
        pl.len().alias("total_lines"),
        pl.col("_units").sum().alias("total_units"),
    ]
    if "cliente" in available:
        total_exprs.append(pl.col("cliente").n_unique().alias("total_customers"))
    else:
        total_exprs.append(pl.lit(None).alias("total_customers"))
    totals = collect_streaming(base.select(total_exprs)).row(0, named=True)

    total_baskets = max(int(totals.get("total_baskets") or 0), 1)
    total_lines = max(int(totals.get("total_lines") or 0), 1)
    total_units = max(float(totals.get("total_units") or 0.0), 1.0)
    total_customers = int(totals.get("total_customers") or 0)

    agg_exprs = [
        pl.col("ticket").n_unique().alias("basket_count"),
        pl.len().alias("line_count"),
        pl.col("_units").sum().alias("units_sum"),
    ]
    if "cliente" in available:
        agg_exprs.append(pl.col("cliente").n_unique().alias("customer_count"))
    else:
        agg_exprs.append(pl.lit(None).alias("customer_count"))
    if "desc_larga_articulo" in available:
        agg_exprs.append(pl.col("desc_larga_articulo").drop_nulls().first().alias("product_description"))
    else:
        agg_exprs.append(pl.lit(None).cast(pl.Utf8).alias("product_description"))
    if "idsector" in available:
        agg_exprs.append(pl.col("idsector").drop_nulls().first().cast(pl.Utf8).alias("sector_id"))
    else:
        agg_exprs.append(pl.lit(None).cast(pl.Utf8).alias("sector_id"))
    if "desc_sector" in available:
        agg_exprs.append(pl.col("desc_sector").drop_nulls().first().alias("sector_description"))
    else:
        agg_exprs.append(pl.lit(None).cast(pl.Utf8).alias("sector_description"))

    customer_threshold = float(cfg.get("baskets.diagnostics.common_customer_penetration_threshold", 0.01))
    basket_threshold = float(cfg.get("baskets.diagnostics.common_basket_penetration_threshold", 0.005))
    line_threshold = float(cfg.get("baskets.diagnostics.common_line_share_threshold", 0.005))

    customer_penetration_expr = (
        pl.col("customer_count") / pl.lit(max(total_customers, 1))
        if total_customers > 0
        else pl.lit(0.0)
    )
    return collect_streaming(
        base.group_by("idarticu")
        .agg(agg_exprs)
        .with_columns(
            [
                customer_penetration_expr.alias("customer_penetration"),
                (pl.col("basket_count") / pl.lit(total_baskets)).alias("basket_penetration"),
                (pl.col("line_count") / pl.lit(total_lines)).alias("line_share"),
                (pl.col("units_sum") / pl.lit(total_units)).alias("unit_share"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("customer_penetration") >= customer_threshold)
                    | (pl.col("basket_penetration") >= basket_threshold)
                    | (pl.col("line_share") >= line_threshold)
                ).alias("common_product_candidate"),
                (
                    (0.45 * pl.col("customer_penetration"))
                    + (0.45 * pl.col("basket_penetration"))
                    + (0.10 * pl.col("line_share"))
                ).alias("commonness_score"),
            ]
        )
    )


def _metadata_lookup(meta: pl.DataFrame) -> dict[Any, dict[str, Any]]:
    return {_python_scalar(row["idarticu"]): row for row in meta.iter_rows(named=True)}


def _meta_value(meta_lookup: dict[Any, dict[str, Any]], product_id: Any, key: str, default: Any = None) -> Any:
    row = meta_lookup.get(_python_scalar(product_id))
    if row is None:
        return default
    value = row.get(key, default)
    return default if value is None else value


def _select_validation_indices(
    product_ids: Sequence[Any],
    meta_lookup: dict[Any, dict[str, Any]],
    sample_product_ids: Sequence[int] | None,
    cfg: PipelineConfig,
) -> tuple[list[int], dict[int, set[str]], dict[str, int]]:
    id_to_index = {_python_scalar(pid): idx for idx, pid in enumerate(product_ids)}
    index_groups: dict[int, set[str]] = {}
    category_match_counts: dict[str, int] = {}

    def add(index: int, group: str) -> None:
        index_groups.setdefault(index, set()).add(group)

    if sample_product_ids:
        for pid in sample_product_ids:
            key = _python_scalar(pid)
            if key in id_to_index:
                add(id_to_index[key], "requested")
        return list(index_groups), index_groups, category_match_counts

    random_indices = deterministic_sample_indices(
        len(product_ids),
        int(cfg.get("embedding_validation.sample_size", 25)),
        cfg.random_seed,
    ).tolist()
    for idx in random_indices:
        add(int(idx), "random")

    staple_n = int(cfg.get("embedding_validation.staple_sample_size", 8))
    if staple_n > 0:
        ranked = sorted(
            range(len(product_ids)),
            key=lambda idx: (
                bool(_meta_value(meta_lookup, product_ids[idx], "common_product_candidate", False)),
                _safe_float(_meta_value(meta_lookup, product_ids[idx], "commonness_score")),
                _safe_float(_meta_value(meta_lookup, product_ids[idx], "basket_penetration")),
            ),
            reverse=True,
        )
        for idx in ranked[:staple_n]:
            add(idx, "staple")

    common_n = int(cfg.get("embedding_validation.common_sample_size", staple_n))
    if common_n > 0:
        common_ranked = sorted(
            range(len(product_ids)),
            key=lambda idx: (
                _safe_float(_meta_value(meta_lookup, product_ids[idx], "basket_penetration")),
                _safe_float(_meta_value(meta_lookup, product_ids[idx], "customer_penetration")),
                _safe_float(_meta_value(meta_lookup, product_ids[idx], "line_share")),
            ),
            reverse=True,
        )
        for idx in common_ranked[:common_n]:
            add(idx, "common_frequency")

    niche_n = int(cfg.get("embedding_validation.niche_sample_size", 8))
    niche_max = float(cfg.get("embedding_validation.niche_basket_penetration_max", 0.001))
    if niche_n > 0:
        niche_candidates = [
            idx
            for idx, pid in enumerate(product_ids)
            if _safe_float(_meta_value(meta_lookup, pid, "basket_penetration"), 1.0) <= niche_max
        ]
        if not niche_candidates:
            niche_candidates = sorted(
                range(len(product_ids)),
                key=lambda idx: _safe_float(_meta_value(meta_lookup, product_ids[idx], "basket_penetration"), 1.0),
            )[:niche_n]
        else:
            niche_candidates = sorted(
                niche_candidates,
                key=lambda idx: (
                    _safe_float(_meta_value(meta_lookup, product_ids[idx], "basket_penetration"), 1.0),
                    -_safe_float(_meta_value(meta_lookup, product_ids[idx], "line_count")),
                ),
            )
        for idx in niche_candidates[:niche_n]:
            add(idx, "niche")

    rare_n = int(cfg.get("embedding_validation.rare_sample_size", niche_n))
    if rare_n > 0:
        rare_ranked = sorted(
            range(len(product_ids)),
            key=lambda idx: (
                _safe_float(_meta_value(meta_lookup, product_ids[idx], "basket_penetration"), 1.0),
                _safe_float(_meta_value(meta_lookup, product_ids[idx], "customer_penetration"), 1.0),
                -_safe_float(_meta_value(meta_lookup, product_ids[idx], "line_count")),
            ),
        )
        for idx in rare_ranked[:rare_n]:
            add(idx, "rare_frequency")

    category_sample_size = int(cfg.get("embedding_validation.niche_category_sample_size", 5))
    if category_sample_size > 0:
        selected_categories = _selected_category_candidates(product_ids, meta_lookup, cfg)
        for category, candidates in selected_categories.items():
            category_group = f"category_{category}"
            category_match_counts[category] = len(candidates)
            candidates = sorted(
                candidates,
                key=lambda idx: (
                    _safe_float(_meta_value(meta_lookup, product_ids[idx], "basket_penetration"), 1.0),
                    -_safe_float(_meta_value(meta_lookup, product_ids[idx], "line_count")),
                ),
            )
            for idx in candidates[:category_sample_size]:
                add(idx, category_group)

    return list(index_groups), index_groups, category_match_counts


def _top_neighbor_indices(
    normalized: np.ndarray,
    query_indices: np.ndarray,
    n_neighbors: int,
    chunk_size: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if normalized.shape[0] <= 1 or n_neighbors <= 0 or len(query_indices) == 0:
        return []

    k = min(n_neighbors, normalized.shape[0] - 1)
    chunks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for start in range(0, len(query_indices), chunk_size):
        batch = query_indices[start : start + chunk_size]
        scores = normalized[batch] @ normalized.T
        for row_pos, query_idx in enumerate(batch):
            scores[row_pos, int(query_idx)] = -np.inf
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        for row_pos, query_idx in enumerate(batch):
            chunks.append((int(query_idx), top[row_pos], top_scores[row_pos]))
    return chunks


def _hubness_audit(
    normalized: np.ndarray,
    product_ids: Sequence[Any],
    meta_lookup: dict[Any, dict[str, Any]],
    cfg: PipelineConfig,
) -> pl.DataFrame:
    n_products = len(product_ids)
    n_neighbors = min(int(cfg.get("embedding_validation.hubness_neighbors", 10)), max(n_products - 1, 0))
    if n_products <= 1 or n_neighbors <= 0:
        return pl.DataFrame()

    sample_size = cfg.get("embedding_validation.hubness_sample_size", None)
    if sample_size is None or int(sample_size) <= 0 or int(sample_size) >= n_products:
        query_indices = np.arange(n_products)
    else:
        query_indices = deterministic_sample_indices(n_products, int(sample_size), cfg.random_seed)

    chunk_size = max(int(cfg.get("embedding_validation.hubness_chunk_size", 256)), 1)
    inbound = np.zeros(n_products, dtype=np.int64)
    cross_sector = np.zeros(n_products, dtype=np.int64)
    similarity_sum = np.zeros(n_products, dtype=np.float64)
    source_sector_sets: list[set[str]] = [set() for _ in range(n_products)]

    for query_idx, neighbors, scores in _top_neighbor_indices(normalized, query_indices, n_neighbors, chunk_size):
        source_pid = product_ids[query_idx]
        source_sector = _meta_value(meta_lookup, source_pid, "sector_description")
        for neighbor_idx, score in zip(neighbors, scores):
            neighbor_idx = int(neighbor_idx)
            neighbor_pid = product_ids[neighbor_idx]
            neighbor_sector = _meta_value(meta_lookup, neighbor_pid, "sector_description")
            inbound[neighbor_idx] += 1
            similarity_sum[neighbor_idx] += float(score)
            if source_sector:
                source_sector_sets[neighbor_idx].add(str(source_sector))
            if source_sector and neighbor_sector and source_sector != neighbor_sector:
                cross_sector[neighbor_idx] += 1

    expected_count = max((len(query_indices) * n_neighbors) / max(n_products, 1), 1e-12)
    rows = []
    for idx, count in enumerate(inbound):
        if count <= 0:
            continue
        pid = product_ids[idx]
        sectors = sorted(source_sector_sets[idx])
        rows.append(
            {
                "idarticu": pid,
                "product_description": _meta_value(meta_lookup, pid, "product_description"),
                "sector_description": _meta_value(meta_lookup, pid, "sector_description"),
                "hub_neighbor_count": int(count),
                "hubness_lift": float(count / expected_count),
                "avg_inbound_similarity": float(similarity_sum[idx] / count),
                "cross_sector_neighbor_count": int(cross_sector[idx]),
                "cross_sector_neighbor_share": float(cross_sector[idx] / count),
                "distinct_source_sectors": len(sectors),
                "source_sector_examples": ", ".join(sectors[:6]),
                "common_product_candidate": bool(_meta_value(meta_lookup, pid, "common_product_candidate", False)),
                "customer_penetration": _safe_float(_meta_value(meta_lookup, pid, "customer_penetration")),
                "basket_penetration": _safe_float(_meta_value(meta_lookup, pid, "basket_penetration")),
                "line_share": _safe_float(_meta_value(meta_lookup, pid, "line_share")),
                "commonness_score": _safe_float(_meta_value(meta_lookup, pid, "commonness_score")),
                "hubness_query_products": int(len(query_indices)),
                "hubness_neighbors": int(n_neighbors),
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(
        ["hub_neighbor_count", "cross_sector_neighbor_count", "distinct_source_sectors"],
        descending=[True, True, True],
    )


def _nearest_neighbor_rows(
    normalized: np.ndarray,
    product_ids: Sequence[Any],
    indices: Sequence[int],
    index_groups: dict[int, set[str]],
    meta_lookup: dict[Any, dict[str, Any]],
    hubness_lookup: dict[Any, dict[str, Any]],
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    rows = []
    n_neighbors = min(int(cfg.get("embedding_validation.neighbors", 8)), max(len(product_ids) - 1, 0))
    common_neighbor_threshold = float(
        cfg.get(
            "embedding_validation.common_neighbor_basket_penetration_threshold",
            cfg.get("baskets.diagnostics.common_basket_penetration_threshold", 0.005),
        )
    )
    if n_neighbors <= 0:
        return rows

    generic_warning_threshold = float(
        cfg.get("embedding_validation.generic_neighbor_warning_share_threshold", 0.50)
    )
    for idx in indices:
        pid = product_ids[idx]
        scores = normalized @ normalized[idx]
        scores[idx] = -np.inf
        order = np.argsort(-scores)[:n_neighbors]
        product_sector = _meta_value(meta_lookup, pid, "sector_description")
        groups = index_groups.get(idx, {"random"})
        query_is_niche_focus = _is_niche_focus_group(groups)
        validation_categories = [
            group.removeprefix("category_") for group in sorted(groups) if group.startswith("category_")
        ]
        neighbor_records: list[dict[str, Any]] = []
        for rank, neighbor_idx in enumerate(order, start=1):
            neighbor_id = product_ids[int(neighbor_idx)]
            neighbor_sector = _meta_value(meta_lookup, neighbor_id, "sector_description")
            neighbor_common = bool(_meta_value(meta_lookup, neighbor_id, "common_product_candidate", False))
            neighbor_basket_penetration = _safe_float(
                _meta_value(meta_lookup, neighbor_id, "basket_penetration")
            )
            neighbor_is_common_filler = neighbor_common or neighbor_basket_penetration >= common_neighbor_threshold
            neighbor_hub = hubness_lookup.get(_python_scalar(neighbor_id), {})
            neighbor_records.append(
                {
                    "sample_group": ";".join(sorted(groups)),
                    "validation_categories": ";".join(validation_categories),
                    "product_id": pid,
                    "product_description": _meta_value(meta_lookup, pid, "product_description"),
                    "product_sector": product_sector,
                    "product_common_product_candidate": bool(
                        _meta_value(meta_lookup, pid, "common_product_candidate", False)
                    ),
                    "product_customer_penetration": _safe_float(
                        _meta_value(meta_lookup, pid, "customer_penetration")
                    ),
                    "product_basket_penetration": _safe_float(_meta_value(meta_lookup, pid, "basket_penetration")),
                    "product_commonness_score": _safe_float(_meta_value(meta_lookup, pid, "commonness_score")),
                    "neighbor_rank": rank,
                    "neighbor_id": neighbor_id,
                    "neighbor_description": _meta_value(meta_lookup, neighbor_id, "product_description"),
                    "neighbor_sector": neighbor_sector,
                    "neighbor_common_product_candidate": neighbor_common,
                    "neighbor_customer_penetration": _safe_float(
                        _meta_value(meta_lookup, neighbor_id, "customer_penetration")
                    ),
                    "neighbor_basket_penetration": neighbor_basket_penetration,
                    "neighbor_commonness_score": _safe_float(
                        _meta_value(meta_lookup, neighbor_id, "commonness_score")
                    ),
                    "neighbor_is_common_filler": bool(neighbor_is_common_filler),
                    "neighbor_is_generic_staple": bool(neighbor_is_common_filler),
                    "neighbor_hubness_lift": _safe_float(neighbor_hub.get("hubness_lift")),
                    "neighbor_hub_neighbor_count": int(neighbor_hub.get("hub_neighbor_count", 0) or 0),
                    "same_sector": bool(product_sector and neighbor_sector and product_sector == neighbor_sector),
                    "cosine_similarity": float(scores[int(neighbor_idx)]),
                }
            )
        if not neighbor_records:
            continue
        generic_neighbor_share = sum(
            1 for row in neighbor_records if row["neighbor_is_generic_staple"]
        ) / len(neighbor_records)
        generic_neighbor_warning = query_is_niche_focus and generic_neighbor_share >= generic_warning_threshold
        for row in neighbor_records:
            row["query_is_niche_focus"] = bool(query_is_niche_focus)
            row["query_generic_neighbor_share"] = float(generic_neighbor_share)
            row["generic_neighbor_warning"] = bool(generic_neighbor_warning)
            row["generic_neighbor_warning_threshold"] = float(generic_warning_threshold)
            rows.append(row)
    return rows


def _summary_metrics(report: pl.DataFrame, hubness: pl.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "sampled_products": report.select("product_id").n_unique() if report.height else 0,
        "neighbor_rows": report.height,
        "mean_similarity": None,
        "same_sector_share": None,
        "common_filler_neighbor_share": None,
        "niche_common_filler_neighbor_share": None,
        "niche_focus_products": 0,
        "generic_neighbor_warning_products": 0,
        "generic_neighbor_warning_product_share": None,
        "max_hub_neighbor_count": None,
        "max_hubness_lift": None,
        "top_hub_common_candidate_share": None,
    }
    if report.height:
        metrics["mean_similarity"] = report["cosine_similarity"].mean()
        metrics["same_sector_share"] = report["same_sector"].mean()
        metrics["common_filler_neighbor_share"] = report["neighbor_is_common_filler"].mean()
        niche = report.filter(pl.col("sample_group").str.contains("niche"))
        if niche.height:
            metrics["niche_common_filler_neighbor_share"] = niche["neighbor_is_common_filler"].mean()
        niche_focus = report.filter(pl.col("query_is_niche_focus"))
        if niche_focus.height:
            niche_focus_products = niche_focus.select("product_id").n_unique()
            warning_products = (
                niche_focus.filter(pl.col("generic_neighbor_warning")).select("product_id").n_unique()
            )
            metrics["niche_focus_products"] = niche_focus_products
            metrics["generic_neighbor_warning_products"] = warning_products
            metrics["generic_neighbor_warning_product_share"] = warning_products / max(niche_focus_products, 1)
    if hubness.height:
        metrics["max_hub_neighbor_count"] = hubness["hub_neighbor_count"].max()
        metrics["max_hubness_lift"] = hubness["hubness_lift"].max()
        top_n = min(25, hubness.height)
        metrics["top_hub_common_candidate_share"] = hubness.head(top_n)["common_product_candidate"].mean()
    return metrics


def _sample_group_product_count(report: pl.DataFrame, group_name: str) -> int:
    subset = _filter_group(report, group_name)
    return subset.select("product_id").n_unique() if subset.height else 0


def _group_quality_row(report: pl.DataFrame, label: str, group_name: str) -> dict[str, Any]:
    subset = _filter_group(report, group_name)
    if subset.is_empty():
        return {
            "group": label,
            "sampled_products": 0,
            "neighbor_rows": 0,
            "mean_cosine": None,
            "same_sector_share": None,
            "generic_neighbor_share": None,
            "warning_products": 0,
        }
    return {
        "group": label,
        "sampled_products": subset.select("product_id").n_unique(),
        "neighbor_rows": subset.height,
        "mean_cosine": subset["cosine_similarity"].mean(),
        "same_sector_share": subset["same_sector"].mean(),
        "generic_neighbor_share": subset["neighbor_is_generic_staple"].mean(),
        "warning_products": subset.filter(pl.col("generic_neighbor_warning")).select("product_id").n_unique(),
    }


def _frequency_quality_table(report: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        [
            _group_quality_row(report, "Common products", "common_frequency"),
            _group_quality_row(report, "Rare products", "rare_frequency"),
        ]
    )


def _category_quality_table(
    report: pl.DataFrame,
    category_match_counts: dict[str, int],
    cfg: PipelineConfig,
) -> pl.DataFrame:
    rows = []
    for category in category_match_counts:
        group_name = f"category_{category}"
        row = _group_quality_row(report, category.replace("_", " ").title(), group_name)
        row["matched_products_in_metadata"] = int(category_match_counts.get(category, 0))
        rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _generic_warning_table(report: pl.DataFrame, max_rows: int = 20) -> pl.DataFrame:
    if report.is_empty() or "generic_neighbor_warning" not in report.columns:
        return pl.DataFrame()
    warning_rows = []
    warnings = report.filter(pl.col("generic_neighbor_warning")).sort(
        ["query_generic_neighbor_share", "product_basket_penetration"],
        descending=[True, False],
    )
    for pid in warnings.select("product_id").unique(maintain_order=True)["product_id"].to_list()[:max_rows]:
        block = warnings.filter(pl.col("product_id") == pid)
        first = block.row(0, named=True)
        warning_rows.append(
            {
                "product_id": pid,
                "product_description": first.get("product_description"),
                "product_sector": first.get("product_sector"),
                "sample_group": first.get("sample_group"),
                "validation_categories": first.get("validation_categories"),
                "basket_penetration": first.get("product_basket_penetration"),
                "generic_neighbor_share": first.get("query_generic_neighbor_share"),
                "generic_neighbor_count": int(block["neighbor_is_generic_staple"].sum()),
                "review_reason": "Niche-focus product neighbors are mostly generic staples.",
            }
        )
    return pl.DataFrame(warning_rows) if warning_rows else pl.DataFrame()


def _first_metric(metrics: pl.DataFrame, group_name: str, metric: str, default: float = 0.0) -> float:
    if metrics.is_empty() or metric not in metrics.columns:
        return default
    row = metrics.filter(pl.col("group") == group_name)
    if row.is_empty():
        return default
    return _safe_float(row.row(0, named=True).get(metric), default)


def _quality_guardrail_status(
    report: pl.DataFrame,
    hubness: pl.DataFrame,
    metrics: dict[str, Any],
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Compute non-blocking Stage 3 guardrail status and upstream levers."""

    frequency_quality = _frequency_quality_table(report)
    common_same_sector = _first_metric(frequency_quality, "Common products", "same_sector_share")
    rare_same_sector = _first_metric(frequency_quality, "Rare products", "same_sector_share")
    common_rare_gap = max(common_same_sector - rare_same_sector, 0.0)
    niche_warning_share = _safe_float(metrics.get("generic_neighbor_warning_product_share"))
    top_hubness_lift = _safe_float(metrics.get("max_hubness_lift"))

    thresholds = {
        "max_generic_warning_product_share": float(
            cfg.get("embedding_validation.guardrails.max_generic_warning_product_share", 0.10)
        ),
        "min_rare_same_sector_share": float(
            cfg.get("embedding_validation.guardrails.min_rare_same_sector_share", 0.35)
        ),
        "max_common_rare_same_sector_gap": float(
            cfg.get("embedding_validation.guardrails.max_common_rare_same_sector_gap", 0.35)
        ),
        "max_top_hub_cross_sector_share": float(
            cfg.get("embedding_validation.guardrails.max_top_hub_cross_sector_share", 0.80)
        ),
        "max_hubness_lift": float(cfg.get("embedding_validation.guardrails.max_hubness_lift", 10.0)),
        "min_cross_sector_hubness_lift": float(
            cfg.get("embedding_validation.guardrails.min_cross_sector_hubness_lift", 3.0)
        ),
        "min_cross_sector_source_sectors": int(
            cfg.get("embedding_validation.guardrails.min_cross_sector_source_sectors", 3)
        ),
    }
    cross_sector_hub_candidates = (
        hubness.filter(
            (pl.col("hubness_lift") >= thresholds["min_cross_sector_hubness_lift"])
            & (pl.col("distinct_source_sectors") >= thresholds["min_cross_sector_source_sectors"])
        )
        if hubness.height
        else pl.DataFrame()
    )
    max_hub_cross_sector_share = (
        _safe_float(cross_sector_hub_candidates["cross_sector_neighbor_share"].max())
        if cross_sector_hub_candidates.height
        else 0.0
    )

    issues: list[str] = []
    upstream_actions: list[str] = []

    if niche_warning_share > thresholds["max_generic_warning_product_share"]:
        issues.append(
            "Niche-focus products have too many generic-staple neighbors "
            f"({niche_warning_share:.1%} of niche-focus products flagged)."
        )
        upstream_actions.append(
            "Rerun Stage 1/2 with stronger common-product dampening: lower "
            "`baskets.downsampling.target_customer_penetration`, lower "
            "`baskets.downsampling.min_keep_probability`, or lower `word2vec.sample`."
        )

    if rare_same_sector and rare_same_sector < thresholds["min_rare_same_sector_share"]:
        issues.append(
            f"Rare-product same-sector neighbor share is low ({rare_same_sector:.1%})."
        )
        upstream_actions.append(
            "Rerun Stage 2 sandbox trials with a tighter local context, for example lower "
            "`word2vec.window` or lower `word2vec.max_tokens_per_basket`, then rebuild from Stage 2."
        )

    if common_rare_gap > thresholds["max_common_rare_same_sector_gap"]:
        issues.append(
            "Rare products are materially less coherent than common products "
            f"(same-sector gap {common_rare_gap:.1%})."
        )
        upstream_actions.append(
            "Compare common-vs-rare neighbor quality after Stage 1 anti-blur changes; if the gap remains, "
            "test Item2Vec parameter variants in the sandbox before rebuilding customer embeddings."
        )

    if max_hub_cross_sector_share > thresholds["max_top_hub_cross_sector_share"]:
        issues.append(
            "Some high-inbound embedding hubs attract neighbors across several sectors "
            f"(max cross-sector inbound share {max_hub_cross_sector_share:.1%})."
        )
        upstream_actions.append(
            "Inspect the top hubness rows. If the hubs are sparse non-staples, raise `word2vec.min_count`; "
            "if they are true common staples, tune Stage 1 common-product exposure; if they are coherent "
            "products split by Carrefour sector taxonomy, keep as a watch item."
        )

    if top_hubness_lift > thresholds["max_hubness_lift"]:
        issues.append(f"Top hubness lift is high ({top_hubness_lift:.2f}).")
        upstream_actions.append(
            "Reduce broad co-occurrence blur upstream and rerun Stage 3; high hubness usually means a few "
            "products are absorbing too much neighborhood mass."
        )

    if not issues:
        status = "pass"
        summary = "Stage 3 guardrails did not find segment-blocking product-neighbor issues."
        upstream_actions.append("Continue to Stage 4; keep Stage 3 diagnostics as evidence for model handoff.")
    elif any(
        text.startswith(("Niche-focus", "Rare products are materially", "Top hubness"))
        for text in issues
    ):
        status = "action_needed"
        summary = "Stage 3 found product-neighbor issues that should be addressed in Stage 1/2 before interpretation."
    else:
        status = "watch"
        summary = "Stage 3 found watch items; proceed only if downstream clusters remain coherent."

    return {
        "status": status,
        "summary": summary,
        "issues": issues,
        "upstream_actions": list(dict.fromkeys(upstream_actions)),
        "metrics": {
            "common_same_sector_share": common_same_sector,
            "rare_same_sector_share": rare_same_sector,
            "common_rare_same_sector_gap": common_rare_gap,
            "generic_neighbor_warning_product_share": niche_warning_share,
            "max_hub_cross_sector_share": max_hub_cross_sector_share,
            "max_hubness_lift": top_hubness_lift,
        },
        "thresholds": thresholds,
    }


def _markdown_table(df: pl.DataFrame, max_rows: int = 15) -> str:
    if df.is_empty():
        return "_No rows._"
    df = df.head(max_rows)
    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for row in df.iter_rows(named=True):
        cells = []
        for col in df.columns:
            value = row.get(col)
            if isinstance(value, float):
                text = f"{value:.4f}"
            elif value is None:
                text = ""
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def _product_neighbor_section(
    report: pl.DataFrame,
    group_name: str,
    max_products: int = 8,
    title: str | None = None,
) -> list[str]:
    section_title = title or f"{group_name.title()} Product Neighbor Checks"
    subset = _filter_group(report, group_name)
    if subset.is_empty():
        return [f"## {section_title}", "", "_No rows._", ""]

    lines = [f"## {section_title}", ""]
    product_ids = subset.select("product_id").unique(maintain_order=True)["product_id"].to_list()[:max_products]
    for pid in product_ids:
        block = subset.filter(pl.col("product_id") == pid).sort("neighbor_rank")
        first = block.row(0, named=True)
        lines.append(f"### {pid} - {first.get('product_description') or 'Unknown product'}")
        lines.append(
            "Sector: "
            f"{first.get('product_sector') or 'Unknown'}; "
            f"basket penetration={_safe_float(first.get('product_basket_penetration')):.4f}; "
            f"common={bool(first.get('product_common_product_candidate'))}"
        )
        if first.get("validation_categories"):
            lines.append(f"Validation categories: `{first.get('validation_categories')}`")
        if first.get("generic_neighbor_warning"):
            lines.append(
                "Generic-neighbor warning: "
                f"{_safe_float(first.get('query_generic_neighbor_share')):.2f} of neighbors are generic staples."
            )
        for row in block.iter_rows(named=True):
            filler = " common-filler" if row.get("neighbor_is_common_filler") else ""
            lines.append(
                f"- {row['neighbor_rank']}. {row['neighbor_id']} - "
                f"{row.get('neighbor_description') or 'Unknown'} "
                f"({row.get('neighbor_sector') or 'Unknown'}), "
                f"cosine={row['cosine_similarity']:.3f}, "
                f"basket_pen={_safe_float(row.get('neighbor_basket_penetration')):.4f}, "
                f"hub_lift={_safe_float(row.get('neighbor_hubness_lift')):.2f}{filler}"
            )
        lines.append("")
    return lines


def _validation_markdown(
    report: pl.DataFrame,
    hubness: pl.DataFrame,
    metrics: dict[str, Any],
    hubness_csv_path: Path,
    category_match_counts: dict[str, int],
    cfg: PipelineConfig,
) -> str:
    top_hubs = (
        hubness.select(
            [
                "idarticu",
                "product_description",
                "sector_description",
                "hub_neighbor_count",
                "hubness_lift",
                "cross_sector_neighbor_share",
                "distinct_source_sectors",
                "common_product_candidate",
                "basket_penetration",
            ]
        )
        if hubness.height
        else pl.DataFrame()
    )
    top_cross_sector_hubs = (
        hubness.sort(
            ["cross_sector_neighbor_count", "distinct_source_sectors", "hub_neighbor_count"],
            descending=[True, True, True],
        ).select(
            [
                "idarticu",
                "product_description",
                "sector_description",
                "cross_sector_neighbor_count",
                "cross_sector_neighbor_share",
                "distinct_source_sectors",
                "source_sector_examples",
                "hubness_lift",
            ]
        )
        if hubness.height
        else pl.DataFrame()
    )
    frequency_quality = _frequency_quality_table(report)
    category_quality = _category_quality_table(report, category_match_counts, cfg)
    generic_warnings = _generic_warning_table(report)
    guardrail_status = _quality_guardrail_status(report, hubness, metrics, cfg)
    category_report_max_products = int(cfg.get("embedding_validation.category_report_max_products", 4))
    issue_lines = guardrail_status["issues"] or ["No segment-blocking product-neighbor issues detected."]
    action_lines = guardrail_status["upstream_actions"]

    lines = [
        "# Product Embedding Validation",
        "",
        "Nearest neighbors should share basket missions, sectors, substitutes, or complements without being dominated by universal basket fillers.",
        "",
        "## Stage 2 Anti-Blur Settings",
        "",
        f"- Basket construction strategy: `{cfg.get('baskets.construction_strategy', 'baseline')}`",
        f"- Item2Vec window: `{cfg.get('word2vec.window')}`",
        f"- Full-basket context: `{bool(cfg.get('word2vec.full_basket_context', False))}`",
        f"- Minimum basket tokens used for Item2Vec: `{cfg.get('word2vec.min_tokens_per_basket', 2)}`",
        f"- Maximum basket tokens used for Item2Vec: `{cfg.get('word2vec.max_tokens_per_basket', None)}`",
        f"- Gensim high-frequency subsampling threshold: `{cfg.get('word2vec.sample')}`",
        f"- Min product count: `{cfg.get('word2vec.min_count')}`",
        f"- Hubness audit CSV: `{hubness_csv_path}`",
        "",
        "## Validation Summary",
        "",
        f"- Sampled products: `{int(metrics['sampled_products'] or 0):,}`",
        f"- Neighbor rows reviewed: `{int(metrics['neighbor_rows'] or 0):,}`",
        f"- Mean sampled-neighbor cosine: `{_safe_float(metrics['mean_similarity']):.3f}`",
        f"- Same-sector neighbor share: `{_safe_float(metrics['same_sector_share']):.3f}`",
        f"- Common-filler neighbor share: `{_safe_float(metrics['common_filler_neighbor_share']):.3f}`",
        f"- Niche common-filler neighbor share: `{_safe_float(metrics['niche_common_filler_neighbor_share']):.3f}`",
        f"- Niche-focus products reviewed: `{int(metrics['niche_focus_products'] or 0):,}`",
        f"- Generic-neighbor warning products: `{int(metrics['generic_neighbor_warning_products'] or 0):,}`",
        f"- Generic-neighbor warning product share: `{_safe_float(metrics['generic_neighbor_warning_product_share']):.3f}`",
        f"- Max hub neighbor count: `{int(metrics['max_hub_neighbor_count'] or 0):,}`",
        f"- Max hubness lift: `{_safe_float(metrics['max_hubness_lift']):.2f}`",
        f"- Share of top hubs that are common-product candidates: `{_safe_float(metrics['top_hub_common_candidate_share']):.3f}`",
        "",
        "## Dynamic Guardrail Status",
        "",
        f"- Status: `{guardrail_status['status']}`",
        f"- Summary: {guardrail_status['summary']}",
        "- Issues:",
        *[f"  - {issue}" for issue in issue_lines],
        "- Upstream actions:",
        *[f"  - {action}" for action in action_lines],
        "",
        "## Common vs Rare Neighbor Quality",
        "",
        _markdown_table(frequency_quality),
        "",
        "## Data-Selected Niche Theme Coverage and Quality",
        "",
        _markdown_table(category_quality, max_rows=50),
        "",
        "## Generic Neighbor Warnings",
        "",
        _markdown_table(generic_warnings, max_rows=25),
        "",
        "## Top Embedding Hubs",
        "",
        _markdown_table(top_hubs),
        "",
        "## Top Cross-Sector Hubs",
        "",
        _markdown_table(top_cross_sector_hubs),
        "",
    ]
    lines.extend(_product_neighbor_section(report, "common_frequency", title="Common Product Neighbor Checks"))
    lines.extend(_product_neighbor_section(report, "rare_frequency", title="Rare Product Neighbor Checks"))
    lines.extend(_product_neighbor_section(report, "staple", title="Staple Product Neighbor Checks"))
    lines.extend(_product_neighbor_section(report, "niche", title="Niche Product Neighbor Checks"))
    for category in category_match_counts:
        lines.extend(
            _product_neighbor_section(
                report,
                f"category_{category}",
                max_products=category_report_max_products,
                title=f"{category.replace('_', ' ').title()} Niche Theme Neighbor Checks",
            )
        )
    lines.extend(_product_neighbor_section(report, "random", title="Random Product Neighbor Checks"))
    lines.extend(
        [
            "## How To Use This",
            "",
            "- If top hubs are common products or appear across many source sectors, reduce common-product exposure before or during Item2Vec training.",
            "- If niche products mostly point to common fillers, prefer smaller windows, stronger subsampling, or stricter Stage 1 downsampling.",
            "- If rare products have much lower sector coherence or meaningfully worse nearest-neighbor quality than common products, rerun the relevant Stage 1/2 upstream changes before interpreting customer clusters.",
            "- If an expected niche theme is missing, confirm whether product text uses different vocabulary or adjust the shared product-theme taxonomy.",
            "- Promote a Stage 2 setting only when guardrails improve and Stage 4/6/8 customer clusters remain coherent and stable.",
            "",
        ]
    )
    return "\n".join(lines)


def product_embedding_guardrail_status(
    validation_csv: str | Path,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Return the dynamic, non-blocking Stage 3 product-neighborhood guardrail status."""

    report = pl.read_csv(validation_csv)
    hubness_path = _embedding_validation_output_dir(cfg) / str(
        cfg.get("embedding_validation.hubness_output_csv", "embedding_hubness.csv")
    )
    hubness = pl.read_csv(hubness_path) if hubness_path.exists() else pl.DataFrame()
    status = _quality_guardrail_status(report, hubness, _summary_metrics(report, hubness), cfg)
    status["validation_csv"] = str(validation_csv)
    status["hubness_csv"] = str(hubness_path)
    return status


def _embedding_validation_output_dir(cfg: PipelineConfig) -> Path:
    return cfg.artifacts / str(cfg.get("embedding_validation.output_dir", "stage3"))


def validate_product_embeddings(
    embeddings_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    sample_product_ids: Sequence[int] | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path | None]:
    """Create nearest-neighbor and hubness reports for product embeddings."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output_dir = _embedding_validation_output_dir(cfg)
    write_markdown_report = bool(cfg.get("embedding_validation.write_markdown_report", True))
    csv_path = Path(output_csv) if output_csv else output_dir / str(cfg.get("embedding_validation.output_csv"))
    md_path = Path(output_md) if output_md else output_dir / str(cfg.get("embedding_validation.output_md"))
    hubness_csv_path = output_dir / str(
        cfg.get("embedding_validation.hubness_output_csv", "embedding_hubness.csv")
    )
    cache_metadata = {
        "stage": "embedding_validation",
        "mode": cfg.mode,
        "product_embeddings": file_fingerprint(embeddings_path),
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "sample_product_ids": [int(pid) for pid in sample_product_ids] if sample_product_ids else None,
        "sample_size": int(cfg.get("embedding_validation.sample_size", 25)),
        "neighbors": int(cfg.get("embedding_validation.neighbors", 8)),
        "staple_sample_size": int(cfg.get("embedding_validation.staple_sample_size", 8)),
        "niche_sample_size": int(cfg.get("embedding_validation.niche_sample_size", 8)),
        "common_sample_size": int(
            cfg.get("embedding_validation.common_sample_size", cfg.get("embedding_validation.staple_sample_size", 8))
        ),
        "rare_sample_size": int(
            cfg.get("embedding_validation.rare_sample_size", cfg.get("embedding_validation.niche_sample_size", 8))
        ),
        "niche_category_sample_size": int(cfg.get("embedding_validation.niche_category_sample_size", 5)),
        "niche_categories": _category_pattern_metadata(cfg),
        "generic_neighbor_warning_share_threshold": float(
            cfg.get("embedding_validation.generic_neighbor_warning_share_threshold", 0.50)
        ),
        "guardrails": cfg.get("embedding_validation.guardrails", {}),
        "write_extract_figures": bool(cfg.get("embedding_validation.write_extract_figures", False)),
        "niche_basket_penetration_max": float(
            cfg.get("embedding_validation.niche_basket_penetration_max", 0.001)
        ),
        "common_neighbor_basket_penetration_threshold": float(
            cfg.get(
                "embedding_validation.common_neighbor_basket_penetration_threshold",
                cfg.get("baskets.diagnostics.common_basket_penetration_threshold", 0.005),
            )
        ),
        "hubness_neighbors": int(cfg.get("embedding_validation.hubness_neighbors", 10)),
        "hubness_sample_size": cfg.get("embedding_validation.hubness_sample_size", None),
        "hubness_chunk_size": int(cfg.get("embedding_validation.hubness_chunk_size", 256)),
        "random_seed": cfg.random_seed,
    }
    cache_outputs = [csv_path, hubness_csv_path]
    if write_markdown_report:
        cache_outputs.append(md_path)
    if all(
        should_use_cache(path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        for path in cache_outputs
    ):
        log_event(
            "Stage 3 embedding validation",
            "cache hit",
            cfg=cfg,
            csv=csv_path,
            markdown=md_path if write_markdown_report else None,
            hubness=hubness_csv_path,
        )
        return csv_path, md_path if write_markdown_report else None

    with stage_timer(
        "Stage 3 embedding validation",
        "building nearest-neighbor and hubness diagnostics",
        cfg=cfg,
        output=md_path if write_markdown_report else csv_path,
    ):
        emb = pl.read_parquet(embeddings_path).sort("idarticu")
        feature_cols = numeric_feature_columns(emb, exclude=("idarticu",))
        matrix = emb.select(feature_cols).to_numpy().astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = matrix / np.maximum(norms, 1e-12)
        product_ids = [_python_scalar(pid) for pid in emb["idarticu"].to_list()]

        lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
        meta_lookup = _metadata_lookup(_product_metadata(lf, cfg=cfg))

        indices, index_groups, category_match_counts = _select_validation_indices(
            product_ids,
            meta_lookup,
            sample_product_ids,
            cfg,
        )
        hubness = _hubness_audit(normalized, product_ids, meta_lookup, cfg)
        hubness_lookup = {
            _python_scalar(row["idarticu"]): row for row in hubness.iter_rows(named=True)
        } if hubness.height else {}
        rows = _nearest_neighbor_rows(normalized, product_ids, indices, index_groups, meta_lookup, hubness_lookup, cfg)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        report = pl.DataFrame(rows)
        report.write_csv(csv_path)
        hubness.write_csv(hubness_csv_path)

        metrics = _summary_metrics(report, hubness)
        if write_markdown_report:
            md_path.write_text(
                _validation_markdown(report, hubness, metrics, hubness_csv_path, category_match_counts, cfg),
                encoding="utf-8",
            )
        for path in cache_outputs:
            write_artifact_metadata(path, cache_metadata)
        log_event(
            "Stage 3 embedding validation",
            "wrote diagnostics",
            cfg=cfg,
            sampled_products=len(indices),
            csv=csv_path,
            hubness=hubness_csv_path,
        )
    return csv_path, md_path if write_markdown_report else None
