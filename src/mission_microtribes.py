"""Mission-first customer microtribes built from product evidence."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.product_themes import detect_product_themes
from src.progress import log_event
from src.utils import collect_streaming, schema_names


MISSION_DEFINITIONS: list[dict[str, Any]] = [
    {
        "mission_key": "gluten_free",
        "mission_label": "Gluten-Free Product Buyers",
        "mission_family": "Special Diet",
        "theme_keys": ["gluten_free"],
    },
    {
        "mission_key": "lactose_free",
        "mission_label": "Lactose-Free Product Buyers",
        "mission_family": "Special Diet",
        "theme_keys": ["lactose_free"],
    },
    {
        "mission_key": "organic_bio",
        "mission_label": "Organic/Bio Product Buyers",
        "mission_family": "Natural & Wellness",
        "theme_keys": ["organic_bio"],
        "min_matching_product_count": 2,
    },
    {
        "mission_key": "plant_based",
        "mission_label": "Plant-Based Product Buyers",
        "mission_family": "Natural & Wellness",
        "theme_keys": ["plant_based"],
    },
    {
        "mission_key": "protein_fitness",
        "mission_label": "Protein & Fitness Buyers",
        "mission_family": "Health & Fitness",
        "theme_keys": ["protein_fitness"],
    },
    {
        "mission_key": "world_foods",
        "mission_label": "World Cuisine Buyers",
        "mission_family": "World Cuisine",
        "theme_keys": [
            "world_foods_asian",
            "world_foods_mexican",
            "world_foods_middle_eastern",
            "world_foods_latin",
        ],
        "min_matching_product_count": 2,
    },
    {
        "mission_key": "world_foods_asian",
        "mission_label": "Asian-Style World Food Buyers",
        "mission_family": "World Cuisine",
        "theme_keys": ["world_foods_asian"],
    },
    {
        "mission_key": "world_foods_mexican",
        "mission_label": "Mexican-Style World Food Buyers",
        "mission_family": "World Cuisine",
        "theme_keys": ["world_foods_mexican"],
    },
    {
        "mission_key": "world_foods_middle_eastern",
        "mission_label": "Middle Eastern-Style World Food Buyers",
        "mission_family": "World Cuisine",
        "theme_keys": ["world_foods_middle_eastern"],
    },
    {
        "mission_key": "world_foods_latin",
        "mission_label": "Latin World Food Buyers",
        "mission_family": "World Cuisine",
        "theme_keys": ["world_foods_latin"],
    },
    {
        "mission_key": "pet",
        "mission_label": "Pet Product Buyers",
        "mission_family": "Household & Pets",
        "theme_keys": ["pet", "pet_dog", "pet_cat"],
    },
    {
        "mission_key": "pet_dog",
        "mission_label": "Dog Product Buyers",
        "mission_family": "Household & Pets",
        "theme_keys": ["pet_dog"],
    },
    {
        "mission_key": "pet_cat",
        "mission_label": "Cat Product Buyers",
        "mission_family": "Household & Pets",
        "theme_keys": ["pet_cat"],
    },
    {
        "mission_key": "baby_kids",
        "mission_label": "Baby & Kids Product Buyers",
        "mission_family": "Family",
        "theme_keys": ["baby", "kids_general", "kids_girls", "kids_boys", "books_toys"],
    },
    {
        "mission_key": "baby",
        "mission_label": "Baby Care Buyers",
        "mission_family": "Family",
        "theme_keys": ["baby"],
    },
    {
        "mission_key": "kids",
        "mission_label": "Kids & Family Product Buyers",
        "mission_family": "Family",
        "theme_keys": ["kids_general", "kids_girls", "kids_boys", "books_toys"],
    },
    {
        "mission_key": "meat_heavy",
        "mission_label": "Meat & Charcuterie Buyers",
        "mission_family": "Fresh Food",
        "theme_keys": ["meat_charcuterie"],
        "min_matching_product_count": 6,
    },
    {
        "mission_key": "seafood",
        "mission_label": "Seafood Buyers",
        "mission_family": "Fresh Food",
        "theme_keys": ["seafood"],
        "min_matching_product_count": 4,
    },
    {
        "mission_key": "alcohol",
        "mission_label": "Beer & Alcohol Buyers",
        "mission_family": "Drinks",
        "theme_keys": ["alcohol"],
        "min_matching_product_count": 2,
    },
    {
        "mission_key": "alcohol_free",
        "mission_label": "Alcohol-Free Beer Buyers",
        "mission_family": "Drinks",
        "theme_keys": ["alcohol_free"],
    },
]


def write_mission_microtribe_artifacts(
    assignments_path: str | Path,
    *,
    transactions: pl.LazyFrame | None = None,
    core_tribe_summary_path: str | Path | None = None,
    output_customer_parquet: str | Path | None = None,
    output_summary_csv: str | Path | None = None,
    output_summary_md: str | Path | None = None,
    output_summary_html: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write mission-first microtribe tags and a presentation summary.

    Missions are multi-label purchase-evidence segments. They do not replace the
    mutually exclusive clustering result; they sit on top of it for activation.
    """

    customer_output = (
        Path(output_customer_parquet)
        if output_customer_parquet
        else cfg.reports / f"customer_mission_tags_{cfg.mode}.parquet"
    )
    summary_csv = (
        Path(output_summary_csv)
        if output_summary_csv
        else cfg.reports / f"mission_microtribe_summary_{cfg.mode}.csv"
    )
    write_companions = bool(cfg.get("mission_microtribes.write_companion_files", False))
    summary_md = Path(output_summary_md) if output_summary_md else (summary_csv.with_suffix(".md") if write_companions else None)
    summary_html = (
        Path(output_summary_html) if output_summary_html else (summary_csv.with_suffix(".html") if write_companions else None)
    )

    assignments = _assignment_frame(assignments_path)
    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    customer_tags, product_counts = _customer_mission_tags(assignments, lf)
    tribe_lookup = _tribe_lookup(core_tribe_summary_path)
    summary = _mission_summary(customer_tags, product_counts, assignments, tribe_lookup, cfg=cfg)
    if summary.is_empty() or "mission_key" not in summary.columns:
        customer_tags = _empty_customer_mission_tags()
    else:
        customer_tags = customer_tags.filter(pl.col("mission_key").is_in(summary["mission_key"].to_list()))

    customer_output.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    customer_tags.write_parquet(customer_output)
    summary.write_csv(summary_csv)
    if summary_md is not None:
        summary_md.write_text(_mission_summary_markdown(summary, cfg=cfg), encoding="utf-8")
    if summary_html is not None:
        summary_html.write_text(_mission_summary_html(summary, cfg=cfg), encoding="utf-8")

    log_event(
        "Stage 9 mission microtribes",
        "wrote mission-first customer segments",
        cfg=cfg,
        customer_tags=customer_output,
        summary_csv=summary_csv,
    )
    paths = {
        "parquet": customer_output,
        "summary_csv": summary_csv,
    }
    if summary_md is not None:
        paths["summary_markdown"] = summary_md
    if summary_html is not None:
        paths["summary_html"] = summary_html
    return paths


def _assignment_frame(assignments_path: str | Path) -> pl.DataFrame:
    lf = pl.scan_parquet(assignments_path)
    columns = set(schema_names(lf))
    select_exprs = [pl.col("cliente"), pl.col("tribe_id")]
    if "assignment_source" in columns:
        select_exprs.append(pl.col("assignment_source"))
    else:
        select_exprs.append(pl.lit(None).cast(pl.Utf8).alias("assignment_source"))
    if "assignment_confidence_score" in columns:
        select_exprs.append(pl.col("assignment_confidence_score").cast(pl.Float64))
    else:
        select_exprs.append(pl.lit(None).cast(pl.Float64).alias("assignment_confidence_score"))
    return collect_streaming(lf.select(select_exprs).unique(subset=["cliente"], keep="first"))


def _customer_mission_tags(
    assignments: pl.DataFrame,
    transactions: pl.LazyFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    transaction_columns = set(schema_names(transactions))
    if "desc_larga_articulo" not in transaction_columns:
        return _empty_customer_mission_tags(), _empty_mission_product_counts()

    allowed_themes = _mission_theme_keys()
    assignment_lf = assignments.lazy().select("cliente")
    customer_product = (
        transactions.select(["cliente", "idarticu", "desc_larga_articulo"])
        .join(assignment_lf, on="cliente", how="inner")
        .rename({"desc_larga_articulo": "product_description"})
        .unique(subset=["cliente", "idarticu"])
    )
    product_theme_map = _product_theme_map(customer_product, allowed_themes)
    if product_theme_map.is_empty():
        return _empty_customer_mission_tags(), _empty_mission_product_counts()

    mission_theme_map = _mission_theme_map()
    customer_theme = (
        customer_product.join(product_theme_map.lazy(), on="idarticu", how="inner")
        .group_by(["cliente", "theme_key"])
        .agg(pl.col("idarticu").n_unique().alias("theme_product_count"))
    )
    customer_mission = (
        customer_theme.join(mission_theme_map.lazy(), on="theme_key", how="inner")
        .group_by(["cliente", "mission_key", "mission_label", "mission_family"])
        .agg(
            [
                pl.col("theme_key").n_unique().alias("matching_theme_count"),
                pl.col("theme_product_count").sum().alias("matching_product_count"),
                pl.col("min_matching_product_count").max().alias("min_matching_product_count"),
            ]
        )
        .filter(pl.col("matching_product_count") >= pl.col("min_matching_product_count"))
    )
    customer_tags = collect_streaming(
        customer_mission.join(assignments.lazy(), on="cliente", how="left").sort(
            ["mission_family", "mission_key", "cliente"]
        )
    )
    eligible_customer_missions = customer_mission.select(["cliente", "mission_key"]).unique()

    product_counts = collect_streaming(
        customer_product.join(product_theme_map.lazy(), on="idarticu", how="inner")
        .join(mission_theme_map.lazy(), on="theme_key", how="inner")
        .join(eligible_customer_missions, on=["cliente", "mission_key"], how="inner")
        .group_by(["mission_key", "mission_label", "idarticu", "product_description"])
        .agg(pl.col("cliente").n_unique().alias("product_customers"))
        .sort(["mission_key", "product_customers"], descending=[False, True])
    )
    return customer_tags, product_counts


def _product_theme_map(customer_product: pl.LazyFrame, allowed_themes: set[str]) -> pl.DataFrame:
    products = collect_streaming(customer_product.select(["idarticu", "product_description"]).unique())
    rows = []
    for row in products.iter_rows(named=True):
        description = "" if row.get("product_description") is None else str(row.get("product_description"))
        for theme in detect_product_themes(description):
            if theme in allowed_themes:
                rows.append({"idarticu": row["idarticu"], "theme_key": theme})
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"idarticu": pl.Utf8, "theme_key": pl.Utf8})


def _mission_theme_map() -> pl.DataFrame:
    rows = []
    for definition in MISSION_DEFINITIONS:
        for theme_key in definition["theme_keys"]:
            rows.append(
                {
                    "mission_key": definition["mission_key"],
                    "mission_label": definition["mission_label"],
                    "mission_family": definition["mission_family"],
                    "theme_key": theme_key,
                    "min_matching_product_count": int(definition.get("min_matching_product_count", 1)),
                }
            )
    return pl.DataFrame(rows)


def _mission_theme_keys() -> set[str]:
    return {theme for definition in MISSION_DEFINITIONS for theme in definition["theme_keys"]}


def _tribe_lookup(core_tribe_summary_path: str | Path | None) -> dict[int, dict[str, Any]]:
    if core_tribe_summary_path is None:
        return {}
    path = Path(core_tribe_summary_path)
    if not path.exists():
        return {}
    summary = pl.read_csv(path)
    name_col = "suggested_tribe_name" if "suggested_tribe_name" in summary.columns else "working_tribe_name"
    lookup = {}
    for row in summary.iter_rows(named=True):
        tribe_id = row.get("tribe_id")
        if tribe_id is None:
            continue
        lookup[int(tribe_id)] = {
            "name": row.get(name_col) or f"Tribe {tribe_id}",
            "n_customers": int(row.get("n_customers") or 0),
        }
    return lookup


def _mission_summary(
    customer_tags: pl.DataFrame,
    product_counts: pl.DataFrame,
    assignments: pl.DataFrame,
    tribe_lookup: dict[int, dict[str, Any]],
    *,
    cfg: PipelineConfig,
) -> pl.DataFrame:
    if customer_tags.is_empty():
        return _empty_mission_summary()

    population_customers = assignments["cliente"].n_unique()
    assigned = assignments.filter(pl.col("tribe_id") >= 0)
    assigned_customers = assigned["cliente"].n_unique()
    tribe_sizes = {
        int(row["tribe_id"]): int(row["len"])
        for row in assigned.group_by("tribe_id").len().iter_rows(named=True)
    }
    rows = []
    min_customers = int(cfg.get("mission_microtribes.min_customers", 20))
    top_n_tribes = int(cfg.get("mission_microtribes.top_n_core_tribes", 4))
    top_n_products = int(cfg.get("mission_microtribes.top_n_products", 5))

    grouped = customer_tags.group_by(["mission_key", "mission_label", "mission_family"]).agg(
        [
            pl.col("cliente").n_unique().alias("mission_customers"),
            pl.col("matching_theme_count").mean().alias("avg_matching_theme_count"),
            pl.col("matching_product_count").mean().alias("avg_matching_product_count"),
            (pl.col("tribe_id") >= 0).sum().alias("customers_with_core_tribe"),
            (
                (pl.col("tribe_id") >= 0)
                & (~pl.col("assignment_source").fill_null("").str.starts_with("nearest_centroid"))
            )
            .sum()
            .alias("core_cluster_customers"),
            pl.col("assignment_source")
            .fill_null("")
            .str.starts_with("nearest_centroid")
            .sum()
            .alias("soft_assigned_customers"),
            pl.col("assignment_confidence_score").mean().alias("avg_assignment_confidence"),
        ]
    )
    for row in grouped.sort(["mission_family", "mission_customers"], descending=[False, True]).iter_rows(named=True):
        mission_key = str(row["mission_key"])
        mission_customers = int(row["mission_customers"] or 0)
        if mission_customers < min_customers:
            continue
        max_core_lift, strongest_core_tribe = _mission_core_lift_stats(
            customer_tags,
            mission_key,
            mission_customers,
            tribe_sizes,
            assigned_customers,
            tribe_lookup,
        )
        avg_matching_product_count = row["avg_matching_product_count"]
        confidence = _mission_confidence(
            mission_customers,
            avg_matching_product_count,
            population_customers,
            max_core_lift,
        )
        rows.append(
            {
                "mission_key": mission_key,
                "mission_label": row["mission_label"],
                "mission_family": row["mission_family"],
                "mission_customers": mission_customers,
                "population_share_pct": round(mission_customers / max(population_customers, 1) * 100.0, 2),
                "customers_with_core_tribe": int(row["customers_with_core_tribe"] or 0),
                "core_cluster_customers": int(row["core_cluster_customers"] or 0),
                "soft_assigned_customers": int(row["soft_assigned_customers"] or 0),
                "avg_matching_theme_count": _round_optional(row["avg_matching_theme_count"]),
                "avg_matching_product_count": _round_optional(avg_matching_product_count),
                "avg_assignment_confidence": _round_optional(row["avg_assignment_confidence"]),
                "max_core_lift": _round_optional(max_core_lift),
                "strongest_core_tribe": strongest_core_tribe,
                "top_core_tribes": _top_core_tribes(
                    customer_tags,
                    mission_key,
                    mission_customers,
                    tribe_sizes,
                    assigned_customers,
                    tribe_lookup,
                    limit=top_n_tribes,
                ),
                "top_products": _top_products(product_counts, mission_key, limit=top_n_products),
                "mission_confidence": confidence,
                "presentation_score": _mission_presentation_score(
                    mission_customers,
                    avg_matching_product_count,
                    max_core_lift,
                ),
                "activation_read": _mission_activation_read(mission_customers, avg_matching_product_count, confidence),
            }
        )
    if not rows:
        return _empty_mission_summary()
    return _select_presented_missions(pl.DataFrame(rows), cfg=cfg)


def _top_core_tribes(
    customer_tags: pl.DataFrame,
    mission_key: str,
    mission_customers: int,
    tribe_sizes: dict[int, int],
    assigned_customers: int,
    tribe_lookup: dict[int, dict[str, Any]],
    *,
    limit: int,
) -> str:
    subset = customer_tags.filter((pl.col("mission_key") == mission_key) & (pl.col("tribe_id") >= 0))
    if subset.is_empty():
        return "n/a"
    tribe_counts = subset.group_by("tribe_id").agg(pl.col("cliente").n_unique().alias("customers"))
    parts = []
    for row in tribe_counts.sort("customers", descending=True).head(limit).iter_rows(named=True):
        tribe_id = int(row["tribe_id"])
        customers = int(row["customers"])
        mission_share = customers / max(mission_customers, 1)
        tribe_rate = tribe_sizes.get(tribe_id, 0) / max(assigned_customers, 1)
        lift = mission_share / max(tribe_rate, 1e-9)
        name = tribe_lookup.get(tribe_id, {}).get("name", f"Tribe {tribe_id}")
        parts.append(f"T{tribe_id} {name} ({customers:,}, {mission_share:.0%}, {lift:.2f}x)")
    return "; ".join(parts)


def _mission_core_lift_stats(
    customer_tags: pl.DataFrame,
    mission_key: str,
    mission_customers: int,
    tribe_sizes: dict[int, int],
    assigned_customers: int,
    tribe_lookup: dict[int, dict[str, Any]],
) -> tuple[float | None, str]:
    subset = customer_tags.filter((pl.col("mission_key") == mission_key) & (pl.col("tribe_id") >= 0))
    if subset.is_empty():
        return None, "n/a"
    best_lift = -1.0
    best_read = "n/a"
    tribe_counts = subset.group_by("tribe_id").agg(pl.col("cliente").n_unique().alias("customers"))
    for row in tribe_counts.iter_rows(named=True):
        tribe_id = int(row["tribe_id"])
        customers = int(row["customers"])
        mission_share = customers / max(mission_customers, 1)
        tribe_rate = tribe_sizes.get(tribe_id, 0) / max(assigned_customers, 1)
        lift = mission_share / max(tribe_rate, 1e-9)
        if lift > best_lift:
            name = tribe_lookup.get(tribe_id, {}).get("name", f"Tribe {tribe_id}")
            best_lift = lift
            best_read = f"T{tribe_id} {name} ({customers:,}, {mission_share:.0%}, {lift:.2f}x)"
    return best_lift, best_read


def _top_products(product_counts: pl.DataFrame, mission_key: str, *, limit: int) -> str:
    if product_counts.is_empty():
        return "n/a"
    subset = product_counts.filter(pl.col("mission_key") == mission_key).head(limit)
    if subset.is_empty():
        return "n/a"
    parts = []
    for row in subset.iter_rows(named=True):
        product = _clean_display_text(row.get("product_description"))
        parts.append(f"{product} ({int(row.get('product_customers') or 0):,})")
    return "; ".join(parts)


def _mission_confidence(
    customer_count: int,
    avg_matching_products: Any,
    population_customers: int,
    max_core_lift: float | None = None,
) -> str:
    avg_products = float(avg_matching_products or 0.0)
    share = customer_count / max(population_customers, 1)
    lift = float(max_core_lift or 0.0)
    if customer_count >= 1000 and avg_products >= 1.5 and (lift >= 1.1 or avg_products >= 2.0):
        return "high"
    if customer_count >= 500 and (avg_products >= 1.2 or share >= 0.02) and (lift >= 1.0 or avg_products >= 1.8):
        return "medium"
    if customer_count >= 20:
        return "niche"
    return "low"


def _mission_presentation_score(customer_count: int, avg_matching_products: Any, max_core_lift: Any) -> float:
    avg_products = float(avg_matching_products or 0.0)
    lift = max(float(max_core_lift or 0.0), 1.0)
    return round((max(customer_count, 0) ** 0.5) * avg_products * lift, 3)


def _select_presented_missions(summary: pl.DataFrame, *, cfg: PipelineConfig) -> pl.DataFrame:
    if summary.is_empty():
        return _empty_mission_summary()
    confidence_rank = {"high": 3, "medium": 2, "niche": 1, "low": 0}
    min_rank = confidence_rank.get(str(cfg.get("mission_microtribes.min_presented_confidence", "medium")).lower(), 2)
    excluded = set(cfg.get("mission_microtribes.exclude_from_presentation", []) or [])
    min_customers = int(cfg.get("mission_microtribes.min_presented_customers", cfg.get("mission_microtribes.min_customers", 20)))
    min_avg_products = float(cfg.get("mission_microtribes.min_presented_avg_matching_products", 1.0))
    max_missions = int(cfg.get("mission_microtribes.max_presented_missions", 10))
    selected = summary.with_columns(
        pl.col("mission_confidence")
        .map_elements(lambda value: confidence_rank.get(str(value).lower(), 0), return_dtype=pl.Int64)
        .alias("_confidence_rank")
    )
    selected = selected.filter(
        (~pl.col("mission_key").is_in(list(excluded)))
        & (pl.col("mission_customers") >= min_customers)
        & (pl.col("avg_matching_product_count").fill_null(0.0) >= min_avg_products)
        & (pl.col("_confidence_rank") >= min_rank)
    )
    if selected.is_empty():
        return _empty_mission_summary()
    selected = (
        selected.sort(
            ["_confidence_rank", "presentation_score", "mission_customers"],
            descending=[True, True, True],
        )
        .head(max_missions)
        .with_row_index("presentation_rank", offset=1)
        .drop("_confidence_rank")
    )
    ordered_cols = [
        "presentation_rank",
        "mission_key",
        "mission_label",
        "mission_family",
        "mission_customers",
        "population_share_pct",
        "customers_with_core_tribe",
        "core_cluster_customers",
        "soft_assigned_customers",
        "avg_matching_theme_count",
        "avg_matching_product_count",
        "avg_assignment_confidence",
        "max_core_lift",
        "strongest_core_tribe",
        "top_core_tribes",
        "top_products",
        "mission_confidence",
        "presentation_score",
        "activation_read",
    ]
    return selected.select([col for col in ordered_cols if col in selected.columns])


def _mission_activation_read(customer_count: int, avg_matching_products: Any, confidence: str) -> str:
    avg_products = float(avg_matching_products or 0.0)
    if confidence == "high":
        return "Strong mission segment; suitable as a client-facing activation layer."
    if confidence == "medium":
        return "Usable mission segment; validate top products before campaign copy."
    return "Niche mission signal; use for targeted tests or product-level audiences."


def _mission_summary_markdown(summary: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    lines = [
        "# Mission Microtribe Summary",
        "",
        f"Run mode: `{cfg.mode}`",
        "",
        "Mission microtribes are multi-label activation segments derived from product evidence. "
        "They sit above the core clustering result for campaign planning and do not replace the official core tribe assignment.",
        "",
        "Use all labels as purchase-behavior evidence, not as demographic, religious, household, or identity truth.",
        "",
    ]
    if summary.is_empty():
        lines.append("No mission microtribes passed the configured evidence thresholds.")
        return "\n".join(lines) + "\n"
    display_cols = [
        "mission_family",
        "mission_label",
        "mission_customers",
        "population_share_pct",
        "mission_confidence",
        "presentation_score",
        "max_core_lift",
        "strongest_core_tribe",
        "top_core_tribes",
        "top_products",
        "activation_read",
    ]
    lines.append(_markdown_table(summary.select([col for col in display_cols if col in summary.columns])))
    return "\n".join(lines) + "\n"


def _mission_summary_html(summary: pl.DataFrame, cfg: PipelineConfig = CONFIG) -> str:
    rows = []
    for row in summary.iter_rows(named=True):
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('mission_family') or ''))}</td>"
            f"<td><strong>{escape(str(row.get('mission_label') or ''))}</strong><br>"
            f"<span class='muted'>{escape(str(row.get('mission_key') or ''))}</span></td>"
            f"<td>{int(row.get('mission_customers') or 0):,}<br>"
            f"<span class='muted'>{float(row.get('population_share_pct') or 0):.2f}% population</span></td>"
            f"<td>{escape(str(row.get('mission_confidence') or ''))}</td>"
            f"<td>{escape(str(row.get('top_core_tribes') or ''))}</td>"
            f"<td>{escape(str(row.get('top_products') or ''))}</td>"
            f"<td>{escape(str(row.get('activation_read') or ''))}</td>"
            "</tr>"
        )
    body = "\n".join(rows) or "<tr><td colspan='7'>No mission microtribes passed the configured thresholds.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mission Microtribe Summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; }}
.muted {{ color: #667085; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
th, td {{ border: 1px solid #d0d5dd; padding: 8px; vertical-align: top; font-size: 13px; }}
th {{ background: #eef2f6; text-align: left; }}
</style>
</head>
<body>
<h1>Mission Microtribe Summary</h1>
<p class="muted">Run mode: {escape(str(cfg.mode))}. Missions are multi-label purchase-evidence segments for activation, not demographic identity claims.</p>
<table>
<thead>
<tr>
<th>Family</th>
<th>Mission</th>
<th>Customers</th>
<th>Confidence</th>
<th>Core Tribe Distribution</th>
<th>Top Products</th>
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


def _markdown_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return ""
    columns = df.columns
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.iter_rows(named=True):
        values = [_clean_markdown_cell(row.get(col)) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _clean_markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "/").replace("\n", " ")


def _clean_display_text(value: Any, max_len: int = 80) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "..."


def _round_optional(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _empty_customer_mission_tags() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "cliente": pl.Utf8,
            "mission_key": pl.Utf8,
            "mission_label": pl.Utf8,
            "mission_family": pl.Utf8,
            "matching_theme_count": pl.Int64,
            "matching_product_count": pl.Int64,
            "min_matching_product_count": pl.Int64,
            "tribe_id": pl.Int64,
            "assignment_source": pl.Utf8,
            "assignment_confidence_score": pl.Float64,
        }
    )


def _empty_mission_product_counts() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "mission_key": pl.Utf8,
            "mission_label": pl.Utf8,
            "idarticu": pl.Utf8,
            "product_description": pl.Utf8,
            "product_customers": pl.Int64,
        }
    )


def _empty_mission_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "mission_key": pl.Utf8,
            "mission_label": pl.Utf8,
            "mission_family": pl.Utf8,
            "mission_customers": pl.Int64,
            "population_share_pct": pl.Float64,
            "customers_with_core_tribe": pl.Int64,
            "core_cluster_customers": pl.Int64,
            "soft_assigned_customers": pl.Int64,
            "avg_matching_theme_count": pl.Float64,
            "avg_matching_product_count": pl.Float64,
            "avg_assignment_confidence": pl.Float64,
            "max_core_lift": pl.Float64,
            "strongest_core_tribe": pl.Utf8,
            "top_core_tribes": pl.Utf8,
            "top_products": pl.Utf8,
            "mission_confidence": pl.Utf8,
            "presentation_score": pl.Float64,
            "activation_read": pl.Utf8,
            "presentation_rank": pl.UInt32,
        }
    )
