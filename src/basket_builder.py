"""Basket sentence construction for Item2Vec training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.progress import log_event, stage_timer
from src.utils import collect_streaming, file_fingerprint, schema_names, should_use_cache, stable_hash, write_artifact_metadata


def _deterministic_basket_order(ticket: str, products: list[str] | None) -> list[str]:
    """Order basket tokens reproducibly without using product-id order as signal."""

    if not products:
        return []
    return sorted([str(product) for product in products], key=lambda product: stable_hash(f"{ticket}|{product}"))


def build_basket_sentences(
    transactions: pl.LazyFrame | None = None,
    output_path: str | Path | None = None,
    construction_strategy: str | None = None,
    repeat_product_by_quantity: bool | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create one ticket-level product-token sentence per basket."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.artifact_path(
        "baskets",
        "output",
        directory=cfg.outputs / "embeddings",
    )
    strategy = str(construction_strategy or cfg.get("baskets.construction_strategy", "baseline")).strip().lower()
    repeat = (
        bool(cfg.get("baskets.repeat_product_by_quantity", False))
        if repeat_product_by_quantity is None
        else repeat_product_by_quantity
    )
    cache_metadata = {
        "stage": "basket_sentences",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "construction_strategy": strategy,
        "repeat_product_by_quantity": repeat,
        "downsampling": _downsampling_metadata(cfg) if strategy == "common_downsampled" else None,
        "common_product_diagnostics": _diagnostic_threshold_metadata(cfg)
        if strategy == "common_downsampled"
        else None,
        "ordering": "deterministic_ticket_hash",
    }
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata):
        log_event("Stage 1 baskets", "cache hit", cfg=cfg, path=output)
        return output

    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)

    with stage_timer(
        "Stage 1 baskets",
        "building basket sentences",
        cfg=cfg,
        output=output,
        strategy=strategy,
        repeat_products=repeat,
    ):
        if strategy == "baseline":
            basket_lf = _baseline_basket_lazyframe(lf, repeat=repeat)
        elif strategy == "common_downsampled":
            if repeat:
                raise ValueError("Common-downsampled basket construction currently expects repeat_product_by_quantity=false.")
            diagnostics = build_basket_staple_diagnostics(transactions=lf, cfg=cfg)
            downsampling_metadata = _downsampling_metadata(cfg)
            keep_plan = _common_product_keep_plan(
                pl.read_parquet(diagnostics["product_diagnostics"]),
                manual_exclude_product_ids=downsampling_metadata["manual_exclude_product_ids"],
                auto_exclude=downsampling_metadata["auto_exclude"],
                target_customer_penetration=float(downsampling_metadata["target_customer_penetration"]),
                keep_probability_exponent=float(downsampling_metadata["keep_probability_exponent"]),
                min_keep_probability=float(downsampling_metadata["min_keep_probability"]),
            )
            basket_lf = _common_downsampled_basket_lazyframe(lf, keep_plan=keep_plan, cfg=cfg)
        else:
            raise ValueError(f"Unknown baskets.construction_strategy: {strategy!r}")

        output.parent.mkdir(parents=True, exist_ok=True)
        baskets = collect_streaming(basket_lf.sort("ticket"))
        baskets = baskets.with_columns(
            pl.struct(["ticket", "products"])
            .map_elements(
                lambda row: _deterministic_basket_order(row["ticket"], row["products"]),
                return_dtype=pl.List(pl.Utf8),
            )
            .alias("products")
        )
        baskets.write_parquet(output)
        write_artifact_metadata(output, cache_metadata)
        log_event("Stage 1 baskets", "wrote artifact", cfg=cfg, baskets=baskets.height, strategy=strategy, path=output)
    return output


def _baseline_basket_lazyframe(lf: pl.LazyFrame, repeat: bool) -> pl.LazyFrame:
    base = lf.select(["ticket", "idarticu"] + (["unidades"] if repeat else []))
    if repeat:
        token_lf = (
            base.with_columns(
                [
                    pl.col("idarticu").cast(pl.Utf8).alias("_product_token"),
                    pl.when(pl.col("unidades").cast(pl.Float64) > 0)
                    .then(pl.col("unidades").cast(pl.Int64))
                    .otherwise(1)
                    .clip(1, 20)
                    .cast(pl.UInt32)
                    .alias("_repeat_count"),
                ]
            )
            .with_columns(pl.col("_product_token").repeat_by("_repeat_count").alias("_tokens"))
            .select(["ticket", "_tokens"])
            .explode("_tokens")
        )
        return token_lf.group_by("ticket").agg(
            [
                pl.col("_tokens").alias("products"),
                pl.len().alias("n_product_tokens"),
            ]
        )
    return base.with_columns(pl.col("idarticu").cast(pl.Utf8).alias("_product_token")).group_by("ticket").agg(
        [
            pl.col("_product_token").unique().alias("products"),
            pl.col("_product_token").n_unique().alias("n_product_tokens"),
        ]
    )


def _common_downsampled_basket_lazyframe(
    lf: pl.LazyFrame,
    keep_plan: pl.DataFrame,
    cfg: PipelineConfig = CONFIG,
) -> pl.LazyFrame:
    sample_modulus = int(_downsampling_metadata(cfg)["sample_modulus"])
    base_pairs = (
        lf.select(["ticket", "idarticu"])
        .with_columns(
            [
                pl.col("ticket").cast(pl.Utf8),
                pl.col("idarticu").cast(pl.Utf8),
            ]
        )
        .unique()
    )
    raw_pairs = base_pairs.join(keep_plan.lazy(), on="idarticu", how="left").with_columns(
        [
            pl.col("_exclude_from_embedding").fill_null(False),
            pl.col("_manual_exclude_from_embedding").fill_null(False),
            pl.col("_auto_exclude_from_embedding").fill_null(False),
            pl.col("_keep_probability").fill_null(1.0),
        ]
    )
    candidates = (
        raw_pairs.filter(~pl.col("_exclude_from_embedding"))
        .with_columns(
            (
                (
                    pl.concat_str(["ticket", "idarticu"], separator="|").hash(seed=cfg.random_seed)
                    % pl.lit(sample_modulus)
                ).cast(pl.Float64)
                / pl.lit(float(sample_modulus))
            ).alias("_sample_value")
        )
        .with_columns((pl.col("_sample_value") < pl.col("_keep_probability")).alias("_keep_selected"))
    )
    status = candidates.group_by("ticket").agg(pl.col("_keep_selected").any().alias("_has_selected"))
    kept = candidates.filter(pl.col("_keep_selected")).select(["ticket", pl.col("idarticu").alias("_product_token")])
    fallback = (
        candidates.join(status, on="ticket", how="left")
        .filter(~pl.col("_has_selected"))
        .sort(["ticket", "_keep_probability", "_sample_value"], descending=[False, True, False])
        .group_by("ticket")
        .agg(pl.col("idarticu").first().alias("_product_token"))
        .select(["ticket", "_product_token"])
    )
    return pl.concat([kept, fallback], how="vertical").group_by("ticket").agg(
        [
            pl.col("_product_token").alias("products"),
            pl.len().alias("n_product_tokens"),
        ]
    )


def basket_summary(basket_path: str | Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(basket_path)
        .select(
            [
                pl.len().alias("n_baskets"),
                pl.col("n_product_tokens").mean().alias("avg_products_per_basket"),
                pl.col("n_product_tokens").median().alias("median_products_per_basket"),
                pl.col("n_product_tokens").max().alias("max_products_per_basket"),
            ]
        )
        .collect()
    )


def build_basket_staple_diagnostics(
    transactions: pl.LazyFrame | None = None,
    output_dir: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Audit whether highly common products may cloud basket co-occurrence signal.

    This stage writes diagnostics that identify common-product candidates and
    quantify how much of each basket is made from those candidates. When
    common-downsampled basket construction is enabled, these diagnostics also
    drive the dynamic keep/exclusion plan.
    """

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    diagnostics_dir = Path(output_dir) if output_dir else cfg.artifacts / str(
        cfg.get("baskets.diagnostics.output_dir", "stage1")
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    product_output = diagnostics_dir / str(
        cfg.get("baskets.diagnostics.product_ubiquity_output", "product_ubiquity_diagnostics.parquet")
    )
    exposure_output = diagnostics_dir / str(
        cfg.get("baskets.diagnostics.basket_exposure_output", "basket_common_product_exposure.parquet")
    )
    common_csv = diagnostics_dir / str(cfg.get("baskets.diagnostics.common_products_csv", "common_product_candidates.csv"))
    write_markdown_report = bool(cfg.get("baskets.diagnostics.write_markdown_report", True))
    summary_md = diagnostics_dir / str(cfg.get("baskets.diagnostics.summary_md", "basket_common_product_diagnostics.md"))

    customer_threshold = float(cfg.get("baskets.diagnostics.common_customer_penetration_threshold", 0.01))
    basket_threshold = float(cfg.get("baskets.diagnostics.common_basket_penetration_threshold", 0.005))
    line_threshold = float(cfg.get("baskets.diagnostics.common_line_share_threshold", 0.005))
    top_n = int(cfg.get("baskets.diagnostics.top_n_common_products", 50))

    cache_metadata = {
        "stage": "basket_staple_diagnostics",
        "mode": cfg.mode,
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "thresholds": {
            "common_customer_penetration_threshold": customer_threshold,
            "common_basket_penetration_threshold": basket_threshold,
            "common_line_share_threshold": line_threshold,
        },
        "top_n_common_products": top_n,
    }
    cache_outputs = [product_output, exposure_output, common_csv]
    if write_markdown_report:
        cache_outputs.append(summary_md)
    if all(
        should_use_cache(path, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        for path in cache_outputs
    ):
        log_event("Stage 1 baskets", "staple diagnostics cache hit", cfg=cfg, path=diagnostics_dir)
        paths = {
            "product_diagnostics": product_output,
            "basket_exposure": exposure_output,
            "common_products_csv": common_csv,
        }
        if write_markdown_report:
            paths["summary_md"] = summary_md
        return paths

    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    columns = set(schema_names(lf))
    select_cols = ["cliente", "ticket", "idarticu"]
    for optional_col in ["unidades", "desc_larga_articulo", "desc_sector", "idsector"]:
        if optional_col in columns:
            select_cols.append(optional_col)

    with stage_timer("Stage 1 baskets", "building common-product diagnostics", cfg=cfg, output=diagnostics_dir):
        base = lf.select(select_cols).with_columns(
            [
                pl.col("cliente").cast(pl.Utf8),
                pl.col("ticket").cast(pl.Utf8),
                pl.col("idarticu").cast(pl.Utf8),
                _units_expression(columns),
            ]
        )
        totals = collect_streaming(
            base.select(
                [
                    pl.col("cliente").n_unique().alias("total_customers"),
                    pl.col("ticket").n_unique().alias("total_baskets"),
                    pl.len().alias("total_lines"),
                    pl.col("_units").sum().alias("total_units"),
                ]
            )
        ).row(0, named=True)
        total_customers = max(int(totals["total_customers"] or 0), 1)
        total_baskets = max(int(totals["total_baskets"] or 0), 1)
        total_lines = max(int(totals["total_lines"] or 0), 1)
        total_units = max(float(totals["total_units"] or 0.0), 1.0)

        metadata_exprs: list[pl.Expr] = []
        if "desc_larga_articulo" in columns:
            metadata_exprs.append(pl.col("desc_larga_articulo").drop_nulls().first().alias("product_description"))
        else:
            metadata_exprs.append(pl.lit(None).cast(pl.Utf8).alias("product_description"))
        if "desc_sector" in columns:
            metadata_exprs.append(pl.col("desc_sector").drop_nulls().first().alias("sector_description"))
        else:
            metadata_exprs.append(pl.lit(None).cast(pl.Utf8).alias("sector_description"))
        if "idsector" in columns:
            metadata_exprs.append(pl.col("idsector").drop_nulls().first().cast(pl.Utf8).alias("sector_id"))
        else:
            metadata_exprs.append(pl.lit(None).cast(pl.Utf8).alias("sector_id"))

        product_diagnostics = collect_streaming(
            base.group_by("idarticu")
            .agg(
                [
                    pl.col("ticket").n_unique().alias("basket_count"),
                    pl.col("cliente").n_unique().alias("customer_count"),
                    pl.len().alias("line_count"),
                    pl.col("_units").sum().alias("units_sum"),
                    *metadata_exprs,
                ]
            )
            .with_columns(
                [
                    (pl.col("customer_count") / pl.lit(total_customers)).alias("customer_penetration"),
                    (pl.col("basket_count") / pl.lit(total_baskets)).alias("basket_penetration"),
                    (pl.col("line_count") / pl.lit(total_lines)).alias("line_share"),
                    (pl.col("units_sum") / pl.lit(total_units)).alias("unit_share"),
                    (pl.col("units_sum") / pl.col("line_count")).alias("avg_units_per_line"),
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
            .sort(["common_product_candidate", "commonness_score"], descending=[True, True])
        )
        product_diagnostics = product_diagnostics.with_row_index("commonness_rank", offset=1)
        product_diagnostics.write_parquet(product_output)
        write_artifact_metadata(product_output, cache_metadata)

        common_product_rows = product_diagnostics.filter(pl.col("common_product_candidate"))
        common_products = common_product_rows.select("idarticu")
        common_product_rows.head(top_n).write_csv(common_csv)
        write_artifact_metadata(common_csv, cache_metadata)

        unique_ticket_product = base.select(["ticket", "idarticu"]).unique()
        basket_totals = unique_ticket_product.group_by("ticket").agg(pl.len().alias("n_unique_products"))
        if common_products.is_empty():
            common_counts = pl.DataFrame(
                schema={"ticket": pl.Utf8, "n_common_product_candidates": pl.Int64}
            ).lazy()
        else:
            common_counts = (
                unique_ticket_product.join(common_products.lazy(), on="idarticu", how="inner")
                .group_by("ticket")
                .agg(pl.len().alias("n_common_product_candidates"))
            )
        basket_exposure = collect_streaming(
            basket_totals.join(common_counts, on="ticket", how="left")
            .with_columns(pl.col("n_common_product_candidates").fill_null(0).cast(pl.Int64))
            .with_columns(
                (pl.col("n_common_product_candidates") / pl.col("n_unique_products")).alias(
                    "common_product_candidate_share"
                )
            )
            .sort("ticket")
        )
        basket_exposure.write_parquet(exposure_output)
        write_artifact_metadata(exposure_output, cache_metadata)

        if write_markdown_report:
            summary_text = _basket_staple_summary_markdown(
                product_diagnostics=product_diagnostics,
                basket_exposure=basket_exposure,
                totals=totals,
                thresholds={
                    "customer_penetration": customer_threshold,
                    "basket_penetration": basket_threshold,
                    "line_share": line_threshold,
                },
                top_n=min(15, top_n),
                cfg=cfg,
            )
            summary_md.write_text(summary_text, encoding="utf-8")
            write_artifact_metadata(summary_md, cache_metadata)
        log_event(
            "Stage 1 baskets",
            "wrote common-product diagnostics",
            cfg=cfg,
            product_diagnostics=product_output,
            basket_exposure=exposure_output,
            common_candidates=common_products.height,
        )

    paths = {
        "product_diagnostics": product_output,
        "basket_exposure": exposure_output,
        "common_products_csv": common_csv,
    }
    if write_markdown_report:
        paths["summary_md"] = summary_md
    return paths


def _units_expression(columns: set[str]) -> pl.Expr:
    if "unidades" not in columns:
        return pl.lit(1.0).alias("_units")
    return (
        pl.when(pl.col("unidades").cast(pl.Float64) > 0)
        .then(pl.col("unidades").cast(pl.Float64))
        .otherwise(1.0)
        .alias("_units")
    )


def _downsampling_metadata(cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    legacy_manual_ids = cfg.get("baskets.downsampling.exclude_product_ids", [])
    manual_ids = cfg.get("baskets.downsampling.manual_exclude_product_ids", legacy_manual_ids) or []
    auto_exclude = cfg.get("baskets.downsampling.auto_exclude", {}) or {}
    return {
        "manual_exclude_product_ids": [str(value) for value in manual_ids],
        "auto_exclude": {
            "enabled": bool(auto_exclude.get("enabled", True)),
            "customer_penetration_threshold": _optional_float(
                auto_exclude.get("customer_penetration_threshold", 0.50)
            ),
            "basket_penetration_threshold": _optional_float(auto_exclude.get("basket_penetration_threshold", 0.10)),
            "line_share_threshold": _optional_float(auto_exclude.get("line_share_threshold")),
            "max_products": int(auto_exclude.get("max_products", 25)),
        },
        "target_customer_penetration": float(cfg.get("baskets.downsampling.target_customer_penetration", 0.01)),
        "keep_probability_exponent": float(cfg.get("baskets.downsampling.keep_probability_exponent", 0.5)),
        "min_keep_probability": float(cfg.get("baskets.downsampling.min_keep_probability", 0.15)),
        "sample_modulus": 1_000_000,
    }


def _diagnostic_threshold_metadata(cfg: PipelineConfig = CONFIG) -> dict[str, float]:
    return {
        "common_customer_penetration_threshold": float(
            cfg.get("baskets.diagnostics.common_customer_penetration_threshold", 0.01)
        ),
        "common_basket_penetration_threshold": float(
            cfg.get("baskets.diagnostics.common_basket_penetration_threshold", 0.005)
        ),
        "common_line_share_threshold": float(cfg.get("baskets.diagnostics.common_line_share_threshold", 0.005)),
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _common_product_keep_plan(
    product_diagnostics: pl.DataFrame,
    manual_exclude_product_ids: list[str] | None,
    auto_exclude: dict[str, Any] | None,
    target_customer_penetration: float,
    keep_probability_exponent: float,
    min_keep_probability: float,
) -> pl.DataFrame:
    product_diagnostics = product_diagnostics.with_columns(pl.col("idarticu").cast(pl.Utf8))
    manual_exclude_ids = sorted({str(value) for value in (manual_exclude_product_ids or [])})
    auto_exclude_ids = _auto_excluded_product_ids(product_diagnostics, auto_exclude)
    target = max(float(target_customer_penetration), 1e-9)
    min_keep = min(max(float(min_keep_probability), 0.0), 1.0)
    exponent = max(float(keep_probability_exponent), 0.0)
    base_ratio = (
        pl.lit(target)
        / pl.when(pl.col("customer_penetration") > target)
        .then(pl.col("customer_penetration"))
        .otherwise(pl.lit(target))
    )
    if exponent != 1.0:
        base_probability = base_ratio ** exponent
    else:
        base_probability = base_ratio
    return product_diagnostics.select(
        [
            "idarticu",
            "common_product_candidate",
            "customer_penetration",
            "basket_penetration",
            "line_share",
            "commonness_score",
        ]
    ).with_columns(
        [
            pl.col("idarticu").is_in(manual_exclude_ids).alias("_manual_exclude_from_embedding"),
            pl.col("idarticu").is_in(auto_exclude_ids).alias("_auto_exclude_from_embedding"),
        ]
    ).with_columns(
        [
            (pl.col("_manual_exclude_from_embedding") | pl.col("_auto_exclude_from_embedding")).alias(
                "_exclude_from_embedding"
            ),
        ]
    ).with_columns(
        [
            (
                pl.when(pl.col("_exclude_from_embedding"))
                .then(0.0)
                .when(pl.col("common_product_candidate"))
                .then(base_probability.clip(min_keep, 1.0))
                .otherwise(1.0)
            )
            .cast(pl.Float64)
            .alias("_keep_probability"),
        ]
    )


def _auto_excluded_product_ids(product_diagnostics: pl.DataFrame, auto_exclude: dict[str, Any] | None) -> list[str]:
    if not auto_exclude or not bool(auto_exclude.get("enabled", True)):
        return []

    customer_threshold = _optional_float(auto_exclude.get("customer_penetration_threshold"))
    basket_threshold = _optional_float(auto_exclude.get("basket_penetration_threshold"))
    line_threshold = _optional_float(auto_exclude.get("line_share_threshold"))
    max_products = max(int(auto_exclude.get("max_products", 25)), 0)
    if max_products == 0:
        return []

    paired_condition: pl.Expr | None = None
    if customer_threshold is not None:
        paired_condition = pl.col("customer_penetration") >= pl.lit(customer_threshold)
    if basket_threshold is not None:
        basket_condition = pl.col("basket_penetration") >= pl.lit(basket_threshold)
        paired_condition = basket_condition if paired_condition is None else paired_condition & basket_condition

    conditions: list[pl.Expr] = []
    if paired_condition is not None:
        conditions.append(paired_condition)
    if line_threshold is not None:
        conditions.append(pl.col("line_share") >= pl.lit(line_threshold))
    if not conditions:
        return []

    exclusion_condition = conditions[0]
    for condition in conditions[1:]:
        exclusion_condition = exclusion_condition | condition

    selected = (
        product_diagnostics.filter(exclusion_condition)
        .sort("commonness_score", descending=True)
        .head(max_products)
        .select("idarticu")
    )
    return [str(value) for value in selected["idarticu"].to_list()]


def _basket_staple_summary_markdown(
    product_diagnostics: pl.DataFrame,
    basket_exposure: pl.DataFrame,
    totals: dict[str, Any],
    thresholds: dict[str, float],
    top_n: int,
    cfg: PipelineConfig = CONFIG,
) -> str:
    common = product_diagnostics.filter(pl.col("common_product_candidate"))
    exposure_values = basket_exposure["common_product_candidate_share"].to_numpy()
    lines = [
        "# Stage 1 Common-Product Diagnostics",
        "",
        f"Run mode: `{cfg.mode}`",
        "",
        "These diagnostics identify products that are common enough to potentially cloud product co-occurrence and customer-vector signal. If common-downsampled basket construction is enabled, the same diagnostics drive the Stage 1 keep/exclusion plan.",
        "",
        "## Thresholds",
        "",
        f"- Customer penetration >= `{thresholds['customer_penetration']:.4f}`",
        f"- Basket penetration >= `{thresholds['basket_penetration']:.4f}`",
        f"- Line share >= `{thresholds['line_share']:.4f}`",
        "",
        "## Population Summary",
        "",
        f"- Customers: `{int(totals.get('total_customers') or 0):,}`",
        f"- Baskets: `{int(totals.get('total_baskets') or 0):,}`",
        f"- Product lines: `{int(totals.get('total_lines') or 0):,}`",
        f"- Products: `{product_diagnostics.height:,}`",
        f"- Common-product candidates: `{common.height:,}`",
        f"- Avg common-candidate share per basket: `{float(exposure_values.mean()) if exposure_values.size else 0.0:.3f}`",
        f"- P90 common-candidate share per basket: `{float(_safe_percentile(exposure_values, 90)):.3f}`",
        "",
        "## Top Common-Product Candidates",
        "",
    ]
    display_cols = [
        "commonness_rank",
        "idarticu",
        "product_description",
        "sector_description",
        "customer_penetration",
        "basket_penetration",
        "line_share",
        "commonness_score",
    ]
    top = common.select([col for col in display_cols if col in common.columns]).head(top_n)
    lines.append(_markdown_table(top))
    lines.extend(
        [
            "",
            "## How To Use This",
            "",
            "- If common candidates dominate many baskets, test IDF/capped weighting or common-product masking in dev before changing the official pipeline.",
            "- Do not remove products solely because they are common; first confirm they reduce Stage 2 neighbor quality or Stage 4/6 tribe distinctiveness.",
            "- Use this report with Stage 3 embedding validation and Stage 8 product-lift profiles.",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_percentile(values: Any, percentile: float) -> float:
    array = [float(value) for value in values if value is not None]
    return float(pl.Series(array).quantile(percentile / 100.0)) if array else 0.0


def _markdown_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "_No rows._"
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
