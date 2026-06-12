"""Commercial tribe profiling with product and sector lift evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.utils import collect_streaming, file_fingerprint, schema_names, should_use_cache, write_artifact_metadata


def _join_list(values: list[Any], fmt: str = "{}") -> str:
    return "; ".join(fmt.format(value) for value in values)


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
    output = Path(output_path) if output_path else cfg.data_processed / f"tribe_profiles_{stem}.parquet"
    candidate_behavior_path = Path(behavior_path) if behavior_path else cfg.artifact_path("behavioral_features", "output")
    cache_metadata = {
        "stage": "tribe_profiles",
        "mode": cfg.mode,
        "assignments": file_fingerprint(assignments_file),
        "prepared_transactions": file_fingerprint(cfg.prepared_transactions_path),
        "behavior": file_fingerprint(candidate_behavior_path),
        "profiling": cfg.get("profiling", {}),
        "population_definition": "all_assigned_customers_including_noise",
    }
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata):
        return output

    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    columns = set(schema_names(lf))
    product_cols = ["cliente", "idarticu"]
    for col in ["desc_larga_articulo", "idsector", "desc_sector", "importe", "unidades", "ticket"]:
        if col in columns:
            product_cols.append(col)

    assignments_all = (
        pl.scan_parquet(assignments_file)
        .select(["cliente", "tribe_id"])
        .unique(subset=["cliente"], keep="first")
    )
    clustered_assignments = assignments_all.filter(pl.col("tribe_id") >= 0)
    assignment_customers = assignments_all.select("cliente")
    population = lf.select(product_cols).join(assignment_customers, on="cliente", how="inner")
    clustered = population.join(clustered_assignments, on="cliente", how="inner")

    cluster_sizes = collect_streaming(
        clustered_assignments.group_by("tribe_id").agg(pl.col("cliente").n_unique().alias("n_customers")).sort("tribe_id")
    )
    total_customers = int(
        collect_streaming(assignments_all.select(pl.col("cliente").n_unique().alias("n_customers")))[0, "n_customers"]
    )
    clustered_customers = int(cluster_sizes["n_customers"].sum()) if cluster_sizes.height else 0
    noise_customers = max(total_customers - clustered_customers, 0)

    customer_product = clustered.select(
        ["tribe_id", "cliente", "idarticu"]
        + [col for col in ["desc_larga_articulo", "desc_sector"] if col in product_cols]
    ).unique(subset=["tribe_id", "cliente", "idarticu"])
    population_customer_product = population.select(
        ["cliente", "idarticu"] + [col for col in ["desc_larga_articulo", "desc_sector"] if col in product_cols]
    ).unique(subset=["cliente", "idarticu"])
    population_product = population_customer_product.group_by("idarticu").agg(
        [
            pl.col("cliente").n_unique().alias("population_customers"),
            pl.col("desc_larga_articulo").first().alias("product_description")
            if "desc_larga_articulo" in product_cols
            else pl.lit(None).alias("product_description"),
            pl.col("desc_sector").first().alias("sector_description")
            if "desc_sector" in product_cols
            else pl.lit(None).alias("sector_description"),
        ]
    )
    cluster_product = customer_product.group_by(["tribe_id", "idarticu"]).agg(
        pl.col("cliente").n_unique().alias("cluster_customers")
    )
    product_lifts = collect_streaming(
        cluster_product.join(population_product, on="idarticu", how="left")
        .join(cluster_sizes.lazy(), on="tribe_id", how="left")
        .with_columns(
            [
                (pl.col("cluster_customers") / pl.col("n_customers")).alias("cluster_rate"),
                (pl.col("population_customers") / max(total_customers, 1)).alias("population_rate"),
            ]
        )
        .with_columns((pl.col("cluster_rate") / pl.col("population_rate")).alias("lift"))
        .filter(pl.col("cluster_customers") >= int(cfg.get("profiling.min_product_customers", 10)))
        .sort(["tribe_id", "lift", "cluster_customers"], descending=[False, True, True])
    )

    if "desc_sector" in product_cols:
        population_sector = population.group_by("desc_sector").agg(pl.len().alias("population_lines"))
        cluster_sector_totals = clustered.group_by("tribe_id").agg(pl.len().alias("cluster_lines_total"))
        population_total = collect_streaming(population.select(pl.len().alias("n_lines")))[0, "n_lines"]
        sector_lifts = collect_streaming(
            clustered.group_by(["tribe_id", "desc_sector"])
            .agg(pl.len().alias("cluster_lines"))
            .join(population_sector, on="desc_sector", how="left")
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

    behavior_summary = None
    if candidate_behavior_path.exists():
        behavior = pl.scan_parquet(candidate_behavior_path)
        behavior_cols = [col for col in schema_names(behavior) if col != "cliente"]
        aggregations = [pl.col(col).mean().alias(f"avg_{col}") for col in behavior_cols if col != "tribe_id"]
        behavior_summary = collect_streaming(
            clustered_assignments.join(behavior, on="cliente", how="left").group_by("tribe_id").agg(aggregations)
        )

    rows = []
    for cluster_row in cluster_sizes.iter_rows(named=True):
        tribe_id = int(cluster_row["tribe_id"])
        products = product_lifts.filter(pl.col("tribe_id") == tribe_id).head(int(cfg.get("profiling.top_n_products", 15)))
        sectors = sector_lifts.filter(pl.col("tribe_id") == tribe_id).head(int(cfg.get("profiling.top_n_sectors", 10)))
        row: dict[str, Any] = {
            "tribe_id": tribe_id,
            "n_customers": int(cluster_row["n_customers"]),
            "population_share": float(cluster_row["n_customers"] / max(total_customers, 1)),
            "profile_population": "all_assigned_customers_including_noise",
            "profile_population_customers": total_customers,
            "profile_noise_customers": noise_customers,
            "top_product_ids": products["idarticu"].to_list() if products.height else [],
            "top_products": products["product_description"].to_list() if "product_description" in products.columns else [],
            "top_product_lifts": products["lift"].round(3).to_list() if products.height else [],
            "top_product_customer_counts": products["cluster_customers"].to_list() if products.height else [],
            "top_sectors": sectors["desc_sector"].to_list() if "desc_sector" in sectors.columns else [],
            "top_sector_lifts": sectors["lift"].round(3).to_list() if sectors.height else [],
        }
        if behavior_summary is not None:
            behavior_match = behavior_summary.filter(pl.col("tribe_id") == tribe_id)
            if behavior_match.height:
                for col in behavior_match.columns:
                    if col != "tribe_id":
                        row[col] = behavior_match[0, col]
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
                "top_product_ids": pl.List(pl.Utf8),
                "top_products": pl.List(pl.Utf8),
                "top_product_lifts": pl.List(pl.Float64),
                "top_product_customer_counts": pl.List(pl.Int64),
                "top_sectors": pl.List(pl.Utf8),
                "top_sector_lifts": pl.List(pl.Float64),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    profiles.write_parquet(output)
    write_artifact_metadata(output, cache_metadata)
    return output


def flatten_profiles_for_csv(profile_path: str | Path, output_csv: str | Path) -> Path:
    profiles = pl.read_parquet(profile_path)
    rows = []
    for row in profiles.iter_rows(named=True):
        flat = dict(row)
        for key in ["top_products", "top_product_ids", "top_product_lifts", "top_sectors", "top_sector_lifts"]:
            if isinstance(flat.get(key), list):
                flat[key] = _join_list(flat[key])
        rows.append(flat)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(output)
    return output


def profile_quality_summary(profile_path: str | Path, cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    profiles = pl.read_parquet(profile_path)
    strong_product_lift = float(cfg.get("profiling.strong_product_lift_threshold", 1.5))
    strong_sector_lift = float(cfg.get("profiling.strong_sector_lift_threshold", 1.2))
    product_lift_counts = []
    sector_lift_counts = []
    max_product_lifts = []
    for row in profiles.iter_rows(named=True):
        product_lifts = row.get("top_product_lifts") or []
        sector_lifts = row.get("top_sector_lifts") or []
        product_lift_counts.append(sum(1 for value in product_lifts if value and value >= strong_product_lift))
        sector_lift_counts.append(sum(1 for value in sector_lifts if value and value >= strong_sector_lift))
        max_product_lifts.append(max(product_lifts) if product_lifts else 0.0)
    return {
        "profiled_clusters": profiles.height,
        "clusters_with_product_lift": sum(1 for count in product_lift_counts if count > 0),
        "clusters_with_sector_lift": sum(1 for count in sector_lift_counts if count > 0),
        "avg_strong_product_lifts_per_cluster": float(sum(product_lift_counts) / max(profiles.height, 1)),
        "avg_max_product_lift": float(sum(max_product_lifts) / max(profiles.height, 1)),
    }
