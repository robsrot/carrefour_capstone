"""Customer behavioral features and feature-set assembly."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
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


def build_feature_set(
    customer_embeddings_path: str | Path,
    behavior_path: str | Path | None = None,
    variant: str = "embeddings_only",
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create model-ready feature variants A and B."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
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
        "standardize_behavior": bool(cfg.get("feature_sets.standardize_behavior", True)),
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
                updates = []
                for col in behavior_cols:
                    values = result[col].to_numpy().astype(np.float64)
                    mean = float(np.nanmean(values))
                    std = float(np.nanstd(values))
                    denom = std if std > 1e-12 else 1.0
                    updates.append(((pl.col(col) - mean) / denom).cast(pl.Float32).alias(f"beh_{col}"))
                result = result.with_columns(updates).drop(behavior_cols)
            result = result.sort("cliente")
        else:
            raise ValueError(f"Unknown feature set variant: {variant}")

        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output)
        write_artifact_metadata(output, cache_metadata)
        log_event("Stage 5 feature set", "wrote artifact", cfg=cfg, variant=variant, rows=result.height, path=output)
    return output


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
    selection_warning = (
        "OK: official selection uses product embeddings only."
        if selection_feature_set == baseline
        else (
            f"WARNING: official selection uses {selection_feature_set!r}; confirm Stage 6/8 evidence "
            "before allowing behavior or auxiliary features to drive organic tribes."
        )
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
