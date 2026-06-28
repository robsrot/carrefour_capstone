"""Dimensionality-reduction helpers used for baselines and visualization."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.progress import log_event, stage_timer
from src.utils import (
    collect_streaming,
    deterministic_sample_indices,
    file_fingerprint,
    frame_to_numpy,
    numeric_feature_columns,
    should_use_cache,
    write_artifact_metadata,
)


def build_pca_representation(
    feature_path: str | Path,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    n_components: int | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.outputs / "features" / "feature_set_pca.parquet"
    summary_output = Path(summary_path) if summary_path else output.with_name(f"{output.stem}_summary.csv")
    requested_components = n_components or int(cfg.get("pca.n_components", 32))
    cache_metadata = {
        "stage": "pca_representation",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "summary_path": str(summary_output),
        "n_components": requested_components,
        "random_seed": cfg.random_seed,
    }
    summary_metadata = {**cache_metadata, "artifact": "pca_summary"}
    if (
        should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata)
        and should_use_cache(
            summary_output,
            force=force,
            use_cached=cfg.get("cache.use_cached", True),
            metadata=summary_metadata,
        )
    ):
        log_event("Stage 6 PCA", "cache hit", cfg=cfg, path=output, summary=summary_output)
        return output

    source_rows = _parquet_row_count(feature_path)
    schema_df = pl.read_parquet(feature_path, n_rows=1)
    feature_cols = numeric_feature_columns(schema_df)
    log_event(
        "Stage 6 PCA",
        "loading dense feature matrix",
        cfg=cfg,
        source_rows=source_rows,
        feature_count=len(feature_cols),
        approx_float32_mb=round(source_rows * max(len(feature_cols), 1) * 4 / (1024**2), 1),
    )
    df = pl.read_parquet(feature_path).sort("cliente")
    X = StandardScaler(copy=False).fit_transform(frame_to_numpy(df, feature_cols)).astype(np.float32, copy=False)
    dims = min(requested_components, X.shape[1], X.shape[0] - 1)
    if dims < 1:
        raise ValueError("PCA requires at least one numeric feature and at least two rows.")
    with stage_timer(
        "Stage 6 PCA",
        "fitting pre-UMAP PCA",
        cfg=cfg,
        source_rows=df.height,
        input_features=len(feature_cols),
        retained_components=dims,
    ):
        pca = PCA(n_components=dims, random_state=cfg.random_seed)
        coords = pca.fit_transform(X)
    out = {"cliente": df["cliente"].to_list()}
    for idx in range(coords.shape[1]):
        out[f"pca_{idx:03d}"] = coords[:, idx].astype("float32")
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out).write_parquet(output)
    _write_pca_summary(
        summary_output,
        feature_path=feature_path,
        output_path=output,
        feature_count=len(feature_cols),
        requested_components=requested_components,
        actual_components=dims,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    write_artifact_metadata(output, cache_metadata)
    write_artifact_metadata(summary_output, summary_metadata)
    log_event(
        "Stage 6 PCA",
        "wrote artifact",
        cfg=cfg,
        rows=df.height,
        retained_components=dims,
        retained_variance_pct=round(float(pca.explained_variance_ratio_.sum() * 100.0), 2),
        path=output,
    )
    return output


def _write_pca_summary(
    output: Path,
    *,
    feature_path: str | Path,
    output_path: str | Path,
    feature_count: int,
    requested_components: int,
    actual_components: int,
    explained_variance_ratio: np.ndarray,
) -> None:
    explained = np.asarray(explained_variance_ratio, dtype=np.float64)
    retained = float(explained.sum())
    row = {
        "stage": "pca_for_umap",
        "purpose": "Pre-reduce standardized customer product-behavior features before UMAP to denoise the neighbor graph and keep UMAP tractable.",
        "feature_path": str(feature_path),
        "pca_path": str(output_path),
        "input_dimension_count": int(feature_count),
        "requested_component_count": int(requested_components),
        "retained_component_count": int(actual_components),
        "dimension_reduction": f"{int(feature_count)} -> {int(actual_components)}",
        "standardized_input": True,
        "retained_variance_ratio": retained,
        "retained_variance_pct": retained * 100.0,
        "pc1_variance_pct": float(explained[0] * 100.0) if explained.size else None,
        "pc5_cumulative_variance_pct": float(explained[:5].sum() * 100.0) if explained.size else None,
        "pc10_cumulative_variance_pct": float(explained[:10].sum() * 100.0) if explained.size else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_csv(output)


def build_umap_representation(
    feature_path: str | Path,
    output_path: str | Path | None = None,
    umap_overrides: dict | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Build a high-dimensional UMAP representation intended for clustering candidates."""

    import umap
    from sklearn.preprocessing import StandardScaler

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    umap_cfg = {**(cfg.get("umap", {}) or {}), **(umap_overrides or {})}
    output = Path(output_path) if output_path else cfg.outputs / "features" / str(
        umap_cfg.get("output", "feature_set_umap_cluster.parquet")
    )
    n_components = int(umap_cfg.get("n_components", 20))
    n_neighbors = int(umap_cfg.get("n_neighbors", 30))
    min_dist = float(umap_cfg.get("min_dist", 0.0))
    metric = str(umap_cfg.get("metric", "cosine"))
    standardize_input = bool(umap_cfg.get("standardize_input", False))
    row_count = _parquet_row_count(feature_path)
    fit_sample_size = _umap_fit_sample_size(umap_cfg, row_count, cfg)
    transform_batch_size = int(umap_cfg.get("transform_batch_size", 100000))
    cache_metadata = {
        "stage": "umap_representation",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "umap": umap_cfg,
        "resolved_fit_sample_size": fit_sample_size,
        "source_rows": row_count,
        "transform_batch_size": transform_batch_size,
        "random_seed": cfg.random_seed,
        "purpose": "clustering_candidate",
    }
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata):
        log_event("Stage 6 UMAP", "cache hit", cfg=cfg, path=output)
        return output

    with stage_timer(
        "Stage 6 UMAP",
        "building clustering representation",
        cfg=cfg,
        components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        source_rows=row_count,
        fit_sample_size=fit_sample_size,
        transform_batch_size=transform_batch_size,
    ):
        schema_df = pl.read_parquet(feature_path, n_rows=1)
        feature_cols = numeric_feature_columns(schema_df)
        log_event(
            "Stage 6 UMAP",
            "resolved feature matrix policy",
            cfg=cfg,
            feature_count=len(feature_cols),
            dense_fit_float32_mb=round(fit_sample_size * max(len(feature_cols), 1) * 4 / (1024**2), 1),
            full_transform_batched=fit_sample_size < row_count,
        )
        fit_indices = deterministic_sample_indices(row_count, fit_sample_size, cfg.random_seed)
        if fit_indices.shape[0] >= row_count:
            df = pl.read_parquet(feature_path).sort("cliente")
            X = frame_to_numpy(df, feature_cols)
            if standardize_input:
                X = StandardScaler(copy=False).fit_transform(X).astype("float32", copy=False)
            else:
                X = X.astype("float32", copy=False)

            reducer = umap.UMAP(
                n_components=min(n_components, X.shape[1], X.shape[0] - 1),
                n_neighbors=min(n_neighbors, X.shape[0] - 1),
                min_dist=min_dist,
                metric=metric,
                random_state=cfg.random_seed,
                low_memory=bool(umap_cfg.get("low_memory", True)),
                n_jobs=int(umap_cfg.get("n_jobs", 1)),
            )
            coords = reducer.fit_transform(X).astype("float32")
            out = {"cliente": df["cliente"].to_list()}
            for idx in range(coords.shape[1]):
                out[f"umap_{idx:03d}"] = coords[:, idx]
            output.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(out).write_parquet(output)
        else:
            sample = _collect_feature_rows_by_index(feature_path, feature_cols, fit_indices)
            X_fit = frame_to_numpy(sample, feature_cols)
            scaler = None
            if standardize_input:
                scaler = StandardScaler(copy=False)
                X_fit = scaler.fit_transform(X_fit).astype("float32", copy=False)
            else:
                X_fit = X_fit.astype("float32", copy=False)

            reducer = umap.UMAP(
                n_components=min(n_components, X_fit.shape[1], X_fit.shape[0] - 1),
                n_neighbors=min(n_neighbors, X_fit.shape[0] - 1),
                min_dist=min_dist,
                metric=metric,
                random_state=cfg.random_seed,
                low_memory=bool(umap_cfg.get("low_memory", True)),
                n_jobs=int(umap_cfg.get("n_jobs", 1)),
            )
            reducer.fit(X_fit)
            _write_umap_transformed_batches(
                feature_path,
                feature_cols,
                reducer,
                output,
                standardize_input=standardize_input,
                scaler=scaler,
                batch_size=transform_batch_size,
                row_count=row_count,
                cfg=cfg,
            )
        write_artifact_metadata(output, cache_metadata)
        log_event(
            "Stage 6 UMAP",
            "wrote artifact",
            cfg=cfg,
            rows=row_count,
            fit_sample_rows=int(fit_indices.shape[0]),
            path=output,
        )
    return output


def _parquet_row_count(path: str | Path) -> int:
    return int(collect_streaming(pl.scan_parquet(path).select(pl.len().alias("n_rows")))[0, "n_rows"])


def _umap_fit_sample_size(umap_cfg: dict[str, Any], row_count: int, cfg: PipelineConfig) -> int:
    raw_value = umap_cfg.get("fit_sample_size", None)
    if raw_value is None:
        raw_value = cfg.get("modeling.fit_sample_size", None)
    if raw_value is None:
        return row_count
    sample_size = int(raw_value)
    if sample_size <= 0:
        return row_count
    return min(sample_size, row_count)


def _collect_feature_rows_by_index(
    feature_path: str | Path,
    feature_cols: list[str],
    indices: np.ndarray,
) -> pl.DataFrame:
    return (
        collect_streaming(
            pl.scan_parquet(feature_path)
            .with_row_index("_row_idx")
            .filter(pl.col("_row_idx").is_in([int(value) for value in indices]))
            .select(["_row_idx", "cliente", *feature_cols])
            .sort("_row_idx")
        )
        .drop("_row_idx")
    )


def _write_umap_transformed_batches(
    feature_path: str | Path,
    feature_cols: list[str],
    reducer: Any,
    output: Path,
    *,
    standardize_input: bool,
    scaler: Any,
    batch_size: int,
    row_count: int,
    cfg: PipelineConfig,
) -> None:
    batch_size = max(int(batch_size), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output.parent / f".{output.stem}_umap_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    try:
        part_index = 0
        for offset in range(0, row_count, batch_size):
            batch = collect_streaming(
                pl.scan_parquet(feature_path)
                .slice(offset, batch_size)
                .select(["cliente", *feature_cols])
            )
            if batch.is_empty():
                continue
            X_batch = frame_to_numpy(batch, feature_cols)
            if standardize_input and scaler is not None:
                X_batch = scaler.transform(X_batch).astype("float32")
            else:
                X_batch = X_batch.astype("float32", copy=False)
            coords = reducer.transform(X_batch).astype("float32")
            out = {"cliente": batch["cliente"].to_list()}
            for idx in range(coords.shape[1]):
                out[f"umap_{idx:03d}"] = coords[:, idx]
            pl.DataFrame(out).write_parquet(parts_dir / f"part_{part_index:05d}.parquet")
            part_index += 1
            log_event(
                "Stage 6 UMAP",
                "transformed batch",
                cfg=cfg,
                rows=batch.height,
                offset=offset,
                total_rows=row_count,
            )
        pl.scan_parquet(str(parts_dir / "part_*.parquet")).sink_parquet(str(output))
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)


def build_umap_visualization(
    feature_path: str | Path,
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    import umap
    from sklearn.preprocessing import StandardScaler

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.outputs / "features" / "umap_visualization.parquet"
    cache_metadata = {
        "stage": "umap_visualization",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "n_components": 2,
        "random_seed": cfg.random_seed,
        "n_jobs": 1,
    }
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata):
        return output

    df = pl.read_parquet(feature_path).sort("cliente")
    feature_cols = numeric_feature_columns(df)
    X = StandardScaler().fit_transform(frame_to_numpy(df, feature_cols))
    coords = umap.UMAP(n_components=2, random_state=cfg.random_seed, n_jobs=1).fit_transform(X)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "cliente": df["cliente"].to_list(),
            "x": coords[:, 0].astype("float32"),
            "y": coords[:, 1].astype("float32"),
        }
    ).write_parquet(output)
    write_artifact_metadata(output, cache_metadata)
    return output
