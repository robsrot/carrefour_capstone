"""Dimensionality-reduction helpers used for baselines and visualization."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.progress import log_event, stage_timer
from src.utils import file_fingerprint, frame_to_numpy, numeric_feature_columns, should_use_cache, write_artifact_metadata


def build_pca_representation(
    feature_path: str | Path,
    output_path: str | Path | None = None,
    n_components: int | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.outputs / "features" / "feature_set_pca.parquet"
    requested_components = n_components or int(cfg.get("pca.n_components", 32))
    cache_metadata = {
        "stage": "pca_representation",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "n_components": requested_components,
        "random_seed": cfg.random_seed,
    }
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True), metadata=cache_metadata):
        return output

    df = pl.read_parquet(feature_path).sort("cliente")
    feature_cols = numeric_feature_columns(df)
    X = StandardScaler().fit_transform(frame_to_numpy(df, feature_cols))
    dims = min(requested_components, X.shape[1], X.shape[0] - 1)
    coords = PCA(n_components=dims, random_state=cfg.random_seed).fit_transform(X)
    out = {"cliente": df["cliente"].to_list()}
    for idx in range(coords.shape[1]):
        out[f"pca_{idx:03d}"] = coords[:, idx].astype("float32")
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out).write_parquet(output)
    write_artifact_metadata(output, cache_metadata)
    return output


def build_umap_representation(
    feature_path: str | Path,
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Build a high-dimensional UMAP representation intended for clustering candidates."""

    import umap
    from sklearn.preprocessing import StandardScaler

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.outputs / "features" / str(
        cfg.get("umap.output", "feature_set_umap_cluster.parquet")
    )
    umap_cfg = cfg.get("umap", {})
    n_components = int(cfg.get("umap.n_components", 20))
    n_neighbors = int(cfg.get("umap.n_neighbors", 30))
    min_dist = float(cfg.get("umap.min_dist", 0.0))
    metric = str(cfg.get("umap.metric", "cosine"))
    standardize_input = bool(cfg.get("umap.standardize_input", False))
    cache_metadata = {
        "stage": "umap_representation",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "umap": umap_cfg,
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
    ):
        df = pl.read_parquet(feature_path).sort("cliente")
        feature_cols = numeric_feature_columns(df)
        X = frame_to_numpy(df, feature_cols)
        if standardize_input:
            X = StandardScaler().fit_transform(X).astype("float32")
        else:
            X = X.astype("float32", copy=False)

        reducer = umap.UMAP(
            n_components=min(n_components, X.shape[1], X.shape[0] - 1),
            n_neighbors=min(n_neighbors, X.shape[0] - 1),
            min_dist=min_dist,
            metric=metric,
            random_state=cfg.random_seed,
            low_memory=bool(cfg.get("umap.low_memory", True)),
            n_jobs=int(cfg.get("umap.n_jobs", 1)),
        )
        coords = reducer.fit_transform(X).astype("float32")
        out = {"cliente": df["cliente"].to_list()}
        for idx in range(coords.shape[1]):
            out[f"umap_{idx:03d}"] = coords[:, idx]
        output.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(out).write_parquet(output)
        write_artifact_metadata(output, cache_metadata)
        log_event("Stage 6 UMAP", "wrote artifact", cfg=cfg, rows=df.height, dims=coords.shape[1], path=output)
    return output


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
