"""Dimensionality-reduction helpers used for baselines and visualization."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.utils import frame_to_numpy, numeric_feature_columns, should_use_cache


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
    output = Path(output_path) if output_path else cfg.data_processed / "feature_set_pca.parquet"
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True)):
        return output

    df = pl.read_parquet(feature_path).sort("cliente")
    feature_cols = numeric_feature_columns(df)
    X = StandardScaler().fit_transform(frame_to_numpy(df, feature_cols))
    dims = min(n_components or int(cfg.get("pca.n_components", 32)), X.shape[1], X.shape[0] - 1)
    coords = PCA(n_components=dims, random_state=cfg.random_seed).fit_transform(X)
    out = {"cliente": df["cliente"].to_list()}
    for idx in range(coords.shape[1]):
        out[f"pca_{idx:03d}"] = coords[:, idx].astype("float32")
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out).write_parquet(output)
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
    output = Path(output_path) if output_path else cfg.data_processed / "umap_visualization.parquet"
    if should_use_cache(output, force=force, use_cached=cfg.get("cache.use_cached", True)):
        return output

    df = pl.read_parquet(feature_path).sort("cliente")
    feature_cols = numeric_feature_columns(df)
    X = StandardScaler().fit_transform(frame_to_numpy(df, feature_cols))
    coords = umap.UMAP(n_components=2, random_state=cfg.random_seed, n_jobs=1).fit_transform(X)
    pl.DataFrame(
        {
            "cliente": df["cliente"].to_list(),
            "x": coords[:, 0].astype("float32"),
            "y": coords[:, 1].astype("float32"),
        }
    ).write_parquet(output)
    return output
