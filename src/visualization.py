"""Reusable visualization functions for model and tribe review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.utils import deterministic_sample_indices, frame_to_numpy, numeric_feature_columns


def plot_model_comparison(
    comparison_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    comparison = pl.read_csv(comparison_path) if str(comparison_path).endswith(".csv") else pl.read_parquet(comparison_path)
    output = Path(output_path) if output_path else cfg.figures / "model_comparison.png"
    pdf = comparison.to_pandas()
    labels = pdf["model_name"].astype(str) + "\n" + pdf["model_variant"].astype(str)
    values = pdf["silhouette"].fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(pdf) * 1.6), 4.8))
    ax.bar(labels, values, color="#2b6f6d")
    ax.set_ylabel("Silhouette")
    ax.set_title("Candidate Model Comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_cluster_sizes(
    assignments_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    assignments = pl.read_parquet(assignments_path)
    sizes = assignments.group_by("tribe_id").agg(pl.len().alias("customers")).sort("tribe_id")
    output = Path(output_path) if output_path else cfg.figures / f"{Path(assignments_path).stem}_cluster_sizes.png"
    pdf = sizes.to_pandas()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#8a8f98" if tribe < 0 else "#3f6fb5" for tribe in pdf["tribe_id"]]
    ax.bar(pdf["tribe_id"].astype(str), pdf["customers"], color=colors)
    ax.set_xlabel("Tribe")
    ax.set_ylabel("Customers")
    ax.set_title("Cluster Size Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_top_lifts(
    profile_path: str | Path,
    output_dir: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> list[Path]:
    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    profiles = pl.read_parquet(profile_path)
    out_dir = Path(output_dir) if output_dir else cfg.figures / "tribe_lifts"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in profiles.iter_rows(named=True):
        tribe_id = row["tribe_id"]
        products = row.get("top_products") or [str(pid) for pid in (row.get("top_product_ids") or [])]
        lifts = row.get("top_product_lifts") or []
        if not products or not lifts:
            continue
        products = [str(product)[:55] for product in products[:10]]
        lifts = lifts[:10]
        fig, ax = plt.subplots(figsize=(9, 5.2))
        y = np.arange(len(products))
        ax.barh(y, lifts, color="#7a4e8a")
        ax.set_yticks(y)
        ax.set_yticklabels(products)
        ax.invert_yaxis()
        ax.set_xlabel("Product Lift")
        ax.set_title(f"Top Lifted Products - Tribe {tribe_id}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = out_dir / f"tribe_{tribe_id}_top_product_lifts.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def build_2d_projection_figures(
    feature_path: str | Path,
    assignments_path: str | Path,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Create PCA and optional UMAP 2D figures for visual review only."""

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    cfg.ensure_directories()
    features = pl.read_parquet(feature_path).sort("cliente")
    assignments = pl.read_parquet(assignments_path).select(["cliente", "tribe_id"])
    joined = features.join(assignments, on="cliente", how="inner")
    feature_cols = numeric_feature_columns(joined, exclude=("cliente", "tribe_id"))
    sample_idx = deterministic_sample_indices(
        joined.height,
        int(cfg.get("visualization.max_scatter_points", 50000)),
        cfg.random_seed,
    )
    sampled = joined[sample_idx]
    X = frame_to_numpy(sampled, feature_cols)
    labels = sampled["tribe_id"].to_numpy()

    outputs: dict[str, Path] = {}
    pca = PCA(n_components=2, random_state=cfg.random_seed)
    coords = pca.fit_transform(X)
    outputs["pca"] = _scatter(coords, labels, cfg.figures / "pca_visualization.png", "PCA Visualization")

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=cfg.random_seed, n_jobs=1)
        coords = reducer.fit_transform(X)
        outputs["umap"] = _scatter(coords, labels, cfg.figures / "umap_visualization.png", "UMAP Visualization")
    except Exception:
        pass
    plt.close("all")
    return outputs


def _scatter(coords: np.ndarray, labels: np.ndarray, output: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=4, cmap="tab20", alpha=0.7, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.colorbar(scatter, ax=ax, shrink=0.8, label="Tribe")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
