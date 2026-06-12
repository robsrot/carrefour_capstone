"""Reusable visualization functions for model and tribe review."""

from __future__ import annotations

import math
from pathlib import Path
from textwrap import shorten

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.progress import log_event, stage_timer
from src.utils import deterministic_sample_indices, frame_to_numpy, numeric_feature_columns


def _fit_umap_2d(X: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    import umap
    from sklearn.preprocessing import StandardScaler

    if bool(cfg.get("visualization.umap.standardize_input", False)):
        X = StandardScaler().fit_transform(X).astype("float32")
    else:
        X = X.astype("float32", copy=False)

    return umap.UMAP(
        n_components=2,
        n_neighbors=min(int(cfg.get("visualization.umap.n_neighbors", 30)), max(2, X.shape[0] - 1)),
        min_dist=float(cfg.get("visualization.umap.min_dist", 0.05)),
        metric=str(cfg.get("visualization.umap.metric", "cosine")),
        random_state=int(cfg.get("visualization.random_state", cfg.random_seed)),
        n_jobs=int(cfg.get("visualization.umap.n_jobs", 1)),
    ).fit_transform(X)


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
    log_event("Stage 8 figures", "wrote model comparison plot", cfg=cfg, path=output)
    return output


def plot_stage6_model_diagnostics(
    diagnostics_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot all Stage 6 candidate rows before product-lift profiling."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    diagnostics = (
        pl.read_csv(diagnostics_path) if str(diagnostics_path).lower().endswith(".csv") else pl.read_parquet(diagnostics_path)
    )
    output = Path(output_path) if output_path else cfg.figures / "stage6_candidate_model_diagnostics.png"
    pdf = diagnostics.to_pandas()
    if pdf.empty:
        raise ValueError(f"No Stage 6 diagnostics rows found in {diagnostics_path}")

    metric = "coverage_adjusted_silhouette" if "coverage_adjusted_silhouette" in pdf.columns else "silhouette"
    pdf[metric] = pdf[metric].fillna(-0.05).astype(float)
    pdf["cluster_count"] = pdf["cluster_count"].fillna(0).astype(float)
    if "coverage_pct" not in pdf.columns:
        pdf["coverage_pct"] = 100.0
    pdf["coverage_pct"] = pdf["coverage_pct"].fillna(100.0).astype(float)
    if "passes_quality_gate" not in pdf.columns:
        pdf["passes_quality_gate"] = False
    pdf["passes_quality_gate"] = pdf["passes_quality_gate"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "selected_within_family" not in pdf.columns:
        pdf["selected_within_family"] = False
    pdf["candidate_label"] = pdf["model_name"].astype(str) + "\n" + pdf["model_variant"].astype(str)
    pdf = pdf.sort_values(["stage6_rank", "candidate_label"]) if "stage6_rank" in pdf.columns else pdf

    models = list(dict.fromkeys(pdf["model_name"].astype(str)))
    palette = plt.get_cmap("tab10")
    colors = {model: palette(idx % 10) for idx, model in enumerate(models)}

    fig, (ax_scatter, ax_ranked) = plt.subplots(
        1,
        2,
        figsize=(15, max(5.5, min(9.5, 4.5 + len(models) * 0.22))),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    for model in models:
        group = pdf[pdf["model_name"].astype(str) == model]
        for passes, marker, alpha, label_suffix in [
            (True, "o", 0.9, "pass"),
            (False, "x", 0.55, "review"),
        ]:
            subset = group[group["passes_quality_gate"] == passes]
            if subset.empty:
                continue
            sizes = 28 + subset["coverage_pct"].clip(lower=0, upper=100) * 0.9
            ax_scatter.scatter(
                subset["cluster_count"],
                subset[metric],
                s=sizes,
                marker=marker,
                color=colors[model],
                alpha=alpha,
                label=f"{model} ({label_suffix})",
                linewidths=1.2,
            )

    selected = pdf[pdf["selected_within_family"].astype(str).str.lower().isin(["true", "1", "yes"])]
    for _, row in selected.iterrows():
        ax_scatter.annotate(
            str(row["model_variant"]),
            (row["cluster_count"], row[metric]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#262626",
        )

    ax_scatter.set_xlabel("Cluster count")
    ax_scatter.set_ylabel(metric.replace("_", " ").title())
    ax_scatter.set_title("Stage 6 Candidate Landscape")
    ax_scatter.grid(alpha=0.25)
    ax_scatter.legend(loc="best", fontsize=7, frameon=False)

    top = pdf.head(min(18, len(pdf))).iloc[::-1]
    bar_colors = [colors[str(model)] for model in top["model_name"]]
    ax_ranked.barh(range(len(top)), top[metric], color=bar_colors, alpha=0.88)
    ax_ranked.set_yticks(range(len(top)))
    ax_ranked.set_yticklabels(top["candidate_label"], fontsize=7)
    ax_ranked.set_xlabel(metric.replace("_", " ").title())
    ax_ranked.set_title("Top Stage 6 Candidates")
    ax_ranked.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Stage 6 diagnostics", "wrote candidate diagnostics plot", cfg=cfg, path=output)
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
    log_event("Stage 8 figures", "wrote cluster size plot", cfg=cfg, path=output)
    return output


def plot_candidate_umap_grid(
    feature_path: str | Path,
    assignment_paths: dict[str, str | Path],
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create a shared UMAP map with one panel per candidate assignment."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage6_candidate_umap_assignment_grid.png"
    if not assignment_paths:
        raise ValueError("No assignment paths were provided for candidate UMAP grid.")

    with stage_timer("Stage 8 figures", "building candidate UMAP assignment grid", cfg=cfg, candidates=len(assignment_paths)):
        features = pl.read_parquet(feature_path).sort("cliente")
        feature_cols = numeric_feature_columns(features, exclude=("cliente",))
        sample_idx = deterministic_sample_indices(
            features.height,
            int(cfg.get("visualization.max_scatter_points", 50000)),
            int(cfg.get("visualization.random_state", cfg.random_seed)),
        )
        sampled_features = features[sample_idx]
        coords = _fit_umap_2d(frame_to_numpy(sampled_features, feature_cols), cfg)
        projection = pl.DataFrame(
            {
                "cliente": sampled_features["cliente"].to_list(),
                "umap_x": coords[:, 0].astype("float32"),
                "umap_y": coords[:, 1].astype("float32"),
            }
        )

        items = sorted(assignment_paths.items(), key=lambda item: item[0])
        ncols = 2 if len(items) <= 4 else 3
        nrows = math.ceil(len(items) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.2 * ncols, 4.7 * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for ax, (candidate_key, assignment_path) in zip(axes_flat, items):
            assignment_frame = pl.read_parquet(assignment_path)
            assignments = assignment_frame.select(
                [col for col in ["cliente", "tribe_id", "model_name", "model_variant"] if col in assignment_frame.columns]
            )
            joined = projection.join(assignments, on="cliente", how="inner")
            pdf = joined.to_pandas()
            labels = pdf["tribe_id"].to_numpy(dtype=np.int32)
            _scatter_labels(ax, pdf["umap_x"].to_numpy(), pdf["umap_y"].to_numpy(), labels)
            title = candidate_key.replace("::", "\n")
            ax.set_title(shorten(title, width=64, placeholder="..."), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        for ax in axes_flat[len(items) :]:
            ax.axis("off")

        fig.suptitle("Candidate Assignments on Shared UMAP Customer Manifold", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=170)
        plt.close(fig)
        log_event("Stage 8 figures", "wrote candidate UMAP grid", cfg=cfg, path=output)
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
    log_event("Stage 8 figures", "wrote lift plots", cfg=cfg, plots=len(paths), output_dir=out_dir)
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
    with stage_timer("Stage 8 figures", "building 2D projection figures", cfg=cfg):
        features = pl.read_parquet(feature_path).sort("cliente")
        assignments = pl.read_parquet(assignments_path).select(["cliente", "tribe_id"])
        joined = features.join(assignments, on="cliente", how="inner")
        feature_cols = numeric_feature_columns(joined, exclude=("cliente", "tribe_id"))
        sample_idx = deterministic_sample_indices(
            joined.height,
            int(cfg.get("visualization.max_scatter_points", 50000)),
            int(cfg.get("visualization.random_state", cfg.random_seed)),
        )
        sampled = joined[sample_idx]
        X = frame_to_numpy(sampled, feature_cols)
        labels = sampled["tribe_id"].to_numpy()

        stem = Path(assignments_path).stem.replace("cluster_assignments_", "")
        outputs: dict[str, Path] = {}
        pca = PCA(n_components=2, random_state=int(cfg.get("visualization.random_state", cfg.random_seed)))
        coords = pca.fit_transform(X)
        outputs["pca"] = _scatter(
            coords,
            labels,
            cfg.figures / f"pca_selected_tribes_{stem}.png",
            f"PCA Selected Tribes - {stem}",
        )

        try:
            coords = _fit_umap_2d(X, cfg)
            outputs["umap"] = _scatter(
                coords,
                labels,
                cfg.figures / f"umap_selected_tribes_{stem}.png",
                f"UMAP Selected Tribes - {stem}",
            )
        except Exception as exc:
            log_event("Stage 8 figures", "UMAP figure skipped", cfg=cfg, reason=type(exc).__name__)
        plt.close("all")
        log_event("Stage 8 figures", "wrote projection figures", cfg=cfg, plots=len(outputs))
    return outputs


def _scatter(coords: np.ndarray, labels: np.ndarray, output: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6))
    _scatter_labels(ax, coords[:, 0], coords[:, 1], labels)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _scatter_labels(ax, x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> None:
    labels = np.asarray(labels).astype(np.int32)
    noise = labels < 0
    if noise.any():
        ax.scatter(x[noise], y[noise], s=4, color="#9ca3af", alpha=0.35, linewidths=0, label="noise")
    valid = ~noise
    if valid.any():
        ax.scatter(
            x[valid],
            y[valid],
            c=labels[valid],
            s=4,
            cmap="tab20",
            alpha=0.78,
            linewidths=0,
        )
