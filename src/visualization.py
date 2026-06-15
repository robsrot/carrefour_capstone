"""Reusable visualization functions for model and tribe review."""

from __future__ import annotations

import math
from pathlib import Path
from textwrap import shorten
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.progress import log_event, stage_timer
from src.utils import collect_streaming, deterministic_sample_indices, frame_to_numpy, numeric_feature_columns, scan_if_path, schema_names


def _pca_nd(X: np.ndarray, n_components: int, cfg: PipelineConfig) -> np.ndarray:
    from sklearn.decomposition import PCA

    if X.shape[0] == 0:
        return np.empty((0, n_components), dtype=np.float32)
    actual_components = min(n_components, X.shape[0], X.shape[1])
    if actual_components <= 0:
        return np.zeros((X.shape[0], n_components), dtype=np.float32)
    coords = PCA(n_components=actual_components, random_state=cfg.random_seed).fit_transform(X)
    if actual_components < n_components:
        padding = np.zeros((X.shape[0], n_components - actual_components), dtype=coords.dtype)
        coords = np.column_stack([coords, padding])
    return coords.astype(np.float32, copy=False)


def _pca_2d(X: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    return _pca_nd(X, 2, cfg)


def _sample_frame(df: pl.DataFrame, max_rows: int, cfg: PipelineConfig) -> pl.DataFrame:
    idx = deterministic_sample_indices(df.height, int(max_rows), int(cfg.get("visualization.random_state", cfg.random_seed)))
    return df[idx]


def _promo_flag_expr() -> pl.Expr:
    promo = pl.col("idpromoc").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    no_promo_values = ["", "0", "none", "null", "nan", "no promo", "no_promo", "sin promo", "sin promocion"]
    return pl.when(promo.is_not_null() & (~promo.is_in(no_promo_values))).then(1).otherwise(0)


def _save_figure(fig, output: Path, cfg: PipelineConfig, stage: str, message: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    log_event(stage, message, cfg=cfg, path=output)
    return output


def plot_prepared_data_overview(
    transactions: pl.DataFrame | pl.LazyFrame | str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create a compact overview of the prepared transaction data consumed by Notebook 03."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    lf = scan_if_path(transactions)
    columns = set(schema_names(lf))
    output = Path(output_path) if output_path else cfg.figures / "stage0_prepared_data_overview.png"

    exprs = [pl.len().alias("ticket_lines")]
    if "cliente" in columns:
        exprs.append(pl.col("cliente").n_unique().alias("customers"))
    if "ticket" in columns:
        exprs.append(pl.col("ticket").n_unique().alias("baskets"))
    if "idarticu" in columns:
        exprs.append(pl.col("idarticu").n_unique().alias("products"))
    if "importe" in columns:
        exprs.append(pl.col("importe").cast(pl.Float64).sum().alias("revenue"))
    overview = collect_streaming(lf.select(exprs)).row(0, named=True)

    monthly = pl.DataFrame()
    if {"fecha", "ticket"}.issubset(columns):
        month_expr = pl.col("fecha").dt.truncate("1mo").alias("month")
        aggregations = [pl.col("ticket").n_unique().alias("baskets")]
        if "importe" in columns:
            aggregations.append(pl.col("importe").cast(pl.Float64).sum().alias("revenue"))
        monthly = collect_streaming(
            lf.with_columns(month_expr)
            .group_by("month")
            .agg(aggregations)
            .sort("month")
        )

    sector_col = "desc_sector" if "desc_sector" in columns else "idsector" if "idsector" in columns else None
    sectors = pl.DataFrame()
    if sector_col:
        sector_metric = pl.col("importe").cast(pl.Float64).sum().alias("value") if "importe" in columns else pl.len().alias("value")
        sectors = collect_streaming(
            lf.group_by(sector_col)
            .agg(sector_metric)
            .sort("value", descending=True)
            .head(10)
            .rename({sector_col: "sector"})
        )

    promo = pl.DataFrame()
    if "idpromoc" in columns:
        promo = collect_streaming(
            lf.with_columns(_promo_flag_expr().alias("_promo_flag"))
            .group_by("_promo_flag")
            .agg(pl.len().alias("lines"))
            .sort("_promo_flag")
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    ax_metrics, ax_monthly, ax_sector, ax_promo = axes.ravel()

    metric_lines = []
    for key, label in [
        ("ticket_lines", "Ticket lines"),
        ("customers", "Customers"),
        ("baskets", "Baskets"),
        ("products", "Products"),
        ("revenue", "Revenue"),
    ]:
        if key in overview and overview[key] is not None:
            value = overview[key]
            metric_lines.append(f"{label}: {value:,.0f}" if isinstance(value, (int, float)) else f"{label}: {value}")
    ax_metrics.text(0.02, 0.95, "\n".join(metric_lines), va="top", ha="left", fontsize=12)
    ax_metrics.set_title("Prepared Data Snapshot")
    ax_metrics.axis("off")

    if monthly.height:
        pdf = monthly.to_pandas()
        y_col = "revenue" if "revenue" in pdf.columns else "baskets"
        ax_monthly.plot(pdf["month"], pdf[y_col], color="#2b6f6d", linewidth=2)
        ax_monthly.fill_between(pdf["month"], pdf[y_col], color="#2b6f6d", alpha=0.15)
        ax_monthly.set_title(f"Monthly {y_col.title()}")
        ax_monthly.tick_params(axis="x", rotation=30)
        ax_monthly.grid(alpha=0.25)
    else:
        ax_monthly.text(0.5, 0.5, "No date field available", ha="center", va="center")
        ax_monthly.set_title("Monthly Trend")
        ax_monthly.axis("off")

    if sectors.height:
        pdf = sectors.to_pandas().iloc[::-1]
        ax_sector.barh(pdf["sector"].astype(str), pdf["value"], color="#4f7cac")
        ax_sector.set_title("Top Product Sectors")
        ax_sector.grid(axis="x", alpha=0.25)
    else:
        ax_sector.text(0.5, 0.5, "No sector field available", ha="center", va="center")
        ax_sector.set_title("Product Sectors")
        ax_sector.axis("off")

    if promo.height:
        promo_pdf = promo.to_pandas()
        labels = ["No promo" if int(flag) == 0 else "Promo" for flag in promo_pdf["_promo_flag"]]
        ax_promo.bar(labels, promo_pdf["lines"], color=["#7f8c8d", "#c65d3b"])
        ax_promo.set_title("Promo Line Mix")
        ax_promo.grid(axis="y", alpha=0.25)
    else:
        ax_promo.text(0.5, 0.5, "No promo field available", ha="center", va="center")
        ax_promo.set_title("Promo Mix")
        ax_promo.axis("off")

    fig.tight_layout()
    path = _save_figure(fig, output, cfg, "Stage 0 figures", "wrote prepared data overview")
    plt.close(fig)
    return path


def plot_basket_summary(
    basket_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Visualize ticket sentence lengths used for Item2Vec training."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage1_basket_sentence_lengths.png"
    baskets = pl.read_parquet(basket_path, columns=["n_product_tokens"])
    values = baskets["n_product_tokens"].to_numpy().astype(float)
    if len(values) == 0:
        raise ValueError(f"No basket rows found in {basket_path}")
    clip_max = max(1.0, float(np.nanpercentile(values, 99)))
    clipped = np.clip(values, 0, clip_max)

    fig, (ax_hist, ax_summary) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1.0]})
    ax_hist.hist(clipped, bins=min(40, max(5, int(clip_max))), color="#3f6fb5", alpha=0.85)
    ax_hist.set_title("Basket Sentence Lengths")
    ax_hist.set_xlabel("Unique product tokens per ticket")
    ax_hist.set_ylabel("Baskets")
    ax_hist.grid(axis="y", alpha=0.25)

    summary = {
        "Baskets": len(values),
        "Mean": float(np.mean(values)),
        "Median": float(np.median(values)),
        "P90": float(np.percentile(values, 90)),
        "P99": float(np.percentile(values, 99)),
        "Max": float(np.max(values)),
    }
    ax_summary.text(
        0.02,
        0.95,
        "\n".join(f"{key}: {value:,.2f}" if key != "Baskets" else f"{key}: {value:,.0f}" for key, value in summary.items()),
        va="top",
        ha="left",
        fontsize=12,
    )
    ax_summary.set_title("Basket Construction Summary")
    ax_summary.axis("off")

    fig.tight_layout()
    path = _save_figure(fig, output, cfg, "Stage 1 figures", "wrote basket summary")
    plt.close(fig)
    return path


def plot_product_embedding_diagnostics(
    embeddings_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot product embedding norm distribution and a sampled PCA map."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage2_product_embedding_diagnostics.png"
    embeddings = pl.read_parquet(embeddings_path).sort("idarticu")
    feature_cols = numeric_feature_columns(embeddings, exclude=("idarticu",))
    if not feature_cols:
        raise ValueError(f"No product embedding columns found in {embeddings_path}")

    X = frame_to_numpy(embeddings, feature_cols)
    norms = np.linalg.norm(X, axis=1)
    sample = _sample_frame(embeddings, int(cfg.get("visualization.max_scatter_points", 50000)), cfg)
    X_sample = frame_to_numpy(sample, feature_cols)
    coords = _pca_2d(X_sample, cfg)

    fig, (ax_norm, ax_pca) = plt.subplots(1, 2, figsize=(12, 5))
    ax_norm.hist(norms, bins=40, color="#8a5a44", alpha=0.85)
    ax_norm.set_title("Product Vector Norms")
    ax_norm.set_xlabel("L2 norm")
    ax_norm.set_ylabel("Products")
    ax_norm.grid(axis="y", alpha=0.25)

    ax_pca.scatter(coords[:, 0], coords[:, 1], s=5, color="#2b6f6d", alpha=0.55, linewidths=0)
    ax_pca.set_title("Product Embedding PCA Preview")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.grid(alpha=0.2)

    fig.tight_layout()
    path = _save_figure(fig, output, cfg, "Stage 2 figures", "wrote product embedding diagnostics")
    plt.close(fig)
    return path


def plot_embedding_validation_summary(
    validation_csv: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Summarize nearest-neighbor validation similarities and sector coherence."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage3_embedding_validation_summary.png"
    report = pl.read_csv(validation_csv)
    if report.height == 0:
        raise ValueError(f"No validation rows found in {validation_csv}")
    pdf = report.to_pandas()
    ranks = sorted(pdf["neighbor_rank"].dropna().unique())

    fig, (ax_box, ax_sector) = plt.subplots(1, 2, figsize=(12, 5))
    groups = [pdf.loc[pdf["neighbor_rank"] == rank, "cosine_similarity"].astype(float).to_numpy() for rank in ranks]
    ax_box.boxplot(groups, tick_labels=[str(rank) for rank in ranks], patch_artist=True)
    for patch in ax_box.patches:
        patch.set_facecolor("#4f7cac")
        patch.set_alpha(0.65)
    ax_box.set_title("Neighbor Similarity by Rank")
    ax_box.set_xlabel("Neighbor rank")
    ax_box.set_ylabel("Cosine similarity")
    ax_box.grid(axis="y", alpha=0.25)

    if {"product_sector", "neighbor_sector"}.issubset(pdf.columns):
        same = []
        for rank in ranks:
            subset = pdf[pdf["neighbor_rank"] == rank]
            valid = subset["product_sector"].notna() & subset["neighbor_sector"].notna()
            same.append(float((subset.loc[valid, "product_sector"] == subset.loc[valid, "neighbor_sector"]).mean() * 100.0) if valid.any() else 0.0)
        ax_sector.bar([str(rank) for rank in ranks], same, color="#c65d3b", alpha=0.82)
        ax_sector.set_ylabel("Same sector share (%)")
        ax_sector.set_ylim(0, 100)
    else:
        ax_sector.text(0.5, 0.5, "Sector fields unavailable", ha="center", va="center")
    ax_sector.set_title("Nearest-Neighbor Sector Coherence")
    ax_sector.set_xlabel("Neighbor rank")
    ax_sector.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = _save_figure(fig, output, cfg, "Stage 3 figures", "wrote embedding validation summary")
    plt.close(fig)
    return path


def plot_customer_embedding_diagnostics(
    customer_embeddings_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot customer-vector norm distribution and a sampled PCA preview."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage4_customer_embedding_diagnostics.png"
    embeddings = pl.read_parquet(customer_embeddings_path).sort("cliente")
    feature_cols = [col for col in embeddings.columns if col.startswith("emb_")]
    if not feature_cols:
        raise ValueError(f"No customer embedding columns found in {customer_embeddings_path}")

    sample = _sample_frame(embeddings, int(cfg.get("visualization.max_scatter_points", 50000)), cfg)
    X = frame_to_numpy(sample, feature_cols)
    norms = np.linalg.norm(X, axis=1)
    coords = _pca_2d(X, cfg)

    fig, (ax_norm, ax_pca) = plt.subplots(1, 2, figsize=(12, 5))
    finite_norms = norms[np.isfinite(norms)]
    if finite_norms.size == 0:
        raise ValueError(f"No finite customer-vector norms found in {customer_embeddings_path}")
    norm_min = float(np.min(finite_norms))
    norm_max = float(np.max(finite_norms))
    norm_mean = float(np.mean(finite_norms))
    if norm_max - norm_min <= 1e-5:
        ax_norm.axvline(norm_mean, color="#7a4e8a", linewidth=3)
        ax_norm.set_xlim(norm_mean - 0.01, norm_mean + 0.01)
        ax_norm.set_ylim(0, max(1, sample.height))
        ax_norm.text(
            0.5,
            0.72,
            f"All sampled vectors have\nL2 norm ~= {norm_mean:.3f}",
            transform=ax_norm.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0"},
        )
    else:
        ax_norm.hist(finite_norms, bins=40, color="#7a4e8a", alpha=0.82)
    ax_norm.set_title("Customer Vector Norms")
    ax_norm.set_xlabel("L2 norm")
    ax_norm.set_ylabel("Sampled customers")
    ax_norm.grid(axis="y", alpha=0.25)

    color = sample["embedded_unique_products"].to_numpy() if "embedded_unique_products" in sample.columns else norms
    scatter = ax_pca.scatter(coords[:, 0], coords[:, 1], c=color, s=4, cmap="viridis", alpha=0.6, linewidths=0)
    ax_pca.set_title("Customer Embedding PCA Preview")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax_pca, fraction=0.046, pad=0.04, label="Embedded unique products" if "embedded_unique_products" in sample.columns else "Vector norm")

    fig.tight_layout()
    path = _save_figure(fig, output, cfg, "Stage 4 figures", "wrote customer embedding diagnostics")
    plt.close(fig)
    return path


def plot_behavioral_feature_summary(
    behavior_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create a compact distribution dashboard for non-demographic behavioral features."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage5_behavioral_feature_summary.png"
    behavior = pl.read_parquet(behavior_path).sort("cliente")
    plot_cols = [
        col
        for col in [
            "ticket_count",
            "total_spend",
            "avg_basket_value",
            "promo_share",
            "unique_products",
            "unique_sectors",
            "recency_days",
            "frequency_per_30d",
        ]
        if col in behavior.columns
    ]
    if not plot_cols:
        raise ValueError(f"No expected behavioral feature columns found in {behavior_path}")

    sample = _sample_frame(behavior, int(cfg.get("visualization.max_scatter_points", 50000)), cfg)
    ncols = 4
    nrows = math.ceil(len(plot_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.3 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    log_scaled = {"ticket_count", "total_spend", "avg_basket_value", "unique_products", "unique_sectors", "recency_days", "frequency_per_30d"}

    for ax, col in zip(axes_flat, plot_cols):
        values = sample[col].to_numpy().astype(float)
        values = values[np.isfinite(values)]
        if col in log_scaled:
            values = np.log1p(np.maximum(values, 0))
            xlabel = f"log1p({col})"
        else:
            xlabel = col
        ax.hist(values, bins=35, color="#2b6f6d", alpha=0.82)
        ax.set_title(col)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", alpha=0.22)

    for ax in axes_flat[len(plot_cols) :]:
        ax.axis("off")

    fig.suptitle("Stage 5 Behavioral Feature Distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = _save_figure(fig, output, cfg, "Stage 5 figures", "wrote behavioral feature summary")
    plt.close(fig)
    return path


def plot_feature_set_summary(
    feature_paths: Mapping[str, str | Path],
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Compare feature-set dimensionality and PCA concentration."""

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage5_feature_set_summary.png"
    rows = []
    for name, path in feature_paths.items():
        df = pl.read_parquet(path).sort("cliente")
        feature_cols = numeric_feature_columns(df)
        if not feature_cols:
            continue
        sample = _sample_frame(df, int(cfg.get("visualization.max_scatter_points", 50000)), cfg)
        X = frame_to_numpy(sample, feature_cols)
        n_components = min(2, X.shape[0], X.shape[1])
        pc1_share = (
            float(PCA(n_components=n_components, random_state=cfg.random_seed).fit(X).explained_variance_ratio_[0])
            if n_components > 0
            else 0.0
        )
        rows.append(
            {
                "name": str(name),
                "rows": int(df.height),
                "feature_count": int(len(feature_cols)),
                "pc1_variance_share": pc1_share,
            }
        )
    if not rows:
        raise ValueError("No feature-set rows were available for plotting.")

    pdf = pl.DataFrame(rows).to_pandas()
    fig, (ax_count, ax_pc) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_count.bar(pdf["name"], pdf["feature_count"], color="#4f7cac", alpha=0.85)
    ax_count.set_title("Feature Count by Set")
    ax_count.set_ylabel("Numeric features")
    ax_count.tick_params(axis="x", rotation=20)
    ax_count.grid(axis="y", alpha=0.25)

    ax_pc.bar(pdf["name"], pdf["pc1_variance_share"], color="#c65d3b", alpha=0.82)
    ax_pc.set_title("PCA Concentration Preview")
    ax_pc.set_ylabel("PC1 explained variance share")
    ax_pc.set_ylim(0, 1)
    ax_pc.tick_params(axis="x", rotation=20)
    ax_pc.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = _save_figure(fig, output, cfg, "Stage 5 figures", "wrote feature-set summary")
    plt.close(fig)
    return path


def _fit_umap_nd(X: np.ndarray, n_components: int, cfg: PipelineConfig) -> np.ndarray:
    import umap
    from sklearn.preprocessing import StandardScaler

    if X.shape[0] <= n_components + 1:
        return _pca_nd(X, n_components, cfg)

    if bool(cfg.get("visualization.umap.standardize_input", False)):
        X = StandardScaler().fit_transform(X).astype("float32")
    else:
        X = X.astype("float32", copy=False)

    return umap.UMAP(
        n_components=n_components,
        n_neighbors=min(int(cfg.get("visualization.umap.n_neighbors", 30)), max(2, X.shape[0] - 1)),
        min_dist=float(cfg.get("visualization.umap.min_dist", 0.05)),
        metric=str(cfg.get("visualization.umap.metric", "cosine")),
        random_state=int(cfg.get("visualization.random_state", cfg.random_seed)),
        n_jobs=int(cfg.get("visualization.umap.n_jobs", 1)),
    ).fit_transform(X)


def _fit_umap_2d(X: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    return _fit_umap_nd(X, 2, cfg)


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


def plot_assignment_provenance(
    assignments_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot core, soft-assigned, and remaining-noise counts for the selected assignment."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    assignments = pl.read_parquet(assignments_path)
    if "assignment_source" not in assignments.columns:
        assignments = assignments.with_columns(pl.lit("unknown").alias("assignment_source"))
    provenance = (
        assignments.with_columns(
            pl.when(pl.col("tribe_id") < 0)
            .then(pl.lit("remaining noise"))
            .when(pl.col("assignment_source").cast(pl.Utf8).str.contains("soft_noise"))
            .then(pl.lit("q95 soft-assigned"))
            .when(pl.col("tribe_id") >= 0)
            .then(pl.lit("HDBSCAN core"))
            .otherwise(pl.lit("other"))
            .alias("assignment_group")
        )
        .group_by(["tribe_id", "assignment_group"])
        .agg(pl.len().alias("customers"))
        .sort("tribe_id")
    )
    pdf = provenance.to_pandas()
    pivot = (
        pdf.pivot_table(index="tribe_id", columns="assignment_group", values="customers", aggfunc="sum", fill_value=0)
        .sort_index()
    )
    for col in ["HDBSCAN core", "q95 soft-assigned", "remaining noise", "other"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["HDBSCAN core", "q95 soft-assigned", "remaining noise", "other"]]

    output = Path(output_path) if output_path else cfg.figures / f"{Path(assignments_path).stem}_assignment_provenance.png"
    colors = {
        "HDBSCAN core": "#2b6f6d",
        "q95 soft-assigned": "#d28b26",
        "remaining noise": "#8a8f98",
        "other": "#b57aa5",
    }
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(len(pivot), dtype=float)
    x = np.arange(len(pivot))
    for col in pivot.columns:
        values = pivot[col].to_numpy(dtype=float)
        if values.sum() == 0:
            continue
        ax.bar(x, values, bottom=bottom, label=col, color=colors[col])
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([str(idx) for idx in pivot.index])
    ax.set_xlabel("Tribe")
    ax.set_ylabel("Customers")
    ax.set_title("Selected Segmentation: Core vs Soft Assignment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Stage 8 figures", "wrote assignment provenance plot", cfg=cfg, path=output)
    return output


def plot_profile_evidence_dashboard(
    profile_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Create a presentation-ready dashboard of selected tribe profile evidence."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    output = Path(output_path) if output_path else cfg.figures / f"{Path(profile_path).stem}_profile_evidence_dashboard.png"
    if profiles.is_empty():
        raise ValueError(f"No profile rows found in {profile_path}")

    rows = [dict(row) for row in profiles.iter_rows(named=True)]
    tribe_labels = [f"T{row.get('tribe_id')}" for row in rows]
    customers = np.array([int(row.get("n_customers") or 0) for row in rows], dtype=float)
    core = np.array([int(row.get("core_customers") or 0) for row in rows], dtype=float)
    soft = np.array([int(row.get("soft_assigned_customers") or 0) for row in rows], dtype=float)
    soft_share = np.array([float(row.get("soft_assigned_share") or 0.0) * 100.0 for row in rows], dtype=float)
    top_theme_lift = np.array([(row.get("top_theme_lifts") or [0.0])[0] for row in rows], dtype=float)
    top_theme_labels = [str((row.get("top_themes") or ["n/a"])[0]).replace("_", " ") for row in rows]
    strong_product = np.array(
        [
            sum(
                1
                for value in (row.get("top_product_lifts") or [])
                if value and value >= float(cfg.get("profiling.strong_product_lift_threshold", 1.5))
            )
            for row in rows
        ],
        dtype=float,
    )
    strong_theme = np.array(
        [
            sum(
                1
                for value in (row.get("top_theme_lifts") or [])
                if value and value >= float(cfg.get("profiling.strong_theme_lift_threshold", 1.2))
            )
            for row in rows
        ],
        dtype=float,
    )
    strong_sector = np.array(
        [
            sum(
                1
                for value in (row.get("top_sector_lifts") or [])
                if value and value >= float(cfg.get("profiling.strong_sector_lift_threshold", 1.2))
            )
            for row in rows
        ],
        dtype=float,
    )

    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    ax_counts, ax_core, ax_theme, ax_evidence = axes.ravel()

    ax_counts.bar(x, customers, color="#3f6fb5")
    ax_counts.set_title("Customer Count Per Tribe")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(tribe_labels)
    ax_counts.set_ylabel("Customers")
    ax_counts.grid(axis="y", alpha=0.25)

    ax_core.bar(x, core, color="#2b6f6d", label="HDBSCAN core")
    ax_core.bar(x, soft, bottom=core, color="#d28b26", label="q95 soft-assigned")
    ax_core.set_title("Core vs Soft-Assigned Composition")
    ax_core.set_xticks(x)
    ax_core.set_xticklabels(tribe_labels)
    ax_core.set_ylabel("Customers")
    ax_core.legend(frameon=False)
    ax_core.grid(axis="y", alpha=0.25)
    for idx, share in enumerate(soft_share):
        ax_core.text(idx, core[idx] + soft[idx], f"{share:.0f}%", ha="center", va="bottom", fontsize=7)

    ax_theme.bar(x, top_theme_lift, color="#6f8f3f")
    ax_theme.set_title("Top Product-Theme Lift Per Tribe")
    ax_theme.set_xticks(x)
    ax_theme.set_xticklabels([shorten(label, width=14, placeholder="...") for label in top_theme_labels], rotation=35, ha="right")
    ax_theme.set_ylabel("Lift")
    ax_theme.axhline(float(cfg.get("profiling.strong_theme_lift_threshold", 1.2)), color="#444444", linewidth=1, linestyle="--")
    ax_theme.grid(axis="y", alpha=0.25)

    ax_evidence.bar(x, strong_product, color="#7a4e8a", label="product")
    ax_evidence.bar(x, strong_theme, bottom=strong_product, color="#6f8f3f", label="theme")
    ax_evidence.bar(x, strong_sector, bottom=strong_product + strong_theme, color="#4d87a8", label="sector")
    ax_evidence.set_title("Distinctive Evidence Count Per Tribe")
    ax_evidence.set_xticks(x)
    ax_evidence.set_xticklabels(tribe_labels)
    ax_evidence.set_ylabel("Strong lift count")
    ax_evidence.legend(frameon=False)
    ax_evidence.grid(axis="y", alpha=0.25)

    fig.suptitle("Selected Organic Tribe Evidence Dashboard", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Stage 8 figures", "wrote profile evidence dashboard", cfg=cfg, path=output)
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


def _read_profile_shortlist(path: str | Path) -> pl.DataFrame:
    diagnostics = pl.read_csv(path) if str(path).lower().endswith(".csv") else pl.read_parquet(path)
    if "status" in diagnostics.columns:
        diagnostics = diagnostics.filter(pl.col("status") == "profiled")
    if "profile_rank" in diagnostics.columns:
        diagnostics = diagnostics.sort("profile_rank")
    return diagnostics


def plot_experiment8_shortlist_dashboard(
    shortlist_diagnostics_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Visual decision dashboard for the Experiment 8 finalist shortlist."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    diagnostics = _read_profile_shortlist(shortlist_diagnostics_path)
    pdf = diagnostics.to_pandas()
    if pdf.empty:
        raise ValueError(f"No profiled shortlist rows found in {shortlist_diagnostics_path}")

    labels = pdf["shortlist_label"].astype(str).tolist()
    y = np.arange(len(pdf))
    output = Path(output_path) if output_path else cfg.figures / "experiment8_shortlist_decision_dashboard.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    ax_coverage, ax_scatter, ax_lift, ax_balance = axes.ravel()

    ax_coverage.barh(y - 0.18, pdf["coverage_pct"].astype(float), height=0.34, color="#2f7f77", label="coverage")
    ax_coverage.barh(y + 0.18, pdf["noise_pct"].astype(float), height=0.34, color="#9aa0a6", label="noise")
    ax_coverage.set_yticks(y)
    ax_coverage.set_yticklabels(labels)
    ax_coverage.set_xlim(0, 100)
    ax_coverage.set_xlabel("% of customers")
    ax_coverage.set_title("Coverage vs Remaining Noise")
    ax_coverage.grid(axis="x", alpha=0.22)
    ax_coverage.legend(frameon=False, loc="lower right")

    sizes = 60 + pdf["coverage_pct"].astype(float).clip(0, 100) * 4
    scatter = ax_scatter.scatter(
        pdf["cluster_count"].astype(float),
        pdf["silhouette"].astype(float),
        s=sizes,
        c=pdf["davies_bouldin"].astype(float),
        cmap="viridis_r",
        alpha=0.82,
        edgecolors="#333333",
        linewidths=0.5,
    )
    for _, row in pdf.iterrows():
        ax_scatter.annotate(
            str(row["shortlist_label"]),
            (float(row["cluster_count"]), float(row["silhouette"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    ax_scatter.axvspan(
        int(cfg.get("modeling.client_hypothesis_min", 10)),
        int(cfg.get("modeling.client_hypothesis_max", 15)),
        color="#d7eadf",
        alpha=0.35,
        label="client range",
    )
    ax_scatter.set_xlabel("Cluster count")
    ax_scatter.set_ylabel("Silhouette")
    ax_scatter.set_title("Separation vs Tribe Count")
    ax_scatter.set_xlim(max(0, float(pdf["cluster_count"].min()) - 1.0), float(pdf["cluster_count"].max()) + 1.2)
    ax_scatter.grid(alpha=0.22)
    fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04, label="Davies-Bouldin")

    ax_lift.barh(y, pdf["avg_max_product_lift"].astype(float), color="#8059a5")
    ax_lift.set_yticks(y)
    ax_lift.set_yticklabels(labels)
    ax_lift.set_xlabel("Average max product lift")
    ax_lift.set_title("Product-Lift Strength")
    ax_lift.grid(axis="x", alpha=0.22)

    ax_balance.barh(y - 0.18, pdf["cluster_size_cv"].astype(float), height=0.34, color="#c56c39", label="cluster CV")
    ax_balance.barh(
        y + 0.18,
        pdf["max_population_share_pct"].astype(float) / 10.0,
        height=0.34,
        color="#5277a3",
        label="max share / 10",
    )
    ax_balance.set_yticks(y)
    ax_balance.set_yticklabels(labels)
    ax_balance.set_xlabel("Lower is better")
    ax_balance.set_title("Balance / Dominance Check")
    ax_balance.grid(axis="x", alpha=0.22)
    ax_balance.legend(frameon=False, loc="lower right")

    fig.suptitle("Experiment 8 Finalist Decision Dashboard", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Experiment 8 figures", "wrote shortlist decision dashboard", cfg=cfg, path=output)
    return output


def plot_experiment8_shortlist_cluster_sizes(
    shortlist_diagnostics_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot cluster-size distributions for Experiment 8 finalists."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    diagnostics = _read_profile_shortlist(shortlist_diagnostics_path)
    rows = diagnostics.iter_rows(named=True)
    items = [dict(row) for row in rows if row.get("profile_path") and Path(str(row.get("profile_path"))).exists()]
    if not items:
        raise ValueError(f"No profile paths found in {shortlist_diagnostics_path}")

    output = Path(output_path) if output_path else cfg.figures / "experiment8_shortlist_cluster_sizes.png"
    ncols = 2 if len(items) <= 4 else 3
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, item in zip(axes_flat, items):
        profiles = pl.read_parquet(str(item["profile_path"])).sort("tribe_id")
        tribe_labels = [str(value) for value in profiles["tribe_id"].to_list()]
        counts = profiles["n_customers"].to_list()
        noise = int(profiles[0, "profile_noise_customers"]) if profiles.height and "profile_noise_customers" in profiles.columns else 0
        labels = [*tribe_labels, "noise"] if noise else tribe_labels
        values = [*counts, noise] if noise else counts
        colors = ["#3f6fb5"] * len(tribe_labels) + (["#9aa0a6"] if noise else [])
        ax.bar(labels, values, color=colors)
        ax.set_title(
            f"{item.get('shortlist_label')} | {int(item.get('cluster_count') or 0)} tribes | "
            f"{float(item.get('coverage_pct') or 0):.1f}% covered",
            fontsize=9,
        )
        ax.set_ylabel("Customers")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.22)

    for ax in axes_flat[len(items) :]:
        ax.axis("off")

    fig.suptitle("Experiment 8 Finalist Cluster Sizes", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Experiment 8 figures", "wrote shortlist cluster-size grid", cfg=cfg, path=output)
    return output


def plot_experiment8_shortlist_theme_heatmap(
    shortlist_diagnostics_path: str | Path,
    output_path: str | Path | None = None,
    top_n_themes: int = 14,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Aggregate top lifted products into theme counts and plot finalist theme coverage."""

    import matplotlib.pyplot as plt

    from src.product_themes import detect_product_themes

    cfg.ensure_directories()
    diagnostics = _read_profile_shortlist(shortlist_diagnostics_path)
    rows = [dict(row) for row in diagnostics.iter_rows(named=True) if row.get("profile_path")]
    if not rows:
        raise ValueError(f"No profile paths found in {shortlist_diagnostics_path}")

    candidate_counts: list[dict[str, Any]] = []
    theme_totals: dict[str, int] = {}
    for row in rows:
        profile_path = Path(str(row["profile_path"]))
        if not profile_path.exists():
            continue
        profiles = pl.read_parquet(profile_path)
        counts: dict[str, int] = {}
        for profile_row in profiles.iter_rows(named=True):
            for product in profile_row.get("top_products") or []:
                for theme in detect_product_themes(product):
                    counts[theme] = counts.get(theme, 0) + 1
                    theme_totals[theme] = theme_totals.get(theme, 0) + 1
        candidate_counts.append({"label": row.get("shortlist_label"), "counts": counts})

    themes = [theme for theme, _ in sorted(theme_totals.items(), key=lambda item: (-item[1], item[0]))[:top_n_themes]]
    matrix = np.array([[item["counts"].get(theme, 0) for theme in themes] for item in candidate_counts], dtype=float)
    labels = [str(item["label"]) for item in candidate_counts]

    output = Path(output_path) if output_path else cfg.figures / "experiment8_shortlist_theme_heatmap.png"
    fig, ax = plt.subplots(figsize=(max(10, len(themes) * 0.72), max(4.8, len(labels) * 0.58)))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(themes)))
    ax.set_xticklabels(themes, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Theme Coverage Across Shortlisted Tribe Solutions")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            if value:
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=7, color="#1f2937")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Top-product theme hits")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Experiment 8 figures", "wrote shortlist theme heatmap", cfg=cfg, path=output)
    return output


def plot_experiment8_shortlist_umap_grid(
    feature_path: str | Path,
    shortlist_diagnostics_path: str | Path,
    output_path: str | Path | None = None,
    max_candidates: int = 4,
    max_points: int = 15000,
    projection_method: str = "pca",
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot finalist assignments on a shared sampled 2D customer map."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    diagnostics = _read_profile_shortlist(shortlist_diagnostics_path)
    items = [
        dict(row)
        for row in diagnostics.iter_rows(named=True)
        if row.get("assignment_path") and Path(str(row.get("assignment_path"))).exists()
    ][:max_candidates]
    if not items:
        raise ValueError(f"No assignment paths found in {shortlist_diagnostics_path}")

    method = projection_method.strip().lower()
    if method not in {"pca", "umap"}:
        raise ValueError("projection_method must be 'pca' or 'umap'")
    output = Path(output_path) if output_path else cfg.figures / f"experiment8_shortlist_{method}_grid.png"
    with stage_timer("Experiment 8 figures", f"building shortlist {method.upper()} grid", cfg=cfg, candidates=len(items)):
        features = pl.read_parquet(feature_path).sort("cliente")
        feature_cols = numeric_feature_columns(features, exclude=("cliente",))
        sample_idx = deterministic_sample_indices(
            features.height,
            int(max_points),
            int(cfg.get("visualization.random_state", cfg.random_seed)),
        )
        sampled_features = features[sample_idx]
        X = frame_to_numpy(sampled_features, feature_cols)
        coords = _fit_umap_2d(X, cfg) if method == "umap" else _pca_2d(X, cfg)
        projection = pl.DataFrame(
            {
                "cliente": sampled_features["cliente"].to_list(),
                "umap_x": coords[:, 0].astype("float32"),
                "umap_y": coords[:, 1].astype("float32"),
            }
        )

        ncols = 2
        nrows = math.ceil(len(items) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.4 * nrows), squeeze=False)
        axes_flat = axes.ravel()
        for ax, item in zip(axes_flat, items):
            assignments = pl.read_parquet(str(item["assignment_path"])).select(["cliente", "tribe_id"])
            joined = projection.join(assignments, on="cliente", how="inner")
            pdf = joined.to_pandas()
            _scatter_labels(
                ax,
                pdf["umap_x"].to_numpy(),
                pdf["umap_y"].to_numpy(),
                pdf["tribe_id"].to_numpy(dtype=np.int32),
            )
            ax.set_title(
                f"{item.get('shortlist_label')}\n"
                f"{int(item.get('cluster_count') or 0)} tribes | "
                f"{float(item.get('coverage_pct') or 0):.1f}% coverage | "
                f"{float(item.get('noise_pct') or 0):.1f}% noise",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        for ax in axes_flat[len(items) :]:
            ax.axis("off")
        fig.suptitle(f"Experiment 8 Finalist Assignments on Shared {method.upper()} Map", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=170)
        plt.close(fig)
    log_event("Experiment 8 figures", f"wrote shortlist {method.upper()} grid", cfg=cfg, path=output)
    return output


def build_interactive_3d_projection_html(
    feature_path: str | Path,
    assignments_path: str | Path,
    output_path: str | Path | None = None,
    projection_method: str | None = None,
    color_by: str = "tribe",
    max_points: int | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Write a standalone interactive 3D customer projection for the selected solution."""

    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("plotly is required for interactive 3D projection HTML exports.") from exc

    cfg.ensure_directories()
    method = (projection_method or cfg.get("visualization.interactive.projection_method", "umap")).strip().lower()
    if method not in {"umap", "pca"}:
        raise ValueError("projection_method must be 'umap' or 'pca'")
    color_mode = color_by.strip().lower()
    if color_mode not in {"tribe", "provenance"}:
        raise ValueError("color_by must be 'tribe' or 'provenance'")

    stem = Path(assignments_path).stem.replace("cluster_assignments_", "")
    suffix = "tribes" if color_mode == "tribe" else "assignment_provenance"
    output = (
        Path(output_path)
        if output_path
        else cfg.figures / f"interactive_3d_{suffix}_{stem}_{method}.html"
    )
    sample_size = int(max_points or cfg.get("visualization.interactive.max_points", 15000))

    with stage_timer(
        "Stage 8 figures",
        f"building interactive 3D {method.upper()} projection",
        cfg=cfg,
        color_by=color_mode,
    ):
        features = pl.read_parquet(feature_path).sort("cliente")
        assignment_frame = pl.read_parquet(assignments_path)
        assignment_columns = [
            col
            for col in [
                "cliente",
                "tribe_id",
                "assignment_source",
                "assignment_probability",
                "assignment_confidence_score",
                "assignment_confidence_type",
                "model_name",
                "model_variant",
            ]
            if col in assignment_frame.columns
        ]
        assignments = assignment_frame.select(assignment_columns)
        joined = features.join(assignments, on="cliente", how="inner")
        feature_cols = numeric_feature_columns(
            joined,
            exclude=(
                "cliente",
                "tribe_id",
                "assignment_source",
                "assignment_probability",
                "assignment_confidence_score",
                "assignment_confidence_type",
                "model_name",
                "model_variant",
            ),
        )
        if not feature_cols:
            raise ValueError(f"No numeric feature columns found in {feature_path}")

        sample_idx = deterministic_sample_indices(
            joined.height,
            sample_size,
            int(cfg.get("visualization.random_state", cfg.random_seed)),
        )
        sampled = joined[sample_idx]
        X = frame_to_numpy(sampled, feature_cols)
        coords = _fit_umap_nd(X, 3, cfg) if method == "umap" else _pca_nd(X, 3, cfg)

        plot_frame = pl.DataFrame(
            {
                "cliente": sampled["cliente"].to_list(),
                "x": coords[:, 0].astype("float32"),
                "y": coords[:, 1].astype("float32"),
                "z": coords[:, 2].astype("float32"),
                "tribe_id": sampled["tribe_id"].to_list(),
            }
        )
        for col in [
            "assignment_source",
            "assignment_probability",
            "assignment_confidence_score",
            "assignment_confidence_type",
            "model_name",
            "model_variant",
        ]:
            if col in sampled.columns:
                plot_frame = plot_frame.with_columns(pl.Series(col, sampled[col].to_list()))
        if "assignment_source" not in plot_frame.columns:
            plot_frame = plot_frame.with_columns(pl.lit("unknown").alias("assignment_source"))
        plot_frame = plot_frame.with_columns(
            [
                pl.when(pl.col("tribe_id") < 0)
                .then(pl.lit("Noise"))
                .otherwise(pl.concat_str([pl.lit("Tribe "), pl.col("tribe_id").cast(pl.Utf8)]))
                .alias("tribe_label"),
                _assignment_group_expr().alias("assignment_group"),
            ]
        )

        pdf = plot_frame.to_pandas()
        color_col = "tribe_label" if color_mode == "tribe" else "assignment_group"
        hover_cols = [
            col
            for col in [
                "cliente",
                "tribe_id",
                "assignment_source",
                "assignment_probability",
                "assignment_confidence_score",
                "assignment_confidence_type",
                "model_name",
                "model_variant",
            ]
            if col in pdf.columns
        ]
        color_map = None
        if color_col == "assignment_group":
            color_map = {
                "HDBSCAN core": "#2b6f6d",
                "q95 soft-assigned": "#d28b26",
                "remaining noise": "#8a8f98",
                "other": "#b57aa5",
            }
        fig = px.scatter_3d(
            pdf,
            x="x",
            y="y",
            z="z",
            color=color_col,
            hover_data=hover_cols,
            opacity=0.74,
            title=(
                f"Interactive 3D {method.upper()} Customer Map - "
                f"{'Tribes' if color_mode == 'tribe' else 'Assignment Provenance'}"
            ),
            color_discrete_map=color_map,
        )
        fig.update_traces(marker={"size": 3, "line": {"width": 0}})
        fig.update_layout(
            legend_title_text="Tribe" if color_mode == "tribe" else "Assignment",
            margin={"l": 0, "r": 0, "t": 55, "b": 0},
            scene={
                "xaxis_title": f"{method.upper()} 1",
                "yaxis_title": f"{method.upper()} 2",
                "zaxis_title": f"{method.upper()} 3",
            },
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        include_plotlyjs = bool(cfg.get("visualization.interactive.include_plotlyjs", True))
        fig.write_html(output, include_plotlyjs=include_plotlyjs, full_html=True)
        log_event("Stage 8 figures", "wrote interactive 3D projection", cfg=cfg, path=output)
    return output


def _assignment_group_expr() -> pl.Expr:
    source = pl.col("assignment_source").cast(pl.Utf8)
    return (
        pl.when(pl.col("tribe_id") < 0)
        .then(pl.lit("remaining noise"))
        .when(source.str.contains("soft_noise"))
        .then(pl.lit("q95 soft-assigned"))
        .when(pl.col("tribe_id") >= 0)
        .then(pl.lit("HDBSCAN core"))
        .otherwise(pl.lit("other"))
    )


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
