"""Reusable visualization functions for model and tribe review."""

from __future__ import annotations

import math
from pathlib import Path
from textwrap import shorten
from typing import Mapping

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.progress import log_event, stage_timer
from src.utils import collect_streaming, deterministic_sample_indices, frame_to_numpy, numeric_feature_columns, scan_if_path, schema_names


def _pca_2d(X: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    from sklearn.decomposition import PCA

    if X.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    n_components = min(2, X.shape[0], X.shape[1])
    if n_components <= 0:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    coords = PCA(n_components=n_components, random_state=cfg.random_seed).fit_transform(X)
    if n_components == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(X.shape[0], dtype=coords.dtype)])
    return coords.astype(np.float32, copy=False)


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
    ax_norm.hist(norms, bins=40, color="#7a4e8a", alpha=0.82)
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
