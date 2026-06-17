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


KING_BLUE = "#0050A4"

FIGURE_COLORS = {
    "primary": KING_BLUE,
    "secondary": KING_BLUE,
    "tertiary": KING_BLUE,
    "accent": KING_BLUE,
    "warning": KING_BLUE,
    "purple": KING_BLUE,
    "green": KING_BLUE,
    "navy": KING_BLUE,
    "neutral": "#667085",
    "muted": "#A0AEC0",
    "text": "#17202A",
    "subtle_text": "#667085",
    "line": "#52616B",
    "grid": "#D7DEE8",
    "highlight": "#E8F2FF",
    "background": "#F7F9FC",
    "white": "#FFFFFF",
}
CHART_COLOR_SEQUENCE = [
    FIGURE_COLORS["primary"],
]
# Keep this separate from the default chart cycle; use it only where colors encode labels.
CLUSTER_COLOR_SEQUENCE = [
    KING_BLUE,
    "#1F7A8C",
    "#E30613",
    "#F2A900",
    "#6D5BD0",
    "#4C956C",
    "#4BA3C7",
    "#183B56",
    "#C2410C",
    "#7C3AED",
]
ASSIGNMENT_COLORS = {
    "HDBSCAN core": KING_BLUE,
    "q95 soft-assigned": "#4BA3C7",
    "remaining noise": FIGURE_COLORS["neutral"],
    "other": "#183B56",
}
EVIDENCE_COLORS = {
    "Product": KING_BLUE,
    "Theme": "#1F7A8C",
    "Term": "#4BA3C7",
    "Sector": "#183B56",
}
SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"
SCATTER_CMAP = "Blues"


def _color(name: str) -> str:
    return FIGURE_COLORS[name]


def apply_visual_theme() -> None:
    """Apply the shared Carrefour visualization theme to matplotlib."""

    try:
        import matplotlib.pyplot as plt
        from cycler import cycler
    except ImportError:
        return

    plt.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=CHART_COLOR_SEQUENCE),
            "axes.facecolor": FIGURE_COLORS["white"],
            "axes.edgecolor": FIGURE_COLORS["grid"],
            "axes.labelcolor": FIGURE_COLORS["text"],
            "axes.titlecolor": FIGURE_COLORS["text"],
            "axes.grid": True,
            "grid.color": FIGURE_COLORS["grid"],
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "figure.facecolor": FIGURE_COLORS["white"],
            "savefig.facecolor": FIGURE_COLORS["white"],
            "text.color": FIGURE_COLORS["text"],
            "xtick.color": FIGURE_COLORS["subtle_text"],
            "ytick.color": FIGURE_COLORS["subtle_text"],
            "legend.frameon": False,
            "font.size": 10,
        }
    )


apply_visual_theme()


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


def _save_figure(fig, output: Path, cfg: PipelineConfig, stage: str, message: str, dpi: int = 160) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
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
        ax_monthly.plot(pdf["month"], pdf[y_col], color=_color("primary"), linewidth=2)
        ax_monthly.fill_between(pdf["month"], pdf[y_col], color=_color("primary"), alpha=0.15)
        ax_monthly.set_title(f"Monthly {y_col.title()}")
        ax_monthly.tick_params(axis="x", rotation=30)
        ax_monthly.grid(alpha=0.25)
    else:
        ax_monthly.text(0.5, 0.5, "No date field available", ha="center", va="center")
        ax_monthly.set_title("Monthly Trend")
        ax_monthly.axis("off")

    if sectors.height:
        pdf = sectors.to_pandas().iloc[::-1]
        ax_sector.barh(pdf["sector"].astype(str), pdf["value"], color=_color("tertiary"))
        ax_sector.set_title("Top Product Sectors")
        ax_sector.grid(axis="x", alpha=0.25)
    else:
        ax_sector.text(0.5, 0.5, "No sector field available", ha="center", va="center")
        ax_sector.set_title("Product Sectors")
        ax_sector.axis("off")

    if promo.height:
        promo_pdf = promo.to_pandas()
        labels = ["No promo" if int(flag) == 0 else "Promo" for flag in promo_pdf["_promo_flag"]]
        ax_promo.bar(labels, promo_pdf["lines"], color=[_color("neutral"), _color("warning")])
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
    ax_hist.hist(clipped, bins=min(40, max(5, int(clip_max))), color=_color("secondary"), alpha=0.85)
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


def plot_basket_staple_diagnostics(
    product_diagnostics_path: str | Path,
    basket_exposure_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Visualize common-product candidates and basket exposure from Stage 1."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage1_common_product_diagnostics.png"
    products = pl.read_parquet(product_diagnostics_path)
    exposure = pl.read_parquet(basket_exposure_path)
    if products.is_empty():
        raise ValueError(f"No product diagnostics found in {product_diagnostics_path}")
    if exposure.is_empty():
        raise ValueError(f"No basket exposure rows found in {basket_exposure_path}")

    customer_threshold = float(cfg.get("baskets.diagnostics.common_customer_penetration_threshold", 0.01))
    basket_threshold = float(cfg.get("baskets.diagnostics.common_basket_penetration_threshold", 0.005))
    common = products.filter(pl.col("common_product_candidate"))
    top_common = common.sort("commonness_score", descending=True).head(15)
    if top_common.is_empty():
        top_common = products.sort("commonness_score", descending=True).head(15)

    customer_penetration = products["customer_penetration"].to_numpy().astype(float) * 100.0
    basket_penetration = products["basket_penetration"].to_numpy().astype(float) * 100.0
    exposure_share = exposure["common_product_candidate_share"].to_numpy().astype(float) * 100.0
    clip_max = max(0.01, float(np.nanpercentile(customer_penetration, 99.5)))
    clipped_customer_penetration = np.clip(customer_penetration, 0, clip_max)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={"height_ratios": [1.0, 1.15]})
    ax_product_hist, ax_top, ax_exposure, ax_summary = axes.ravel()

    ax_product_hist.hist(clipped_customer_penetration, bins=40, color=_color("secondary"), alpha=0.85)
    ax_product_hist.axvline(customer_threshold * 100.0, color=_color("warning"), linestyle="--", linewidth=1.8)
    ax_product_hist.set_title("Product Customer Penetration")
    ax_product_hist.set_xlabel("Customers buying product (%)")
    ax_product_hist.set_ylabel("Products")
    ax_product_hist.grid(axis="y", alpha=0.25)
    ax_product_hist.text(
        0.98,
        0.95,
        f"clipped at p99.5 = {clip_max:.2f}%",
        ha="right",
        va="top",
        transform=ax_product_hist.transAxes,
        fontsize=9,
        color=_color("subtle_text"),
    )

    labels = [
        shorten(str(row.get("product_description") or row.get("idarticu")), width=38, placeholder="...")
        for row in top_common.iter_rows(named=True)
    ]
    scores = top_common["commonness_score"].to_numpy().astype(float)
    y_pos = np.arange(len(labels))
    ax_top.barh(y_pos, scores, color=_color("primary"), alpha=0.88)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(labels, fontsize=8)
    ax_top.invert_yaxis()
    ax_top.set_title("Top Common-Product Candidates")
    ax_top.set_xlabel("Commonness score")
    ax_top.grid(axis="x", alpha=0.25)

    ax_exposure.hist(exposure_share, bins=30, color=_color("accent"), alpha=0.85)
    ax_exposure.set_title("Common-Product Share Per Basket")
    ax_exposure.set_xlabel("Common-product candidates / unique products (%)")
    ax_exposure.set_ylabel("Baskets")
    ax_exposure.grid(axis="y", alpha=0.25)

    summary_lines = [
        f"Products reviewed: {products.height:,}",
        f"Common-product candidates: {common.height:,}",
        f"Customer threshold: {customer_threshold * 100:.2f}%",
        f"Basket threshold: {basket_threshold * 100:.2f}%",
        f"Avg basket exposure: {float(np.nanmean(exposure_share)):.1f}%",
        f"Median basket exposure: {float(np.nanmedian(exposure_share)):.1f}%",
        f"P90 basket exposure: {float(np.nanpercentile(exposure_share, 90)):.1f}%",
        f"Max product customer penetration: {float(np.nanmax(customer_penetration)):.2f}%",
        f"Max product basket penetration: {float(np.nanmax(basket_penetration)):.2f}%",
    ]
    ax_summary.text(0.02, 0.96, "\n".join(summary_lines), va="top", ha="left", fontsize=12)
    ax_summary.set_title("Stage 1 Staple-Clouding Check")
    ax_summary.axis("off")

    fig.suptitle("Stage 1 Common-Product Diagnostics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = _save_figure(fig, output, cfg, "Stage 1 figures", "wrote common-product diagnostics")
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
    ax_norm.hist(norms, bins=40, color=_color("navy"), alpha=0.85)
    ax_norm.set_title("Product Vector Norms")
    ax_norm.set_xlabel("L2 norm")
    ax_norm.set_ylabel("Products")
    ax_norm.grid(axis="y", alpha=0.25)

    ax_pca.scatter(coords[:, 0], coords[:, 1], s=5, color=_color("primary"), alpha=0.55, linewidths=0)
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
        patch.set_facecolor(_color("tertiary"))
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
        ax_sector.bar([str(rank) for rank in ranks], same, color=_color("warning"), alpha=0.82)
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


def _validation_group_mask(pdf, group_name: str):
    return pdf["sample_group"].fillna("").astype(str).str.split(";").apply(lambda groups: group_name in groups)


def _validation_group_metrics(pdf, group_name: str, label: str) -> dict[str, Any]:
    subset = pdf.loc[_validation_group_mask(pdf, group_name)]
    if subset.empty:
        return {
            "group": label,
            "sampled_products": 0,
            "mean_cosine": 0.0,
            "same_sector_share": 0.0,
            "generic_neighbor_share": 0.0,
            "warning_products": 0,
        }
    return {
        "group": label,
        "sampled_products": int(subset["product_id"].nunique()),
        "mean_cosine": float(subset["cosine_similarity"].astype(float).mean()),
        "same_sector_share": float(subset["same_sector"].astype(float).mean() * 100.0),
        "generic_neighbor_share": float(subset["neighbor_is_generic_staple"].astype(float).mean() * 100.0),
        "warning_products": int(subset.loc[subset["generic_neighbor_warning"].astype(bool), "product_id"].nunique()),
    }


def plot_embedding_validation_quality_extracts(
    validation_csv: str | Path,
    output_dir: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Extract stricter Stage 3 validation checks into standalone figures."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    figure_dir = Path(output_dir) if output_dir else cfg.figures
    report = pl.read_csv(validation_csv)
    if report.height == 0:
        raise ValueError(f"No validation rows found in {validation_csv}")
    pdf = report.to_pandas()
    required = {
        "sample_group",
        "product_id",
        "cosine_similarity",
        "same_sector",
        "neighbor_is_generic_staple",
        "generic_neighbor_warning",
        "query_generic_neighbor_share",
        "product_description",
        "product_sector",
        "product_basket_penetration",
    }
    missing = required.difference(pdf.columns)
    if missing:
        raise ValueError(f"Validation CSV is missing Stage 3 extract columns: {sorted(missing)}")

    paths: dict[str, Path] = {}

    frequency_rows = [
        _validation_group_metrics(pdf, "common_frequency", "Common"),
        _validation_group_metrics(pdf, "rare_frequency", "Rare"),
    ]
    labels = [row["group"] for row in frequency_rows]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, [row["mean_cosine"] for row in frequency_rows], width, label="Mean cosine", color=_color("primary"))
    ax.bar(
        x,
        [row["same_sector_share"] / 100.0 for row in frequency_rows],
        width,
        label="Same-sector share",
        color=_color("secondary"),
    )
    ax.bar(
        x + width,
        [row["generic_neighbor_share"] / 100.0 for row in frequency_rows],
        width,
        label="Generic-neighbor share",
        color=_color("warning"),
    )
    ax.set_title("Stage 3 Common vs Rare Neighbor Quality")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share / cosine")
    for idx, row in enumerate(frequency_rows):
        ax.text(idx, 1.02, f"n={row['sampled_products']}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths["common_vs_rare"] = _save_figure(
        fig,
        figure_dir / "stage3_embedding_validation_common_vs_rare.png",
        cfg,
        "Stage 3 figures",
        "wrote common-vs-rare embedding validation extract",
    )
    plt.close(fig)

    category_names = []
    for value in pdf["sample_group"].fillna("").astype(str):
        for group in value.split(";"):
            if group.startswith("category_") and group not in category_names:
                category_names.append(group)
    category_rows = [
        _validation_group_metrics(pdf, group, group.removeprefix("category_").replace("_", " ").title())
        for group in category_names
    ]
    if category_rows:
        y = np.arange(len(category_rows))
        fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(category_rows) + 2)))
        ax.barh(
            y - 0.16,
            [row["same_sector_share"] for row in category_rows],
            0.32,
            label="Same-sector share",
            color=_color("secondary"),
        )
        ax.barh(
            y + 0.16,
            [row["generic_neighbor_share"] for row in category_rows],
            0.32,
            label="Generic-neighbor share",
            color=_color("warning"),
        )
        ax.set_yticks(y, [row["group"] for row in category_rows])
        ax.set_xlim(0, 115)
        ax.set_xlabel("Neighbor share (%)")
        ax.set_title("Stage 3 Data-Selected Niche Theme Quality")
        for idx, row in enumerate(category_rows):
            ax.text(101, idx, f"n={row['sampled_products']} warn={row['warning_products']}", va="center", fontsize=8)
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No configured niche-category samples were present.", ha="center", va="center")
    paths["niche_categories"] = _save_figure(
        fig,
        figure_dir / "stage3_embedding_validation_niche_categories.png",
        cfg,
        "Stage 3 figures",
        "wrote niche-category embedding validation extract",
    )
    plt.close(fig)

    warnings = (
        pdf.loc[pdf["generic_neighbor_warning"].astype(bool)]
        .sort_values(["query_generic_neighbor_share", "product_basket_penetration"], ascending=[False, True])
        .drop_duplicates(subset=["product_id"])
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * max(len(warnings), 1) + 2)))
    if warnings.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No generic-neighbor warnings in this Stage 3 validation run.", ha="center", va="center")
    else:
        labels = [
            shorten(str(row.product_description or row.product_id), width=42, placeholder="...")
            for row in warnings.itertuples()
        ]
        y = np.arange(len(warnings))
        values = warnings["query_generic_neighbor_share"].astype(float).to_numpy() * 100.0
        ax.barh(y, values, color=_color("warning"), alpha=0.85)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Generic-neighbor share (%)")
        ax.set_title("Stage 3 Generic Neighbor Warnings")
        for idx, row in enumerate(warnings.itertuples()):
            sector = shorten(str(row.product_sector or "Unknown"), width=22, placeholder="...")
            ax.text(min(values[idx] + 1, 98), idx, sector, va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    paths["generic_warnings"] = _save_figure(
        fig,
        figure_dir / "stage3_embedding_validation_generic_warnings.png",
        cfg,
        "Stage 3 figures",
        "wrote generic-neighbor warning extract",
    )
    plt.close(fig)

    return paths


def plot_customer_embedding_diagnostics(
    customer_embeddings_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot customer-vector health plus Stage 4 coverage and dominance gates."""

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

    diagnostics_cfg = cfg.get("customer_embeddings.diagnostics", {}) or {}
    diagnostics_path = (
        cfg.artifacts
        / str(diagnostics_cfg.get("output_dir", "stage4"))
        / f"{diagnostics_cfg.get('output_prefix', 'customer_embedding')}_weight_diagnostics.csv"
    )
    diagnostics: dict[str, Any] = {}
    diagnostics_error: str | None = None
    if diagnostics_path.exists():
        try:
            diagnostics_df = pl.read_csv(diagnostics_path)
            if diagnostics_df.height > 0:
                diagnostics = diagnostics_df.row(0, named=True)
        except Exception as exc:  # pragma: no cover - plot should still render if a CSV is mid-write/corrupt.
            diagnostics_error = str(exc)

    def _metric(name: str) -> float | None:
        value = diagnostics.get(name)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _display_value(value: Any) -> str:
        if value is None or value == "":
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.3g}"
        return str(value)

    fig = plt.figure(figsize=(13.6, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    ax_norm = fig.add_subplot(gs[0, 0])
    ax_pca = fig.add_subplot(gs[0, 1])
    ax_gates = fig.add_subplot(gs[1, 0])
    ax_summary = fig.add_subplot(gs[1, 1])

    finite_norms = norms[np.isfinite(norms)]
    if finite_norms.size == 0:
        raise ValueError(f"No finite customer-vector norms found in {customer_embeddings_path}")
    norm_min = float(np.min(finite_norms))
    norm_max = float(np.max(finite_norms))
    norm_mean = float(np.mean(finite_norms))
    if norm_max - norm_min <= 1e-5:
        ax_norm.axvline(norm_mean, color=_color("primary"), linewidth=3.5)
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
            bbox={"boxstyle": "round,pad=0.35", "facecolor": _color("white"), "edgecolor": _color("grid")},
        )
    else:
        ax_norm.hist(finite_norms, bins=40, color=_color("primary"), alpha=0.82)
    ax_norm.set_title("Customer Vector Norms")
    ax_norm.set_xlabel("L2 norm")
    ax_norm.set_ylabel("Sampled customers")
    ax_norm.grid(axis="y", alpha=0.25)

    ax_pca.scatter(coords[:, 0], coords[:, 1], s=6, color=_color("primary"), alpha=0.35, linewidths=0)
    ax_pca.set_title("Customer Embedding PCA Preview")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.grid(alpha=0.2)

    gates = cfg.get("customer_embeddings.gates", {}) or {}
    gate_specs = [
        ("Line coverage", "line_coverage_pct", "min_line_coverage_pct", "min"),
        ("Unit coverage", "unit_coverage_pct", "min_unit_coverage_pct", "min"),
        ("Customer coverage", "customer_coverage_pct", "min_customer_coverage_pct", "min"),
        ("Common product weight", "common_product_weight_share_pct", "max_common_product_weight_share_pct", "max"),
        ("Top product weight", "top_product_weight_share_pct", "max_top_product_weight_share_pct", "max"),
        ("Top 10 product weight", "top_10_product_weight_share_pct", "max_top_10_product_weight_share_pct", "max"),
    ]
    gate_rows = [
        (label, value, gates.get(threshold_key), direction)
        for label, value_key, threshold_key, direction in gate_specs
        if (value := _metric(value_key)) is not None
    ]
    if gate_rows:
        y = np.arange(len(gate_rows))
        values = [max(0.0, min(100.0, row[1])) for row in gate_rows]
        ax_gates.barh(y, values, color=_color("primary"), alpha=0.82)
        ax_gates.set_yticks(y, [row[0] for row in gate_rows])
        ax_gates.set_xlim(0, 100)
        ax_gates.set_xlabel("Percent")
        ax_gates.set_title("Stage 4 Gate Metrics")
        ax_gates.grid(axis="x", alpha=0.25)
        ax_gates.invert_yaxis()
        for idx, (_, value, threshold, direction) in enumerate(gate_rows):
            ax_gates.text(min(value + 1.0, 98.5), idx, f"{value:.1f}%", va="center", fontsize=8)
            if threshold is not None and threshold != "":
                threshold_value = float(threshold)
                ax_gates.plot(
                    [threshold_value, threshold_value],
                    [idx - 0.36, idx + 0.36],
                    color=_color("line"),
                    linestyle="--",
                    linewidth=1.2,
                )
                prefix = "min" if direction == "min" else "max"
                ax_gates.text(
                    min(threshold_value + 1.0, 98.0),
                    idx + 0.31,
                    f"{prefix} {threshold_value:.0f}%",
                    fontsize=7,
                    color=_color("subtle_text"),
                )
    else:
        ax_gates.axis("off")
        missing_message = "Stage 4 weight diagnostics CSV not found yet."
        if diagnostics_error:
            missing_message = f"Stage 4 weight diagnostics could not be read:\n{diagnostics_error}"
        ax_gates.text(0.5, 0.5, missing_message, ha="center", va="center")

    strategy = diagnostics.get("weight_strategy", cfg.get("customer_embeddings.weight_strategy", "quantity"))
    issues = str(diagnostics.get("stage4_gate_issues", "n/a"))
    if len(issues) > 120:
        issues = shorten(issues, width=120, placeholder="...")
    summary_rows = [
        ("Weight strategy", strategy),
        ("Quantity transform", diagnostics.get("quantity_transform", cfg.get("customer_embeddings.quantity_transform"))),
        ("IDF weighting", "yes" if "idf" in str(strategy).lower() else "no"),
        ("Normalize vectors", diagnostics.get("normalize_vectors", cfg.get("customer_embeddings.normalize_vectors"))),
        ("Recency weighting", diagnostics.get("recency_weighting_enabled")),
        ("Frequency weighting", diagnostics.get("frequency_weighting_enabled")),
        ("Embedded customers", diagnostics.get("weighted_customers", embeddings.height)),
        ("Embedded products", diagnostics.get("weighted_products", len(feature_cols))),
        ("Gate status", diagnostics.get("stage4_gate_status", "n/a")),
        ("Gate issues", issues),
    ]
    ax_summary.axis("off")
    ax_summary.set_title("Recipe and Gate Status", loc="left")
    y_pos = 0.96
    for label, value in summary_rows:
        ax_summary.text(0.02, y_pos, f"{label}:", fontweight="bold", va="top")
        ax_summary.text(0.42, y_pos, _display_value(value), va="top", wrap=True)
        y_pos -= 0.09 if label != "Gate issues" else 0.15

    fig.suptitle("Stage 4 Customer Embedding Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
        ax.hist(values, bins=35, color=_color("primary"), alpha=0.82)
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
    """Compare feature-set dimensionality and single-axis PCA concentration."""

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
    ax_count.bar(pdf["name"], pdf["feature_count"], color=_color("tertiary"), alpha=0.85)
    ax_count.set_title("Feature Count by Set")
    ax_count.set_ylabel("Numeric features")
    ax_count.tick_params(axis="x", rotation=20)
    ax_count.grid(axis="y", alpha=0.25)

    ax_pc.bar(pdf["name"], pdf["pc1_variance_share"], color=_color("warning"), alpha=0.82)
    ax_pc.set_title("Single-Axis Variance Concentration")
    ax_pc.set_ylabel("Share explained by PC1 only")
    ax_pc.set_ylim(0, 1)
    ax_pc.tick_params(axis="x", rotation=20)
    ax_pc.grid(axis="y", alpha=0.25)
    ax_pc.text(
        0.02,
        -0.22,
        "Diagnostic only: this is not total retained variance.",
        transform=ax_pc.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=_color("muted"),
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
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
    ax.bar(labels, values, color=_color("primary"))
    ax.set_ylabel("Silhouette")
    ax.set_title("Candidate Model Comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    log_event("Stage 7 figures", "wrote model comparison plot", cfg=cfg, path=output)
    return output


def plot_stage6_model_diagnostics(
    diagnostics_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot Stage 6 model diagnostics before stability and product-lift profiling."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    diagnostics = (
        pl.read_csv(diagnostics_path) if str(diagnostics_path).lower().endswith(".csv") else pl.read_parquet(diagnostics_path)
    )
    output = Path(output_path) if output_path else cfg.figures / "stage6_model_diagnostics.png"
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
    colors = {model: CHART_COLOR_SEQUENCE[idx % len(CHART_COLOR_SEQUENCE)] for idx, model in enumerate(models)}

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
            color=_color("text"),
        )

    ax_scatter.set_xlabel("Cluster count")
    ax_scatter.set_ylabel(metric.replace("_", " ").title())
    ax_scatter.set_title("Stage 6 Core Tribe Diagnostics")
    ax_scatter.grid(alpha=0.25)
    ax_scatter.legend(loc="best", fontsize=7, frameon=False)

    top = pdf.head(min(18, len(pdf))).iloc[::-1]
    bar_colors = [colors[str(model)] for model in top["model_name"]]
    ax_ranked.barh(range(len(top)), top[metric], color=bar_colors, alpha=0.88)
    ax_ranked.set_yticks(range(len(top)))
    ax_ranked.set_yticklabels(top["candidate_label"], fontsize=7)
    ax_ranked.set_xlabel(metric.replace("_", " ").title())
    ax_ranked.set_title("Stage 6 Ranked Runs")
    ax_ranked.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Stage 6 diagnostics", "wrote candidate diagnostics plot", cfg=cfg, path=output)
    return output


def plot_stage6_quality_evidence(
    quality_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot Stage 6.5 UMAP retention and HDBSCAN validity against target ranges."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    quality = pl.read_csv(quality_path)
    if quality.is_empty():
        raise ValueError(f"No Stage 6.5 quality rows found in {quality_path}")
    row = quality.row(0, named=True)
    output = Path(output_path) if output_path else cfg.figures / "stage6_5_quality_evidence.png"

    specs = [
        {
            "key": "umap_trustworthiness",
            "label": "UMAP trustworthiness",
            "ideal": ">= 0.95 strong; >= 0.90 usable",
            "score": _score_high_good(_as_float_for_plot(row.get("umap_trustworthiness")), low=0.90, high=0.95),
            "value": row.get("umap_trustworthiness"),
        },
        {
            "key": "umap_mean_knn_overlap_pct",
            "label": "UMAP kNN overlap",
            "ideal": ">= 25% useful; >= 15% review",
            "score": _score_high_good(_as_float_for_plot(row.get("umap_mean_knn_overlap_pct")), low=15.0, high=25.0),
            "value": row.get("umap_mean_knn_overlap_pct"),
        },
        {
            "key": "umap_distance_spearman",
            "label": "UMAP distance rank",
            "ideal": ">= 0.70 strong; >= 0.60 usable",
            "score": _score_high_good(_as_float_for_plot(row.get("umap_distance_spearman")), low=0.60, high=0.70),
            "value": row.get("umap_distance_spearman"),
        },
        {
            "key": "hdbscan_dbcv_score",
            "label": "HDBSCAN DBCV",
            "ideal": ">= 0.25 strong; >= 0.10 usable",
            "score": _score_high_good(_as_float_for_plot(row.get("hdbscan_dbcv_score")), low=0.10, high=0.25),
            "value": row.get("hdbscan_dbcv_score"),
        },
        {
            "key": "noise_pct",
            "label": "Noise share",
            "ideal": "<= 60% gate; <= 40% strong",
            "score": _score_low_good(_as_float_for_plot(row.get("noise_pct")), low=40.0, high=60.0),
            "value": row.get("noise_pct"),
        },
        {
            "key": "core_coverage_pct",
            "label": "Core coverage",
            "ideal": ">= 40% useful; >= 60% strong",
            "score": _score_high_good(_as_float_for_plot(row.get("core_coverage_pct")), low=40.0, high=60.0),
            "value": row.get("core_coverage_pct"),
        },
        {
            "key": "silhouette_core_only",
            "label": "Core-only silhouette",
            "ideal": "supporting only; >= 0.40 strong; >= 0.25 usable",
            "score": _score_high_good(_as_float_for_plot(row.get("silhouette_core_only")), low=0.25, high=0.40),
            "value": row.get("silhouette_core_only"),
            "supporting": True,
        },
        {
            "key": "coverage_adjusted_silhouette",
            "label": "Coverage-adjusted silhouette",
            "ideal": ">= 0.25 strong; >= 0.15 usable",
            "score": _score_high_good(_as_float_for_plot(row.get("coverage_adjusted_silhouette")), low=0.15, high=0.25),
            "value": row.get("coverage_adjusted_silhouette"),
        },
        {
            "key": "avg_assignment_confidence",
            "label": "Assignment confidence",
            "ideal": ">= 0.40 stronger; >= 0.30 review",
            "score": _score_high_good(_as_float_for_plot(row.get("avg_assignment_confidence")), low=0.30, high=0.40),
            "value": row.get("avg_assignment_confidence"),
        },
    ]
    specs = [spec for spec in specs if spec["value"] is not None and str(spec["value"]) != ""]
    if not specs:
        raise ValueError(f"Stage 6.5 quality row did not contain plottable metrics: {quality_path}")

    labels = [str(spec["label"]) for spec in specs][::-1]
    scores = [float(spec["score"]) * 100.0 for spec in specs][::-1]
    values = [_format_plot_value(spec["value"]) for spec in specs][::-1]
    ideals = [str(spec["ideal"]) for spec in specs][::-1]
    supporting = [bool(spec.get("supporting", False)) for spec in specs][::-1]
    colors = [
        "#4E79A7" if is_supporting else _evidence_score_color(score / 100.0)
        for score, is_supporting in zip(scores, supporting)
    ]

    fig_height = max(6.0, 1.02 * len(specs))
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    y = np.arange(len(specs))
    ax.barh(y, [100.0] * len(specs), color="#EEF2F6", height=0.64)
    ax.barh(y, scores, color=colors, height=0.64)
    ax.axvline(50.0, color="#475467", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.text(50.7, len(specs) - 0.35, "usable/review threshold", fontsize=9, color="#475467", va="top")

    for idx, (score, value, ideal) in enumerate(zip(scores, values, ideals)):
        ax.text(min(max(score + 2.0, 4.0), 98.0), idx, f"{score:.0f}", va="center", ha="left", fontsize=9)
        ax.text(103.0, idx, f"{value} | {ideal}", va="center", ha="left", fontsize=9, color=_color("text"))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 160)
    ax.set_xlabel("Evidence score, 0-100")
    ax.set_title(title or "Stage 6.5 Representation and Density Evidence")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#2E7D32", label="strong/pass"),
        Patch(facecolor="#F2A900", label="usable/review"),
        Patch(facecolor="#C2410C", label="below threshold"),
        Patch(facecolor="#4E79A7", label="supporting only"),
    ]
    fig.legend(
        handles=[
            *legend_handles,
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        fontsize=8,
    )
    fig.text(
        0.125,
        0.015,
        "Scores are pragmatic diagnostics, not model objectives. Core-only silhouette ignores noise; use it with coverage, DBCV, Stage 6.6 stability, and Stage 7 product-lift evidence.",
        fontsize=9,
        color=_color("subtle_text"),
    )

    fig.tight_layout(rect=(0, 0.095, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Stage 6.5 diagnostics", "wrote quality evidence scorecard", cfg=cfg, path=output)
    return output


def plot_stage6_cluster_readiness(
    readiness_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot Stage 6.6 cluster-level readiness before product profiling."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    readiness = (
        pl.read_csv(readiness_path)
        if str(readiness_path).lower().endswith(".csv")
        else pl.read_parquet(readiness_path)
    )
    if readiness.is_empty():
        raise ValueError(f"No Stage 6.6 cluster readiness rows found in {readiness_path}")
    pdf = readiness.to_pandas()
    output = Path(output_path) if output_path else cfg.figures / "stage6_6_cluster_readiness.png"

    readiness_colors = {
        "strong": "#2E7D32",
        "usable": "#F2A900",
        "review": "#C2410C",
    }
    pdf["profile_readiness"] = pdf["profile_readiness"].fillna("review").astype(str)
    pdf["mean_assignment_confidence"] = pdf["mean_assignment_confidence"].fillna(0.0).astype(float)
    pdf["customers"] = pdf["customers"].fillna(0).astype(float)
    pdf["core_customer_share_pct"] = pdf["core_customer_share_pct"].fillna(0.0).astype(float)
    min_cluster_size = float(pdf["min_cluster_size_reference"].dropna().iloc[0]) if "min_cluster_size_reference" in pdf else 0.0

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    for readiness_label, group in pdf.groupby("profile_readiness"):
        sizes = 70.0 + group["core_customer_share_pct"].clip(lower=0, upper=35) * 24.0
        ax.scatter(
            group["customers"],
            group["mean_assignment_confidence"],
            s=sizes,
            color=readiness_colors.get(readiness_label, _color("neutral")),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
            label=readiness_label,
        )
        for _, row in group.iterrows():
            ax.annotate(
                str(int(row["tribe_id"])),
                (row["customers"], row["mean_assignment_confidence"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color=_color("text"),
            )

    if min_cluster_size > 0:
        ax.axvline(min_cluster_size, color="#475467", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(min_cluster_size, ax.get_ylim()[1] * 0.96, "min cluster size", rotation=90, va="top", ha="right", fontsize=8)
    ax.axhline(0.30, color="#475467", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.text(ax.get_xlim()[1] * 0.99, 0.305, "confidence review line", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("Customers in core tribe")
    ax.set_ylabel("Mean assignment confidence")
    ax.set_title(title or "Stage 6.6 Cluster Readiness Before Profiling")
    ax.grid(alpha=0.22)
    ax.legend(title="Profile readiness", frameon=False, loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Stage 6.6 diagnostics", "wrote cluster readiness plot", cfg=cfg, path=output)
    return output


def _as_float_for_plot(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def _score_high_good(value: float | None, *, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if value >= high:
        return 1.0
    if value <= low:
        return max(0.0, 0.45 * value / low) if low > 0 else 0.0
    return 0.50 + 0.50 * ((value - low) / (high - low))


def _score_low_good(value: float | None, *, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if value <= low:
        return 1.0
    if value >= high:
        return max(0.0, 0.45 * (1.0 - min(value - high, high) / max(high, 1e-12)))
    return 1.0 - 0.50 * ((value - low) / (high - low))


def _evidence_score_color(score: float) -> str:
    if score >= 0.75:
        return "#2E7D32"
    if score >= 0.50:
        return "#F2A900"
    return "#C2410C"


def _format_plot_value(value: Any) -> str:
    value_float = _as_float_for_plot(value)
    if value_float is None:
        return str(value)
    if abs(value_float) >= 100:
        return f"{value_float:.0f}"
    if abs(value_float) >= 10:
        return f"{value_float:.1f}"
    return f"{value_float:.3f}".rstrip("0").rstrip(".")


def plot_stage6_umap_representation(
    umap_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot the first two dimensions of the Stage 6.1 UMAP representation."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage6_1_umap_representation.png"
    umap_df = pl.read_parquet(umap_path)
    umap_cols = [col for col in numeric_feature_columns(umap_df) if col.startswith("umap_")]
    if len(umap_cols) < 2:
        raise ValueError(f"UMAP representation needs at least two numeric components for plotting: {umap_path}")
    sample = _sample_frame(umap_df, int(cfg.get("visualization.max_scatter_points", 50000)), cfg)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.scatter(
        sample[umap_cols[0]].to_numpy(),
        sample[umap_cols[1]].to_numpy(),
        s=4,
        color=_color("primary"),
        alpha=0.32,
        linewidths=0,
    )
    ax.set_title("Stage 6.1 UMAP Customer Manifold")
    ax.set_xlabel(umap_cols[0])
    ax.set_ylabel(umap_cols[1])
    ax.grid(alpha=0.22)
    fig.tight_layout()
    path = _save_figure(fig, output, cfg, "Stage 6.1 figures", "wrote UMAP representation preview")
    plt.close(fig)
    return path


def plot_stage6_noise_umap_probe(
    umap_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot the first two dimensions of the remaining-noise-only UMAP probe."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage6_7_remaining_noise_umap_probe.png"
    umap_df = pl.read_parquet(umap_path)
    umap_cols = [col for col in numeric_feature_columns(umap_df) if col.startswith("umap_")]
    if len(umap_cols) < 2:
        raise ValueError(f"Noise UMAP probe needs at least two numeric components for plotting: {umap_path}")
    sample = _sample_frame(umap_df, int(cfg.get("visualization.max_scatter_points", 50000)), cfg)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.scatter(
        sample[umap_cols[0]].to_numpy(),
        sample[umap_cols[1]].to_numpy(),
        s=4,
        color=_color("neutral"),
        alpha=0.34,
        linewidths=0,
    )
    ax.set_title("Stage 6.7 Remaining-Noise UMAP Probe")
    ax.set_xlabel(umap_cols[0])
    ax.set_ylabel(umap_cols[1])
    ax.grid(alpha=0.22)
    ax.text(
        0.02,
        -0.14,
        "Visual review only: run HDBSCAN only if coherent sub-structure is visible.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=_color("subtle_text"),
    )
    fig.tight_layout()
    path = _save_figure(fig, output, cfg, "Stage 6.7 figures", "wrote remaining-noise UMAP probe")
    plt.close(fig)
    return path


def plot_stage68_evidence_overview(
    profile_path: str | Path,
    *,
    noise_vs_core_path: str | Path | None = None,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot the compact Stage 6.8 evidence handoff without reopening raw inputs."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    output = (
        Path(output_path)
        if output_path
        else cfg.figures / f"stage6_8_tribe_evidence_overview_{cfg.mode}.png"
    )
    if profiles.is_empty():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No Stage 6.8 tribe evidence available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        path = _save_figure(fig, output, cfg, "Stage 6.8 figures", "wrote empty evidence overview")
        plt.close(fig)
        return path

    rows = [dict(row) for row in profiles.iter_rows(named=True)]
    tribe_labels = [f"T{int(row.get('tribe_id'))}" for row in rows]
    customers = np.array([float(row.get("n_customers") or 0.0) for row in rows], dtype=float)
    noise_customers = _first_numeric_profile_value(
        profiles,
        ["unassigned_noise_customers_global", "profile_noise_customers", "noise_customers"],
    )

    max_product_lift = np.array(
        [_max_numeric_list(row.get("top_product_lifts_vs_rest") or row.get("top_product_lifts")) for row in rows],
        dtype=float,
    )
    top_product_reach = np.array([_max_numeric_list(row.get("top_product_reach_pct")) for row in rows], dtype=float)
    strong_threshold = float(cfg.get("profiling.strong_product_lift_threshold", 1.5))

    behavior_fields = [
        ("avg_ticket_count", "Tickets"),
        ("avg_unique_products", "Products"),
        ("avg_frequency_per_30d", "Frequency"),
        ("avg_basket_value", "Basket value"),
        ("avg_promo_share", "Promo share"),
        ("mean_recency_days", "Recency"),
    ]
    behavior_fields = [(col, label) for col, label in behavior_fields if col in profiles.columns]
    behavior_matrix = _weighted_behavior_ratio_matrix(rows, customers, behavior_fields)

    noise_rows = _read_noise_vs_core_rows(noise_vs_core_path)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    ax_size, ax_lift, ax_behavior, ax_noise = axes.ravel()

    size_labels = [*tribe_labels, "Unassigned\nnoise"] if noise_customers else tribe_labels
    size_values = [*customers.tolist(), float(noise_customers)] if noise_customers else customers.tolist()
    size_colors = [_color("secondary")] * len(tribe_labels) + ([_color("neutral")] if noise_customers else [])
    ax_size.bar(np.arange(len(size_labels)), size_values, color=size_colors)
    ax_size.set_title("Assigned Tribe Customers And Remaining Noise")
    ax_size.set_xticks(np.arange(len(size_labels)))
    ax_size.set_xticklabels(size_labels)
    ax_size.set_ylabel("Customers")
    ax_size.grid(axis="y", alpha=0.22)

    x = np.arange(len(tribe_labels))
    ax_lift.bar(x, max_product_lift, color=EVIDENCE_COLORS["Product"], label="max lift vs rest")
    ax_lift_twin = ax_lift.twinx()
    ax_lift_twin.plot(x, top_product_reach, color=_color("line"), marker="o", linewidth=1.5, label="max reach")
    ax_lift.axhline(strong_threshold, color=_color("warning"), linewidth=1, linestyle="--")
    ax_lift.set_title("Product Evidence Strength")
    ax_lift.set_xticks(x)
    ax_lift.set_xticklabels(tribe_labels)
    ax_lift.set_ylabel("Max product lift vs rest")
    ax_lift_twin.set_ylabel("Max product reach %")
    ax_lift.grid(axis="y", alpha=0.22)

    if behavior_fields and behavior_matrix.size:
        image = ax_behavior.imshow(
            behavior_matrix,
            aspect="auto",
            cmap=DIVERGING_CMAP,
            vmin=0.5,
            vmax=1.5,
        )
        ax_behavior.set_title("Behavior KPIs Vs Assigned Population")
        ax_behavior.set_xticks(np.arange(len(behavior_fields)))
        ax_behavior.set_xticklabels([label for _, label in behavior_fields], rotation=35, ha="right")
        ax_behavior.set_yticks(np.arange(len(tribe_labels)))
        ax_behavior.set_yticklabels(tribe_labels)
        fig.colorbar(image, ax=ax_behavior, fraction=0.046, pad=0.04, label="ratio")
    else:
        ax_behavior.text(0.5, 0.5, "No behavioral KPI columns found.", ha="center", va="center")
        ax_behavior.axis("off")

    if noise_rows:
        metric_labels = [shorten(row["label"], width=18, placeholder="...") for row in noise_rows]
        ratios = np.array([row["ratio"] for row in noise_rows], dtype=float)
        y = np.arange(len(noise_rows))
        colors = [_color("secondary") if value >= 1.0 else _color("neutral") for value in ratios]
        ax_noise.barh(y, ratios, color=colors)
        ax_noise.axvline(1.0, color=_color("line"), linewidth=1, linestyle="--")
        ax_noise.set_yticks(y)
        ax_noise.set_yticklabels(metric_labels)
        ax_noise.set_xlabel("Noise / core mean")
        ax_noise.set_title("Noise Population Behavioral Profile")
        ax_noise.grid(axis="x", alpha=0.22)
    else:
        ax_noise.text(0.5, 0.5, "Noise-vs-core metrics unavailable.", ha="center", va="center")
        ax_noise.axis("off")

    fig.suptitle("Stage 6.8 Tribe Evidence Assembly Overview", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = _save_figure(fig, output, cfg, "Stage 6.8 figures", "wrote tribe evidence overview")
    plt.close(fig)
    return path


def _first_numeric_profile_value(profiles: pl.DataFrame, columns: Sequence[str]) -> int:
    for column in columns:
        if column not in profiles.columns:
            continue
        values = [
            float(value)
            for value in profiles[column].drop_nulls().to_list()
            if value is not None and math.isfinite(float(value))
        ]
        if values:
            return int(round(values[0]))
    return 0


def _max_numeric_list(values: Any) -> float:
    if values is None:
        return 0.0
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = [values]
    cleaned: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            cleaned.append(numeric)
    return max(cleaned) if cleaned else 0.0


def _weighted_behavior_ratio_matrix(
    rows: list[dict[str, Any]],
    weights: np.ndarray,
    fields: list[tuple[str, str]],
) -> np.ndarray:
    matrix = np.empty((len(rows), len(fields)), dtype=float)
    matrix[:] = np.nan
    for col_idx, (column, _) in enumerate(fields):
        values = np.array(
            [
                np.nan if row.get(column) is None else float(row.get(column))
                for row in rows
            ],
            dtype=float,
        )
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            continue
        population_mean = float(np.average(values[valid], weights=weights[valid]))
        if abs(population_mean) < 1e-12:
            continue
        matrix[:, col_idx] = values / population_mean
    return matrix


def _read_noise_vs_core_rows(noise_vs_core_path: str | Path | None) -> list[dict[str, Any]]:
    if noise_vs_core_path is None or not Path(noise_vs_core_path).exists():
        return []
    frame = pl.read_csv(noise_vs_core_path)
    if frame.is_empty() or "noise_vs_core_ratio" not in frame.columns:
        return []
    preferred = [
        "ticket_count",
        "frequency_per_30d",
        "basket_value",
        "total_spend",
        "unique_products",
        "recency_days",
        "promo_share",
    ]
    label_map = {
        "ticket_count": "Tickets",
        "frequency_per_30d": "Frequency",
        "basket_value": "Basket value",
        "total_spend": "Total spend",
        "unique_products": "Products",
        "recency_days": "Recency",
        "promo_share": "Promo share",
    }
    rows = []
    for metric in preferred:
        metric_frame = frame.filter(pl.col("metric") == metric)
        if metric_frame.is_empty():
            continue
        row = metric_frame.row(0, named=True)
        ratio = row.get("noise_vs_core_ratio")
        if ratio is None or not math.isfinite(float(ratio)):
            continue
        rows.append({"label": label_map.get(metric, metric), "ratio": float(ratio)})
    if rows:
        return rows
    for row in frame.head(7).iter_rows(named=True):
        ratio = row.get("noise_vs_core_ratio")
        if ratio is None or not math.isfinite(float(ratio)):
            continue
        rows.append({"label": str(row.get("metric") or "metric"), "ratio": float(ratio)})
    return rows


def plot_stage6_hdbscan_assignment_map(
    umap_path: str | Path,
    assignment_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot hard HDBSCAN tribe assignments on the Stage 6.1 UMAP plane."""

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.figures / "stage6_2_hdbscan_assignment_map.png"
    umap_df = pl.read_parquet(umap_path)
    assignments = pl.read_parquet(assignment_path).select(["cliente", "tribe_id"])
    umap_cols = [col for col in numeric_feature_columns(umap_df) if col.startswith("umap_")]
    if len(umap_cols) < 2:
        raise ValueError(f"UMAP representation needs at least two numeric components for plotting: {umap_path}")

    plot_df = umap_df.select(["cliente", umap_cols[0], umap_cols[1]]).join(assignments, on="cliente", how="inner")
    sample = _sample_frame(plot_df, int(cfg.get("visualization.max_scatter_points", 50000)), cfg)
    coords = sample.select([umap_cols[0], umap_cols[1]]).to_numpy().astype(np.float32, copy=False)
    labels = sample["tribe_id"].to_numpy().astype(np.int32, copy=False)
    path = _scatter(
        coords,
        labels,
        output,
        title or "Stage 6.2 Hard HDBSCAN Core Tribes on UMAP",
        allocation_summary=_tribe_allocation_summary(assignments),
    )
    log_event("Stage 6.2 figures", "wrote hard HDBSCAN assignment map", cfg=cfg, path=path)
    return path


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
    colors = [_color("neutral") if tribe < 0 else _color("secondary") for tribe in pdf["tribe_id"]]
    ax.bar(pdf["tribe_id"].astype(str), pdf["customers"], color=colors)
    ax.set_xlabel("Tribe")
    ax.set_ylabel("Customers")
    ax.set_title("Cluster Size Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    log_event("Stage 7 figures", "wrote cluster size plot", cfg=cfg, path=output)
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
    colors = ASSIGNMENT_COLORS
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
    log_event("Stage 7 figures", "wrote assignment provenance plot", cfg=cfg, path=output)
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

    ax_counts.bar(x, customers, color=_color("secondary"))
    ax_counts.set_title("Customer Count Per Tribe")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(tribe_labels)
    ax_counts.set_ylabel("Customers")
    ax_counts.grid(axis="y", alpha=0.25)

    ax_core.bar(x, core, color=ASSIGNMENT_COLORS["HDBSCAN core"], label="HDBSCAN core")
    ax_core.bar(x, soft, bottom=core, color=ASSIGNMENT_COLORS["q95 soft-assigned"], label="q95 soft-assigned")
    ax_core.set_title("Core vs Soft-Assigned Composition")
    ax_core.set_xticks(x)
    ax_core.set_xticklabels(tribe_labels)
    ax_core.set_ylabel("Customers")
    ax_core.legend(frameon=False)
    ax_core.grid(axis="y", alpha=0.25)
    for idx, share in enumerate(soft_share):
        ax_core.text(idx, core[idx] + soft[idx], f"{share:.0f}%", ha="center", va="bottom", fontsize=7)

    ax_theme.bar(x, top_theme_lift, color=EVIDENCE_COLORS["Theme"])
    ax_theme.set_title("Top Product-Theme Lift Per Tribe")
    ax_theme.set_xticks(x)
    ax_theme.set_xticklabels([shorten(label, width=14, placeholder="...") for label in top_theme_labels], rotation=35, ha="right")
    ax_theme.set_ylabel("Lift")
    ax_theme.axhline(float(cfg.get("profiling.strong_theme_lift_threshold", 1.2)), color=_color("line"), linewidth=1, linestyle="--")
    ax_theme.grid(axis="y", alpha=0.25)

    ax_evidence.bar(x, strong_product, color=EVIDENCE_COLORS["Product"], label="product")
    ax_evidence.bar(x, strong_theme, bottom=strong_product, color=EVIDENCE_COLORS["Theme"], label="theme")
    ax_evidence.bar(x, strong_sector, bottom=strong_product + strong_theme, color=EVIDENCE_COLORS["Sector"], label="sector")
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
    log_event("Stage 7 figures", "wrote profile evidence dashboard", cfg=cfg, path=output)
    return output


def plot_tribe_vs_population_evidence_dashboard(
    profile_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot per-tribe evidence against the rest of the assigned population."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    output = (
        Path(output_path)
        if output_path
        else cfg.figures / f"stage_07_tribe_vs_population_evidence_dashboard_{cfg.mode}.png"
    )
    if profiles.is_empty():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No tribe profiles available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return _save_figure(fig, output, cfg, "Stage 7 figures", "wrote empty tribe evidence dashboard")

    rows = [dict(row) for row in profiles.iter_rows(named=True)]
    tribe_ids = [int(row.get("tribe_id")) for row in rows]
    def _organic_label(row: Mapping[str, Any]) -> str:
        label = row.get("llm_working_label")
        if not label:
            products = row.get("top_products") or []
            label = f"{products[0]} buyers" if products else ""
        return shorten(str(label), width=24, placeholder="...")

    tribe_labels = [f"T{row.get('tribe_id')} {_organic_label(row)}".strip() for row in rows]
    compact_labels = [f"T{tribe_id}" for tribe_id in tribe_ids]
    customers = np.array([float(row.get("n_customers") or 0.0) for row in rows], dtype=float)
    core = np.array([float(row.get("core_customers") or 0.0) for row in rows], dtype=float)
    soft = np.zeros(len(rows), dtype=float)
    population_share = np.array([float(row.get("population_share") or 0.0) for row in rows], dtype=float)

    def _float_list(values: Any) -> list[float]:
        return [float(value) for value in (values or []) if value is not None and not math.isnan(float(value))]

    def _strong_count(row: Mapping[str, Any], column: str, threshold: float) -> int:
        return sum(1 for value in _float_list(row.get(column)) if value >= threshold)

    def _max_list(values: Any) -> float:
        cleaned = _float_list(values)
        return max(cleaned) if cleaned else 0.0

    def _max_customer_rate_lift_vs_rest(row: Mapping[str, Any], lift_col: str, count_col: str) -> float:
        tribe_customers = float(row.get("n_customers") or 0.0)
        total_customers = float(row.get("profile_population_customers") or 0.0)
        if tribe_customers <= 0 or total_customers <= tribe_customers:
            return _max_list(row.get(lift_col))
        lifts = row.get(lift_col) or []
        counts = row.get(count_col) or []
        rest_lifts: list[float] = []
        for lift_raw, count_raw in zip(lifts, counts):
            if lift_raw is None or count_raw is None:
                continue
            lift = float(lift_raw)
            cluster_count = float(count_raw)
            if lift <= 0 or cluster_count <= 0:
                continue
            cluster_rate = cluster_count / tribe_customers
            population_rate = cluster_rate / lift
            rest_count = max(population_rate * total_customers - cluster_count, 0.0)
            rest_rate = rest_count / max(total_customers - tribe_customers, 1.0)
            if rest_rate > 0:
                rest_lifts.append(cluster_rate / rest_rate)
        return max(rest_lifts) if rest_lifts else _max_list(row.get(lift_col))

    max_lifts = {
        "Product": np.array(
            [_max_customer_rate_lift_vs_rest(row, "top_product_lifts_vs_rest", "top_product_customer_counts") for row in rows],
            dtype=float,
        ),
        "Sector": np.array([_max_list(row.get("top_sector_lifts")) for row in rows], dtype=float),
    }
    evidence_counts = {
        "Product": np.array(
            [
                _strong_count(row, "top_product_lifts_vs_rest", float(cfg.get("profiling.strong_product_lift_threshold", 1.5)))
                for row in rows
            ],
            dtype=float,
        ),
        "Sector": np.array(
            [_strong_count(row, "top_sector_lifts", float(cfg.get("profiling.strong_sector_lift_threshold", 1.2))) for row in rows],
            dtype=float,
        ),
    }

    behavior_candidates = [
        ("avg_ticket_count", "Tickets"),
        ("avg_unique_products", "Unique products"),
        ("avg_frequency_per_30d", "Frequency"),
        ("avg_promo_share", "Promo share"),
        ("avg_basket_value", "Basket value"),
        ("avg_recency_days", "Recency"),
    ]
    behavior_fields = [item for item in behavior_candidates if item[0] in profiles.columns]
    behavior_matrix = np.empty((len(rows), len(behavior_fields)), dtype=float)
    behavior_matrix[:] = np.nan
    total_weight = float(customers.sum())
    for col_idx, (column, _) in enumerate(behavior_fields):
        values = np.array(
            [
                np.nan if row.get(column) is None else float(row.get(column))
                for row in rows
            ],
            dtype=float,
        )
        valid = np.isfinite(values)
        if not valid.any():
            continue
        total_sum = float(np.nansum(values[valid] * customers[valid]))
        for row_idx, value in enumerate(values):
            if not np.isfinite(value):
                continue
            rest_weight = total_weight - customers[row_idx]
            if rest_weight <= 0:
                continue
            rest_mean = (total_sum - value * customers[row_idx]) / rest_weight
            if rest_mean > 0:
                behavior_matrix[row_idx, col_idx] = value / rest_mean

    n_tribes = len(rows)
    fig_height = max(10.0, 0.55 * n_tribes + 7.0)
    fig, axes = plt.subplots(2, 2, figsize=(17, fig_height))
    ax_size, ax_lift, ax_counts, ax_behavior = axes.ravel()

    y = np.arange(n_tribes)
    ax_size.barh(y, core, color=ASSIGNMENT_COLORS["HDBSCAN core"], label="Core assigned")
    ax_size.set_yticks(y)
    ax_size.set_yticklabels(tribe_labels)
    ax_size.invert_yaxis()
    ax_size.set_xlabel("Customers")
    ax_size.set_title("Hard Organic Tribe Size")
    ax_size.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=1)
    ax_size.grid(axis="x", alpha=0.25)
    x_max = max(float(customers.max()) if customers.size else 1.0, 1.0)
    ax_size.set_xlim(0, x_max * 1.23)
    for idx, value in enumerate(customers):
        ax_size.text(
            value + x_max * 0.015,
            idx,
            f"{int(value):,} ({population_share[idx] * 100:.1f}%)",
            va="center",
            fontsize=8,
        )

    x = np.arange(n_tribes)
    width = 0.34
    lift_colors = {key: EVIDENCE_COLORS[key] for key in ["Product", "Sector"]}
    all_lift_values = np.concatenate([values[np.isfinite(values)] for values in max_lifts.values()])
    lift_cap = 4.0
    if all_lift_values.size:
        lift_cap = max(3.0, min(8.0, float(np.nanpercentile(all_lift_values, 90)) * 1.15))
    for offset, (label, values) in enumerate(max_lifts.items()):
        plot_values = np.clip(values, 0, lift_cap)
        ax_lift.bar(x + (offset - 0.5) * width, plot_values, width=width, color=lift_colors[label], label=label)
    ax_lift.axhline(1.0, color=_color("line"), linewidth=1, linestyle="--")
    ax_lift.set_xticks(x)
    ax_lift.set_xticklabels(compact_labels)
    ax_lift.set_ylabel("Max lift")
    ax_lift.set_title(f"Strongest Product/Sector Signal Vs Rest (capped at {lift_cap:.1f}x)")
    ax_lift.legend(frameon=False)
    ax_lift.grid(axis="y", alpha=0.25)

    bottom = np.zeros(n_tribes, dtype=float)
    for label, values in evidence_counts.items():
        ax_counts.bar(x, values, bottom=bottom, color=lift_colors[label], label=label)
        bottom += values
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(compact_labels)
    ax_counts.set_ylabel("Strong lifted signals")
    ax_counts.set_title("Distinctive Evidence Count Per Tribe")
    ax_counts.legend(frameon=False)
    ax_counts.grid(axis="y", alpha=0.25)

    if behavior_fields:
        display_matrix = np.clip(behavior_matrix, 0.5, 1.5)
        masked = np.ma.masked_invalid(display_matrix)
        cmap = plt.get_cmap(DIVERGING_CMAP).copy()
        cmap.set_bad(_color("background"))
        im = ax_behavior.imshow(masked, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.5)
        ax_behavior.set_yticks(np.arange(n_tribes))
        ax_behavior.set_yticklabels(compact_labels)
        ax_behavior.set_xticks(np.arange(len(behavior_fields)))
        ax_behavior.set_xticklabels([label for _, label in behavior_fields], rotation=35, ha="right")
        ax_behavior.set_title("Behavior/KPI Ratio Vs Rest Of Population")
        for row_idx in range(behavior_matrix.shape[0]):
            for col_idx in range(behavior_matrix.shape[1]):
                value = behavior_matrix[row_idx, col_idx]
                if np.isfinite(value):
                    ax_behavior.text(col_idx, row_idx, f"{value:.1f}x", ha="center", va="center", fontsize=7)
        cbar = fig.colorbar(im, ax=ax_behavior, fraction=0.046, pad=0.04)
        cbar.set_label("Tribe / rest ratio")
    else:
        ax_behavior.text(0.5, 0.5, "No behavioral profiling fields available.", ha="center", va="center")
        ax_behavior.axis("off")

    fig.suptitle(
        "Stage 7 Tribe Evidence Dashboard: Each Tribe Compared With The Rest Of The Assigned Population",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = _save_figure(fig, output, cfg, "Stage 7 figures", "wrote tribe-vs-population evidence dashboard")
    plt.close(fig)
    return path


def plot_tribe_profile_comparison_heatmap(
    profile_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot a between-tribe comparison heatmap using ratios versus the rest."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    output = (
        Path(output_path)
        if output_path
        else cfg.figures / f"stage_07_tribe_profile_comparison_heatmap_{cfg.mode}.png"
    )
    if profiles.is_empty():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No tribe profiles available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return _save_figure(fig, output, cfg, "Stage 7 figures", "wrote empty tribe profile comparison heatmap")

    rows = [dict(row) for row in profiles.iter_rows(named=True)]
    metric_values: list[tuple[str, list[float | None]]] = [
        ("Top product lift", [_max_ratio(row.get("top_product_lifts_vs_rest") or row.get("top_product_lifts")) for row in rows]),
        ("Top sector lift", [_max_ratio(row.get("top_sector_lifts_vs_rest") or row.get("top_sector_lifts")) for row in rows]),
        ("Top term lift", [_max_ratio(row.get("top_product_term_lifts_vs_rest") or row.get("top_product_term_lifts")) for row in rows]),
    ]
    for column, label in _profile_heatmap_behavior_fields(profiles):
        metric_values.append((label, [_profile_rest_ratio(profiles, row, column) for row in rows]))

    metric_values = [
        (label, values)
        for label, values in metric_values
        if any(value is not None and math.isfinite(float(value)) for value in values)
    ]
    if not metric_values:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No comparable profile ratios available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return _save_figure(fig, output, cfg, "Stage 7 figures", "wrote empty tribe profile comparison heatmap")

    matrix = np.full((len(rows), len(metric_values)), np.nan, dtype=float)
    annotation = [["" for _ in metric_values] for _ in rows]
    for col_idx, (_, values) in enumerate(metric_values):
        for row_idx, value in enumerate(values):
            if value is None or not math.isfinite(float(value)) or value <= 0:
                continue
            matrix[row_idx, col_idx] = math.log2(float(value))
            annotation[row_idx][col_idx] = f"{float(value):.1f}x"

    masked = np.ma.masked_invalid(np.clip(matrix, -1.25, 2.0))
    fig_width = max(12.0, 0.9 * len(metric_values) + 5.0)
    fig_height = max(7.0, 0.55 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.get_cmap(DIVERGING_CMAP).copy()
    cmap.set_bad(_color("background"))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-1.25, vmax=2.0)

    tribe_labels = [
        f"T{row.get('tribe_id')} {shorten(str(row.get('suggested_tribe_name') or ''), width=26, placeholder='...')}".strip()
        for row in rows
    ]
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(tribe_labels)
    ax.set_xticks(np.arange(len(metric_values)))
    ax.set_xticklabels([label for label, _ in metric_values], rotation=35, ha="right")
    ax.set_title("Stage 7 Between-Tribe Profile Comparison")
    ax.set_xlabel("Metric ratio versus rest of assigned population")
    ax.set_ylabel("Tribe")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            if annotation[row_idx][col_idx]:
                ax.text(col_idx, row_idx, annotation[row_idx][col_idx], ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2 ratio vs rest; 0 means same as rest")
    fig.tight_layout()
    path = _save_figure(fig, output, cfg, "Stage 7 figures", "wrote tribe profile comparison heatmap")
    plt.close(fig)
    return path


def _profile_heatmap_behavior_fields(profiles: pl.DataFrame) -> list[tuple[str, str]]:
    candidates = [
        ("avg_ticket_count", "Tickets"),
        ("avg_total_units", "Units"),
        ("avg_unique_products", "Unique products"),
        ("avg_unique_sectors", "Unique sectors"),
        ("avg_frequency_per_30d", "Frequency"),
        ("avg_promo_share", "Promo share"),
        ("avg_avg_basket_value", "Basket value"),
    ]
    return [(column, label) for column, label in candidates if column in profiles.columns]


def _profile_rest_ratio(profiles: pl.DataFrame, row: Mapping[str, Any], column: str) -> float | None:
    value = _finite_float(row.get(column))
    tribe_id = int(row.get("tribe_id"))
    if value is None:
        return None
    total_weight = 0.0
    total_value = 0.0
    own_weight = 0.0
    own_value = 0.0
    for item in profiles.iter_rows(named=True):
        item_value = _finite_float(item.get(column))
        weight = _finite_float(item.get("n_customers")) or 0.0
        if item_value is None or weight <= 0:
            continue
        total_weight += weight
        total_value += item_value * weight
        if int(item.get("tribe_id")) == tribe_id:
            own_weight += weight
            own_value += item_value * weight
    rest_weight = total_weight - own_weight
    if rest_weight <= 0:
        return None
    rest_mean = (total_value - own_value) / rest_weight
    return value / rest_mean if rest_mean > 0 else None


def _max_ratio(values: Any) -> float | None:
    cleaned = [_finite_float(value) for value in (values or [])]
    finite = [value for value in cleaned if value is not None and value > 0]
    return max(finite) if finite else None


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def plot_tribe_theme_lift_heatmap(
    profile_path: str | Path,
    output_path: str | Path | None = None,
    *,
    max_themes: int = 14,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot selected tribe differentiation through strategic product-theme lift."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    profiles = pl.read_parquet(profile_path).sort("tribe_id")
    output = (
        Path(output_path)
        if output_path
        else cfg.figures / f"stage_07_tribe_theme_lift_heatmap_{cfg.mode}.png"
    )
    if profiles.is_empty():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No tribe profiles available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return _save_figure(fig, output, cfg, "Stage 7 figures", "wrote empty tribe-theme heatmap")

    theme_scores: dict[str, float] = {}
    rows = []
    for row in profiles.iter_rows(named=True):
        tribe_id = int(row.get("tribe_id"))
        themes = row.get("top_themes") or []
        lifts = row.get("top_theme_lifts") or []
        for idx, theme in enumerate(themes):
            if idx >= len(lifts) or lifts[idx] is None:
                continue
            theme_key = str(theme)
            lift = float(lifts[idx])
            rows.append({"tribe_id": tribe_id, "theme": theme_key, "lift": lift})
            theme_scores[theme_key] = max(theme_scores.get(theme_key, 0.0), lift)

    if not rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No product-theme lift evidence available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return _save_figure(fig, output, cfg, "Stage 7 figures", "wrote empty tribe-theme heatmap")

    selected_themes = [
        theme for theme, _ in sorted(theme_scores.items(), key=lambda item: item[1], reverse=True)[:max_themes]
    ]
    tribe_ids = [int(value) for value in profiles["tribe_id"].to_list()]
    matrix = np.zeros((len(tribe_ids), len(selected_themes)), dtype=float)
    tribe_index = {tribe_id: idx for idx, tribe_id in enumerate(tribe_ids)}
    theme_index = {theme: idx for idx, theme in enumerate(selected_themes)}
    for row in rows:
        if row["theme"] in theme_index:
            matrix[tribe_index[int(row["tribe_id"])], theme_index[row["theme"]]] = float(row["lift"])

    fig_width = max(11, 0.75 * len(selected_themes) + 4)
    fig_height = max(6, 0.42 * len(tribe_ids) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=max(float(matrix.max()), 1.5))
    ax.set_yticks(np.arange(len(tribe_ids)))
    ax.set_yticklabels([f"T{tribe_id}" for tribe_id in tribe_ids])
    ax.set_xticks(np.arange(len(selected_themes)))
    ax.set_xticklabels(
        [shorten(theme.replace("_", " "), width=18, placeholder="...") for theme in selected_themes],
        rotation=35,
        ha="right",
    )
    ax.set_xlabel("Product-theme evidence")
    ax.set_ylabel("Selected core tribe")
    ax.set_title("Stage 7 Product-Theme Lift By Selected Tribe")
    threshold = float(cfg.get("profiling.strong_theme_lift_threshold", 1.2))
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            if value >= threshold:
                ax.text(x, y, f"{value:.1f}x", ha="center", va="center", fontsize=7, color=_color("text"))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Theme lift vs assigned population")
    fig.tight_layout()
    return _save_figure(fig, output, cfg, "Stage 7 figures", "wrote tribe-theme heatmap")


def plot_mission_microtribe_overview(
    mission_summary_path: str | Path,
    output_path: str | Path | None = None,
    *,
    max_missions: int = 20,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot the client-facing shopping mission layer by size and confidence."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = (
        Path(output_path)
        if output_path
        else cfg.figures / f"stage_09_shopping_mission_overview_{cfg.mode}.png"
    )
    summary = pl.read_csv(mission_summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    if summary.is_empty():
        ax.text(0.5, 0.5, "No shopping missions passed the configured thresholds.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output, dpi=170)
        plt.close(fig)
        log_event("Stage 9 figures", "wrote empty shopping mission overview", cfg=cfg, path=output)
        return output

    pdf = (
        summary.sort("mission_customers", descending=True)
        .head(max_missions)
        .sort("mission_customers")
        .to_pandas()
    )
    families = list(dict.fromkeys(pdf["mission_family"].astype(str)))
    colors = {family: CHART_COLOR_SEQUENCE[idx % len(CHART_COLOR_SEQUENCE)] for idx, family in enumerate(families)}
    bar_colors = [colors[str(family)] for family in pdf["mission_family"]]
    labels = [
        shorten(str(label), width=36, placeholder="...")
        for label in pdf["mission_label"].astype(str).to_list()
    ]

    y = np.arange(len(pdf))
    ax.barh(y, pdf["mission_customers"].astype(float), color=bar_colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Customers with mission evidence")
    ax.set_title("Stage 9 Shopping Missions: Evidence-Backed Activation Segments")
    ax.grid(axis="x", alpha=0.22)
    max_customers = max(float(pdf["mission_customers"].max()), 1.0)
    ax.set_xlim(0, max_customers * 1.35)

    for idx, row in pdf.iterrows():
        confidence = str(row.get("mission_confidence", ""))
        share = float(row.get("population_share_pct", 0.0))
        customers = int(row.get("mission_customers", 0))
        ax.text(
            customers,
            idx,
            f" {customers:,} | {share:.1f}% | {confidence}",
            va="center",
            fontsize=7,
            color=_color("text"),
        )

    handles = [
        plt.Line2D([0], [0], marker="s", color=_color("white"), label=family, markerfacecolor=color, markersize=8)
        for family, color in colors.items()
    ]
    ax.legend(handles=handles, title="Mission family", loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Stage 9 figures", "wrote shopping mission overview", cfg=cfg, path=output)
    return output


def plot_core_mission_lift_heatmap(
    customer_mission_tags_path: str | Path,
    core_summary_path: str | Path,
    output_path: str | Path | None = None,
    *,
    max_missions: int = 18,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Plot how strongly each shopping mission concentrates inside each core tribe."""

    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    output = (
        Path(output_path)
        if output_path
        else cfg.figures / f"stage_09_core_tribe_by_shopping_mission_lift_{cfg.mode}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    tags = pl.read_parquet(customer_mission_tags_path)
    core = pl.read_csv(core_summary_path)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    required_cols = {"cliente", "tribe_id", "mission_key", "mission_label"}
    if tags.is_empty() or not required_cols.issubset(tags.columns) or "n_customers" not in core.columns:
        ax.text(0.5, 0.5, "Mission/core evidence is not available for the heatmap.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output, dpi=170)
        plt.close(fig)
        log_event("Stage 9 figures", "wrote empty core-mission heatmap", cfg=cfg, path=output)
        return output

    tags = tags.filter(pl.col("tribe_id") >= 0)
    if tags.is_empty():
        ax.text(0.5, 0.5, "No mission-tagged customers have a core tribe assignment.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output, dpi=170)
        plt.close(fig)
        log_event("Stage 9 figures", "wrote empty core-mission heatmap", cfg=cfg, path=output)
        return output

    mission_totals = (
        tags.group_by(["mission_key", "mission_label"])
        .agg(pl.col("cliente").n_unique().alias("mission_customers"))
        .sort("mission_customers", descending=True)
        .head(max_missions)
    )
    top_keys = mission_totals["mission_key"].to_list()
    tribe_sizes = core.select(["tribe_id", "n_customers"]).with_columns(pl.col("tribe_id").cast(pl.Int64))
    total_assigned = float(tribe_sizes["n_customers"].sum() or 0.0)
    if total_assigned <= 0:
        ax.text(0.5, 0.5, "Core tribe sizes are missing.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output, dpi=170)
        plt.close(fig)
        log_event("Stage 9 figures", "wrote empty core-mission heatmap", cfg=cfg, path=output)
        return output

    counts = (
        tags.filter(pl.col("mission_key").is_in(top_keys))
        .group_by(["tribe_id", "mission_key", "mission_label"])
        .agg(pl.col("cliente").n_unique().alias("customers"))
        .join(mission_totals, on=["mission_key", "mission_label"], how="left")
        .join(tribe_sizes, on="tribe_id", how="left")
        .with_columns(
            (
                (pl.col("customers") / pl.col("mission_customers"))
                / (pl.col("n_customers") / pl.lit(total_assigned))
            )
            .round(2)
            .alias("mission_lift_vs_core_size")
        )
    )
    pdf = counts.select(["tribe_id", "mission_label", "mission_lift_vs_core_size"]).to_pandas()
    if pdf.empty:
        ax.text(0.5, 0.5, "No mission/core intersections were found.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output, dpi=170)
        plt.close(fig)
        log_event("Stage 9 figures", "wrote empty core-mission heatmap", cfg=cfg, path=output)
        return output

    mission_order = mission_totals["mission_label"].to_list()
    matrix = (
        pdf.pivot_table(
            index="tribe_id",
            columns="mission_label",
            values="mission_lift_vs_core_size",
            aggfunc="max",
            fill_value=0.0,
        )
        .reindex(columns=mission_order)
        .sort_index()
    )
    values = matrix.to_numpy(dtype=float)
    vmax = max(1.5, min(4.0, float(np.nanmax(values)) if values.size else 1.5))
    image = ax.imshow(values, cmap=SEQUENTIAL_CMAP, aspect="auto", vmin=0.0, vmax=vmax)

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [shorten(str(label), width=22, placeholder="...") for label in matrix.columns],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([f"T{int(value)}" for value in matrix.index], fontsize=8)
    ax.set_xlabel("Shopping mission")
    ax.set_ylabel("Organic core tribe")
    ax.set_title("Stage 9 Bridge: Shopping Mission Lift by Core Tribe")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if value >= 1.2:
                ax.text(col_idx, row_idx, f"{value:.1f}x", ha="center", va="center", fontsize=7, color=_color("text"))

    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Mission concentration lift")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    log_event("Stage 9 figures", "wrote core tribe by shopping mission heatmap", cfg=cfg, path=output)
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

    with stage_timer("Stage 7 figures", "building candidate UMAP assignment grid", cfg=cfg, candidates=len(assignment_paths)):
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
        log_event("Stage 7 figures", "wrote candidate UMAP grid", cfg=cfg, path=output)
    return output


def plot_top_lifts(
    profile_path: str | Path,
    output_dir: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> list[Path]:
    import matplotlib.pyplot as plt

    cfg.ensure_directories()
    profiles = pl.read_parquet(profile_path)
    out_dir = Path(output_dir) if output_dir else cfg.figures
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
        ax.barh(y, lifts, color=EVIDENCE_COLORS["Product"])
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
    log_event("Stage 7 figures", "wrote lift plots", cfg=cfg, plots=len(paths), output_dir=out_dir)
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

    ax_coverage.barh(y - 0.18, pdf["coverage_pct"].astype(float), height=0.34, color=_color("secondary"), label="coverage")
    ax_coverage.barh(y + 0.18, pdf["noise_pct"].astype(float), height=0.34, color=_color("muted"), label="noise")
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
        cmap=f"{SCATTER_CMAP}_r",
        alpha=0.82,
        edgecolors=_color("text"),
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
        color=_color("highlight"),
        alpha=0.35,
        label="client range",
    )
    ax_scatter.set_xlabel("Cluster count")
    ax_scatter.set_ylabel("Silhouette")
    ax_scatter.set_title("Separation vs Tribe Count")
    ax_scatter.set_xlim(max(0, float(pdf["cluster_count"].min()) - 1.0), float(pdf["cluster_count"].max()) + 1.2)
    ax_scatter.grid(alpha=0.22)
    fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04, label="Davies-Bouldin")

    ax_lift.barh(y, pdf["avg_max_product_lift"].astype(float), color=EVIDENCE_COLORS["Product"])
    ax_lift.set_yticks(y)
    ax_lift.set_yticklabels(labels)
    ax_lift.set_xlabel("Average max product lift")
    ax_lift.set_title("Product-Lift Strength")
    ax_lift.grid(axis="x", alpha=0.22)

    ax_balance.barh(y - 0.18, pdf["cluster_size_cv"].astype(float), height=0.34, color=_color("warning"), label="cluster CV")
    ax_balance.barh(
        y + 0.18,
        pdf["max_population_share_pct"].astype(float) / 10.0,
        height=0.34,
        color=_color("tertiary"),
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
        noise = _first_numeric_profile_value(
            profiles,
            ["unassigned_noise_customers_global", "profile_noise_customers", "noise_customers"],
        )
        labels = [*tribe_labels, "noise"] if noise else tribe_labels
        values = [*counts, noise] if noise else counts
        colors = [_color("secondary")] * len(tribe_labels) + ([_color("muted")] if noise else [])
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
    image = ax.imshow(matrix, cmap=SEQUENTIAL_CMAP, aspect="auto")
    ax.set_xticks(np.arange(len(themes)))
    ax.set_xticklabels(themes, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Theme Coverage Across Shortlisted Tribe Solutions")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            if value:
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=7, color=_color("text"))
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
        "Stage 7 figures",
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
            color_map = ASSIGNMENT_COLORS
        elif color_col == "tribe_label":
            color_map = {
                label: _tribe_color(-1 if label == "Noise" else int(str(label).replace("Tribe ", "")))
                for label in sorted(pdf[color_col].dropna().astype(str).unique())
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
        log_event("Stage 7 figures", "wrote interactive 3D projection", cfg=cfg, path=output)
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
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
    stage_label: str = "Stage 7 figures",
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Create PCA and optional UMAP 2D figures for visual review only."""

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    cfg.ensure_directories()
    with stage_timer(stage_label, "building 2D projection figures", cfg=cfg):
        out_dir = Path(output_dir) if output_dir else cfg.figures
        features = pl.read_parquet(feature_path).sort("cliente")
        assignments = pl.read_parquet(assignments_path).select(["cliente", "tribe_id"])
        allocation_summary = _tribe_allocation_summary(assignments)
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
        prefix = output_prefix or f"selected_tribes_{stem}"
        outputs: dict[str, Path] = {}
        pca = PCA(n_components=2, random_state=int(cfg.get("visualization.random_state", cfg.random_seed)))
        coords = pca.fit_transform(X)
        outputs["pca"] = _scatter(
            coords,
            labels,
            out_dir / f"{prefix}_pca.png",
            f"PCA Selected Tribes - {stem}",
            allocation_summary,
        )

        try:
            coords = _fit_umap_2d(X, cfg)
            outputs["umap"] = _scatter(
                coords,
                labels,
                out_dir / f"{prefix}_umap.png",
                f"UMAP Selected Tribes - {stem}",
                allocation_summary,
            )
        except Exception as exc:
            log_event(stage_label, "UMAP figure skipped", cfg=cfg, reason=type(exc).__name__)
        plt.close("all")
        log_event(stage_label, "wrote projection figures", cfg=cfg, plots=len(outputs))
    return outputs


def _tribe_allocation_summary(assignments: pl.DataFrame) -> list[dict[str, Any]]:
    if assignments.is_empty():
        return []

    total_customers = assignments["cliente"].n_unique()
    if total_customers <= 0:
        return []

    counts = (
        assignments.group_by("tribe_id")
        .agg(pl.col("cliente").n_unique().alias("customers"))
        .sort("tribe_id")
    )
    observed = {int(row["tribe_id"]): int(row["customers"]) for row in counts.iter_rows(named=True)}
    tribe_ids = sorted(tribe_id for tribe_id in observed if tribe_id >= 0)
    if any(tribe_id < 0 for tribe_id in observed):
        tribe_ids.append(-1)

    rows: list[dict[str, Any]] = []
    for tribe_id in tribe_ids:
        customers = sum(count for key, count in observed.items() if key < 0) if tribe_id < 0 else observed.get(tribe_id, 0)
        rows.append(
            {
                "tribe_id": tribe_id,
                "label": "Noise" if tribe_id < 0 else f"T{tribe_id}",
                "customers": customers,
                "share_pct": 100.0 * customers / total_customers,
            }
        )
    return rows


def _scatter(coords: np.ndarray, labels: np.ndarray, output: Path, title: str, allocation_summary: Sequence[Mapping[str, Any]] | None = None) -> Path:
    import matplotlib.pyplot as plt

    has_summary = bool(allocation_summary)
    if has_summary:
        fig, (ax_summary, ax) = plt.subplots(
            1,
            2,
            figsize=(10.8, 6),
            gridspec_kw={"width_ratios": [1.0, 2.65], "wspace": 0.12},
        )
        _draw_tribe_allocation_panel(ax_summary, allocation_summary)
    else:
        fig, ax = plt.subplots(figsize=(7.5, 6))
    _scatter_labels(ax, coords[:, 0], coords[:, 1], labels)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if has_summary:
        fig.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.11, wspace=0.14)
    else:
        fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _draw_tribe_allocation_panel(ax, allocation_summary: Sequence[Mapping[str, Any]]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.0, 0.98, "Tribe Allocation", fontsize=12, fontweight="bold", va="top")
    ax.text(0.0, 0.915, "Tribe", fontsize=8.5, color=_color("subtle_text"), va="top")
    ax.text(0.38, 0.915, "Customers", fontsize=8.5, color=_color("subtle_text"), va="top")
    ax.text(0.78, 0.915, "Share", fontsize=8.5, color=_color("subtle_text"), va="top")
    ax.plot([0.0, 0.98], [0.89, 0.89], color=_color("grid"), linewidth=0.8)

    row_count = max(len(allocation_summary), 1)
    row_gap = min(0.058, 0.82 / row_count)
    y = 0.855
    for row in allocation_summary:
        tribe_id = int(row.get("tribe_id", -1))
        color = _tribe_color(tribe_id)
        label = str(row.get("label", "Noise" if tribe_id < 0 else f"T{tribe_id}"))
        customers = int(row.get("customers", 0) or 0)
        share_pct = float(row.get("share_pct", 0.0) or 0.0)

        ax.scatter([0.025], [y], s=42, color=color, alpha=0.9, linewidths=0)
        ax.text(0.07, y, label, fontsize=9.5, va="center")
        ax.text(0.38, y, f"{customers:,}", fontsize=9.5, va="center")
        ax.text(0.78, y, f"{share_pct:.1f}%", fontsize=9.5, va="center")
        y -= row_gap


def _tribe_color(tribe_id: int) -> str:
    if tribe_id < 0:
        return _color("muted")
    return CLUSTER_COLOR_SEQUENCE[int(tribe_id) % len(CLUSTER_COLOR_SEQUENCE)]


def _scatter_labels(ax, x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> None:
    labels = np.asarray(labels).astype(np.int32)
    noise = labels < 0
    if noise.any():
        ax.scatter(x[noise], y[noise], s=4, color=_tribe_color(-1), alpha=0.35, linewidths=0, label="noise")
    valid = ~noise
    if valid.any():
        color_lookup = {int(label): _tribe_color(int(label)) for label in np.unique(labels[valid])}
        colors = [color_lookup[int(label)] for label in labels[valid]]
        ax.scatter(
            x[valid],
            y[valid],
            c=colors,
            s=4,
            alpha=0.78,
            linewidths=0,
        )

