"""Human-readable summaries for experiment diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.progress import log_event


DEFAULT_SUMMARY_COLUMNS = [
    "stage6_rank",
    "embedding_sandbox_rank",
    "customer_embedding_sandbox_rank",
    "feature_set_sandbox_rank",
    "selected_in_embedding_sandbox",
    "selected_in_customer_embedding_sandbox",
    "selected_in_feature_set_sandbox",
    "selected_within_family",
    "passes_quality_gate",
    "trial_name",
    "feature_set_name",
    "model_name",
    "algorithm_name",
    "model_variant",
    "candidate_id",
    "k",
    "weight_strategy",
    "normalize_vectors",
    "cluster_count",
    "coverage_adjusted_silhouette",
    "stage6_score",
    "silhouette",
    "davies_bouldin",
    "noise_pct",
    "coverage_pct",
    "cluster_size_cv",
    "embedding_quality_score",
    "customer_embedding_quality_score",
    "feature_set_quality_score",
    "product_vocab_coverage_pct",
    "same_sector_at_1_pct",
    "same_sector_neighbor_share_pct",
    "mean_neighbor_cosine",
    "vector_dims",
    "feature_count",
    "finite_pct",
    "zero_vector_pct",
    "dead_dimension_count",
    "dead_feature_count",
    "effective_dimension",
    "pca_components_for_80pct",
    "pca_components_for_90pct",
    "mean_nearest_neighbor_cosine",
    "selection_reason",
]


def write_summary_artifacts(
    df: pl.DataFrame,
    output_base: str | Path,
    title: str,
    priority_columns: list[str] | None = None,
    cfg: PipelineConfig = CONFIG,
    max_markdown_rows: int = 20,
    write_markdown: bool = True,
) -> dict[str, Path]:
    """Write compact CSV and Markdown summaries next to a canonical artifact."""

    base = Path(output_base)
    summary_csv, summary_md = _summary_paths(base)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = _summary_frame(df, priority_columns or [])
    summary.write_csv(summary_csv)
    if write_markdown:
        summary_md.write_text(
            _markdown_summary(title, summary, max_rows=max_markdown_rows),
            encoding="utf-8",
        )
    log_event(
        "Experiment summary",
        "wrote human-readable summaries",
        cfg=cfg,
        rows=summary.height,
        csv=summary_csv,
        markdown=summary_md if write_markdown else None,
    )
    result = {"summary_csv": summary_csv}
    if write_markdown:
        result["summary_md"] = summary_md
    return result


def _summary_paths(output_base: Path) -> tuple[Path, Path]:
    stem = output_base.stem
    for suffix in ["_diagnostics", "_manifest"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    summary_stem = f"{stem}_summary"
    return output_base.with_name(summary_stem).with_suffix(".csv"), output_base.with_name(summary_stem).with_suffix(".md")


def _summary_frame(df: pl.DataFrame, priority_columns: list[str]) -> pl.DataFrame:
    if df.is_empty():
        return df

    ordered_columns: list[str] = []
    for col in [*priority_columns, *DEFAULT_SUMMARY_COLUMNS]:
        if col in df.columns and col not in ordered_columns:
            ordered_columns.append(col)

    selected = ordered_columns or df.columns
    return df.select(selected) if selected else df


def _markdown_summary(title: str, df: pl.DataFrame, max_rows: int) -> str:
    lines = [
        f"# {title}",
        "",
        "Rows are ranked from best to worst according to the sandbox scoring heuristic.",
        "Use this as the quick experiment readout, then inspect the canonical Parquet if you need full fidelity.",
        "",
    ]
    if df.is_empty():
        lines.append("No rows were produced.")
        return "\n".join(lines) + "\n"

    markdown_df = df.head(max_rows)
    lines.append(_markdown_table(markdown_df))
    if df.height > max_rows:
        lines.extend(["", f"Showing top {max_rows} of {df.height} rows."])
    return "\n".join(lines) + "\n"


def _markdown_table(df: pl.DataFrame) -> str:
    columns = df.columns
    header = "| " + " | ".join(_escape_markdown(col) for col in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for row in df.iter_rows(named=True):
        rows.append("| " + " | ".join(_format_markdown_value(row.get(col)) for col in columns) + " |")
    return "\n".join(rows)


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:.2f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).replace("\n", " ").replace("\r", " ")
    if len(text) > 180:
        text = text[:177] + "..."
    return _escape_markdown(text)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")
