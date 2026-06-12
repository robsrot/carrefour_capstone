"""Final exports and evidence-based model-selection decision log."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.profiling import flatten_profiles_for_csv, profile_quality_summary


def build_model_comparison(
    candidate_results: list[dict[str, Any]],
    profile_paths: dict[str, Path] | None = None,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Write a client-readable model comparison table."""

    cfg.ensure_directories()
    rows = []
    for result in candidate_results:
        row = dict(result)
        profile_key = f"{row.get('model_name')}::{row.get('model_variant')}"
        profile_path = (profile_paths or {}).get(profile_key) or (profile_paths or {}).get(row.get("model_name"))
        if profile_path and Path(profile_path).exists():
            quality = profile_quality_summary(profile_path)
            row.update(quality)
            row["interpretability_summary"] = (
                f"{quality['clusters_with_product_lift']}/{quality['profiled_clusters']} tribes have strong product lift; "
                f"{quality['clusters_with_sector_lift']}/{quality['profiled_clusters']} have sector lift."
            )
            row["profile_path"] = str(profile_path)
        else:
            row["interpretability_summary"] = "Not profiled yet."
            row["profile_path"] = None
        row["cluster_balance"] = (
            f"min share={_fmt_pct(row.get('min_cluster_share'))}, "
            f"max share={_fmt_pct(row.get('max_cluster_share'))}, "
            f"noise={row.get('noise_pct', 0):.2f}%"
        )
        rows.append(row)

    scored = _score_rows(rows)
    output = (
        Path(output_path)
        if output_path
        else cfg.reports / cfg.get("exports.model_comparison_template").format(mode=cfg.mode)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(scored).write_csv(output)
    return output


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        silhouette = row.get("silhouette") or 0.0
        db = row.get("davies_bouldin")
        balance_cv = row.get("cluster_size_cv")
        noise_pct = row.get("noise_pct") or 0.0
        confidence = row.get("avg_assignment_confidence") or 0.0
        product_lift = row.get("avg_strong_product_lifts_per_cluster") or 0.0
        score = float(silhouette)
        score += min(float(product_lift), 5.0) * 0.04
        score += float(confidence) * 0.05
        if db is not None:
            score -= min(float(db), 5.0) * 0.03
        if balance_cv is not None:
            score -= min(float(balance_cv), 3.0) * 0.03
        score -= min(float(noise_pct), 100.0) / 100.0 * 0.08
        row["selection_score"] = round(score, 6)

    ranked = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["final_rank"] = idx
        row["recommendation"] = "Selected" if idx == 1 else "Rejected"
    return ranked


def selected_model(comparison_path: str | Path) -> dict[str, Any]:
    comparison = pl.read_csv(comparison_path)
    selected = comparison.sort("final_rank").row(0, named=True)
    return selected


def export_final_assignments(
    selected_assignment_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    output = (
        Path(output_path)
        if output_path
        else cfg.reports / cfg.get("exports.assignments_template").format(mode=cfg.mode)
    )
    assignments = pl.read_parquet(selected_assignment_path).select(
        ["cliente", "tribe_id", "model_name", "assignment_probability"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    assignments.write_parquet(output)
    return output


def export_final_profiles(
    selected_profile_path: str | Path,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    output = (
        Path(output_path)
        if output_path
        else cfg.reports / cfg.get("profiling.output_csv_template").format(mode=cfg.mode)
    )
    return flatten_profiles_for_csv(selected_profile_path, output)


def write_decision_log(
    comparison_path: str | Path,
    selected_profile_path: str | Path | None = None,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Write the final evidence-based decision log requested by the brief."""

    comparison = pl.read_csv(comparison_path).sort("final_rank")
    selected = comparison.row(0, named=True)
    cluster_count = int(selected.get("cluster_count") or 0)
    low = int(cfg.get("modeling.client_hypothesis_min", 10))
    high = int(cfg.get("modeling.client_hypothesis_max", 15))
    agrees = low <= cluster_count <= high
    profile_quality = profile_quality_summary(selected_profile_path) if selected_profile_path else {}

    lines = [
        "# Decision Log: Final Tribe Model Selection",
        "",
        f"Run mode: `{cfg.mode}`",
        "",
        "## Selected Model",
        "",
        f"Selected model: **{selected.get('model_name')} ({selected.get('model_variant')})**.",
        f"Selected tribe count: **{cluster_count}**.",
        (
            f"This {'agrees' if agrees else 'does not agree'} with the client hypothesis "
            f"of {low}-{high} tribes."
        ),
        "",
        "The client expectation of 10-15 tribes was treated as a hypothesis rather than a constraint. "
        "The selected number of tribes was chosen based on empirical evaluation and interpretability, "
        "not because it matched the expected range.",
        "",
        "## Why This Model Was Selected",
        "",
        (
            f"The selected solution ranked first on the evidence score with silhouette "
            f"{_fmt_metric(selected.get('silhouette'))}, Davies-Bouldin "
            f"{_fmt_metric(selected.get('davies_bouldin'))}, noise share "
            f"{_fmt_metric(selected.get('noise_pct'), suffix='%')}, and cluster balance "
            f"{selected.get('cluster_balance')}."
        ),
        (
            f"Interpretability evidence: {selected.get('interpretability_summary')}. "
            f"Average maximum product lift was "
            f"{_fmt_metric(profile_quality.get('avg_max_product_lift'))} where available."
        ),
        "",
        "## Why Alternatives Were Not Selected",
        "",
    ]

    for row in comparison.iter_rows(named=True):
        if int(row.get("final_rank") or 0) == 1:
            continue
        lines.append(
            f"- {row.get('model_name')} ({row.get('model_variant')}) was not selected: "
            f"rank {row.get('final_rank')}, silhouette {_fmt_metric(row.get('silhouette'))}, "
            f"Davies-Bouldin {_fmt_metric(row.get('davies_bouldin'))}, "
            f"noise {_fmt_metric(row.get('noise_pct'), suffix='%')}, "
            f"interpretability: {row.get('interpretability_summary')}."
        )

    lines.extend(
        [
            "",
            "## Evidence Used",
            "",
            "- Model comparison metrics: silhouette, Davies-Bouldin, Calinski-Harabasz where available.",
            "- Cluster size balance: minimum and maximum population share, plus size dispersion.",
            "- Interpretability: product lift, sector lift, and profile completeness.",
            "- Assignment confidence for probabilistic GMM candidates.",
            "- Noise share for HDBSCAN candidates.",
            "- Stability evidence should be added from repeated seeds before production rollout if not already cached.",
            "",
            "## Limitations",
            "",
            "- Product descriptions and sector metadata are only as reliable as the prepared product master.",
            "- Autoencoder and HDBSCAN candidates can be sensitive to scaling and sampling choices.",
            "- The current decision log reflects the artifacts available in this run; production should rerun with the full prepared data.",
            "",
            "## Recommended Next Steps",
            "",
            "- Review top lifted products and sectors with Carrefour category experts.",
            "- Rerun the selected model across multiple random seeds and compare tribe stability.",
            "- Validate tribe actionability through campaign, assortment, and promo use cases.",
            "- Run the same notebook in production mode after dev evidence is approved.",
            "",
        ]
    )

    output = (
        Path(output_path)
        if output_path
        else cfg.reports / cfg.get("exports.decision_log_template").format(mode=cfg.mode)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _fmt_metric(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.4f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"
