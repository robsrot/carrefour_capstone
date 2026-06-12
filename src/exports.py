"""Final exports and evidence-based model-selection decision log."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.profiling import flatten_profiles_for_csv, profile_quality_summary
from src.evaluation import quality_gate_result


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
            quality = profile_quality_summary(profile_path, cfg=cfg)
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

    ranked = _rank_rows(rows, cfg=cfg)
    output = (
        Path(output_path)
        if output_path
        else cfg.reports / cfg.get("exports.model_comparison_template").format(mode=cfg.mode)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(ranked).write_csv(output)
    return output


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _sort_metric(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rank_rows(rows: list[dict[str, Any]], cfg: PipelineConfig = CONFIG) -> list[dict[str, Any]]:
    require_profile_lift = bool(cfg.get("quality_gates.require_profile_lift", False))
    min_profile_lift_count = float(cfg.get("quality_gates.min_strong_product_lifts_per_cluster", 1.0))

    for row in rows:
        if "passes_quality_gate" not in row or row.get("passes_quality_gate") in {None, ""}:
            passes_gate, gate_reason = quality_gate_result(row, cfg=cfg)
            row["passes_quality_gate"] = passes_gate
            row["quality_gate_reason"] = gate_reason
        else:
            row["passes_quality_gate"] = _as_bool(row.get("passes_quality_gate"))
            row["quality_gate_reason"] = row.get("quality_gate_reason") or "pass"

        profile_lifts = _sort_metric(row.get("avg_strong_product_lifts_per_cluster"), 0.0)
        profile_available = row.get("profiled_clusters") not in {None, "", 0}
        profile_passes = profile_lifts >= min_profile_lift_count
        row["profile_passes_quality_gate"] = bool(profile_passes)

        blockers = []
        if not row["passes_quality_gate"]:
            blockers.append(str(row.get("quality_gate_reason") or "failed metric gate"))
        if require_profile_lift and (not profile_available or not profile_passes):
            blockers.append(
                f"avg_strong_product_lifts_per_cluster<{min_profile_lift_count}"
                if profile_available
                else "profile_lift_not_available"
            )

        row["eligible_for_selection"] = not blockers
        row["selection_blockers"] = "; ".join(blockers) if blockers else "pass"
        row["selection_basis"] = (
            "Eligible candidates are ranked by coverage_adjusted_silhouette descending, "
            "then Davies-Bouldin ascending, cluster_size_cv ascending, noise_pct ascending, "
            "and product-lift evidence descending. No hidden weighted score is used."
        )

    def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            0.0 if row["eligible_for_selection"] else 1.0,
            -_sort_metric(row.get("coverage_adjusted_silhouette"), _sort_metric(row.get("silhouette"), -999.0)),
            _sort_metric(row.get("davies_bouldin"), 999.0),
            _sort_metric(row.get("cluster_size_cv"), 999.0),
            _sort_metric(row.get("noise_pct"), 999.0),
            -_sort_metric(row.get("avg_strong_product_lifts_per_cluster"), 0.0),
        )

    ranked = sorted(rows, key=_rank_key)
    has_selected = any(row["eligible_for_selection"] for row in ranked)
    for idx, row in enumerate(ranked, start=1):
        row["final_rank"] = idx
        if idx == 1 and row["eligible_for_selection"]:
            row["recommendation"] = "Selected"
        elif not has_selected and idx == 1:
            row["recommendation"] = "No Valid Automatic Selection"
        elif row["eligible_for_selection"]:
            row["recommendation"] = "Rejected"
        else:
            row["recommendation"] = "Review Only"
    return ranked


def selected_model(comparison_path: str | Path) -> dict[str, Any]:
    comparison = pl.read_csv(comparison_path)
    selected_rows = comparison.filter(pl.col("recommendation") == "Selected").sort("final_rank")
    if selected_rows.height == 0:
        raise ValueError(
            "No candidate passed the configured quality gates. Inspect the model comparison before exporting final tribes."
        )
    selected = selected_rows.row(0, named=True)
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
    assignments = pl.read_parquet(selected_assignment_path)
    columns = [
        col
        for col in [
            "cliente",
            "tribe_id",
            "model_name",
            "model_variant",
            "assignment_probability",
            "assignment_confidence_type",
            "assignment_source",
        ]
        if col in assignments.columns
    ]
    assignments = assignments.select(columns)
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
    selected_rows = comparison.filter(pl.col("recommendation") == "Selected")
    if selected_rows.height == 0:
        output = (
            Path(output_path)
            if output_path
            else cfg.reports / cfg.get("exports.decision_log_template").format(mode=cfg.mode)
        )
        lines = [
            "# Decision Log: Final Tribe Model Selection",
            "",
            f"Run mode: `{cfg.mode}`",
            "",
            "## Selection Outcome",
            "",
            "No model was automatically selected because no candidate passed the configured quality gates.",
            "",
            "## Candidate Review",
            "",
        ]
        for row in comparison.iter_rows(named=True):
            lines.append(
                f"- {row.get('model_name')} ({row.get('model_variant')}): "
                f"rank {row.get('final_rank')}, recommendation {row.get('recommendation')}, "
                f"blockers: {row.get('selection_blockers')}."
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
        return output

    selected = selected_rows.sort("final_rank").row(0, named=True)
    cluster_count = int(selected.get("cluster_count") or 0)
    low = int(cfg.get("modeling.client_hypothesis_min", 10))
    high = int(cfg.get("modeling.client_hypothesis_max", 15))
    agrees = low <= cluster_count <= high
    profile_quality = profile_quality_summary(selected_profile_path, cfg=cfg) if selected_profile_path else {}

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
            f"The selected solution passed the configured quality gates and ranked first among eligible candidates "
            f"using coverage-adjusted silhouette, Davies-Bouldin, balance, noise, and product-lift evidence. "
            f"Its silhouette was "
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
