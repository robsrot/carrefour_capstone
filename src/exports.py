"""Final exports and evidence-based model-selection decision log."""

from __future__ import annotations

import os
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.evaluation import quality_gate_result
from src.profiling import (
    PRODUCT_RANKING_BASIS,
    flatten_profiles_for_csv,
    profile_quality_summary,
    stage68_artifact_paths,
)
from src.progress import log_event, stage_timer


FIGURE_ARTIFACT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".html"}


def collect_cached_presentation_figures(
    *,
    cfg: PipelineConfig = CONFIG,
    include_model_selection: bool = True,
) -> dict[str, Path]:
    """Return presentation-ready figure artifacts already present in the output cache."""

    figures_dir = cfg.figures
    figures: dict[str, Path] = {}
    seen_paths: set[str] = set()

    def add(label: str, path: Path) -> None:
        if not path.exists() or path.suffix.lower() not in FIGURE_ARTIFACT_SUFFIXES:
            return
        path_key = _stage9_path_identity(path)
        if path_key in seen_paths:
            return
        figures[_unique_stage9_label(label, figures)] = path
        seen_paths.add(path_key)

    known_presentation_files = [
        ("Selected Tribe Vs Population Dashboard", f"stage_07_tribe_vs_population_evidence_dashboard_{cfg.mode}.png"),
        ("Selected Tribe Theme Lift Heatmap", f"stage_07_tribe_theme_lift_heatmap_{cfg.mode}.png"),
        ("Core Tribe Sizes", f"stage_09_core_tribe_sizes_{cfg.mode}.png"),
        ("Assignment Provenance", f"stage_09_assignment_provenance_{cfg.mode}.png"),
        ("Shopping Mission Overview", f"stage_09_shopping_mission_overview_{cfg.mode}.png"),
        ("Core Mission Lift Heatmap", f"stage_09_core_tribe_by_shopping_mission_lift_{cfg.mode}.png"),
        ("Stage 6.6 PCA Projection", f"stage_06_6_winner_projection_pca.png"),
        ("Stage 6.6 UMAP Projection", f"stage_06_6_winner_projection_umap.png"),
        ("Stage 6.7 Remaining Noise UMAP Probe", "stage6_7_remaining_noise_umap_probe.png"),
        ("Stage 9 PCA Projection", f"stage_09_final_projection_pca.png"),
        ("Stage 9 UMAP Projection", f"stage_09_final_projection_umap.png"),
    ]
    for label, filename in known_presentation_files:
        add(label, figures_dir / filename)

    projection_paths: list[Path] = []
    if figures_dir.exists():
        projection_paths.extend(sorted(figures_dir.glob("stage_*_projection_*.png")))
    for path in projection_paths:
        add(_cached_stage9_figure_label(path, cfg), path)

    if figures_dir.exists():
        for path in sorted(figures_dir.iterdir()):
            add(_cached_stage9_figure_label(path, cfg), path)
    return figures


def merge_stage9_figure_paths(*figure_maps: Mapping[str, Any] | None) -> dict[str, Path]:
    """Merge nested figure path maps while keeping the first label for each path."""

    merged: dict[str, Path] = {}
    seen_paths: set[str] = set()
    for figure_map in figure_maps:
        for label, path in _flatten_stage9_paths(figure_map or {}):
            path = Path(path)
            path_key = _stage9_path_identity(path)
            if path_key in seen_paths:
                continue
            merged[_unique_stage9_label(label, merged)] = path
            seen_paths.add(path_key)
    return merged


def collect_cached_stage9_evidence_paths(*, cfg: PipelineConfig = CONFIG) -> dict[str, Path]:
    """Return final-handoff support exports that should appear in the Stage 9 manifest."""

    support_dir = cfg.artifacts / "stage7" / "final_handoff" / "supporting_tables"
    stage68_paths = stage68_artifact_paths(cfg)
    candidates = {
        "Stage 6.8 Tribe Evidence Manifest": stage68_paths["manifest_json"],
        "Stage 6.8 Product Lift Table": stage68_paths["product_lifts_path"],
        "Stage 6.8 Sector Lift Table": stage68_paths["sector_lifts_path"],
        "Stage 6.8 Customer Metric Tests": stage68_paths["customer_metric_tests_csv"],
        "Stage 6.8 Raw Transaction Export Directory": stage68_paths["transaction_export_dir"],
        "Stage 6.8 Customer Summary Export Directory": stage68_paths["customer_export_dir"],
        "Noise Vs Core Customer Metric Diagnostic": (
            support_dir / f"stage7_final_noise_vs_core_customer_metrics_{cfg.mode}.csv"
        ),
        "Per-Tribe Raw Transaction Export Manifest": (
            support_dir / "tribe_raw_transactions" / f"tribe_raw_transactions_manifest_{cfg.mode}.csv"
        ),
        "Per-Tribe Raw Transaction Export Directory": support_dir / "tribe_raw_transactions",
        "Per-Tribe Customer Summary Export Manifest": (
            support_dir / "tribe_customer_summaries" / f"tribe_customer_summaries_manifest_{cfg.mode}.csv"
        ),
        "Per-Tribe Customer Summary Export Directory": support_dir / "tribe_customer_summaries",
    }
    return {label: path for label, path in candidates.items() if path.exists()}


def build_model_comparison(
    candidate_results: list[dict[str, Any]],
    profile_paths: dict[str, Path] | None = None,
    output_path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Write a client-readable model comparison table."""

    cfg.ensure_directories()
    with stage_timer("Stage 9 exports", "building model comparison", cfg=cfg, candidates=len(candidate_results)):
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
                    f"{quality['clusters_with_sector_lift']}/{quality['profiled_clusters']} have sector lift; "
                    f"{quality.get('clusters_with_theme_lift', 0)}/{quality['profiled_clusters']} have theme lift; "
                    f"avg soft-assigned share {float(quality.get('avg_soft_assigned_share') or 0.0):.2%}."
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
            else cfg.reports
            / str(cfg.get("exports.model_comparison_template", "model_comparison_{mode}.csv")).format(mode=cfg.mode)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(ranked).write_csv(output)
        selected = next((row for row in ranked if row.get("recommendation") == "Selected"), None)
        log_event(
            "Stage 9 exports",
            "wrote model comparison",
            cfg=cfg,
            selected=selected.get("model_variant") if selected else "none",
            path=output,
        )
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
        else cfg.reports
        / str(cfg.get("exports.assignments_template", "customer_tribe_assignments_{mode}.parquet")).format(mode=cfg.mode)
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
            "assignment_confidence_score",
            "assignment_confidence_type",
            "assignment_source",
        ]
        if col in assignments.columns
    ]
    assignments = assignments.select(columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    assignments.write_parquet(output)
    log_event("Stage 9 exports", "wrote final assignments", cfg=cfg, rows=assignments.height, path=output)
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
    exported = flatten_profiles_for_csv(selected_profile_path, output)
    log_event("Stage 9 exports", "wrote final profiles", cfg=cfg, path=exported)
    return exported


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
            else cfg.reports
            / str(cfg.get("exports.decision_log_template", "decision_log_{mode}.md")).format(mode=cfg.mode)
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
        log_event("Stage 9 exports", "wrote no-selection decision log", cfg=cfg, path=output)
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
        "## Tribe Discovery Progression",
        "",
        "The final segmentation should be read as a three-layer discovery process:",
        "",
        "1. Organic core tribes: dense product-purchase groups discovered by the clustering model without forcing every customer into a tribe.",
        "2. Confidence-scored soft assignment: nearby non-core customers are attached for campaign usability while preserving assignment provenance and confidence.",
        "3. Evidence-backed subtribes: within-tribe product, sector, theme, and product-term lifts explain what makes each tribe distinctive.",
        "",
        "Core members carry the strongest organic evidence. Soft-assigned members and subtribes are useful for activation, but should be interpreted through their confidence and lift evidence.",
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
            f"{_fmt_metric(profile_quality.get('avg_max_product_lift'))} where available. "
            f"Average strong theme lifts per tribe: "
            f"{_fmt_metric(profile_quality.get('avg_strong_theme_lifts_per_cluster'))}."
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
            "- Product-theme lift: strategic purchase themes derived from product descriptions.",
            "- Assignment provenance: core HDBSCAN assignments versus q95 soft-assigned customers.",
            "- Assignment confidence for probabilistic candidates and distance-percentile soft assignments.",
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
        else cfg.reports / str(cfg.get("exports.decision_log_template", "decision_log_{mode}.md")).format(mode=cfg.mode)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    log_event("Stage 9 exports", "wrote decision log", cfg=cfg, path=output)
    return output


def _fmt_metric(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.4f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def presentation_report_dir(cfg: PipelineConfig = CONFIG) -> Path:
    """Return the compact client-facing artifact directory."""

    return cfg.artifacts / str(cfg.get("exports.presentation_dir", "presentation"))


def evidence_report_dir(cfg: PipelineConfig = CONFIG) -> Path:
    """Return the detailed audit/evidence artifact directory."""

    return cfg.artifacts / str(cfg.get("exports.evidence_dir", "evidence"))


def write_stage9_presentation_pack(
    *,
    selected: Mapping[str, Any],
    core_summary_path: str | Path,
    mission_summary_path: str | Path,
    clustering_atlas_path: str | Path,
    assignment_path: str | Path,
    profile_path: str | Path,
    figure_paths: Mapping[str, Any] | None = None,
    evidence_paths: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write the minimal storyboard for Stage 9 presentation and handoff."""

    out_dir = Path(output_dir) if output_dir else presentation_report_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_output = out_dir / f"00_stage9_presentation_pack_{cfg.mode}.md"
    html_output = out_dir / f"00_stage9_presentation_pack_{cfg.mode}.html"
    manifest_output = out_dir / f"00_stage9_artifact_manifest_{cfg.mode}.csv"

    stats = _stage9_stats(core_summary_path, mission_summary_path)
    core = _read_csv(core_summary_path)
    missions = _read_csv(mission_summary_path)
    figures_for_manifest = merge_stage9_figure_paths(
        collect_cached_presentation_figures(cfg=cfg),
        figure_paths or {},
    )
    evidence_for_manifest = {
        **collect_cached_stage9_evidence_paths(cfg=cfg),
        **(evidence_paths or {}),
    }
    manifest = _stage9_artifact_manifest(
        core_summary_path=core_summary_path,
        mission_summary_path=mission_summary_path,
        clustering_atlas_path=clustering_atlas_path,
        assignment_path=assignment_path,
        profile_path=profile_path,
        figure_paths=figures_for_manifest,
        evidence_paths=evidence_for_manifest,
        base_dir=out_dir,
    )
    manifest.write_csv(manifest_output)
    md_output.write_text(
        _stage9_storyboard_markdown(
            selected=selected,
            stats=stats,
            core=core,
            missions=missions,
            manifest=manifest,
            manifest_output=manifest_output,
            cfg=cfg,
        ),
        encoding="utf-8",
    )
    html_output.write_text(
        _stage9_storyboard_html(
            selected=selected,
            stats=stats,
            core=core,
            missions=missions,
            manifest=manifest,
            manifest_output=manifest_output,
            cfg=cfg,
        ),
        encoding="utf-8",
    )
    log_event(
        "Stage 9 presentation pack",
        "wrote curated presentation index",
        cfg=cfg,
        markdown=md_output,
        html=html_output,
        manifest=manifest_output,
    )
    return {"markdown": md_output, "html": html_output, "manifest": manifest_output}


def _stage9_stats(core_summary_path: str | Path, mission_summary_path: str | Path) -> dict[str, Any]:
    core = _read_csv(core_summary_path)
    missions = _read_csv(mission_summary_path)
    return {
        "core_tribes": core.height,
        "assigned_customers": int(core["n_customers"].sum()) if "n_customers" in core.columns and core.height else 0,
        "mission_segments": missions.height,
        "customer_mission_memberships": (
            int(missions["mission_customers"].sum())
            if "mission_customers" in missions.columns and missions.height
            else 0
        ),
    }


def _stage9_artifact_manifest(
    *,
    core_summary_path: str | Path,
    mission_summary_path: str | Path,
    clustering_atlas_path: str | Path,
    assignment_path: str | Path,
    profile_path: str | Path,
    figure_paths: Mapping[str, Any],
    evidence_paths: Mapping[str, Any],
    base_dir: Path,
) -> pl.DataFrame:
    rows: list[dict[str, str]] = [
        _stage9_artifact_row("presentation", "01 Core tribe summary", core_summary_path, base_dir),
        _stage9_artifact_row("presentation", "02 Shopping mission summary", mission_summary_path, base_dir),
        _stage9_artifact_row("presentation", "03 Clustering atlas", clustering_atlas_path, base_dir),
        _stage9_artifact_row("official export", "Customer tribe assignments", assignment_path, base_dir),
        _stage9_artifact_row("official export", "Flattened tribe profiles", profile_path, base_dir),
    ]
    for label, path in _flatten_stage9_paths(figure_paths):
        rows.append(_stage9_artifact_row("presentation visual", label, path, base_dir))
    for label, path in _flatten_stage9_paths(evidence_paths):
        rows.append(_stage9_artifact_row(_stage9_tier_for_extra_artifact(label), label, path, base_dir))
    return pl.DataFrame(rows)


def _stage9_artifact_row(tier: str, label: str, path: str | Path, base_dir: Path) -> dict[str, str]:
    return {
        "tier": tier,
        "artifact": label,
        "path": str(path),
        "relative_link": _stage9_artifact_href(path, base_dir),
        "purpose": _stage9_purpose_for_label(label, tier),
    }


def _flatten_stage9_paths(value: Any, prefix: str = "") -> list[tuple[str, Path]]:
    if not value:
        return []
    if isinstance(value, Mapping):
        flattened: list[tuple[str, Path]] = []
        for key, nested in value.items():
            label = f"{prefix} {_stage9_label_from_key(key)}".strip()
            flattened.extend(_flatten_stage9_paths(nested, label))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for idx, nested in enumerate(value, start=1):
            label = f"{prefix} {idx}".strip()
            flattened.extend(_flatten_stage9_paths(nested, label))
        return flattened
    if isinstance(value, (str, Path)):
        return [(prefix or Path(value).stem.replace("_", " ").title(), Path(value))]
    return []


def _stage9_label_from_key(key: Any) -> str:
    label = str(key).replace("_", " ").strip()
    return label if any(char.isupper() for char in label) else label.title()


def _stage9_path_identity(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def _unique_stage9_label(label: str, existing: Mapping[str, Any]) -> str:
    base = str(label).strip() or "Figure"
    if base not in existing:
        return base
    idx = 2
    while f"{base} {idx}" in existing:
        idx += 1
    return f"{base} {idx}"


def _stage9_tier_for_extra_artifact(label: str) -> str:
    text = label.lower()
    if "raw transaction export" in text or "customer summary export" in text:
        return "official export"
    return "evidence"


def _cached_stage9_figure_label(path: Path, cfg: PipelineConfig) -> str:
    stem = path.stem
    mode_suffix = f"_{cfg.mode}"
    if stem.endswith(mode_suffix):
        stem = stem[: -len(mode_suffix)]
    if stem.startswith("umap_selected_tribes_"):
        return "UMAP 2D Customer Map"
    if stem.startswith("pca_selected_tribes_"):
        return "PCA 2D Customer Map"
    label = stem.replace("_", " ").title()
    for old, new in {
        "Pca": "PCA",
        "Umap": "UMAP",
        "Html": "HTML",
        "2D": "2D",
        "Vs": "Vs",
    }.items():
        label = label.replace(old, new)
    return label


def _stage9_purpose_for_label(label: str, tier: str) -> str:
    text = label.lower()
    if "raw transaction export" in text:
        return "Analyst per-tribe prepared transaction-line parquet exports with tribe_id and assignment confidence attached."
    if "customer summary export" in text:
        return "Analyst per-tribe customer-level KPI parquet exports with tribe_id and assignment confidence attached."
    if "noise vs core" in text:
        return "Noise population diagnostic comparing behavioral KPIs against assigned core customers; profiling context only."
    if "stage 6.8" in text and "manifest" in text:
        return "Cache manifest for the precomputed tribe evidence bundle consumed by Stage 7."
    if "stage 6.8" in text and "product lift" in text:
        return "Full tribe-by-product lift table precomputed before Stage 7 interpretation."
    if "stage 6.8" in text and "sector lift" in text:
        return "Full tribe-by-sector lift table precomputed before Stage 7 interpretation."
    if "stage 6.8" in text and "customer metric" in text:
        return "Precomputed customer-level behavioral ANOVA table for Stage 7 and appendix review."
    if "stage 6.8" in text and "raw transaction" in text:
        return "Analyst per-tribe prepared transaction-line parquet exports written before Stage 7."
    if "stage 6.8" in text and "customer summary" in text:
        return "Analyst per-tribe customer-level behavioral summary parquet exports written before Stage 7."
    if "remaining noise" in text and "umap probe" in text:
        return "Visual review of remaining HDBSCAN noise before any optional candidate-only third clustering pass."
    if "core tribe summary" in text:
        return "First read: compact list of organic core tribes, sizes, soft-assignment share, and product evidence."
    if "shopping mission" in text:
        return "Client-facing activation layer: multi-label purchase missions supported by product evidence."
    if "clustering atlas" in text:
        return "Detailed storybook connecting core tribes, subtribes, products, terms, and visuals."
    if "heatmap" in text or "mission lift" in text:
        return "Visual proof of how shopping missions concentrate across organic core tribes."
    if "2d" in text or "map" in text:
        return "Visual review of the selected customer segmentation in a reduced 2D feature space."
    if "overview" in text:
        return "Visual size and confidence summary for the shopping mission layer."
    if "assignment" in text:
        return "Operational customer-level export with tribe ID, assignment provenance, and confidence fields."
    if "profile" in text:
        return "Detailed product, sector, theme, term, and KPI evidence for audit and appendix use."
    if tier == "evidence":
        return "Supporting table for audit, appendix, or deeper category expert review."
    return "Supporting Stage 9 artifact."


def _stage9_storyboard_markdown(
    *,
    selected: Mapping[str, Any],
    stats: Mapping[str, Any],
    core: pl.DataFrame,
    missions: pl.DataFrame,
    manifest: pl.DataFrame,
    manifest_output: Path,
    cfg: PipelineConfig,
) -> str:
    presentation = manifest.filter(pl.col("tier").str.contains("presentation"))
    exports = manifest.filter(pl.col("tier") == "official export")
    evidence = manifest.filter(pl.col("tier") == "evidence")
    lines = [
        "# Stage 9 Tribe and Shopping Mission Storyboard",
        "",
        f"Run mode: `{cfg.mode}`",
        "",
        "This is the clean, minimal reading path for the final segmentation. Use it to explain the result without getting lost in intermediate outputs.",
        "",
        "## One-Slide Story",
        "",
        f"- Model: **{selected.get('model_name')} ({selected.get('model_variant')})**",
        f"- Core tribes: **{stats.get('core_tribes')}**",
        f"- Assigned customers: **{int(stats.get('assigned_customers') or 0):,}**",
        f"- Shopping missions surfaced: **{stats.get('mission_segments')}**",
        f"- Customer-mission memberships: **{int(stats.get('customer_mission_memberships') or 0):,}**",
        "",
        "## Story Flow",
        "",
        "1. **Organic core tribes**: the clustering model finds mutually exclusive product-behavior groups.",
        "2. **Assignment confidence**: core customers are strongest; soft assignment extends campaign coverage.",
        "3. **Shopping missions**: product evidence surfaces multi-label activation audiences across the core tribes.",
        "4. **Evidence backup**: the atlas and evidence folder show the products, themes, terms, and confidence behind the claims.",
        "",
        "Labels are purchase-behavior summaries, not demographic, religious, household, or identity claims.",
        PRODUCT_RANKING_BASIS,
        "",
        "## Core Tribe Preview",
        "",
        _stage9_simple_markdown_table(_core_story_preview(core)),
        "",
        "## Shopping Mission Preview",
        "",
        "Mission size shows audience scale. Use the mission-lift heatmap to judge which missions are meaningfully concentrated in specific core tribes.",
        "",
        _stage9_simple_markdown_table(_mission_story_preview(missions)),
        "",
        "## Open These First",
        "",
        _stage9_manifest_markdown_table(presentation),
        "",
        "## Backup Evidence",
        "",
        _stage9_manifest_markdown_table(exports),
        "",
        _stage9_manifest_markdown_table(evidence),
        "",
        f"Full artifact manifest: `{manifest_output}`",
        "",
    ]
    return "\n".join(lines)


def _stage9_storyboard_html(
    *,
    selected: Mapping[str, Any],
    stats: Mapping[str, Any],
    core: pl.DataFrame,
    missions: pl.DataFrame,
    manifest: pl.DataFrame,
    manifest_output: Path,
    cfg: PipelineConfig,
) -> str:
    exports = manifest.filter(pl.col("tier") == "official export")
    evidence = manifest.filter(pl.col("tier") == "evidence")
    core_preview = _core_story_preview(core)
    mission_preview = _mission_story_preview(missions)
    core_sizes = _stage9_manifest_href(manifest, "Core Tribe Sizes")
    provenance = _stage9_manifest_href(manifest, "Assignment Provenance")
    umap_customer_map = _stage9_manifest_href(manifest, "UMAP 2D Customer Map")
    pca_customer_map = _stage9_manifest_href(manifest, "PCA 2D Customer Map")
    mission_overview = _stage9_manifest_href(manifest, "Shopping Mission Overview")
    mission_heatmap = _stage9_manifest_href(manifest, "Core Mission Lift Heatmap")
    atlas = _stage9_manifest_href(manifest, "03 Clustering atlas")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 9 Tribe and Shopping Mission Storyboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #17202a; background: #f6f7f9; }}
header {{ background: #ffffff; border-bottom: 1px solid #d0d5dd; padding: 26px 34px; }}
main {{ padding: 24px 34px 42px; }}
h1, h2, h3 {{ margin: 0; }}
h1 {{ font-size: 26px; }}
h2 {{ margin-top: 28px; margin-bottom: 12px; font-size: 20px; }}
h3 {{ font-size: 15px; margin-bottom: 8px; }}
.muted {{ color: #667085; font-size: 13px; }}
.snapshot {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 18px; }}
.metric, .story-step, .artifact-card, details {{ background: #ffffff; border: 1px solid #d0d5dd; border-radius: 8px; padding: 14px; }}
.metric strong {{ display: block; font-size: 23px; margin-top: 4px; }}
.flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-top: 14px; }}
.story-step b {{ display: inline-flex; width: 26px; height: 26px; border-radius: 50%; align-items: center; justify-content: center; background: #175cd3; color: #ffffff; margin-right: 7px; }}
.visual-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
.visual {{ background: #ffffff; border: 1px solid #d0d5dd; border-radius: 8px; padding: 12px; }}
.visual img {{ display: block; width: 100%; max-height: 520px; object-fit: contain; background: #ffffff; margin-top: 10px; }}
.artifact-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
.artifact-card strong {{ display: block; margin-bottom: 6px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; background: #ffffff; }}
th, td {{ border: 1px solid #d0d5dd; padding: 9px; vertical-align: top; font-size: 13px; }}
th {{ background: #eef2f6; text-align: left; }}
details {{ margin-top: 12px; }}
summary {{ cursor: pointer; font-weight: 700; }}
a {{ color: #175cd3; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
<h1>Stage 9 Tribe and Shopping Mission Storyboard</h1>
<p class="muted">Run mode: {escape(str(cfg.mode))}. Model: {escape(str(selected.get('model_name')))} ({escape(str(selected.get('model_variant')))}).</p>
<p class="muted">Labels are purchase-behavior summaries, not demographic, religious, household, or identity claims.</p>
<p class="muted">{escape(PRODUCT_RANKING_BASIS)}</p>
<div class="snapshot">
<div class="metric">Core tribes<strong>{stats.get('core_tribes')}</strong></div>
<div class="metric">Assigned customers<strong>{int(stats.get('assigned_customers') or 0):,}</strong></div>
<div class="metric">Shopping missions<strong>{stats.get('mission_segments')}</strong></div>
<div class="metric">Mission memberships<strong>{int(stats.get('customer_mission_memberships') or 0):,}</strong></div>
</div>
</header>
<main>
<h2>The Story Flow</h2>
<div class="flow">
<div class="story-step"><h3><b>1</b>Organic core tribes</h3><p class="muted">The model discovers mutually exclusive product-behavior groups. These are the statistical backbone.</p></div>
<div class="story-step"><h3><b>2</b>Assignment confidence</h3><p class="muted">Core members are strongest. Soft-assigned customers extend coverage while preserving provenance and confidence.</p></div>
<div class="story-step"><h3><b>3</b>Shopping missions</h3><p class="muted">Specific activation audiences are identified from product evidence and can overlap for the same customer.</p></div>
<div class="story-step"><h3><b>4</b>Evidence backup</h3><p class="muted">The atlas and evidence folder provide the products, themes, terms, and lifts behind every claim.</p></div>
</div>

<h2>Visual Evidence</h2>
<div class="visual-grid">
{_stage9_visual_card("Core tribe size", core_sizes, "Shows the organic tribe count and customer distribution.")}
{_stage9_visual_card("Assignment confidence", provenance, "Separates HDBSCAN core customers from soft-assigned coverage.")}
{_stage9_visual_card("UMAP 2D customer map", umap_customer_map, "Visual review aid for local neighborhood structure; not an automatic model winner.")}
{_stage9_visual_card("PCA 2D customer map", pca_customer_map, "Linear projection baseline for comparing whether the selected tribes remain readable.")}
{_stage9_visual_card("Shopping mission overview", mission_overview, "Shows the size and confidence of each mission audience.")}
{_stage9_visual_card("Mission concentration by core tribe", mission_heatmap, "Shows which missions are genuinely concentrated in which organic tribes.")}
</div>

<h2>Core Tribe Preview</h2>
{_stage9_story_table_html(core_preview)}

<h2>Shopping Mission Preview</h2>
<p class="muted">Mission size shows audience scale. Use the mission-lift heatmap above to judge which missions are meaningfully concentrated in specific core tribes.</p>
{_stage9_story_table_html(mission_preview)}

<h2>Minimal Evidence Pack</h2>
<div class="artifact-row">
{_stage9_artifact_card("Core tribes", _stage9_manifest_href(manifest, "01 Core tribe summary"), "Compact tribe list with sizes, soft-assignment share, and product evidence.")}
{_stage9_artifact_card("Shopping missions", _stage9_manifest_href(manifest, "02 Shopping mission summary"), "Multi-label activation audiences backed by product evidence.")}
{_stage9_artifact_card("Clustering atlas", atlas, "Deeper proof layer for stakeholders who ask how a label was derived.")}
</div>

<details>
<summary>Backup evidence and operational exports</summary>
<h3>Operational exports</h3>
{_stage9_html_table(exports)}
<h3>Detailed evidence</h3>
{_stage9_html_table(evidence)}
</details>
<p class="muted">Full artifact manifest: {escape(str(manifest_output))}</p>
</main>
</body>
</html>
"""


def _core_story_preview(core: pl.DataFrame) -> pl.DataFrame:
    if core.is_empty():
        return pl.DataFrame()
    rows = []
    for row in core.sort("tribe_id").iter_rows(named=True):
        rows.append(
            {
                "tribe": f"T{row.get('tribe_id')}",
                "working_label": _stage9_truncate(row.get("suggested_tribe_name"), 42),
                "customers": _stage9_fmt_int(row.get("n_customers")),
                "population_share": _stage9_fmt_pct_value(row.get("population_share_pct")),
                "soft_assigned": _stage9_fmt_pct_value(row.get("soft_assigned_share_pct")),
                "top_evidence": _stage9_truncate(row.get("top_themes") or row.get("top_product_terms") or row.get("top_products"), 120),
            }
        )
    return pl.DataFrame(rows)


def _mission_story_preview(missions: pl.DataFrame, limit: int = 12) -> pl.DataFrame:
    if missions.is_empty():
        return pl.DataFrame()
    rows = []
    for row in missions.sort("mission_customers", descending=True).head(limit).iter_rows(named=True):
        rows.append(
            {
                "mission_family": row.get("mission_family"),
                "shopping_mission": _stage9_truncate(row.get("mission_label"), 44),
                "customers": _stage9_fmt_int(row.get("mission_customers")),
                "population_share": _stage9_fmt_pct_value(row.get("population_share_pct")),
                "confidence": row.get("mission_confidence"),
                "largest_core_tribe_memberships": _stage9_truncate(row.get("top_core_tribes"), 130),
            }
        )
    return pl.DataFrame(rows)


def _stage9_manifest_href(manifest: pl.DataFrame, artifact: str) -> str:
    if manifest.is_empty() or "artifact" not in manifest.columns:
        return ""
    match = manifest.filter(pl.col("artifact") == artifact)
    if match.is_empty():
        return ""
    return str(match[0, "relative_link"] or "")


def _stage9_visual_card(title: str, href: str, caption: str) -> str:
    if not href:
        return (
            "<div class='visual'>"
            f"<h3>{escape(title)}</h3>"
            f"<p class='muted'>{escape(caption)}</p>"
            "<p class='muted'>Visual not generated in this run.</p>"
            "</div>"
        )
    if Path(href).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        visual = f"<a href='{escape(href)}'><img src='{escape(href)}' alt='{escape(title)}'></a>"
    else:
        visual = f"<p><a href='{escape(href)}'>Open artifact</a></p>"
    return (
        "<div class='visual'>"
        f"<h3>{escape(title)}</h3>"
        f"<p class='muted'>{escape(caption)}</p>"
        f"{visual}"
        "</div>"
    )


def _stage9_artifact_card(title: str, href: str, caption: str) -> str:
    link = f"<a href='{escape(href)}'>Open</a>" if href else "<span class='muted'>Not generated</span>"
    return (
        "<div class='artifact-card'>"
        f"<strong>{escape(title)}</strong>"
        f"<p class='muted'>{escape(caption)}</p>"
        f"{link}"
        "</div>"
    )


def _stage9_story_table_html(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "<p class='muted'>No rows available.</p>"
    header = "".join(f"<th>{escape(str(col).replace('_', ' ').title())}</th>" for col in df.columns)
    rows = []
    for row in df.iter_rows(named=True):
        cells = "".join(f"<td>{escape(str(row.get(col) or ''))}</td>" for col in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _stage9_simple_markdown_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "No rows available."
    columns = df.columns
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.iter_rows(named=True):
        values = [str(row.get(col) or "").replace("|", "/").replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _stage9_manifest_markdown_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "No artifacts."
    cols = [col for col in ["tier", "artifact", "relative_link", "purpose"] if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.select(cols).iter_rows(named=True):
        values = [str(row.get(col) or "").replace("|", "/").replace("\n", " ") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _stage9_html_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "<p class='muted'>No artifacts.</p>"
    rows = []
    for row in df.iter_rows(named=True):
        href = escape(str(row.get("relative_link") or row.get("path") or ""))
        artifact = escape(str(row.get("artifact") or ""))
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('tier') or ''))}</td>"
            f"<td><a href='{href}'>{artifact}</a></td>"
            f"<td>{escape(str(row.get('purpose') or ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Tier</th><th>Artifact</th><th>Purpose</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _stage9_fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return str(value)


def _stage9_fmt_pct_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _stage9_truncate(value: Any, max_len: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3].rstrip() + "..."


def _read_csv(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path)


def _stage9_artifact_href(path: str | Path, base_dir: Path) -> str:
    try:
        return Path(os.path.relpath(Path(path), base_dir)).as_posix()
    except ValueError:
        return Path(path).as_posix()


