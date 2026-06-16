"""Consolidated dimensionality-reduction clustering experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.dimensionality import build_umap_representation
from src.evaluation import evaluate_labels, quality_gate_result
from src.experiment_reporting import write_summary_artifacts
from src.model_selection import run_umap_hdbscan_experiments
from src.utils import frame_to_numpy, numeric_feature_columns


def consolidated_umap_hdbscan_trials() -> list[dict[str, Any]]:
    """Single-stage UMAP-HDBSCAN trials carried forward from Experiments 6, 7, and 7B."""

    trials = [
        # Experiment 6 carry-forward baselines.
        _trial("local_leaf", 20, 30, 0.0, "cosine", 300, 1, "leaf"),
        _trial("balanced_eom", 20, 50, 0.0, "cosine", 500, 1, "eom"),
        _trial("broad_eom", 15, 75, 0.0, "cosine", 750, 1, "eom"),
        # Experiment 7 carry-forward baselines.
        _trial("u5_n15_leaf_mcs300", 5, 15, 0.0, "cosine", 300, 1, "leaf"),
        _trial("u5_n30_leaf_mcs400", 5, 30, 0.0, "cosine", 400, 1, "leaf"),
        _trial("u8_n15_leaf_mcs350", 8, 15, 0.0, "cosine", 350, 1, "leaf"),
        _trial("u8_n30_leaf_mcs450", 8, 30, 0.0, "cosine", 450, 1, "leaf"),
        _trial("u10_n30_eom_mcs300", 10, 30, 0.0, "cosine", 300, 1, "eom"),
        _trial("u10_n50_leaf_mcs500", 10, 50, 0.0, "cosine", 500, 1, "leaf"),
        # Experiment 7B carry-forward baselines.
        _trial("u8_n20_leaf_mcs325", 8, 20, 0.0, "cosine", 325, 1, "leaf"),
        _trial("u8_n25_leaf_mcs350", 8, 25, 0.0, "cosine", 350, 1, "leaf"),
        _trial("u8_n30_leaf_mcs350", 8, 30, 0.0, "cosine", 350, 1, "leaf"),
        _trial("u10_n40_leaf_mcs350", 10, 40, 0.0, "cosine", 350, 1, "leaf"),
        _trial("u10_n50_leaf_mcs350", 10, 50, 0.0, "cosine", 350, 1, "leaf"),
        _trial("u10_n50_leaf_mcs400", 10, 50, 0.0, "cosine", 400, 1, "leaf"),
        _trial("u8_n20_eom_mcs150", 8, 20, 0.0, "cosine", 150, 1, "eom"),
        _trial("u10_n30_eom_mcs150", 10, 30, 0.0, "cosine", 150, 1, "eom"),
        _trial("u10_n50_eom_mcs200", 10, 50, 0.0, "cosine", 200, 1, "eom"),
        # Smaller min_cluster_size probes around the best target-range LEAF area.
        _trial("u8_n15_leaf_mcs200", 8, 15, 0.0, "cosine", 200, 1, "leaf"),
        _trial("u8_n15_leaf_mcs250", 8, 15, 0.0, "cosine", 250, 1, "leaf"),
        _trial("u8_n20_leaf_mcs250", 8, 20, 0.0, "cosine", 250, 1, "leaf"),
        _trial("u8_n30_leaf_mcs250", 8, 30, 0.0, "cosine", 250, 1, "leaf"),
        _trial("u10_n30_leaf_mcs200", 10, 30, 0.0, "cosine", 200, 1, "leaf"),
        _trial("u10_n30_leaf_mcs250", 10, 30, 0.0, "cosine", 250, 1, "leaf"),
        _trial("u10_n30_leaf_mcs300", 10, 30, 0.0, "cosine", 300, 1, "leaf"),
        # Broader-neighborhood LEAF probes. These test whether broad manifolds can split cleanly.
        _trial("u10_n75_leaf_mcs400", 10, 75, 0.0, "cosine", 400, 1, "leaf"),
        _trial("u10_n75_leaf_mcs500", 10, 75, 0.0, "cosine", 500, 1, "leaf"),
        _trial("u15_n75_leaf_mcs500", 15, 75, 0.0, "cosine", 500, 1, "leaf"),
        _trial("u15_n75_leaf_mcs650", 15, 75, 0.0, "cosine", 650, 1, "leaf"),
        _trial("u15_n100_leaf_mcs650", 15, 100, 0.0, "cosine", 650, 1, "leaf"),
        _trial("u20_n100_leaf_mcs750", 20, 100, 0.0, "cosine", 750, 1, "leaf"),
    ]
    return _dedupe_trials(trials)


def focused_umap_hdbscan_noise_reduction_trials(
    *,
    n_components_values: Sequence[int] = (8, 10),
    n_neighbors_values: Sequence[int] = (75, 100, 125, 150),
    min_cluster_size_values: Sequence[int] = (300, 350, 400, 500),
    min_samples_values: Sequence[int] = (2, 3, 5),
) -> list[dict[str, Any]]:
    """Targeted hard UMAP-HDBSCAN grid for reducing noise without forcing soft assignment.

    The grid is intentionally centered on LEAF clustering because EOM collapsed
    the current customer manifold into a few broad macro groups. It varies UMAP
    smoothing and density strictness while keeping noise unassigned.
    """

    trials: list[dict[str, Any]] = []
    for n_components in n_components_values:
        for n_neighbors in n_neighbors_values:
            for min_cluster_size in min_cluster_size_values:
                for min_samples in min_samples_values:
                    trials.append(
                        _trial(
                            (
                                f"u{int(n_components)}_n{int(n_neighbors)}_leaf_"
                                f"mcs{int(min_cluster_size)}_ms{int(min_samples)}"
                            ),
                            int(n_components),
                            int(n_neighbors),
                            0.0,
                            "cosine",
                            int(min_cluster_size),
                            int(min_samples),
                            "leaf",
                        )
                    )
    return _dedupe_trials(trials)


def run_noise_reduction_grid_experiment(
    feature_path: str | Path,
    experiment_name: str = "umap_hdbscan_noise_reduction_grid",
    trials: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Run the focused Stage 6 hard UMAP-HDBSCAN noise-reduction grid."""

    selected_trials = trials or focused_umap_hdbscan_noise_reduction_trials()
    if limit is not None:
        selected_trials = selected_trials[: int(limit)]
    return run_umap_hdbscan_experiments(
        feature_path,
        trials=selected_trials,
        experiment_name=experiment_name,
        force=force,
        cfg=cfg,
    )


def two_stage_hdbscan_trials() -> list[dict[str, Any]]:
    """Hierarchical EOM -> LEAF trials for splitting the largest broad parent cluster."""

    return [
        _two_stage_trial("u8_n20_eom150_leaf250_noise", 8, 20, 150, 250, "noise"),
        _two_stage_trial("u8_n20_eom150_leaf250_parent_fallback", 8, 20, 150, 250, "parent_fallback"),
        _two_stage_trial("u10_n30_eom150_leaf250_noise", 10, 30, 150, 250, "noise"),
        _two_stage_trial("u10_n30_eom150_leaf250_parent_fallback", 10, 30, 150, 250, "parent_fallback"),
        _two_stage_trial("u10_n50_eom200_leaf300_noise", 10, 50, 200, 300, "noise"),
        _two_stage_trial("u10_n50_eom200_leaf300_parent_fallback", 10, 50, 200, 300, "parent_fallback"),
        _two_stage_trial("u15_n75_eom500_leaf500_noise", 15, 75, 500, 500, "noise"),
        _two_stage_trial("u15_n75_eom500_leaf500_parent_fallback", 15, 75, 500, 500, "parent_fallback"),
    ]


def run_consolidated_single_stage_experiment(
    feature_path: str | Path,
    experiment_name: str = "umap_dimensionality_reduction_master_search",
    trials: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Run the consolidated single-stage UMAP-HDBSCAN search."""

    selected_trials = trials or consolidated_umap_hdbscan_trials()
    if limit is not None:
        selected_trials = selected_trials[: int(limit)]
    return run_umap_hdbscan_experiments(
        feature_path,
        trials=selected_trials,
        experiment_name=experiment_name,
        force=force,
        cfg=cfg,
    )


def run_two_stage_hdbscan_experiments(
    feature_path: str | Path,
    experiment_name: str = "umap_dimensionality_reduction_master_search",
    trials: list[dict[str, Any]] | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Any]:
    """Run EOM parent clustering, then split the largest parent cluster with LEAF."""

    if not cfg.experiments_enabled:
        raise RuntimeError(f"Two-stage HDBSCAN experiments are disabled for mode {cfg.mode!r}.")

    experiment_slug = _trial_slug(experiment_name)
    experiment_dir = cfg.experiments / experiment_slug
    experiment_dir.mkdir(parents=True, exist_ok=True)
    force = cfg.get("cache.force", False) if force is None else force
    selected_trials = trials or two_stage_hdbscan_trials()

    rows: list[dict[str, Any]] = []
    assignment_paths: dict[str, Path] = {}

    for trial in selected_trials:
        trial_name = _trial_slug(str(trial.get("name", "two_stage")))
        umap_overrides = dict(trial.get("umap", {}) or {})
        parent_cfg = dict(trial.get("parent_hdbscan", {}) or {})
        child_cfg = dict(trial.get("child_hdbscan", {}) or {})
        child_noise_strategy = str(trial.get("child_noise_strategy", "noise"))
        if child_noise_strategy not in {"noise", "parent_fallback"}:
            raise ValueError(f"Unsupported child_noise_strategy: {child_noise_strategy}")

        umap_path = build_umap_representation(
            feature_path,
            output_path=experiment_dir / f"{trial_name}_umap.parquet",
            umap_overrides=umap_overrides,
            force=force,
            cfg=cfg,
        )
        df = pl.read_parquet(umap_path).sort("cliente")
        feature_cols = numeric_feature_columns(df)
        X = frame_to_numpy(df, feature_cols)
        clientes = df["cliente"].to_list()

        parent_labels, _ = _fit_hdbscan_labels(X, parent_cfg)
        valid_parent_labels, parent_counts = _valid_label_counts(parent_labels)
        if not valid_parent_labels:
            final_labels = np.full(parent_labels.shape[0], -1, dtype=np.int32)
            split_parent_label = None
            split_parent_size = 0
            child_labels = np.array([], dtype=np.int32)
            child_indices = np.array([], dtype=np.int64)
        else:
            largest_pos = int(np.argmax(parent_counts))
            split_parent_label = int(valid_parent_labels[largest_pos])
            child_indices = np.where(parent_labels == split_parent_label)[0]
            split_parent_size = int(child_indices.shape[0])
            child_labels, _ = _fit_hdbscan_labels(X[child_indices], child_cfg)
            final_labels = _combine_parent_child_labels(
                parent_labels=parent_labels,
                child_indices=child_indices,
                child_labels=child_labels,
                split_parent_label=split_parent_label,
                child_noise_strategy=child_noise_strategy,
            )

        metrics = evaluate_labels(X, final_labels, cfg=cfg)
        passes_gate, gate_reason = quality_gate_result(metrics, cfg=cfg)
        parent_metrics = evaluate_labels(X, parent_labels, cfg=cfg)
        child_metrics = (
            evaluate_labels(X[child_indices], child_labels, cfg=cfg)
            if child_indices.shape[0] and np.unique(child_labels[child_labels >= 0]).shape[0] >= 1
            else {"cluster_count": 0, "noise_pct": 100.0}
        )
        variant = (
            f"{trial_name}_parent_mcs{parent_cfg.get('min_cluster_size')}_"
            f"child_mcs{child_cfg.get('min_cluster_size')}_{child_noise_strategy}"
        )
        model_name = f"experiment_{experiment_slug}_{trial_name}"
        candidate_id = f"{model_name}::{variant}"
        assignment_path = experiment_dir / f"cluster_assignments_{variant}.parquet"
        _write_assignment(
            assignment_path=assignment_path,
            clientes=clientes,
            labels=final_labels,
            model_name=model_name,
            model_variant=variant,
            assignment_source=f"two_stage_hdbscan_{child_noise_strategy}",
        )
        assignment_paths[candidate_id] = assignment_path

        rows.append(
            {
                "model": "UMAP Two-Stage HDBSCAN Experiment",
                "model_id": model_name,
                "model_name": model_name,
                "algorithm_name": "UMAP_TwoStageHDBSCAN",
                "trial_name": trial_name,
                "model_variant": variant,
                "candidate_id": candidate_id,
                "feature_space": f"umap_customer_embeddings_{experiment_slug}_{trial_name}",
                "representation_path": str(umap_path),
                "split_parent_label": split_parent_label,
                "split_parent_size": split_parent_size,
                "split_parent_share_pct": 100.0 * split_parent_size / max(len(parent_labels), 1),
                "child_noise_strategy": child_noise_strategy,
                "parent_cluster_count": parent_metrics.get("cluster_count"),
                "parent_noise_pct": parent_metrics.get("noise_pct"),
                "child_cluster_count": child_metrics.get("cluster_count"),
                "child_noise_pct_within_parent": child_metrics.get("noise_pct"),
                **metrics,
                "passes_quality_gate": passes_gate,
                "quality_gate_reason": gate_reason,
                "assignment_path": str(assignment_path),
            }
        )

    diagnostics = _write_ranked_diagnostics(
        rows,
        output_base=experiment_dir / f"{experiment_slug}_two_stage_diagnostics",
        title="UMAP Two-Stage HDBSCAN Diagnostics",
        cfg=cfg,
    )
    return {"diagnostics": diagnostics, "assignment_paths": assignment_paths}


def run_soft_noise_assignment_experiments(
    diagnostics_path: str | Path,
    experiment_name: str = "umap_dimensionality_reduction_master_search",
    output_suffix: str = "soft_assignment",
    strategies: Sequence[str] = ("q95", "all"),
    target_cluster_min: int | None = None,
    target_cluster_max: int | None = None,
    max_candidates: int | None = 12,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Assign HDBSCAN noise customers to nearest core-cluster centroids as a flagged follow-up."""

    if not cfg.experiments_enabled:
        raise RuntimeError(f"Soft-assignment experiments are disabled for mode {cfg.mode!r}.")

    experiment_slug = _trial_slug(experiment_name)
    experiment_dir = cfg.experiments / experiment_slug
    diagnostics = pl.read_parquet(diagnostics_path).sort("stage6_rank")
    rows = _candidate_rows_for_soft_assignment(
        diagnostics,
        min_clusters=target_cluster_min or int(cfg.get("modeling.client_hypothesis_min", 10)),
        max_clusters=target_cluster_max or int(cfg.get("modeling.client_hypothesis_max", 15)),
        max_candidates=max_candidates,
    )

    output_rows: list[dict[str, Any]] = []
    for base_row in rows:
        assignment_path = base_row.get("assignment_path")
        if not assignment_path or not Path(str(assignment_path)).exists():
            continue
        representation_path = _representation_path_for_row(base_row, experiment_dir)
        if representation_path is None or not representation_path.exists():
            continue

        assignment = pl.read_parquet(assignment_path).sort("cliente")
        representation = pl.read_parquet(representation_path).sort("cliente")
        feature_cols = numeric_feature_columns(representation)
        joined = assignment.join(representation, on="cliente", how="inner").sort("cliente")
        X = frame_to_numpy(joined, feature_cols)
        labels = joined["tribe_id"].to_numpy().astype(np.int32)
        if labels.shape[0] == 0 or not np.any(labels < 0) or np.unique(labels[labels >= 0]).shape[0] < 2:
            continue

        for strategy in strategies:
            soft_labels, assigned_count, threshold, confidence_scores = _soft_assign_noise_labels(X, labels, strategy=strategy)
            metrics = evaluate_labels(X, soft_labels, cfg=cfg)
            passes_gate, gate_reason = quality_gate_result(metrics, cfg=cfg)
            variant = f"{base_row.get('model_variant')}_soft_{strategy}"
            model_name = f"{base_row.get('model_name')}_soft"
            candidate_id = f"{model_name}::{variant}"
            soft_assignment_path = experiment_dir / f"cluster_assignments_soft_{_short_hash(candidate_id)}_{strategy}.parquet"
            _write_assignment(
                assignment_path=soft_assignment_path,
                clientes=joined["cliente"].to_list(),
                labels=soft_labels,
                model_name=model_name,
                model_variant=variant,
                assignment_source=f"nearest_centroid_soft_noise_{strategy}",
                assignment_confidence_score=confidence_scores,
                assignment_confidence_type=f"nearest_centroid_distance_percentile_{strategy}",
            )
            assigned_confidences = confidence_scores[np.isfinite(confidence_scores)]
            output_rows.append(
                {
                    "model": "Soft Noise Assignment Experiment",
                    "model_id": model_name,
                    "model_name": model_name,
                    "algorithm_name": f"{base_row.get('algorithm_name')}_SoftNoiseAssignment",
                    "trial_name": base_row.get("trial_name"),
                    "model_variant": variant,
                    "candidate_id": candidate_id,
                    "base_candidate_id": base_row.get("candidate_id"),
                    "feature_space": base_row.get("feature_space"),
                    "representation_path": str(representation_path),
                    "soft_assignment_strategy": strategy,
                    "soft_assignment_distance_threshold": threshold,
                    "core_noise_pct": base_row.get("noise_pct"),
                    "soft_assigned_customers": assigned_count,
                    "soft_assigned_pct": 100.0 * assigned_count / max(labels.shape[0], 1),
                    "soft_assignment_confidence_mean": float(np.mean(assigned_confidences))
                    if assigned_confidences.size
                    else None,
                    "soft_assignment_confidence_p10": float(np.quantile(assigned_confidences, 0.10))
                    if assigned_confidences.size
                    else None,
                    **metrics,
                    "final_noise_pct": metrics.get("noise_pct"),
                    "passes_quality_gate": passes_gate,
                    "quality_gate_reason": gate_reason,
                    "assignment_path": str(soft_assignment_path),
                }
            )

    if not output_rows:
        output_base = experiment_dir / f"{experiment_slug}_{output_suffix}_diagnostics"
        empty = pl.DataFrame()
        parquet_path = output_base.with_suffix(".parquet")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        empty.write_parquet(parquet_path)
        summaries = write_summary_artifacts(
            empty,
            output_base=output_base,
            title="UMAP HDBSCAN Soft Noise Assignment Diagnostics",
            priority_columns=[],
            cfg=cfg,
        )
        return {"parquet": parquet_path, **summaries}

    return _write_ranked_diagnostics(
        output_rows,
        output_base=experiment_dir / f"{experiment_slug}_{output_suffix}_diagnostics",
        title="UMAP HDBSCAN Soft Noise Assignment Diagnostics",
        cfg=cfg,
    )


def build_combined_dimensionality_diagnostics(
    diagnostics_paths: Iterable[str | Path],
    output_path: str | Path,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Combine single-stage, two-stage, and soft-assignment diagnostics into one ranked file."""

    rows: list[dict[str, Any]] = []
    for diagnostics_path in diagnostics_paths:
        path = Path(diagnostics_path)
        if not path.exists():
            continue
        for row in pl.read_parquet(path).iter_rows(named=True):
            item = dict(row)
            item["source_diagnostics_path"] = str(path)
            item["candidate_id"] = item.get("candidate_id") or f"{item.get('model_name')}::{item.get('model_variant')}"
            rows.append(item)

    return _write_ranked_diagnostics(
        rows,
        output_base=Path(output_path).with_suffix(""),
        title="Consolidated Dimensionality Reduction Search Diagnostics",
        cfg=cfg,
    )


def build_dimensionality_family_comparison(
    diagnostics_path: str | Path,
    output_path: str | Path,
    target_cluster_min: int | None = None,
    target_cluster_max: int | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Summarize dimensionality experiment families side by side."""

    target_min = target_cluster_min or int(cfg.get("modeling.client_hypothesis_min", 10))
    target_max = target_cluster_max or int(cfg.get("modeling.client_hypothesis_max", 15))
    diagnostics = pl.read_parquet(diagnostics_path)

    rows_by_family: dict[str, list[dict[str, Any]]] = {}
    if not diagnostics.is_empty():
        for row in diagnostics.iter_rows(named=True):
            item = dict(row)
            family = _experiment_family(item)
            item["experiment_family"] = family
            rows_by_family.setdefault(family, []).append(item)

    comparison_rows = []
    for family, label, family_order in _expected_experiment_families():
        family_rows = rows_by_family.pop(family, [])
        comparison_rows.append(
            _family_comparison_row(
                family=family,
                label=label,
                family_order=family_order,
                rows=family_rows,
                target_min=target_min,
                target_max=target_max,
            )
        )

    for offset, family in enumerate(sorted(rows_by_family), start=100):
        comparison_rows.append(
            _family_comparison_row(
                family=family,
                label=family.replace("_", " ").title(),
                family_order=offset,
                rows=rows_by_family[family],
                target_min=target_min,
                target_max=target_max,
            )
        )

    output_base = Path(output_path)
    parquet_path = output_base if output_base.suffix == ".parquet" else output_base.with_suffix(".parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.from_dicts(sorted(comparison_rows, key=lambda row: row["comparison_rank"]), infer_schema_length=None)
    df.write_parquet(parquet_path)
    summaries = write_summary_artifacts(
        df,
        output_base=parquet_path.with_suffix(""),
        title="Dimensionality Experiment Family Comparison",
        priority_columns=[
            "comparison_rank",
            "experiment_family_label",
            "status",
            "candidate_count",
            "target_range_candidate_count",
            "passing_candidate_count",
            "best_overall_trial",
            "best_overall_clusters",
            "best_overall_score",
            "best_overall_silhouette",
            "best_overall_noise_pct",
            "best_overall_coverage_pct",
            "best_target_trial",
            "best_target_clusters",
            "best_target_score",
            "best_target_silhouette",
            "best_target_noise_pct",
            "best_target_coverage_pct",
            "best_target_passes_gate",
            "comparison_note",
        ],
        cfg=cfg,
        max_markdown_rows=50,
    )
    return {"parquet": parquet_path, **summaries}


def default_dimensionality_profile_shortlist() -> list[dict[str, Any]]:
    """Finalist candidates worth profiling after the Experiment 8 master search."""

    return [
        {
            "label": "main_10_q95",
            "role": "Main 10-tribe q95 candidate",
            "algorithm_name": "UMAP_HDBSCAN_SoftNoiseAssignment",
            "trial_name": "u10_n75_leaf_mcs400",
            "soft_assignment_strategy": "q95",
        },
        {
            "label": "granular_15_q95",
            "role": "Main 15-tribe q95 candidate",
            "algorithm_name": "UMAP_HDBSCAN_SoftNoiseAssignment",
            "trial_name": "u8_n15_leaf_mcs350",
            "soft_assignment_strategy": "q95",
        },
        {
            "label": "coverage_15_q95",
            "role": "Higher-coverage 15-tribe q95 backup",
            "algorithm_name": "UMAP_HDBSCAN_SoftNoiseAssignment",
            "trial_name": "u8_n30_leaf_mcs350",
            "soft_assignment_strategy": "q95",
        },
        {
            "label": "core_10_raw",
            "role": "High-confidence raw core reference",
            "algorithm_name": "UMAP_HDBSCAN",
            "trial_name": "u10_n75_leaf_mcs400",
        },
        {
            "label": "broad_7_two_stage_q95",
            "role": "Broad two-stage baseline",
            "algorithm_name": "UMAP_TwoStageHDBSCAN_SoftNoiseAssignment",
            "trial_name": "u15_n75_eom500_leaf500_parent_fallback",
            "soft_assignment_strategy": "q95",
        },
    ]


def ranked_dimensionality_profile_shortlist(
    diagnostics_path: str | Path,
    *,
    max_candidates: int = 5,
    target_cluster_min: int | None = None,
    target_cluster_max: int | None = None,
    cfg: PipelineConfig = CONFIG,
) -> list[dict[str, Any]]:
    """Select top-ranked target-range candidates from a diagnostics file for profiling."""

    diagnostics = pl.read_parquet(diagnostics_path)
    if diagnostics.is_empty():
        return []

    target_min = target_cluster_min or int(cfg.get("modeling.client_hypothesis_min", 10))
    target_max = target_cluster_max or int(cfg.get("modeling.client_hypothesis_max", 15))
    ranked = diagnostics.sort("stage6_rank") if "stage6_rank" in diagnostics.columns else diagnostics
    target = ranked.filter(
        (pl.col("cluster_count") >= target_min)
        & (pl.col("cluster_count") <= target_max)
        & (pl.col("assignment_path").is_not_null())
    )
    if target.is_empty():
        target = ranked.filter(pl.col("assignment_path").is_not_null())

    shortlist = []
    for idx, row in enumerate(target.head(max_candidates).iter_rows(named=True), start=1):
        trial_name = str(row.get("trial_name") or f"candidate_{idx}")
        clusters = row.get("cluster_count")
        noise = row.get("noise_pct")
        label = f"rank{idx}_{_trial_slug(trial_name)}"
        shortlist.append(
            {
                "label": label,
                "role": (
                    f"Rank {idx} target-range candidate"
                    f" ({clusters} tribes, {float(noise):.1f}% noise)"
                    if noise is not None
                    else f"Rank {idx} target-range candidate"
                ),
                "candidate_id": row.get("candidate_id"),
            }
        )
    return shortlist


def profile_dimensionality_shortlist(
    diagnostics_path: str | Path,
    experiment_name: str = "umap_dimensionality_reduction_master_search",
    shortlist: Sequence[dict[str, Any]] | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Profile shortlisted Experiment 8 candidates and compare interpretability."""

    from src.profiling import flatten_profiles_for_csv, profile_quality_summary, profile_tribes

    experiment_slug = _trial_slug(experiment_name)
    experiment_dir = cfg.experiments / experiment_slug
    experiment_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = pl.read_parquet(diagnostics_path).sort("stage6_rank")
    selected_specs = list(default_dimensionality_profile_shortlist() if shortlist is None else shortlist)
    comparison_rows: list[dict[str, Any]] = []
    detail_sections: list[str] = []

    for profile_rank, spec in enumerate(selected_specs, start=1):
        label = _trial_slug(str(spec.get("label", f"candidate_{profile_rank}")))
        role = str(spec.get("role", label))
        candidate = _find_shortlist_candidate(diagnostics, spec)
        if candidate is None:
            comparison_rows.append(
                {
                    "profile_rank": profile_rank,
                    "shortlist_label": label,
                    "shortlist_role": role,
                    "status": "missing_candidate",
                    "comparison_note": "Candidate was not found in the diagnostics file.",
                }
            )
            continue

        assignment_path_value = candidate.get("assignment_path")
        if not assignment_path_value or not Path(str(assignment_path_value)).exists():
            comparison_rows.append(
                {
                    "profile_rank": profile_rank,
                    "shortlist_label": label,
                    "shortlist_role": role,
                    "status": "missing_assignment",
                    "algorithm_name": candidate.get("algorithm_name"),
                    "trial_name": candidate.get("trial_name"),
                    "model_variant": candidate.get("model_variant"),
                    "comparison_note": "Candidate exists, but its assignment parquet was not found.",
                }
            )
            continue

        profile_path = experiment_dir / f"tribe_profiles_{label}.parquet"
        profile_csv = experiment_dir / f"tribe_profiles_{label}.csv"
        profile_path = profile_tribes(
            assignment_path_value,
            output_path=profile_path,
            force=force,
            cfg=cfg,
        )
        flatten_profiles_for_csv(profile_path, profile_csv)

        quality = profile_quality_summary(profile_path, cfg=cfg)
        profiles = pl.read_parquet(profile_path).sort("tribe_id")
        profile_stats = _profile_stats(profiles)
        comparison_rows.append(
            {
                "profile_rank": profile_rank,
                "shortlist_label": label,
                "shortlist_role": role,
                "status": "profiled",
                "algorithm_name": candidate.get("algorithm_name"),
                "trial_name": candidate.get("trial_name"),
                "model_variant": candidate.get("model_variant"),
                "soft_assignment_strategy": candidate.get("soft_assignment_strategy"),
                "cluster_count": candidate.get("cluster_count"),
                "coverage_adjusted_silhouette": candidate.get("coverage_adjusted_silhouette"),
                "silhouette": candidate.get("silhouette"),
                "davies_bouldin": candidate.get("davies_bouldin"),
                "noise_pct": candidate.get("noise_pct"),
                "coverage_pct": candidate.get("coverage_pct"),
                "cluster_size_cv": candidate.get("cluster_size_cv"),
                "passes_quality_gate": candidate.get("passes_quality_gate"),
                "core_noise_pct": candidate.get("core_noise_pct"),
                "soft_assigned_pct": candidate.get("soft_assigned_pct"),
                "soft_assigned_customers": candidate.get("soft_assigned_customers"),
                **quality,
                **profile_stats,
                "assignment_path": str(assignment_path_value),
                "profile_path": str(profile_path),
                "profile_csv": str(profile_csv),
                "candidate_id": candidate.get("candidate_id"),
                "comparison_note": _profile_comparison_note(candidate, quality, profile_stats),
            }
        )
        detail_sections.append(_profile_detail_section(label, role, candidate, profiles))

    output_base = experiment_dir / f"{experiment_slug}_profile_shortlist_diagnostics"
    parquet_path = output_base.with_suffix(".parquet")
    df = pl.from_dicts(comparison_rows, infer_schema_length=None) if comparison_rows else pl.DataFrame()
    df.write_parquet(parquet_path)
    summaries = write_summary_artifacts(
        df,
        output_base=output_base,
        title="Experiment 8 Shortlist Profile Comparison",
        priority_columns=[
            "profile_rank",
            "shortlist_label",
            "shortlist_role",
            "status",
            "cluster_count",
            "coverage_adjusted_silhouette",
            "silhouette",
            "davies_bouldin",
            "noise_pct",
            "coverage_pct",
            "cluster_size_cv",
            "profiled_clusters",
            "clusters_with_product_lift",
            "clusters_with_sector_lift",
            "avg_strong_product_lifts_per_cluster",
            "avg_max_product_lift",
            "min_cluster_customers",
            "median_cluster_customers",
            "max_cluster_customers",
            "max_population_share_pct",
            "mean_top_product_overlap",
            "mean_top_sector_overlap",
            "comparison_note",
            "profile_csv",
        ],
        cfg=cfg,
        max_markdown_rows=50,
    )

    details_md = experiment_dir / f"{experiment_slug}_profile_shortlist_details.md"
    details_md.write_text(
        "# Experiment 8 Shortlist Profile Details\n\n"
        + "\n\n".join(detail_sections or ["No shortlist candidates were profiled."])
        + "\n",
        encoding="utf-8",
    )
    return {"parquet": parquet_path, "details_md": details_md, **summaries}


def _trial(
    name: str,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "umap": {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
        },
        "hdbscan": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": cluster_selection_method,
        },
    }


def _expected_experiment_families() -> list[tuple[str, str, int]]:
    return [
        ("single_stage_umap_hdbscan", "Single-stage UMAP-HDBSCAN", 1),
        ("two_stage_hdbscan", "Two-stage EOM-to-LEAF HDBSCAN", 2),
        ("single_stage_soft_assignment", "Single-stage HDBSCAN + soft noise assignment", 3),
        ("two_stage_soft_assignment", "Two-stage HDBSCAN + soft noise assignment", 4),
    ]


def _find_shortlist_candidate(diagnostics: pl.DataFrame, spec: dict[str, Any]) -> dict[str, Any] | None:
    ignored = {"label", "role"}
    for row in diagnostics.iter_rows(named=True):
        candidate = dict(row)
        matched = True
        for key, expected in spec.items():
            if key in ignored:
                continue
            value = candidate.get(key)
            if expected is None:
                if value not in {None, ""}:
                    matched = False
                    break
            elif str(value) != str(expected):
                matched = False
                break
        if matched:
            return candidate
    return None


def _experiment_family(row: dict[str, Any]) -> str:
    algorithm_name = str(row.get("algorithm_name") or "")
    if "SoftNoiseAssignment" in algorithm_name and "TwoStage" in algorithm_name:
        return "two_stage_soft_assignment"
    if "SoftNoiseAssignment" in algorithm_name:
        return "single_stage_soft_assignment"
    if "TwoStage" in algorithm_name:
        return "two_stage_hdbscan"
    if algorithm_name == "UMAP_HDBSCAN":
        return "single_stage_umap_hdbscan"
    return _trial_slug(algorithm_name or "other")


def _family_comparison_row(
    family: str,
    label: str,
    family_order: int,
    rows: list[dict[str, Any]],
    target_min: int,
    target_max: int,
) -> dict[str, Any]:
    rows = sorted(rows, key=_rank_key)
    target_rows = [
        row
        for row in rows
        if target_min <= int(row.get("cluster_count") or 0) <= target_max
    ]
    passing_rows = [row for row in rows if _as_bool(row.get("passes_quality_gate"))]

    best_overall = rows[0] if rows else None
    best_target = sorted(target_rows, key=_rank_key)[0] if target_rows else None
    best_passing = sorted(passing_rows, key=_rank_key)[0] if passing_rows else None
    comparison_rank = family_order

    if not rows:
        return {
            "comparison_rank": comparison_rank,
            "experiment_family": family,
            "experiment_family_label": label,
            "status": "missing_outputs",
            "candidate_count": 0,
            "target_range_candidate_count": 0,
            "passing_candidate_count": 0,
            "target_cluster_min": target_min,
            "target_cluster_max": target_max,
            **_candidate_prefix("best_overall", None),
            **_candidate_prefix("best_target", None),
            **_candidate_prefix("best_passing", None),
            "comparison_note": "Not run yet, or diagnostics were not found in the combined file.",
        }

    note = "Has target-range candidate." if best_target else "No candidate in target range."
    if best_target and not _as_bool(best_target.get("passes_quality_gate")):
        note = "Best target-range candidate exists but does not pass the current quality gate."

    return {
        "comparison_rank": comparison_rank,
        "experiment_family": family,
        "experiment_family_label": label,
        "status": "available",
        "candidate_count": len(rows),
        "target_range_candidate_count": len(target_rows),
        "passing_candidate_count": len(passing_rows),
        "target_cluster_min": target_min,
        "target_cluster_max": target_max,
        **_candidate_prefix("best_overall", best_overall),
        **_candidate_prefix("best_target", best_target),
        **_candidate_prefix("best_passing", best_passing),
        "comparison_note": note,
    }


def _candidate_prefix(prefix: str, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            f"{prefix}_trial": None,
            f"{prefix}_algorithm_name": None,
            f"{prefix}_model_variant": None,
            f"{prefix}_clusters": None,
            f"{prefix}_score": None,
            f"{prefix}_silhouette": None,
            f"{prefix}_davies_bouldin": None,
            f"{prefix}_noise_pct": None,
            f"{prefix}_coverage_pct": None,
            f"{prefix}_cluster_size_cv": None,
            f"{prefix}_passes_gate": None,
            f"{prefix}_candidate_id": None,
        }

    return {
        f"{prefix}_trial": row.get("trial_name"),
        f"{prefix}_algorithm_name": row.get("algorithm_name"),
        f"{prefix}_model_variant": row.get("model_variant"),
        f"{prefix}_clusters": row.get("cluster_count"),
        f"{prefix}_score": row.get("coverage_adjusted_silhouette") or row.get("stage6_score"),
        f"{prefix}_silhouette": row.get("silhouette"),
        f"{prefix}_davies_bouldin": row.get("davies_bouldin"),
        f"{prefix}_noise_pct": row.get("noise_pct"),
        f"{prefix}_coverage_pct": row.get("coverage_pct"),
        f"{prefix}_cluster_size_cv": row.get("cluster_size_cv"),
        f"{prefix}_passes_gate": _as_bool(row.get("passes_quality_gate")),
        f"{prefix}_candidate_id": row.get("candidate_id"),
    }


def _profile_stats(profiles: pl.DataFrame) -> dict[str, Any]:
    if profiles.is_empty():
        return {
            "min_cluster_customers": None,
            "median_cluster_customers": None,
            "max_cluster_customers": None,
            "max_population_share_pct": None,
            "mean_top_product_overlap": None,
            "mean_top_sector_overlap": None,
        }
    sizes = profiles["n_customers"].to_list()
    shares = profiles["population_share"].to_list()
    return {
        "min_cluster_customers": int(min(sizes)),
        "median_cluster_customers": float(np.median(np.asarray(sizes, dtype=np.float64))),
        "max_cluster_customers": int(max(sizes)),
        "max_population_share_pct": float(max(shares) * 100.0) if shares else None,
        "mean_top_product_overlap": _mean_pairwise_jaccard(profiles["top_product_ids"].to_list()),
        "mean_top_sector_overlap": _mean_pairwise_jaccard(profiles["top_sectors"].to_list()),
    }


def _mean_pairwise_jaccard(values: Sequence[Sequence[Any]]) -> float | None:
    sets = [set(str(item) for item in items if item is not None and str(item)) for items in values]
    scores = []
    for left_idx in range(len(sets)):
        for right_idx in range(left_idx + 1, len(sets)):
            union = sets[left_idx] | sets[right_idx]
            if not union:
                continue
            scores.append(len(sets[left_idx] & sets[right_idx]) / len(union))
    return float(np.mean(scores)) if scores else None


def _profile_comparison_note(
    candidate: dict[str, Any],
    quality: dict[str, Any],
    stats: dict[str, Any],
) -> str:
    clusters = int(candidate.get("cluster_count") or 0)
    coverage = _as_float(candidate.get("coverage_pct"), 0.0)
    noise = _as_float(candidate.get("noise_pct"), 100.0)
    product_clusters = int(quality.get("clusters_with_product_lift") or 0)
    profiled = int(quality.get("profiled_clusters") or 0)
    product_overlap = stats.get("mean_top_product_overlap")
    overlap_note = ""
    if product_overlap is not None:
        overlap_note = f" Mean top-product overlap is {product_overlap:.2f}."
    return (
        f"{clusters} clusters, {coverage:.1f}% coverage, {noise:.1f}% noise; "
        f"{product_clusters}/{profiled} clusters have strong product lift."
        f"{overlap_note}"
    )


def _profile_detail_section(
    label: str,
    role: str,
    candidate: dict[str, Any],
    profiles: pl.DataFrame,
) -> str:
    from src.product_themes import detect_product_themes, normalize_product_text

    lines = [
        f"## {label}",
        "",
        f"Role: {role}",
        "",
        (
            f"Candidate: `{candidate.get('algorithm_name')}` / `{candidate.get('trial_name')}` / "
            f"`{candidate.get('model_variant')}`"
        ),
        "",
        "| Tribe | Customers | Share % | Top themes | Top products | Top sectors |",
        "| ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in profiles.sort("tribe_id").iter_rows(named=True):
        raw_products = row.get("top_products") or row.get("top_product_ids") or []
        products = _preview_list([normalize_product_text(value) for value in raw_products], 4)
        themes = _top_themes(raw_products, detect_product_themes)
        sectors = _preview_list(row.get("top_sectors") or [], 4)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("tribe_id")),
                    str(row.get("n_customers")),
                    f"{100.0 * float(row.get('population_share') or 0.0):.2f}",
                    themes,
                    products,
                    sectors,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _top_themes(values: Sequence[Any], detector: Any, limit: int = 5) -> str:
    counts: dict[str, int] = {}
    for value in values:
        for theme in detector(value):
            counts[theme] = counts.get(theme, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{theme} ({count})" for theme, count in ranked[:limit])


def _preview_list(values: Sequence[Any], limit: int) -> str:
    cleaned = [str(value) for value in values if value is not None and str(value)]
    return "; ".join(cleaned[:limit])


def _two_stage_trial(
    name: str,
    n_components: int,
    n_neighbors: int,
    parent_mcs: int,
    child_mcs: int,
    child_noise_strategy: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "umap": {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": 0.0,
            "metric": "cosine",
        },
        "parent_hdbscan": {
            "min_cluster_size": parent_mcs,
            "min_samples": 1,
            "cluster_selection_method": "eom",
        },
        "child_hdbscan": {
            "min_cluster_size": child_mcs,
            "min_samples": 1,
            "cluster_selection_method": "leaf",
        },
        "child_noise_strategy": child_noise_strategy,
    }


def _dedupe_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for trial in trials:
        name = _trial_slug(str(trial.get("name", "trial")))
        if name in seen:
            continue
        seen.add(name)
        deduped.append(trial)
    return deduped


def _fit_hdbscan_labels(X: np.ndarray, hdbscan_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        import hdbscan
    except ImportError:
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=int(hdbscan_cfg.get("min_cluster_size", 500)),
            min_samples=int(hdbscan_cfg.get("min_samples", 10)),
            cluster_selection_method=str(hdbscan_cfg.get("cluster_selection_method", "eom")),
        )
        labels = model.fit_predict(X)
        return np.asarray(labels).astype(np.int32), None

    model = hdbscan.HDBSCAN(
        min_cluster_size=int(hdbscan_cfg.get("min_cluster_size", 500)),
        min_samples=int(hdbscan_cfg.get("min_samples", 10)),
        cluster_selection_method=str(hdbscan_cfg.get("cluster_selection_method", "eom")),
    )
    labels = model.fit_predict(X)
    probabilities = getattr(model, "probabilities_", None)
    return np.asarray(labels).astype(np.int32), None if probabilities is None else np.asarray(probabilities)


def _valid_label_counts(labels: np.ndarray) -> tuple[list[int], np.ndarray]:
    valid_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    return [int(label) for label in valid_labels], counts.astype(np.int64)


def _combine_parent_child_labels(
    parent_labels: np.ndarray,
    child_indices: np.ndarray,
    child_labels: np.ndarray,
    split_parent_label: int,
    child_noise_strategy: str,
) -> np.ndarray:
    final = np.full(parent_labels.shape[0], -1, dtype=np.int32)
    next_label = 0

    for parent_label in sorted(int(label) for label in np.unique(parent_labels) if label >= 0):
        if parent_label == split_parent_label:
            continue
        final[parent_labels == parent_label] = next_label
        next_label += 1

    if child_noise_strategy == "parent_fallback":
        final[child_indices] = next_label
        next_label += 1

    for child_label in sorted(int(label) for label in np.unique(child_labels) if label >= 0):
        final[child_indices[child_labels == child_label]] = next_label
        next_label += 1

    return final


def _soft_assign_noise_labels(
    X: np.ndarray,
    labels: np.ndarray,
    strategy: str = "q95",
) -> tuple[np.ndarray, int, float | None, np.ndarray]:
    confidence_scores = np.full(labels.shape[0], np.nan, dtype=np.float32)
    valid_labels = sorted(int(label) for label in np.unique(labels) if label >= 0)
    centroids = np.vstack([X[labels == label].mean(axis=0) for label in valid_labels]).astype(np.float32)
    valid_indices = np.where(labels >= 0)[0]
    valid_label_to_idx = {label: idx for idx, label in enumerate(valid_labels)}
    own_centroid_idx = np.array([valid_label_to_idx[int(label)] for label in labels[valid_indices]], dtype=np.int32)
    own_distances = np.linalg.norm(X[valid_indices] - centroids[own_centroid_idx], axis=1)

    threshold = None
    if strategy.startswith("q"):
        quantile = float(strategy[1:]) / 100.0
        threshold = float(np.quantile(own_distances, quantile))
    elif strategy != "all":
        raise ValueError(f"Unsupported soft assignment strategy: {strategy}")

    noise_indices = np.where(labels < 0)[0]
    distances = np.linalg.norm(X[noise_indices, None, :] - centroids[None, :, :], axis=2)
    nearest_centroid_idx = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(distances.shape[0]), nearest_centroid_idx]
    assign_mask = np.ones(noise_indices.shape[0], dtype=bool)
    if threshold is not None:
        assign_mask = nearest_distances <= threshold

    soft_labels = labels.copy()
    assigned_noise_indices = noise_indices[assign_mask]
    soft_labels[assigned_noise_indices] = np.array(valid_labels, dtype=np.int32)[nearest_centroid_idx[assign_mask]]
    core_distance_reference = np.sort(own_distances)
    if core_distance_reference.size:
        nearest_percentiles = np.searchsorted(core_distance_reference, nearest_distances, side="left") / float(
            core_distance_reference.size
        )
        nearest_confidence = np.clip(1.0 - nearest_percentiles, 0.0, 1.0).astype(np.float32)
        confidence_scores[assigned_noise_indices] = nearest_confidence[assign_mask]
    return soft_labels.astype(np.int32), int(assigned_noise_indices.shape[0]), threshold, confidence_scores


def _candidate_rows_for_soft_assignment(
    diagnostics: pl.DataFrame,
    min_clusters: int,
    max_clusters: int,
    max_candidates: int | None,
) -> list[dict[str, Any]]:
    if diagnostics.is_empty() or "cluster_count" not in diagnostics.columns or "noise_pct" not in diagnostics.columns:
        return []
    target = diagnostics.filter(
        (pl.col("cluster_count") >= min_clusters) & (pl.col("cluster_count") <= max_clusters) & (pl.col("noise_pct") > 0)
    )
    if target.height == 0:
        target = diagnostics.filter(pl.col("noise_pct") > 0).head(max_candidates or 12)
    elif max_candidates is not None:
        target = target.head(max_candidates)
    return [dict(row) for row in target.iter_rows(named=True)]


def _representation_path_for_row(row: dict[str, Any], experiment_dir: Path) -> Path | None:
    value = row.get("representation_path")
    if value:
        return Path(str(value))
    trial_name = row.get("trial_name")
    if trial_name:
        return experiment_dir / f"{_trial_slug(str(trial_name))}_umap.parquet"
    return None


def _write_assignment(
    assignment_path: Path,
    clientes: list[str],
    labels: np.ndarray,
    model_name: str,
    model_variant: str,
    assignment_source: str,
    assignment_confidence_score: np.ndarray | None = None,
    assignment_confidence_type: str | None = None,
) -> None:
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    confidence_score_values = (
        [None] * len(clientes)
        if assignment_confidence_score is None
        else np.asarray(assignment_confidence_score, dtype=np.float32)
    )
    pl.DataFrame(
        {
            "cliente": clientes,
            "tribe_id": np.asarray(labels).astype(np.int32),
            "model_name": [model_name] * len(clientes),
            "model_variant": [model_variant] * len(clientes),
            "assignment_probability": [None] * len(clientes),
            "assignment_confidence_score": confidence_score_values,
            "assignment_confidence_type": [assignment_confidence_type] * len(clientes),
            "assignment_source": [assignment_source] * len(clientes),
        }
    ).sort("cliente").write_parquet(assignment_path)


def _write_ranked_diagnostics(
    rows: list[dict[str, Any]],
    output_base: Path,
    title: str,
    cfg: PipelineConfig,
) -> dict[str, Path]:
    if not rows:
        raise ValueError("No dimensionality experiment rows were produced.")

    ranked = sorted(rows, key=_rank_key)
    for rank, row in enumerate(ranked, start=1):
        row["stage6_rank"] = rank
        row["stage6_score"] = row.get("coverage_adjusted_silhouette")
        row["selected_within_family"] = rank == 1 and bool(row.get("passes_quality_gate"))

    output_base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = output_base.with_suffix(".parquet")
    df = pl.from_dicts(ranked, infer_schema_length=None)
    df.write_parquet(parquet_path)
    summaries = write_summary_artifacts(
        df,
        output_base=output_base,
        title=title,
        priority_columns=[
            "stage6_rank",
            "passes_quality_gate",
            "algorithm_name",
            "trial_name",
            "model_variant",
            "cluster_count",
            "stage6_score",
            "coverage_adjusted_silhouette",
            "silhouette",
            "davies_bouldin",
            "noise_pct",
            "cluster_size_cv",
            "selected_within_family",
        ],
        cfg=cfg,
    )
    return {"parquet": parquet_path, **summaries}


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    return (
        0.0 if _as_bool(row.get("passes_quality_gate")) else 1.0,
        -_as_float(row.get("coverage_adjusted_silhouette"), _as_float(row.get("silhouette"), -999.0)),
        _as_float(row.get("davies_bouldin"), 999.0),
        _as_float(row.get("cluster_size_cv"), 999.0),
        _as_float(row.get("noise_pct"), 999.0),
        str(row.get("candidate_id") or row.get("model_variant")),
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trial_slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.strip().lower()).strip("_") or "trial"


def _short_hash(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=6).hexdigest()
