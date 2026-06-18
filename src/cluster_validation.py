"""Cluster validity and lightweight stability diagnostics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.evaluation import cluster_size_summary
from src.experiment_reporting import write_summary_artifacts
from src.progress import log_event, stage_timer
from src.utils import (
    collect_streaming,
    deterministic_sample_indices,
    file_fingerprint,
    frame_to_numpy,
    numeric_feature_columns,
    should_use_cache,
    write_artifact_metadata,
)


def build_cluster_validity_stability_report(
    model_suite: dict[str, Any],
    feature_path: str | Path,
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write a compact report on candidate validity and label-geometry stability.

    This is intentionally not a substitute for a full repeated-run stability study.
    It checks whether each candidate's labels remain recoverable from the original
    feature geometry after small feature perturbations.
    """

    cfg.ensure_directories()
    output = Path(output_path) if output_path else cfg.model_selection / "cluster_validity_stability.parquet"
    force = bool(cfg.get("cache.force", False)) if force is None else force
    sample_size = int(cfg.get("stability.sample_size", 10000))
    repeats = int(cfg.get("stability.repeats", 5))
    jitter_scale = float(cfg.get("stability.jitter_scale", 0.02))
    artifact_paths = _cluster_stability_artifact_paths(output, cfg)
    cache_metadata = _cluster_stability_cache_metadata(
        model_suite=model_suite,
        feature_path=feature_path,
        sample_size=sample_size,
        repeats=repeats,
        jitter_scale=jitter_scale,
        cfg=cfg,
    )

    if _cluster_stability_cache_hit(
        artifact_paths,
        cache_metadata=cache_metadata,
        force=force,
        cfg=cfg,
    ):
        log_event("Cluster validity", "cache hit", cfg=cfg, path=output)
        return artifact_paths

    with stage_timer(
        "Cluster validity",
        "building validity and stability report",
        cfg=cfg,
        feature_path=feature_path,
        repeats=repeats,
    ):
        feature_cols = numeric_feature_columns(pl.read_parquet(feature_path, n_rows=1))

        rows = []
        cluster_rows = []
        for candidate_key, assignment_path in sorted((model_suite.get("assignment_paths") or {}).items()):
            assignment_df = pl.read_parquet(assignment_path)
            if "assignment_confidence_score" not in assignment_df.columns:
                assignment_df = assignment_df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("assignment_confidence_score")
                )
            if "assignment_source" not in assignment_df.columns:
                assignment_df = assignment_df.with_columns(pl.lit("unknown").alias("assignment_source"))
            assignments = assignment_df.select(
                [
                    "cliente",
                    "tribe_id",
                    "model_name",
                    "model_variant",
                    "assignment_confidence_score",
                    "assignment_source",
                ]
            ).unique(subset=["cliente"], keep="first")
            labels = assignments["tribe_id"].to_numpy().astype(int)
            metrics = cluster_size_summary(labels)
            stability_features = _sample_candidate_features(assignments, feature_path, feature_cols, sample_size, cfg)
            stability = _jitter_stability(
                stability_features["X"],
                stability_features["labels"],
                sample_size=sample_size,
                repeats=repeats,
                jitter_scale=jitter_scale,
                seed=cfg.random_seed,
            )
            cluster_recovery = stability.pop("_cluster_recovery_accuracy_mean_by_label", {})
            cluster_rows.extend(
                _cluster_readiness_rows(
                    assignments,
                    candidate_key=candidate_key,
                    total_rows=int(assignments.height),
                    cluster_recovery=cluster_recovery,
                    min_cluster_size=_candidate_min_cluster_size(candidate_key, model_suite, cfg),
                )
            )
            rows.append(
                {
                    "candidate_id": candidate_key,
                    "model_name": assignments["model_name"][0] if assignments.height else None,
                    "model_variant": assignments["model_variant"][0] if assignments.height else None,
                    "rows_evaluated": int(assignments.height),
                    "feature_count": int(len(feature_cols)),
                    "stability_sample_size": int(stability_features["X"].shape[0]),
                    "stability_repeats": repeats,
                    "jitter_scale": jitter_scale,
                    **metrics,
                    **stability,
                    "validity_note": (
                        "Checks internal geometry and perturbation sensitivity. "
                        "A final submission should still include repeated-seed or bootstrap model reruns."
                    ),
                }
            )

    report = pl.DataFrame(rows).sort("candidate_id") if rows else pl.DataFrame()
    output.parent.mkdir(parents=True, exist_ok=True)
    report.write_parquet(output)
    cluster_output = artifact_paths["cluster_parquet"]
    cluster_report = (
        pl.DataFrame(cluster_rows).sort(["candidate_id", "tribe_id"]) if cluster_rows else pl.DataFrame()
    )
    cluster_report.write_parquet(cluster_output)
    summaries = write_summary_artifacts(
        report,
        output_base=output,
        title="Cluster Validity And Stability Diagnostics",
        priority_columns=[
            "candidate_id",
            "cluster_count",
            "noise_pct",
            "cluster_size_cv",
            "jitter_ari_mean",
            "jitter_ari_std",
            "jitter_label_recovery_accuracy_mean",
            "stability_sample_size",
            "validity_note",
        ],
        cfg=cfg,
        write_markdown=bool(cfg.get("model_selection.write_summary_markdown", True)),
    )
    cluster_summaries = write_summary_artifacts(
        cluster_report,
        output_base=cluster_output,
        title="Cluster Profile Readiness Diagnostics",
        priority_columns=[
            "candidate_id",
            "tribe_id",
            "customers",
            "customer_share_pct",
            "core_customer_share_pct",
            "mean_assignment_confidence",
            "p10_assignment_confidence",
            "jitter_label_recovery_accuracy_mean",
            "profile_readiness",
            "readiness_issues",
        ],
        cfg=cfg,
        write_markdown=bool(cfg.get("model_selection.write_summary_markdown", True)),
    )
    log_event("Cluster validity", "wrote validity and stability report", cfg=cfg, path=output)
    result = {
        "parquet": output,
        **summaries,
        "cluster_parquet": cluster_output,
        "cluster_summary_csv": cluster_summaries["summary_csv"],
        "cluster_summary_md": cluster_summaries.get("summary_md"),
    }
    for path in result.values():
        if path is not None:
            write_artifact_metadata(path, cache_metadata)
    return {key: value for key, value in result.items() if value is not None}


def _cluster_stability_artifact_paths(output: Path, cfg: PipelineConfig) -> dict[str, Path]:
    cluster_output = output.with_name(f"{output.stem}_clusters.parquet")
    summary_csv, summary_md = _summary_paths_for(output)
    cluster_summary_csv, cluster_summary_md = _summary_paths_for(cluster_output)
    paths = {
        "parquet": output,
        "summary_csv": summary_csv,
        "cluster_parquet": cluster_output,
        "cluster_summary_csv": cluster_summary_csv,
    }
    if bool(cfg.get("model_selection.write_summary_markdown", True)):
        paths["summary_md"] = summary_md
        paths["cluster_summary_md"] = cluster_summary_md
    return paths


def _summary_paths_for(output_base: Path) -> tuple[Path, Path]:
    stem = output_base.stem
    for suffix in ["_diagnostics", "_manifest"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    summary_stem = f"{stem}_summary"
    return output_base.with_name(summary_stem).with_suffix(".csv"), output_base.with_name(summary_stem).with_suffix(".md")


def _cluster_stability_cache_hit(
    artifact_paths: dict[str, Path],
    *,
    cache_metadata: dict[str, Any],
    force: bool,
    cfg: PipelineConfig,
) -> bool:
    use_cached = bool(cfg.get("cache.use_cached", True))
    return all(
        should_use_cache(path, force=force, use_cached=use_cached, metadata=cache_metadata)
        for path in artifact_paths.values()
    )


def _cluster_stability_cache_metadata(
    *,
    model_suite: dict[str, Any],
    feature_path: str | Path,
    sample_size: int,
    repeats: int,
    jitter_scale: float,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    assignment_paths = {
        str(candidate_key): file_fingerprint(path)
        for candidate_key, path in sorted((model_suite.get("assignment_paths") or {}).items())
    }
    candidate_results = [
        {
            "model_name": candidate.get("model_name"),
            "model_variant": candidate.get("model_variant"),
        }
        for candidate in (model_suite.get("candidate_results") or [])
    ]
    return {
        "stage": "cluster_validity_stability_report",
        "mode": cfg.mode,
        "feature_path": file_fingerprint(feature_path),
        "assignment_paths": assignment_paths,
        "candidate_results": candidate_results,
        "official_candidate_key": model_suite.get("official_candidate_key"),
        "stability": {
            "sample_size": sample_size,
            "repeats": repeats,
            "jitter_scale": jitter_scale,
        },
        "hdbscan": {
            "default_min_cluster_size": cfg.get("hdbscan.min_cluster_size", 0),
            "official_umap_min_cluster_size": cfg.get("official_model_suite.umap_hdbscan.hdbscan.min_cluster_size"),
        },
        "profile_readiness_logic_version": 1,
        "random_seed": cfg.random_seed,
    }


def _sample_candidate_features(
    assignments: pl.DataFrame,
    feature_path: str | Path,
    feature_cols: list[str],
    sample_size: int,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    core = assignments.filter(pl.col("tribe_id") >= 0)
    if core.is_empty():
        return {"X": np.empty((0, len(feature_cols)), dtype=np.float32), "labels": np.array([], dtype=np.int32)}

    sample_idx = deterministic_sample_indices(core.height, min(sample_size, core.height), cfg.random_seed)
    sampled_assignments = (
        core[sample_idx]
        .select(["cliente", "tribe_id"])
        .with_row_index("_sample_order")
    )
    sampled_customers = sampled_assignments["cliente"].to_list()
    sampled_features = collect_streaming(
        pl.scan_parquet(feature_path)
        .select(["cliente", *feature_cols])
        .filter(pl.col("cliente").is_in(sampled_customers))
    )
    joined = sampled_assignments.join(sampled_features, on="cliente", how="inner").sort("_sample_order")
    if joined.is_empty():
        return {"X": np.empty((0, len(feature_cols)), dtype=np.float32), "labels": np.array([], dtype=np.int32)}
    X = _standardize(frame_to_numpy(joined, feature_cols))
    labels = joined["tribe_id"].to_numpy().astype(np.int32)
    return {"X": X, "labels": labels}


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X.astype(np.float32, copy=False), copy=False)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return ((X - mean) / np.maximum(std, 1e-12)).astype(np.float32)


def _candidate_min_cluster_size(candidate_key: str, model_suite: dict[str, Any], cfg: PipelineConfig) -> int:
    """Resolve the min-cluster-size reference for candidate-level readiness checks."""

    candidate = _candidate_result(candidate_key, model_suite)
    parsed = _parse_min_cluster_size(candidate.get("model_variant") if candidate else None)
    if parsed is not None:
        return parsed

    if _is_official_umap_hdbscan_candidate(candidate_key, candidate, model_suite, cfg):
        promoted = cfg.get("official_model_suite.umap_hdbscan.hdbscan.min_cluster_size")
        if promoted is not None:
            return int(promoted)

    return int(cfg.get("hdbscan.min_cluster_size", 0) or 0)


def _candidate_result(candidate_key: str, model_suite: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in model_suite.get("candidate_results", []) or []:
        key = f"{candidate.get('model_name')}::{candidate.get('model_variant')}"
        if key == candidate_key:
            return candidate
    return None


def _parse_min_cluster_size(model_variant: Any) -> int | None:
    if model_variant is None:
        return None
    match = re.search(r"(?:^|_)mcs(\d+)(?:_|$)", str(model_variant))
    return int(match.group(1)) if match else None


def _is_official_umap_hdbscan_candidate(
    candidate_key: str,
    candidate: dict[str, Any] | None,
    model_suite: dict[str, Any],
    cfg: PipelineConfig,
) -> bool:
    if candidate_key == model_suite.get("official_candidate_key"):
        return True
    official_model_name = cfg.get("official_model_suite.umap_hdbscan.model_name")
    return bool(candidate and official_model_name and candidate.get("model_name") == official_model_name)


def _jitter_stability(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    repeats: int,
    jitter_scale: float,
    seed: int,
) -> dict[str, float | None]:
    from sklearn.metrics import adjusted_rand_score

    valid_idx = np.where(labels >= 0)[0]
    if len(valid_idx) == 0 or len(np.unique(labels[valid_idx])) < 2:
        return {
            "jitter_ari_mean": None,
            "jitter_ari_std": None,
            "jitter_label_recovery_accuracy_mean": None,
            "jitter_label_recovery_accuracy_std": None,
            "_cluster_recovery_accuracy_mean_by_label": {},
        }

    aris: list[float] = []
    accuracies: list[float] = []
    chosen_base = valid_idx[
        deterministic_sample_indices(len(valid_idx), min(sample_size, len(valid_idx)), seed)
    ]
    X_sample = X[chosen_base]
    y_sample = labels[chosen_base]
    unique_labels = np.sort(np.unique(y_sample))
    cluster_accuracies: dict[int, list[float]] = {int(label): [] for label in unique_labels}

    centroids = np.vstack([X_sample[y_sample == label].mean(axis=0) for label in unique_labels])
    for repeat in range(max(repeats, 1)):
        rng = np.random.default_rng(seed + repeat + 1)
        jittered = X_sample + rng.normal(0.0, jitter_scale, size=X_sample.shape).astype(np.float32)
        distances = ((jittered[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        predicted = unique_labels[np.argmin(distances, axis=1)]
        aris.append(float(adjusted_rand_score(y_sample, predicted)))
        accuracies.append(float(np.mean(predicted == y_sample)))
        for label in unique_labels:
            mask = y_sample == label
            if np.any(mask):
                cluster_accuracies[int(label)].append(float(np.mean(predicted[mask] == label)))

    return {
        "jitter_ari_mean": float(np.mean(aris)),
        "jitter_ari_std": float(np.std(aris)),
        "jitter_label_recovery_accuracy_mean": float(np.mean(accuracies)),
        "jitter_label_recovery_accuracy_std": float(np.std(accuracies)),
        "_cluster_recovery_accuracy_mean_by_label": {
            str(label): float(np.mean(values))
            for label, values in cluster_accuracies.items()
            if values
        },
    }


def _cluster_readiness_rows(
    joined: pl.DataFrame,
    *,
    candidate_key: str,
    total_rows: int,
    cluster_recovery: dict[str, float],
    min_cluster_size: int,
) -> list[dict[str, Any]]:
    core = joined.filter(pl.col("tribe_id") >= 0)
    if core.is_empty():
        return []
    core_rows = max(core.height, 1)
    summary = (
        core.group_by("tribe_id")
        .agg(
            pl.len().alias("customers"),
            pl.col("assignment_confidence_score").mean().alias("mean_assignment_confidence"),
            pl.col("assignment_confidence_score").quantile(0.10).alias("p10_assignment_confidence"),
            pl.col("assignment_confidence_score").min().alias("min_assignment_confidence"),
            pl.col("assignment_source").n_unique().alias("assignment_source_count"),
        )
        .sort("tribe_id")
    )
    rows: list[dict[str, Any]] = []
    for row in summary.iter_rows(named=True):
        tribe_id = int(row["tribe_id"])
        customers = int(row["customers"])
        recovery = cluster_recovery.get(str(tribe_id))
        confidence = row.get("mean_assignment_confidence")
        p10_confidence = row.get("p10_assignment_confidence")
        readiness, issues = _profile_readiness(
            customers=customers,
            min_cluster_size=min_cluster_size,
            recovery=recovery,
            confidence=confidence,
            p10_confidence=p10_confidence,
        )
        rows.append(
            {
                "candidate_id": candidate_key,
                "tribe_id": tribe_id,
                "customers": customers,
                "customer_share_pct": customers / max(total_rows, 1) * 100.0,
                "core_customer_share_pct": customers / core_rows * 100.0,
                "min_cluster_size_reference": min_cluster_size,
                "mean_assignment_confidence": confidence,
                "p10_assignment_confidence": p10_confidence,
                "min_assignment_confidence": row.get("min_assignment_confidence"),
                "assignment_source_count": int(row.get("assignment_source_count") or 0),
                "jitter_label_recovery_accuracy_mean": recovery,
                "profile_readiness": readiness,
                "readiness_issues": "; ".join(issues) if issues else "pass",
            }
        )
    return rows


def _profile_readiness(
    *,
    customers: int,
    min_cluster_size: int,
    recovery: float | None,
    confidence: float | None,
    p10_confidence: float | None,
) -> tuple[str, list[str]]:
    confidence = _finite_or_none(confidence)
    p10_confidence = _finite_or_none(p10_confidence)
    recovery = _finite_or_none(recovery)
    issues: list[str] = []
    if min_cluster_size and customers < min_cluster_size:
        issues.append(f"customers<{min_cluster_size}")
    if recovery is None:
        issues.append("missing_jitter_recovery")
    elif recovery < 0.60:
        issues.append("jitter_recovery<0.60")
    if confidence is not None and confidence < 0.20:
        issues.append("mean_assignment_confidence<0.20")
    if p10_confidence is not None and p10_confidence < 0.05:
        issues.append("p10_assignment_confidence<0.05")

    if not issues and recovery is not None and recovery >= 0.85 and (confidence is None or confidence >= 0.30):
        return "strong", issues
    if not issues:
        return "usable", issues
    return "review", issues


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if np.isfinite(value_float) else None
