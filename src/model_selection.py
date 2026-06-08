"""Helpers for config-driven model selection in the ML pipeline."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


AUTO = "auto"


def is_auto(value: str | None) -> bool:
    """Return True when a pipeline component should be selected dynamically."""
    return str(value or "").lower() == AUTO


def validate_choice(choice: str, candidates: Iterable[str], component: str) -> str:
    """Validate a selected model name against configured candidates."""
    selected = str(choice)
    valid = [str(c) for c in candidates]
    if selected not in valid:
        raise ValueError(
            f"Unknown {component}={selected!r}; configured candidates are {valid}"
        )
    return selected


def resolve_choice(
    requested: str,
    *,
    fallback: str,
    candidates: Iterable[str],
    component: str,
) -> str:
    """Resolve an auto value to its fallback before benchmark results exist."""
    selected = fallback if is_auto(requested) else requested
    return validate_choice(selected, candidates, component)


def _records(frame: Any) -> list[dict[str, Any]]:
    """Convert pandas, Polars, or records-like objects into list-of-dict rows."""
    if frame is None:
        return []
    if isinstance(frame, list):
        return [dict(row) for row in frame]
    if hasattr(frame, "to_dicts"):
        return [dict(row) for row in frame.to_dicts()]
    if hasattr(frame, "to_dict"):
        maybe = frame.to_dict(orient="records")
        return [dict(row) for row in maybe]
    raise TypeError(f"Cannot convert {type(frame)!r} to records")


def _number(value: Any, default: float = math.inf) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isnan(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _coverage_adj_silhouette(row: dict[str, Any]) -> float:
    sil = _number(row.get("silhouette"), default=0.0)
    noise = _number(row.get("noise_pct"), default=100.0)
    return sil * max(0.0, 1.0 - noise / 100.0)


def select_vector_source(
    comparison: Any,
    *,
    target_clusters: int,
) -> dict[str, Any]:
    """Select the vector source with the best benchmarked clustering behavior."""
    rows = _records(comparison)
    if not rows:
        raise ValueError("Vector-source selection requires a non-empty comparison table.")

    scored = []
    for row in rows:
        clusters = int(_number(row.get("assigned_n_clusters"), default=0))
        scored.append({
            **row,
            "target_gap": abs(clusters - int(target_clusters)),
            "coverage_adj_silhouette": (
                _number(row.get("assigned_silhouette"), default=0.0)
                * max(0.0, 1.0 - _number(row.get("assigned_from_noise_pct"), default=100.0) / 100.0)
            ),
        })

    usable = [row for row in scored if int(_number(row.get("assigned_n_clusters"), 0)) >= 2]
    ranked = sorted(
        usable or scored,
        key=lambda row: (
            int(row["target_gap"]),
            _number(row.get("assigned_from_noise_pct")),
            _number(row.get("assigned_davies_bouldin")),
            -_number(row.get("coverage_adj_silhouette"), default=0.0),
        ),
    )
    winner = dict(ranked[0])
    winner["selected"] = str(winner["vector_source"])
    return winner


def select_reducer(comparison: Any) -> dict[str, Any]:
    """Select the reducer with the strongest benchmark score."""
    rows = _records(comparison)
    if not rows:
        raise ValueError("Reducer selection requires a non-empty comparison table.")
    ranked = sorted(
        rows,
        key=lambda row: (
            -_number(row.get("silhouette"), default=-math.inf),
            _number(row.get("davies_bouldin")),
        ),
    )
    winner = dict(ranked[0])
    winner["selected"] = str(winner["reducer"])
    return winner


def select_clusterer(
    comparison: Any,
    *,
    target_clusters: int,
) -> dict[str, Any]:
    """Select the clusterer closest to target tribe count, then by geometry."""
    rows = _records(comparison)
    if not rows:
        raise ValueError("Clusterer selection requires a non-empty comparison table.")

    scored = []
    for row in rows:
        clusters = int(_number(row.get("n_clusters"), default=0))
        scored.append({
            **row,
            "target_gap": abs(clusters - int(target_clusters)),
            "coverage_adj_silhouette": _coverage_adj_silhouette(row),
        })

    usable = [row for row in scored if int(_number(row.get("n_clusters"), 0)) >= 2]
    ranked = sorted(
        usable or scored,
        key=lambda row: (
            int(row["target_gap"]),
            -_number(row.get("coverage_adj_silhouette"), default=0.0),
            _number(row.get("noise_pct")),
            _number(row.get("davies_bouldin")),
        ),
    )
    winner = dict(ranked[0])
    winner["selected"] = str(winner.get("candidate", winner.get("method")))
    return winner


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_selection_artifact(
    path: Path,
    *,
    component: str,
    requested: str,
    selected: str,
    fallback: str | None = None,
    candidates: Iterable[str] | None = None,
    winner: dict[str, Any] | None = None,
    comparison_path: Path | None = None,
) -> dict[str, Any]:
    """Persist the selected model and the row that justified it."""
    payload = {
        "component": component,
        "requested": requested,
        "selected": selected,
        "fallback": fallback,
        "candidates": list(candidates or []),
        "winner": winner or {},
        "comparison_path": str(comparison_path) if comparison_path else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return payload


def load_selection_artifact(path: Path) -> dict[str, Any] | None:
    """Load a persisted selection artifact if it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text())
