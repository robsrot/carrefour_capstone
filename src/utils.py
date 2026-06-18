"""Shared helpers for deterministic, cache-aware pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def metadata_hash(metadata: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_safe(metadata), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_metadata_path(path: Path) -> Path:
    file_path = Path(path)
    parts = file_path.parts
    for idx, part in enumerate(parts[:-1]):
        if part.lower() == "outputs" and idx + 1 < len(parts):
            return Path(*parts[: idx + 2]) / ".artifact_metadata.json"
    return file_path.parent / ".artifact_metadata.json"


def _legacy_artifact_metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.meta.json")


def _artifact_metadata_key(path: Path, manifest_path: Path) -> str:
    try:
        return path.relative_to(manifest_path.parent).as_posix()
    except ValueError:
        return str(path)


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(file_path), "exists": False}
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def should_use_cache(
    path: Path,
    force: bool = False,
    use_cached: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    return bool(cache_status(path, force=force, use_cached=use_cached, metadata=metadata)["cache_hit"])


def cache_status(
    path: Path | str,
    force: bool = False,
    use_cached: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a readable cache decision for one artifact."""

    path = Path(path)
    status: dict[str, Any] = {
        "path": path,
        "exists": path.exists(),
        "use_cached": bool(use_cached),
        "force": bool(force),
        "metadata_required": metadata is not None,
        "metadata_manifest": None,
        "metadata_status": "not_required" if metadata is None else "unchecked",
        "cache_hit": False,
        "reason": "",
    }
    if not use_cached or force or not path.exists():
        if not use_cached:
            status["reason"] = "cache disabled by config"
        elif force:
            status["reason"] = "force rebuild enabled"
        else:
            status["reason"] = "artifact missing"
        return status
    if metadata is None:
        status["cache_hit"] = True
        status["reason"] = "artifact exists and no metadata was required"
        return status

    meta_path = artifact_metadata_path(path)
    status["metadata_manifest"] = meta_path
    try:
        if meta_path.exists():
            cached = read_json(meta_path)
            key = _artifact_metadata_key(path, meta_path)
            entry = cached.get("artifacts", {}).get(key)
            if entry:
                status["metadata_status"] = "match" if entry.get("metadata_hash") == metadata_hash(metadata) else "mismatch"
                status["cache_hit"] = status["metadata_status"] == "match"
                status["reason"] = "metadata matched" if status["cache_hit"] else "metadata hash mismatch"
                return status
            status["metadata_status"] = "missing_entry"
            status["reason"] = "metadata manifest has no entry for this artifact"
            return status

        legacy_meta_path = _legacy_artifact_metadata_path(path)
        if legacy_meta_path.exists():
            cached = read_json(legacy_meta_path)
            status["metadata_manifest"] = legacy_meta_path
            status["metadata_status"] = "match" if cached.get("metadata_hash") == metadata_hash(metadata) else "mismatch"
            status["cache_hit"] = status["metadata_status"] == "match"
            status["reason"] = "legacy metadata matched" if status["cache_hit"] else "legacy metadata hash mismatch"
            return status
    except (OSError, json.JSONDecodeError):
        status["metadata_status"] = "unreadable"
        status["reason"] = "metadata could not be read"
        return status
    status["metadata_status"] = "missing_manifest"
    status["reason"] = "metadata manifest missing"
    return status


def write_artifact_metadata(path: Path, metadata: Mapping[str, Any]) -> Path:
    meta_path = artifact_metadata_path(path)
    if meta_path.exists():
        try:
            payload = read_json(meta_path)
        except (OSError, json.JSONDecodeError):
            payload = {}
    else:
        payload = {}
    payload.setdefault("version", 1)
    payload.setdefault("artifacts", {})
    payload["artifacts"][_artifact_metadata_key(path, meta_path)] = {
        "metadata_hash": metadata_hash(metadata),
        "metadata": _json_safe(metadata),
    }
    return write_json(meta_path, payload)


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame with Polars' streaming engine when available."""

    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def scan_if_path(frame_or_path: pl.DataFrame | pl.LazyFrame | str | Path) -> pl.LazyFrame:
    if isinstance(frame_or_path, pl.LazyFrame):
        return frame_or_path
    if isinstance(frame_or_path, pl.DataFrame):
        return frame_or_path.lazy()
    return pl.scan_parquet(str(frame_or_path))


def schema_names(lf: pl.LazyFrame) -> list[str]:
    try:
        return list(lf.collect_schema().names())
    except AttributeError:
        return list(lf.schema.keys())


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_hash(value: Any) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass


def numeric_feature_columns(df: pl.DataFrame, exclude: Sequence[str] = ("cliente",)) -> list[str]:
    excluded = set(exclude)
    numeric_types = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    return [name for name, dtype in df.schema.items() if name not in excluded and dtype in numeric_types]


def embedding_columns(df_or_schema: pl.DataFrame | dict[str, Any]) -> list[str]:
    names = list(df_or_schema.schema.keys()) if isinstance(df_or_schema, pl.DataFrame) else list(df_or_schema)
    return [name for name in names if name.startswith("emb_") or name.startswith("x_") or name.startswith("latent_")]


def frame_to_numpy(df: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        raise ValueError("No numeric feature columns were provided.")
    return df.select(list(columns)).to_numpy().astype(np.float32, copy=False)


def deterministic_sample_indices(n_rows: int, sample_size: int | None, seed: int) -> np.ndarray:
    if sample_size is None or sample_size <= 0 or sample_size >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=sample_size, replace=False))


def list_mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None
