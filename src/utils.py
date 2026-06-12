"""Shared helpers for deterministic, cache-aware pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def should_use_cache(path: Path, force: bool = False, use_cached: bool = True) -> bool:
    return use_cached and not force and path.exists()


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
