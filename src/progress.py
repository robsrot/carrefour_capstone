"""Small notebook-friendly progress logging helpers."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


def _enabled(cfg: Any | None) -> bool:
    if cfg is None or not hasattr(cfg, "get"):
        return True
    return bool(cfg.get("progress.enabled", True))


def _format_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def log_event(stage: str, message: str, cfg: Any | None = None, **details: Any) -> None:
    """Print one concise timestamped progress line when progress logging is enabled."""

    if not _enabled(cfg):
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    detail_text = ""
    if details:
        rendered = ", ".join(f"{key}={_format_value(value)}" for key, value in details.items() if value is not None)
        detail_text = f" | {rendered}" if rendered else ""
    print(f"[{timestamp}] {stage}: {message}{detail_text}", flush=True)


@contextmanager
def stage_timer(stage: str, message: str, cfg: Any | None = None, **details: Any) -> Iterator[None]:
    """Log start/finish lines with elapsed seconds around a stage."""

    started_at = perf_counter()
    log_event(stage, message, cfg=cfg, **details)
    try:
        yield
    finally:
        elapsed = perf_counter() - started_at
        log_event(stage, "finished", cfg=cfg, elapsed_s=round(elapsed, 1))
