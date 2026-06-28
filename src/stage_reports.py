from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.config import CONFIG, PipelineConfig


ReportItems = Mapping[str, Any] | Sequence[tuple[str, Any]]


def stage_report_path(
    stage: str | int,
    *,
    cfg: PipelineConfig = CONFIG,
    reports_dir: str | Path | None = None,
) -> Path:
    """Return the one notebook-facing markdown report path for a pipeline stage."""
    root = Path(reports_dir) if reports_dir is not None else cfg.reports
    return root / f"stage_{_stage_token(stage)}.md"


def write_stage_report(
    stage: str | int,
    title: str,
    *,
    summary: str | Iterable[str] | None = None,
    metrics: Any | None = None,
    figures: ReportItems | None = None,
    artifacts: ReportItems | None = None,
    cfg: PipelineConfig = CONFIG,
    max_metric_rows: int = 12,
    max_value_width: int = 120,
) -> Path:
    """Write a compact per-stage markdown report and return its path.

    The report is intentionally short: status bullets, optional key metrics, visual
    evidence paths, and a small artifact list. Detailed CSV/Parquet evidence can
    still exist elsewhere, but the notebook has one human-facing report per stage.
    """
    path = stage_report_path(stage, cfg=cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Stage {_stage_display(stage)}: {title}",
        "",
        f"- Mode: `{cfg.mode}`",
    ]

    summary_lines = _as_lines(summary)
    if summary_lines:
        lines.extend(["", "## Status"])
        lines.extend(f"- {_format_value(line, max_value_width=max_value_width)}" for line in summary_lines)

    metric_rows = _metric_rows(metrics, max_rows=max_metric_rows, max_value_width=max_value_width)
    if metric_rows:
        has_ideal = any(ideal for _, _, ideal in metric_rows)
        if has_ideal:
            lines.extend(["", "## Key Metrics", "", "| Metric | Value | Ideal Value / Range |", "|---|---|---|"])
            lines.extend(f"| {metric} | {value} | {ideal or ''} |" for metric, value, ideal in metric_rows)
        else:
            lines.extend(["", "## Key Metrics", "", "| Metric | Value |", "|---|---|"])
            lines.extend(f"| {metric} | {value} |" for metric, value, _ in metric_rows)

    figure_rows = _item_rows(figures, max_value_width=max_value_width)
    if figure_rows:
        lines.extend(["", "## Visual Evidence"])
        lines.extend(f"- {label}: `{value}`" for label, value in figure_rows)

    artifact_rows = _item_rows(artifacts, max_value_width=max_value_width)
    if artifact_rows:
        lines.extend(["", "## Artifacts"])
        lines.extend(f"- {label}: `{value}`" for label, value in artifact_rows)

    lines.extend(["", "_Notebook-facing compact report. Detailed data artifacts are kept for reproducibility._", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def compact_stage_report_markdown(path: str | Path, *, max_lines: int = 80) -> str:
    """Read a stage report for concise notebook display."""
    report_path = Path(path)
    lines = report_path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    clipped = lines[:max_lines]
    clipped.extend(["", f"_Showing first {max_lines} lines. Full report: `{report_path}`_"])
    return "\n".join(clipped)


def display_stage_report(path: str | Path, *, max_lines: int = 80) -> None:
    """Display a compact report in a notebook without dumping long markdown."""
    from IPython.display import Markdown, display

    display(Markdown(compact_stage_report_markdown(path, max_lines=max_lines)))


def _stage_token(stage: str | int) -> str:
    text = str(stage).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if token.isdigit():
        return f"{int(token):02d}"
    return token or "unknown"


def _stage_display(stage: str | int) -> str:
    text = str(stage).strip()
    match = re.fullmatch(r"0*(\d+)[_.](\d+)", text)
    if match:
        return f"{int(match.group(1))}.{match.group(2)}"
    return text


def _as_lines(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(line).strip() for line in value if str(line).strip()]


def _metric_rows(metrics: Any | None, *, max_rows: int, max_value_width: int) -> list[tuple[str, str, str | None]]:
    if metrics is None:
        return []

    if isinstance(metrics, Mapping):
        return [
            (_escape_cell(str(key)), _escape_cell(_format_value(value, max_value_width=max_value_width)), None)
            for key, value in metrics.items()
        ][:max_rows]

    rows = _records_from_frame_like(metrics)
    if not rows:
        return [("value", _escape_cell(_format_value(metrics, max_value_width=max_value_width)), None)]

    if all(set(row.keys()) >= {"metric", "value"} for row in rows):
        return [
            (
                _escape_cell(_format_value(row["metric"], max_value_width=max_value_width)),
                _escape_cell(_format_value(row["value"], max_value_width=max_value_width)),
                _escape_cell(
                    _format_value(
                        row.get("ideal_value_range", row.get("ideal_range", row.get("ideal"))),
                        max_value_width=max_value_width,
                    )
                )
                if row.get("ideal_value_range", row.get("ideal_range", row.get("ideal"))) is not None
                else None,
            )
            for row in rows[:max_rows]
        ]

    rendered: list[tuple[str, str, str | None]] = []
    for index, row in enumerate(rows[:max_rows], start=1):
        metric = str(row.get("metric", row.get("name", f"row_{index}")))
        values = {
            key: value
            for key, value in row.items()
            if key not in {"metric", "name", "ideal_value_range", "ideal_range", "ideal"} and value is not None
        }
        ideal = row.get("ideal_value_range", row.get("ideal_range", row.get("ideal")))
        rendered.append(
            (
                _escape_cell(metric),
                _escape_cell(_format_value(values, max_value_width=max_value_width)),
                _escape_cell(_format_value(ideal, max_value_width=max_value_width)) if ideal is not None else None,
            )
        )
    return rendered


def _records_from_frame_like(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "head") and hasattr(value, "to_dicts"):
        return list(value.head(100).to_dicts())
    if hasattr(value, "head") and hasattr(value, "to_dict"):
        frame = value.head(100)
        records = frame.to_dict(orient="records")
        return list(records)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                records.append(dict(item))
            elif isinstance(item, tuple) and len(item) == 2:
                records.append({"metric": item[0], "value": item[1]})
            elif isinstance(item, tuple) and len(item) == 3:
                records.append({"metric": item[0], "value": item[1], "ideal_value_range": item[2]})
        return records
    return []


def _item_rows(items: ReportItems | None, *, max_value_width: int) -> list[tuple[str, str]]:
    if items is None:
        return []
    pairs = items.items() if isinstance(items, Mapping) else items
    return [
        (
            _escape_cell(str(label)),
            _format_path_or_value(value, max_value_width=max_value_width),
        )
        for label, value in pairs
        if value is not None
    ]


def _format_path_or_value(value: Any, *, max_value_width: int) -> str:
    if isinstance(value, Path):
        return str(value)
    return _format_value(value, max_value_width=max_value_width)


def _format_value(value: Any, *, max_value_width: int) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            text = str(value)
        elif abs(value) >= 1000:
            text = f"{value:,.0f}"
        elif abs(value) >= 1:
            text = f"{value:,.3f}".rstrip("0").rstrip(".")
        else:
            text = f"{value:.4f}".rstrip("0").rstrip(".")
    elif isinstance(value, Mapping):
        text = ", ".join(
            f"{key}={_format_value(val, max_value_width=max_value_width)}" for key, val in value.items()
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        text = ", ".join(_format_value(item, max_value_width=max_value_width) for item in value)
    else:
        text = str(value)

    text = text.replace("\n", " ").strip()
    if len(text) > max_value_width:
        return f"{text[: max_value_width - 1]}..."
    return text


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|")
