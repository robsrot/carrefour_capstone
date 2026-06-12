"""Minimal quality-report helper for prepared artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.utils import collect_streaming


def build_quality_report(force: bool = False, cfg: PipelineConfig = CONFIG) -> dict[str, Any]:
    """Return the cached Notebook 02 quality report or create a lightweight prepared-data report."""

    report_path = cfg.data_prod / "quality_report.json"
    if report_path.exists() and not force:
        with report_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    lf = load_prepared_transactions(path=cfg.data_prod / cfg.get("data.prepared_transactions"), cfg=cfg)
    required = cfg.get("data.required_columns", [])
    row = collect_streaming(
        lf.select(
            [
                pl.len().alias("rows"),
                pl.col("cliente").n_unique().alias("unique_customers"),
                pl.col("ticket").n_unique().alias("unique_tickets"),
                pl.col("idarticu").n_unique().alias("unique_products"),
                *[pl.col(col).null_count().alias(f"{col}_nulls") for col in required],
            ]
        )
    ).row(0, named=True)
    report = {
        "prepared_data_audit": row,
        "_summary": {"all_passed": all(row.get(f"{col}_nulls", 0) == 0 for col in required)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
