"""Data loading utilities for raw conversion and prepared ML inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.utils import collect_streaming, schema_names


RAW_SOURCES = {
    "maestra_articulos": {
        "csv": "ie_maestra_articulos.csv",
        "parquet": "maestra_articulos.parquet",
        "columns": ["idarticu", "desc_larga_articulo", "idsector", "desc_sector"],
    },
    "linea_tickets": {
        "csv": "ie_linea_ticket.csv",
        "parquet": "linea_tickets.parquet",
        "columns": [
            "idempres",
            "fecha",
            "hora",
            "ticket",
            "cliente",
            "idarticu",
            "unidades",
            "importe",
            "idpromoc",
            "idtiprod",
        ],
    },
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_csv_checksums(record: bool = False, cfg: PipelineConfig = CONFIG) -> dict[str, str]:
    """Verify or record SHA-256 checksums for the raw CSV inputs."""

    checksum_path = cfg.raw_csv / "checksums.sha256.json"
    actual = {}
    for name, spec in RAW_SOURCES.items():
        path = cfg.raw_csv / spec["csv"]
        if not path.exists():
            raise FileNotFoundError(f"Missing raw CSV for {name}: {path}")
        actual[name] = _sha256(path)

    if record or not checksum_path.exists():
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with checksum_path.open("w", encoding="utf-8") as f:
            json.dump(actual, f, indent=2)
        return actual

    import json

    with checksum_path.open("r", encoding="utf-8") as f:
        expected = json.load(f)
    mismatches = {k: (expected.get(k), v) for k, v in actual.items() if expected.get(k) != v}
    if mismatches:
        raise ValueError(f"Raw CSV checksum mismatch: {mismatches}")
    return actual


def convert_csv_to_parquet(force: bool = False, cfg: PipelineConfig = CONFIG) -> dict[str, Path]:
    """Convert raw semicolon-delimited Carrefour CSV files to Parquet."""

    cfg.raw_parquet.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name, spec in RAW_SOURCES.items():
        csv_path = cfg.raw_csv / spec["csv"]
        parquet_path = cfg.raw_parquet / spec["parquet"]
        outputs[name] = parquet_path
        if parquet_path.exists() and not force:
            continue
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing raw CSV: {csv_path}")
        lf = pl.scan_csv(
            csv_path,
            separator=";",
            encoding="latin1",
            infer_schema_length=10000,
            ignore_errors=False,
        )
        lf.sink_parquet(parquet_path)
    return outputs


def load_maestra_articulos(cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    path = cfg.product_master_path
    if not path.exists():
        raise FileNotFoundError(f"Product master parquet not found: {path}")
    return pl.read_parquet(path)


def load_linea_tickets(cfg: PipelineConfig = CONFIG) -> pl.LazyFrame:
    path = cfg.raw_parquet / RAW_SOURCES["linea_tickets"]["parquet"]
    if not path.exists():
        raise FileNotFoundError(f"Raw ticket parquet not found: {path}")
    return pl.scan_parquet(path)


def validate_required_fields(lf: pl.LazyFrame, required: Iterable[str]) -> None:
    names = set(schema_names(lf))
    missing = [col for col in required if col not in names]
    if missing:
        raise ValueError(f"Prepared transaction data is missing required columns: {missing}")


def load_prepared_transactions(
    path: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
    merge_metadata_if_missing: bool = True,
) -> pl.LazyFrame:
    """Load Notebook 02 output for Notebook 03 without touching raw CSVs."""

    transaction_path = Path(path) if path is not None else cfg.prepared_transactions_path
    if not transaction_path.exists():
        raise FileNotFoundError(
            f"Prepared transactions not found at {transaction_path}. Run Notebook 02 first."
        )
    lf = pl.scan_parquet(transaction_path)
    validate_required_fields(lf, cfg.get("data.required_columns", []))

    optional_metadata = {"desc_larga_articulo", "idsector", "desc_sector"}
    names = set(schema_names(lf))
    if merge_metadata_if_missing and not optional_metadata.issubset(names):
        if not cfg.product_master_path.exists():
            raise FileNotFoundError(
                "Prepared transactions do not include product metadata and the product master "
                f"was not found at {cfg.product_master_path}."
            )
        product_lf = pl.scan_parquet(cfg.product_master_path).select(
            ["idarticu", "desc_larga_articulo", "idsector", "desc_sector"]
        )
        lf = lf.join(product_lf, on="idarticu", how="left")
    return lf


def peek(lf: pl.LazyFrame | pl.DataFrame, n: int = 5) -> pl.DataFrame:
    if isinstance(lf, pl.DataFrame):
        return lf.head(n)
    return collect_streaming(lf.limit(n))
