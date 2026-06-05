import hashlib
import json

import polars as pl
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CSV_DIR = _ROOT / "data" / "raw" / "csv"
_PARQUET_DIR = _ROOT / "data" / "raw" / "parquet"

# single source of truth for file mappings
_SOURCES = {
    "maestra_articulos": {
        "csv": "ie_maestra_articulos.csv",
        "parquet": "maestra_articulos.parquet",
    },
    "linea_tickets": {
        "csv": "ie_linea_ticket.csv",
        "parquet": "linea_tickets.parquet",
    },
}


# Canonical SHA-256 digests of the source CSV files.
# If a file's digest doesn't match, it has been modified or re-saved since the
# canonical version was hashed — downstream parquets will diverge across machines.
# Update these values whenever the source files are intentionally replaced.
_CSV_CHECKSUMS: dict[str, str] = {
    # "ie_maestra_articulos.csv": "<sha256>",
    # "ie_linea_ticket.csv":      "<sha256>",
}
_CHECKSUMS_FILE = _CSV_DIR / "checksums.json"


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_csv_checksums(*, record: bool = False) -> None:
    """Verify (or record) SHA-256 checksums of the raw CSV files.

    Call with record=True once after receiving the canonical CSV files to write
    checksums.json. On every subsequent machine, call with record=False (default)
    to confirm the files are identical to the canonical copy.

    Raises RuntimeError if any checksum mismatches (record=False only).
    """
    results: dict[str, str] = {}
    for name, paths in _SOURCES.items():
        csv_path = _CSV_DIR / paths["csv"]
        if not csv_path.exists():
            print(f"  {name}: CSV not found — skipping checksum")
            continue
        digest = _sha256(csv_path)
        results[paths["csv"]] = digest
        if record:
            print(f"  {name}: {digest}")
        else:
            expected = _CSV_CHECKSUMS.get(paths["csv"])
            if expected is None:
                print(f"  {name}: no reference checksum stored — run verify_csv_checksums(record=True) first")
            elif digest == expected:
                print(f"  {name}: OK")
            else:
                raise RuntimeError(
                    f"CSV checksum mismatch for {paths['csv']}!\n"
                    f"  Expected : {expected}\n"
                    f"  Got      : {digest}\n"
                    "This file differs from the canonical copy. "
                    "Re-download it from the shared source before converting."
                )
    if record:
        _CHECKSUMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CHECKSUMS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nChecksums written → {_CHECKSUMS_FILE}")
        print("Copy the digests into _CSV_CHECKSUMS in src/data_loader.py and commit.")


def _stream_csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    """stream csv → parquet in 256 mb chunks; never loads the full file into memory."""
    reader = pa_csv.open_csv(
        csv_path,
        parse_options=pa_csv.ParseOptions(delimiter=";"),
        read_options=pa_csv.ReadOptions(
            encoding="latin-1",
            block_size=256 * 1024 * 1024,
        ),
    )
    rows = 0
    with pq.ParquetWriter(parquet_path, reader.schema) as writer:
        for batch in reader:
            writer.write_batch(batch)
            rows += len(batch)
            print(f"\r  {rows:,} rows...", end="", flush=True)
    print(f"\r  done — {rows:,} rows total")


def convert_csv_to_parquet(force: bool = False) -> None:
    """convert raw csv files to parquet; skip existing unless force=True."""
    _PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    for name, paths in _SOURCES.items():
        out_path = _PARQUET_DIR / paths["parquet"]
        if out_path.exists() and not force:
            print(f"{name}: already converted, skipping")
            continue

        print(f"{name}: converting...")
        _stream_csv_to_parquet(_CSV_DIR / paths["csv"], out_path)
        print(f"{name}: saved → {out_path.name}")


def peek(dataset: str, n: int = 5) -> pl.DataFrame:
    """return the first n rows without loading the full file."""
    path = _PARQUET_DIR / _SOURCES[dataset]["parquet"]
    pf = pq.ParquetFile(path)
    print(f"{dataset}: {pf.metadata.num_rows:,} rows × {pf.metadata.num_columns} cols")
    return pl.scan_parquet(path).head(n).collect()


def load_maestra_articulos() -> pl.DataFrame:
    """load product master into memory. ~34 MB — safe to hold as a full Polars DataFrame."""
    return pl.read_parquet(_PARQUET_DIR / _SOURCES["maestra_articulos"]["parquet"])


def load_linea_tickets(columns: list[str] | None = None) -> pl.LazyFrame:
    """return a lazy frame over the full transaction dataset. pass columns= to select a subset."""
    lf = pl.scan_parquet(_PARQUET_DIR / _SOURCES["linea_tickets"]["parquet"])
    return lf.select(columns) if columns else lf
