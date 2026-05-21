import pandas as pd
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


def load_maestra_articulos() -> pd.DataFrame:
    """load product master into memory. 34 MB — safe to hold as a full pandas DataFrame."""
    return pd.read_parquet(
        _PARQUET_DIR / _SOURCES["maestra_articulos"]["parquet"]
    )


def load_linea_tickets(columns: list[str] | None = None) -> pl.LazyFrame:
    """return a lazy frame over the full transaction dataset. pass columns= to select a subset."""
    lf = pl.scan_parquet(_PARQUET_DIR / _SOURCES["linea_tickets"]["parquet"])
    return lf.select(columns) if columns else lf
