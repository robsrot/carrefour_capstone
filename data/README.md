# Data Directory

Data files are local artifacts and are not committed to this repository.

## Purpose

`data/` holds raw inputs and prepared transaction tables only. The ML pipeline writes generated embeddings, features, models, reports, figures, profiles, and experiments to `outputs/<mode>/`.

## Expected Layout

```text
data/
|-- raw/
|   |-- csv/          Original source CSV files
|   `-- parquet/      Generated raw Parquet files
|-- processed/        Production prepared data
`-- dev/              Stratified dev subset
```

## Required Raw Files

Place these files in `data/raw/csv/` after cloning:

| File | Description |
|---|---|
| `ie_maestra_articulos.csv` | Product master / article catalogue |
| `ie_linea_ticket.csv` | Transactional ticket lines |

Then run:

```python
from src.data_loader import verify_csv_checksums, convert_csv_to_parquet

verify_csv_checksums()
convert_csv_to_parquet()
```

## Prepared Data Artifacts

Production preprocessing writes to `data/processed/`:

| Artifact | Purpose |
|---|---|
| `quality_report.json` | Full data quality gate results |
| `df_combined.parquet` | Clean joined ticket/product table |
| `customer_kpis.parquet` | Per-customer spend, visit, promo, and basket metrics for profiling |
| `product_eda_*.parquet` | Optional product EDA tables |

Dev subset generation writes to `data/dev/`:

| Artifact | Purpose |
|---|---|
| `df_combined.parquet` | Stratified dev transaction subset |
| `subset_metadata.json` | Dev subset parameters and validation hashes |

## Generated ML Artifacts

Generated ML artifacts belong under:

```text
outputs/<mode>/
  embeddings/
  features/
  figures/
  models/
  profiles/
  reports/
  experiments/   # dev only
```

If embeddings, cluster labels, figures, model binaries, or reports appear under `data/dev/` or `data/processed/`, treat them as stale local clutter unless a current source module explicitly reads them.

## Rules

- Never commit files under `data/raw/`, `data/processed/`, or `data/dev/` except `.gitkeep` and documentation.
- Keep prod and dev prepared data separate by setting `CARREFOUR_MODE` before running pipeline code.
- Regenerate downstream caches with the relevant `force=True` flag after changing upstream feature definitions or hyperparameters.
