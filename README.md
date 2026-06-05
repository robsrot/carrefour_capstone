# Carrefour Data Challenge

Product-first behavioral customer segmentation for Carrefour checkout data. The goal is to discover actionable customer tribes from what people buy, not from demographic attributes.

## Current Status

As of 2026-06-05:

- Production preprocessing and EDA are complete: raw CSVs have been converted, cleaned, joined, quality-checked, and summarized into `data/processed/`.
- The ML pipeline has been run end-to-end in dev mode on a stratified 44k-customer subset, through product embeddings, customer vectors, UMAP/PCA, HDBSCAN, K-Means baselines, and visual outputs.
- The production-scale ML run is still pending. Tribe profiles and commercial tribe names are also still missing.

For the detailed status, gaps, and product-first improvement plan, see [docs/PROJECT_STATUS_AND_ROADMAP.md](docs/PROJECT_STATUS_AND_ROADMAP.md).

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate carrefour
python -m ipykernel install --user --name=carrefour --display-name "Python (carrefour)"
```

### 2. Add the raw CSV files

Raw data is not committed. Place the source files here:

```text
data/raw/csv/ie_maestra_articulos.csv
data/raw/csv/ie_linea_ticket.csv
```

### 3. Convert raw CSVs to Parquet

Run once from Python or from a notebook:

```python
from src.data_loader import verify_csv_checksums, convert_csv_to_parquet

verify_csv_checksums()
convert_csv_to_parquet()
```

Use `verify_csv_checksums(record=True)` only on the machine that establishes the canonical raw files.

### 4. Run the notebooks in order

```bash
jupyter notebook notebooks/01_exploration.ipynb
jupyter notebook notebooks/02_pre-analysis.ipynb
```

After `02_pre-analysis.ipynb` has created `data/processed/df_combined.parquet` and `data/processed/customer_kpis.parquet`, generate the dev subset:

```powershell
$env:CARREFOUR_MODE = "prod"
python -m src.generate_dev_subset
```

Then run the ML pipeline in dev mode:

```powershell
$env:CARREFOUR_MODE = "dev"
jupyter notebook notebooks/03_ml_pipeline.ipynb
```

Prod mode is the default when `CARREFOUR_MODE` is unset.

## Repository Layout

```text
configs/       Hyperparameters and dev/prod overrides
data/          Local raw, processed, and dev data artifacts; never committed
docs/          Project context, current status, and roadmap
models/        Local trained models; never committed
notebooks/     Ordered analysis and pipeline notebooks
outputs/       Local plots and exported visuals; never committed
src/           Reusable pipeline modules
tests/         Reproducibility, data loader, and data quality tests
```

## Operating Conventions

- Keep reusable logic in `src/`; notebooks should orchestrate and explain, not duplicate pipeline code.
- Put hyperparameters in `configs/base.yaml` and `configs/dev.yaml`; expose new values through `src/config.py`.
- Use `CARREFOUR_MODE=dev` for fast iteration and `CARREFOUR_MODE=prod` for full artifacts.
- Do not commit raw data, Parquet caches, trained models, or output images.
- Always join customer-level data with `join(on="cliente")`; do not rely on positional row order.
- Current UMAP/PCA artifact names still say `20d`, but the active config uses 50 clustering dimensions. Treat the filename as historical until the cache naming is migrated deliberately.

## Documentation

- [CLAUDE.md](CLAUDE.md) is the agent/operator guide for this repo.
- [docs/PROJECT_STATUS_AND_ROADMAP.md](docs/PROJECT_STATUS_AND_ROADMAP.md) is the current state, missing work, and next experiment plan.
- [docs/Carrefour_Data_Challenge_Project_Context.md](docs/Carrefour_Data_Challenge_Project_Context.md) preserves the original project brief and methodological constraints.
