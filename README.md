# Carrefour Data Challenge

Product-first behavioral customer segmentation for Carrefour checkout data. The goal is to discover actionable customer tribes from what people buy, not from demographic attributes.

## Current Status

As of 2026-06-13:

- Production preprocessing and EDA are complete: raw CSVs have been converted, cleaned, joined, quality-checked, and summarized into `data/processed/`.
- The ML pipeline is product-first: official customer vectors aggregate product embeddings using product quantities only, not spend. Spend remains available for profiling and business interpretation after clustering.
- UMAP is used as a dimensionality-reduction aid and candidate clustering representation. It is not treated as an automatic winner.
- Official Stage 6 compares committed model families without forcing a curated 10-15 cluster target. The client expectation is treated as a hypothesis checked after the model is evaluated.
- Dev-mode experiment sandboxes live in `notebooks/04_experiment_sandbox.ipynb`; IDF downweighting and broader UMAP-HDBSCAN sweeps belong there before any setting is promoted into YAML.
- Generated Stage 4+ artifacts should be treated as stale unless rebuilt after the latest quantity-only vectorization update.
- The production-scale ML run is still pending after the latest pipeline updates.

For the detailed status, gaps, and product-first improvement plan, see [docs/PROJECT_STATUS_AND_ROADMAP.md](docs/PROJECT_STATUS_AND_ROADMAP.md).

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate carrefour
python -m ipykernel install --user --name=carrefour --display-name "Python (carrefour)"
```

Verify the HDBSCAN-compatible dependency set:

```powershell
python -c "import sklearn, hdbscan, polars; print('sklearn', sklearn.__version__); print('hdbscan ok'); print('polars', polars.__version__)"
```

Expected: `sklearn 1.7.2`. If this prints `1.8.x`, recreate the environment or run:

```powershell
conda activate carrefour
conda install -c conda-forge scikit-learn=1.7.2
pip install --force-reinstall --no-deps hdbscan==0.8.40
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

Optionally run experiment sandboxes in dev mode before committing final hyperparameters:

```powershell
$env:CARREFOUR_MODE = "dev"
jupyter notebook notebooks/04_experiment_sandbox.ipynb
```

Promote only the winning settings from the sandbox summaries into YAML. Then run the clean ML pipeline in dev mode:

```powershell
$env:CARREFOUR_MODE = "dev"
jupyter notebook notebooks/03_ml_pipeline.ipynb
```

After pulling the current repo or changing vectorization/modeling code, rerun from Stage 4 onward before interpreting Stage 6+ results.

Prod mode is the default when `CARREFOUR_MODE` is unset.

## Colleague Handoff Checklist

Before handing the repo to another teammate:

```powershell
python -c "import sklearn, hdbscan; print(sklearn.__version__); print('hdbscan ok')"
pytest
git status --short
```

The source handoff must include the new config, notebook, and experiment modules. Generated data/model/output artifacts stay local and should not be committed.

For experimentation:

1. Work in `CARREFOUR_MODE=dev`.
2. Use `notebooks/04_experiment_sandbox.ipynb`.
3. Inspect each experiment's compact `*_summary.csv` or `*_summary.md`.
4. Promote only evidence-backed settings into YAML.
5. Rerun `notebooks/03_ml_pipeline.ipynb` cleanly from the affected upstream stage.

## Repository Layout

```text
configs/       Hyperparameters and dev/prod overrides
data/          Local raw, processed, and dev data artifacts; never committed
docs/          Project context, current status, and roadmap
notebooks/     Ordered analysis and pipeline notebooks
outputs/       Mode-scoped generated artifacts; never committed
src/           Reusable pipeline modules
tests/         Reproducibility, data loader, and data quality tests
```

Generated ML artifacts stay under `outputs/<mode>/`. Dev includes an experiment workbench; prod does not.

```text
outputs/dev/
  .artifact_metadata.json  Central cache metadata manifest
  embeddings/   Basket sentences and product embedding tables
  features/     Customer vectors and model-ready feature sets
  figures/      Client-facing plots and diagnostics
  models/       Trained local model binaries and internal model-selection caches
  profiles/     Tribe profile parquet outputs
  reports/      Human-readable summaries, exports, and final selection evidence
  experiments/  Optional sandbox runs, one flat folder per experiment

outputs/prod/
  .artifact_metadata.json  Central cache metadata manifest after prod stages run
  embeddings/   Basket sentences and product embedding tables
  features/     Customer vectors and model-ready feature sets
  figures/      Client-facing plots and diagnostics
  models/       Trained local model binaries and internal model-selection caches
  profiles/     Tribe profile parquet outputs
  reports/      Human-readable summaries, exports, and final selection evidence
```

## Operating Conventions

- Keep reusable logic in `src/`; notebooks should orchestrate and explain, not duplicate pipeline code.
- Put shared hyperparameters in `configs/base.yaml`, dev overrides in `configs/dev.yaml`, and production-scale overrides in `configs/prod.yaml`; expose new values through `src/config.py`.
- Use `CARREFOUR_MODE=dev` for fast iteration and `CARREFOUR_MODE=prod` for full artifacts.
- Use `notebooks/04_experiment_sandbox.ipynb` for hyperparameter experiments and `notebooks/03_ml_pipeline.ipynb` for the official YAML-driven run.
- Do not commit raw data, Parquet caches, trained models, or output images.
- Keep sandbox outputs in `outputs/dev/experiments/<experiment_name>/` with no nested subfolders; promote only the selected settings back into YAML.
- Experiments are disabled in prod. Production should only run the official pipeline with the scale-aware settings in `configs/prod.yaml`.
- Sandbox diagnostics keep Parquet as the canonical artifact and also write compact `*_summary.csv` / `*_summary.md` files so experiments can be inspected quickly.
- Keep `reports/` for concise human-facing outputs. Official Stage 6 compares only the committed candidates from `official_model_suite`; broader sweeps belong in the sandbox notebook.
- Official customer vectorization must not use `importe` or other spend fields. Use quantity-only weighting in the pipeline and test IDF variants only in the sandbox.
- Stage 7 writes cluster validity and perturbation-stability diagnostics after Stage 6, so selection is based on more than silhouette/noise alone.
- Stage 6 working files such as assignments and per-family result caches live under `outputs/<mode>/models/model_selection/`.
- Cache metadata is centralized in `outputs/<mode>/.artifact_metadata.json`; avoid per-file `.meta.json` sidecars.
- Always join customer-level data with `join(on="cliente")`; do not rely on positional row order.

## Documentation

- [AGENTS.md](AGENTS.md) is the agent/operator guide for this repo.
- [docs/PROJECT_STATUS_AND_ROADMAP.md](docs/PROJECT_STATUS_AND_ROADMAP.md) is the current state, missing work, and next experiment plan.
- [docs/Carrefour_Data_Challenge_Project_Context.md](docs/Carrefour_Data_Challenge_Project_Context.md) preserves the original project brief and methodological constraints.
