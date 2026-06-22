# Carrefour Data Challenge

Product-first behavioral customer segmentation for Carrefour checkout data. The goal is to discover actionable customer tribes from what people buy and how much they buy, not from demographics or spend-driven shortcuts.

## Current Status

As of 2026-06-22:

- Production preprocessing and dev-subset generation are complete in the local workspace: prepared tables live under `data/processed/` and `data/dev/`, while generated ML artifacts live under `outputs/<mode>/`.
- The official modeling recipe is product-first and YAML-driven. Stage 4 uses `customer_embeddings.weight_strategy: quantity_idf`, `quantity_transform: log1p`, product-purchase recency decay, product-specific basket-count frequency scaling, and normalized customer vectors.
- Official customer vectors do not use `importe`, total spend, average basket value, revenue tier, or demographics. Spend and KPIs are interpretation context only after clustering.
- Official model selection uses the `embeddings_only` feature set. Behavioral and product-exposure features can be built for diagnostics, challenger evidence, and profiling, but they are not the default clustering signal.
- Stage 6 is now a hard three-stage UMAP-HDBSCAN flow: PCA pre-reduction, UMAP representation, first HDBSCAN pass, stricter second pass over first-pass noise, third pass over remaining noise, three-pass merge, product-lift filtering, density evidence, readiness checks, remaining-customer evidence, and Stage 6.8 evidence assembly.
- Stage 6.6 readiness uses jitter recovery as the stability gate: `strong` requires no blockers and jitter recovery >= 0.80; `usable` has no blockers but is below the strong target; `review` is used for blockers such as jitter recovery < 0.60 or assignment-confidence issues.
- HDBSCAN noise stays honest as `tribe_id = -1`. Stage 6.7 can inspect remaining noise and optionally run candidate-only HDBSCAN after visual review, but it does not alter the official assignment.
- Stage 7 is a read-only communication layer. It consumes the Stage 6.8 evidence bundle and writes the final story, final index, manifest, tribe cards, product summaries, comparison tables, customer-metric context, and aggregate-only LLM evidence.
- UMAP is a clustering representation aid, not automatic proof. The 10-15 tribe range is a client hypothesis, not a hard clustering constraint.
- Cache metadata is centralized in `outputs/<mode>/.artifact_metadata.json`. Existing artifacts are reused when caching is enabled and `force=False`; metadata status is diagnostic, so use `force=True`, disable cache, or delete targeted generated files when rebuilding after logic/config changes.
- Dev-mode experiment sandboxes live in `notebooks/04_experiment_sandbox.ipynb`; alternate vector recipes and broader sweeps belong there before any setting is promoted into YAML.

## Quick Start

### 1. Create the Environment

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

### 2. Add the Raw CSV Files

Raw data is not committed. Place the source files here:

```text
data/raw/csv/ie_maestra_articulos.csv
data/raw/csv/ie_linea_ticket.csv
```

### 3. Convert Raw CSVs to Parquet

Run once from Python or from a notebook:

```python
from src.data_loader import verify_csv_checksums, convert_csv_to_parquet

verify_csv_checksums()
convert_csv_to_parquet()
```

Use `verify_csv_checksums(record=True)` only on the machine that establishes the canonical raw files.

### 4. Run the Notebooks in Order

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

Promote only evidence-backed settings into YAML. Then run the official ML pipeline:

```powershell
$env:CARREFOUR_MODE = "dev"
jupyter notebook notebooks/03_ml_pipeline.ipynb
```

Use `CARREFOUR_MODE=prod` for the full production run. Prod mode is the default when `CARREFOUR_MODE` is unset.

After pulling the current repo or changing vectorization/modeling code, rerun from the affected upstream stage before interpreting Stage 6+ or Stage 7 outputs. The current Stage 4 `quantity_idf` recipe requires rebuilding Stage 4 onward when changed.

## Colleague Handoff Checklist

Before handing the repo to another teammate:

```powershell
python -c "import sklearn, hdbscan; print(sklearn.__version__); print('hdbscan ok')"
pytest
git status --short
```

The source handoff must include configs, notebooks, docs, tests, `environment.yml`, and all `src/*.py` modules used by the official flow. Generated data/model/output artifacts stay local and should not be committed.

For experimentation:

1. Work in `CARREFOUR_MODE=dev`.
2. Use `notebooks/04_experiment_sandbox.ipynb`.
3. Inspect each experiment's compact `*_summary.csv` or `*_summary.md`.
4. Promote only evidence-backed settings into YAML.
5. Rerun `notebooks/03_ml_pipeline.ipynb` cleanly from the affected upstream stage.

## Repository Layout

```text
configs/          Hyperparameters and dev/prod overrides
data/             Local raw, processed, and dev data artifacts; never committed except docs/.gitkeep
docs/             Project context and methodological notes
notebooks/        Ordered analysis, official pipeline, and sandbox notebooks
outputs/          Mode-scoped generated artifacts; never committed
src/              Reusable pipeline modules
tests/            Unit tests for config, caching, embeddings, model selection, profiling, exports, and visuals
```

Generated ML artifacts stay under `outputs/<mode>/`. Dev includes an experiment workbench; prod does not.

```text
outputs/<mode>/
  .artifact_metadata.json  Central cache metadata manifest
  artifacts/    Stage diagnostics and handoff support files
    stage1/
    stage2/
    stage3/
    stage4/
    stage5/
    stage6/
      stage6_8_evidence/
    stage7/
      final_handoff/
  embeddings/   Basket sentences and product embedding tables
  features/     Customer vectors, feature sets, PCA/UMAP representations
  figures/      Notebook and presentation figures
    tribe_lifts/
    stage7_tribe_cards/
  models/       Trained local model binaries and assignment/result caches
    model_selection/
  profiles/     Legacy/profile parquet outputs retained for compatibility
  reports/      Compact notebook-facing stage reports
  experiments/  Optional sandbox runs in dev only
```

## Operating Conventions

- Keep reusable logic in `src/`; notebooks should orchestrate and explain, not duplicate pipeline code.
- Put shared hyperparameters in `configs/base.yaml`, dev overrides in `configs/dev.yaml`, and production-scale overrides in `configs/prod.yaml`; expose new values through `src/config.py`.
- Use `CARREFOUR_MODE=dev` for fast iteration and `CARREFOUR_MODE=prod` for full artifacts.
- Use `notebooks/04_experiment_sandbox.ipynb` for hyperparameter experiments and `notebooks/03_ml_pipeline.ipynb` for the official YAML-driven run.
- Do not commit raw data, Parquet caches, trained models, output images, `.env`, or secrets.
- Keep sandbox outputs in `outputs/dev/experiments/<experiment_name>/`; promote only the selected settings back into YAML.
- Experiments are disabled in prod. Production should only run the official pipeline with the scale-aware settings in `configs/prod.yaml`.
- Official customer vectorization must not use `importe` or other spend fields. The current official recipe is `quantity_idf` with `log1p(unidades)`, recency decay, product basket-frequency weighting, and vector normalization.
- Official clustering should remain on `embeddings_only`. Behavior/spend-derived columns are allowed for profiling and business interpretation after clustering.
- Stage 6 working files such as assignments and per-family result caches live under `outputs/<mode>/models/model_selection/`; diagnostics live under `outputs/<mode>/artifacts/stage6/`.
- Stage 6.8 is the raw-evidence boundary. After it runs, Stage 7 should read the saved evidence bundle and per-tribe exports rather than reopening global transactions or assignments.
- Stage 7 accepts Stage 6.6 readiness as final and does not apply extra promotion gates.
- Always join customer-level data with `join(on="cliente")`; do not rely on positional row order.

## Documentation

- [AGENTS.md](AGENTS.md) is the agent/operator guide for this repo.
- [data/README.md](data/README.md) documents local data expectations.
- [docs/Carrefour_Data_Challenge_Project_Context.md](docs/Carrefour_Data_Challenge_Project_Context.md) preserves the original project brief and methodological constraints.
