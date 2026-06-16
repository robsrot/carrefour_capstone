# Agent And Operator Guide

This repository builds product-first customer tribes from Carrefour checkout data. The modeling signal must come from products purchased and quantities purchased. Do not use demographic attributes.

## Current Modeling Contract

- Official customer embeddings use product identity plus `unidades` with the configured `log1p` quantity transform, product-purchase recency decay, and product-specific basket-count frequency scaling.
- Official customer embeddings do not use `importe`, total spend, average basket value, or revenue tier.
- Spend and KPIs are allowed for profiling and business interpretation after clustering.
- UMAP is a dimensionality-reduction aid, not an automatic winner.
- The 10-15 tribe range is a client hypothesis, not a clustering constraint.
- Alternative customer-vector recipes remain sandbox experiments until deliberately promoted into YAML.

## Modes

| Mode | Prepared data | Generated artifacts | Experiments |
|---|---|---|---|
| `dev` | `data/dev/` | `outputs/dev/` | Enabled |
| `prod` | `data/processed/` | `outputs/prod/` | Disabled |

Set mode explicitly before notebooks:

```powershell
$env:CARREFOUR_MODE = "dev"
```

Inside `notebooks/03_ml_pipeline.ipynb`, the Stage 0 `RUN_MODE` cell is authoritative for the active kernel.

## Environment

Use the environment file as the source of truth:

```powershell
conda env create -f environment.yml
conda activate carrefour
python -m ipykernel install --user --name=carrefour --display-name "Python (carrefour)"
```

Verify before running HDBSCAN:

```powershell
python -c "import sklearn, hdbscan; print(sklearn.__version__); print('hdbscan ok')"
```

Expected sklearn version: `1.7.2`. `hdbscan==0.8.40` can fail with sklearn `1.8.x` in sampled production HDBSCAN.

## Run Order

1. `notebooks/01_exploration.ipynb`
2. `notebooks/02_pre-analysis.ipynb`
3. `python -m src.generate_dev_subset`
4. Optional dev experiments in `notebooks/04_experiment_sandbox.ipynb`
5. Official run in `notebooks/03_ml_pipeline.ipynb`

After changing vectorization, feature construction, UMAP, or clustering logic, rebuild downstream stages. The current quantity plus recency/frequency vectorization recipe requires rerunning from Stage 4 onward when changed.

## Artifact Rules

- Source code, configs, notebooks, docs, tests, and environment files can be committed.
- Raw CSVs, Parquet files, trained models, output figures, `.env`, and secrets must not be committed.
- Official generated artifacts belong under `outputs/<mode>/`.
- Dev sandbox artifacts belong under `outputs/dev/experiments/<experiment_name>/`.
- Keep experiment folders flat and limited to essential artifacts plus compact summaries.

## Handoff Checks

Run these before handing the repo to another teammate:

```powershell
pytest
git status --short
python -c "import sklearn, hdbscan; print(sklearn.__version__); print('hdbscan ok')"
```

If `git status --short` shows untracked source files such as notebooks, configs, or `src/*.py`, include them in the handoff. Generated files can remain local.
