# Agent And Operator Guide

This repository builds product-first customer tribes from Carrefour checkout data. The modeling signal must come from products purchased and quantities purchased. Do not use demographic attributes.

## Current Modeling Contract

- Official customer embeddings use product identity plus `unidades` with `customer_embeddings.weight_strategy: quantity_idf`, the configured `log1p` quantity transform, product-purchase recency decay, product-specific basket-count frequency scaling, and normalized customer vectors.
- Product IDF is now part of the promoted YAML recipe. Other customer-vector variants remain sandbox experiments until deliberately promoted into YAML.
- Official customer embeddings do not use `importe`, total spend, average basket value, revenue tier, or demographic fields.
- Spend and KPIs are allowed for profiling and business interpretation after clustering.
- Official model selection uses `modeling.feature_set_for_selection: embeddings_only`.
- UMAP is a dimensionality-reduction aid, not an automatic winner.
- The 10-15 tribe range is a client hypothesis, not a clustering constraint.
- HDBSCAN core-discovery noise remains `tribe_id = -1`; do not silently soft-assign noise. The only allowed post-hoc assignment is the explicitly flagged Stage 6.5a centroid-rescue activation layer.

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

After changing vectorization, feature construction, UMAP, clustering, evidence assembly, or profiling logic, rebuild downstream stages. The current `quantity_idf` plus recency/frequency vectorization recipe requires rerunning from Stage 4 onward when changed.

## Official Stage 6, Stage 7, And Stage 8 Contract

- Stage 6.1 builds the PCA-pre-reduced UMAP representation from `embeddings_only`.
- Stage 6.2 runs the first hard HDBSCAN pass.
- Stage 6.3 runs a stricter hard HDBSCAN pass only on first-pass noise.
- Stage 6.4 runs a third hard HDBSCAN pass only on customers still left as noise.
- Stage 6.5 merges all three passes and applies the product-lift filter. This merged, lift-filtered assignment is the official assignment.
- Stage 6.5a builds a separate centroid-rescue activation layer from hard-core tribe centroids. It improves coverage for messaging and dashboard lookup, but it does not change the core-only profiling assignment.
- The Stage 6 quality-evidence cell writes representation and density checks before readiness.
- Stage 6.6 checks cluster stability, confidence, and profile readiness. `strong` requires no blockers and jitter recovery >= 0.80; `usable` has no blockers but is below the strong target; `review` is used for blockers such as jitter recovery < 0.60, missing recovery, low assignment confidence, or undersized clusters.
- Stage 6.7 analyzes customers still unassigned after Stage 6.5a. These remaining-customer groups are descriptive business segments, not hidden hard tribes.
- Stage 6.8 writes the core-only raw-data evidence bundle that Stage 7 consumes. The rescued assignment is available for activation coverage and customer lookup, not for product-lift profiling.
- Stage 7 is read-only: it accepts Stage 6.6 readiness carried through Stage 6.8 and writes the final handoff pack without reopening global transactions or applying new promotion gates. Stage 7.2 separates final promoted tribes from potential review tribes; Stage 7.3 onward presents all retained tribes with `final_strong`, `final_usable`, or `potential_review` status labels and may publish soft-audience opportunities without mutating official assignments.
- Stage 8 is the final dashboard semantic publishing layer. It packages structured Stage 6.8 and Stage 7 outputs under `outputs/<mode>/artifacts/stage8/`, with no Markdown input dependencies and with readiness/relational-integrity checks.

## Artifact Rules

- Source code, configs, notebooks, docs, tests, and environment files can be committed.
- Raw CSVs, Parquet files, trained models, output figures, `.env`, and secrets must not be committed.
- Official generated artifacts belong under `outputs/<mode>/`.
- Stage diagnostics and final handoff support live under `outputs/<mode>/artifacts/`.
- Stage 8 dashboard-ready relational artifacts live under `outputs/<mode>/artifacts/stage8/`.
- Dev sandbox artifacts belong under `outputs/dev/experiments/<experiment_name>/`.
- Cache metadata is centralized in `outputs/<mode>/.artifact_metadata.json`; do not add per-file `.meta.json` sidecars.

## Handoff Checks

Run these before handing the repo to another teammate:

```powershell
pytest
git status --short
python -c "import sklearn, hdbscan; print(sklearn.__version__); print('hdbscan ok')"
```

If `git status --short` shows untracked source files such as notebooks, configs, tests, docs, or `src/*.py`, include them in the handoff. Generated files can remain local.
