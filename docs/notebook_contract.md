# Notebook Contract

Last updated: 2026-06-13

The notebooks should orchestrate the pipeline, show compact diagnostics, and explain decisions. Reusable logic belongs in `src/`.

## Mode Contract

`CARREFOUR_MODE` controls mode-specific paths. The Stage 0 cell in `notebooks/03_ml_pipeline.ipynb` is authoritative inside a notebook kernel: changing the mode there reloads `src.config` and should determine the rest of the run.

| Mode | Prepared data | Generated ML artifacts |
|---|---|---|
| `dev` | `data/dev/` | `outputs/dev/` |
| `prod` | `data/processed/` | `outputs/prod/` |

Prod is the default when `CARREFOUR_MODE` is unset. Experiments are enabled only in dev.

## Environment Contract

The active kernel must use the project environment from `environment.yml`.

Before running Stage 6 HDBSCAN, verify:

```powershell
python -c "import sklearn, hdbscan; print(sklearn.__version__); print('hdbscan ok')"
```

Expected pairing:

- `scikit-learn=1.7.2`
- `hdbscan==0.8.40`

If the kernel reports `scikit-learn 1.8.x`, reconcile the environment before interpreting UMAP-HDBSCAN results.

## Notebook 1: Exploration

`notebooks/01_exploration.ipynb` is exploratory. It may inspect raw CSV/Parquet files and dataset distributions, but it should not define reusable pipeline logic.

## Notebook 2: Pre-Analysis

`notebooks/02_pre-analysis.ipynb` prepares production data and triggers dev subset generation.

Required handoff artifacts:

- `data/processed/df_combined.parquet`
- `data/processed/customer_kpis.parquet`
- `data/processed/quality_report.json`
- `data/dev/df_combined.parquet`
- `data/dev/subset_metadata.json`

`src.generate_dev_subset` uses `data.min_tickets_per_customer` from YAML through `src.config.MIN_TICKETS_PER_CUSTOMER`.

## Notebook 3: Official ML Pipeline

`notebooks/03_ml_pipeline.ipynb` is the official run. It should use YAML settings and avoid hardcoded experiment grids.

Stages:

1. Mode setup and input validation.
2. Basket sentence generation.
3. Item2Vec product embedding training/export.
4. Product embedding validation.
5. Customer embedding aggregation.
6. Feature-set construction and dimensionality representations.
7. Official candidate model comparison.
8. Cluster validity and perturbation-stability diagnostics.
9. Tribe profiling and final exports.

Official customer embeddings are quantity weighted:

- Uses `cliente`, `idarticu`, `unidades`, and product embedding columns.
- Does not use `importe`.
- IDF-downweighted variants are sandbox-only unless promoted to YAML after evidence review.

Official clustering should default to the product/quantity feature set. Feature sets that include behavior or spend-derived profile columns are for ablation, diagnostics, or sandbox evidence unless explicitly promoted after review.

Official Stage 6 compares configured model families without forcing the client hypothesis of 10-15 tribes. The final selected number of tribes is judged after evaluation through metrics, stability, cluster balance, product/sector lift, and commercial interpretability.

UMAP is a dimensionality-reduction aid. It is useful when it improves clustering evidence; it is not automatically selected because it exists.

## Notebook 4: Experiment Sandbox

`notebooks/04_experiment_sandbox.ipynb` is dev-only and self-contained. Use it before changing official YAML settings.

Sandbox outputs go to:

```text
outputs/dev/experiments/<experiment_name>/
```

Each experiment folder should stay flat and limited:

- canonical detailed Parquet diagnostics
- compact `*_summary.csv`
- compact `*_summary.md`
- only the minimum trial artifacts needed to reproduce/inspect the comparison

Do not run sandboxes in prod.

## Cache Contract

Pipeline stages are cached with centralized metadata in `outputs/<mode>/.artifact_metadata.json`.

When changing upstream logic, rebuild all affected downstream artifacts with `force=True` or by deleting only the impacted generated files.

Dependency chain:

```text
df_combined
-> basket_sentences
-> word2vec_product.model
-> product_embeddings
-> customer_embeddings
-> feature_sets / UMAP / PCA
-> cluster assignments
-> validity/stability diagnostics
-> tribe profiles
-> exports and figures
```

The latest quantity-only vectorization change requires rerunning from Stage 4 onward before interpreting new clustering results.
