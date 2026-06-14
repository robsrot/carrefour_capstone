# Project File Inventory

Last updated: 2026-06-13

This inventory lists the current source-controlled files that matter for operating and maintaining the Carrefour segmentation pipeline. Generated artifacts live outside the committed repo state.

## Root

| File | Purpose |
|---|---|
| `README.md` | Human-facing project overview, run order, artifact layout, and operating conventions. |
| `AGENTS.md` | Agent/operator guidance for this repository. |
| `environment.yml` | Conda environment specification. |
| `pytest.ini` | Pytest configuration. |

## Configuration

| File | Purpose |
|---|---|
| `configs/base.yaml` | Shared pipeline defaults and official modeling hyperparameters. |
| `configs/dev.yaml` | Dev-mode overrides for faster local experimentation. |
| `configs/prod.yaml` | Production-scale overrides; same modeling recipe as dev, scaled to larger data. |

Official customer embeddings use `customer_embeddings.weight_strategy: quantity`. IDF variants belong in the experiment sandbox until promoted deliberately.

## Source Modules

| Module | Purpose |
|---|---|
| `src/config.py` | Loads YAML configuration, mode-specific paths, and exported constants. |
| `src/data_loader.py` | Raw CSV checksum verification and CSV-to-Parquet conversion. |
| `src/data_quality.py` | Production quality report. |
| `src/generate_dev_subset.py` | Builds the stratified dev subset from production processed data. |
| `src/basket_builder.py` | Lower-level basket construction helpers used by embedding stages. |
| `src/item2vec.py` | Item2Vec training and product embedding export. |
| `src/embeddings.py` | Basket sentence construction and product embedding validation helpers. |
| `src/embedding_validation.py` | Nearest-neighbor product embedding validation reports. |
| `src/customer_embeddings.py` | Quantity-only official customer vector aggregation plus optional IDF strategies for sandbox use. |
| `src/customer_vectors.py` | Legacy/customer-vector compatibility helpers retained for older workflows. |
| `src/feature_engineering.py` | Behavioral/profile feature engineering; spend is used for diagnostics/profiling, not official vector weighting. |
| `src/dimensionality.py` | UMAP and PCA feature representations. |
| `src/clustering.py` | GMM, HDBSCAN, PCA-KMeans, evaluation, and profile support. |
| `src/evaluation.py` | Shared cluster metric helpers. |
| `src/model_selection.py` | Official Stage 6 candidate suite and focused UMAP-HDBSCAN experiments. |
| `src/cluster_validation.py` | Stage 7 cluster validity and perturbation-stability report. |
| `src/experiment_reporting.py` | Compact experiment summaries for sandbox outputs. |
| `src/experiment_sandbox.py` | Dev-only experiment helpers for embeddings, vectors, feature sets, and focused clustering. |
| `src/profiling.py` | Product, sector, KPI, and lift-based tribe profiles. |
| `src/product_filtering.py` | Product filtering utilities used before embedding/modeling where needed. |
| `src/product_themes.py` | Product theme and interpretation helpers. |
| `src/tribe_namer.py` | Optional LLM-based tribe naming from lift profiles. |
| `src/autoencoder.py` | Experimental autoencoder utilities; not part of the official default pipeline. |
| `src/exports.py` | Final assignment/profile/report exports. |
| `src/visualization.py` | Figure generation for pipeline diagnostics and final explanation. |
| `src/progress.py` | Lightweight progress logging. |
| `src/utils.py` | Shared IO, metadata, sampling, and numeric helper utilities. |

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_exploration.ipynb` | Initial raw-data exploration. |
| `notebooks/02_pre-analysis.ipynb` | Production preprocessing and dev subset generation. |
| `notebooks/03_ml_pipeline.ipynb` | Official YAML-driven ML pipeline. |
| `notebooks/04_experiment_sandbox.ipynb` | Dev-only hyperparameter and method experiments. |

## Documentation

| File | Purpose |
|---|---|
| `docs/PROJECT_STATUS_AND_ROADMAP.md` | Current modeling contract, pipeline status, and next steps. |
| `docs/notebook_contract.md` | Notebook operating contract and stage outputs. |
| `docs/CROSS_MACHINE_REPRODUCIBILITY.md` | Reproducibility guardrails. |
| `docs/Carrefour_Data_Challenge_Project_Context.md` | Original project context. |
| `docs/team_sprint_plan.md` | Historical sprint-planning archive; not the current source of truth. |

## Generated Artifacts

Generated files are local and should not be committed.

```text
outputs/<mode>/
  .artifact_metadata.json
  embeddings/
  features/
  figures/
  models/
  profiles/
  reports/
  experiments/   # dev only
```

`data/raw/`, `data/processed/`, and `data/dev/` hold local raw/prepared data. The official ML pipeline writes generated ML artifacts to `outputs/<mode>/`.
