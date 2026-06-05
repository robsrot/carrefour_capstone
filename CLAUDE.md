# CLAUDE.md

Guidance for Claude Code, Codex, and other coding agents working in this repository.

## Product Mandate

This project builds a product-first customer segmentation pipeline for Carrefour checkout data. The segmentation must be based on purchase behavior, especially the products customers buy and the product combinations they repeat. Do not use demographic attributes.

The final deliverable is not just a clustering table. It is a commercially interpretable set of customer tribes, each named and explained through distinctive products, product categories, basket missions, promo behavior, store affinity, and business KPIs.

## Current Snapshot

Status as of 2026-06-05:

| Area | Status |
|---|---|
| Raw conversion | CSV-to-Parquet path exists in `src.data_loader`. |
| Data quality | `data/processed/quality_report.json` exists and passes 8/8 checks. |
| Production preprocessing | `data/processed/df_combined.parquet` and `customer_kpis.parquet` exist. |
| Dev subset | `data/dev/df_combined.parquet` exists: 44,000 customers, 7,490,843 rows. |
| Dev ML pipeline | Completed through embeddings, customer vectors, UMAP/PCA, HDBSCAN, K-Means baselines, and plots. |
| Prod ML pipeline | Not yet run end-to-end after the current pipeline updates. |
| Tribe profiles | `profile_tribes()` exists, but cached tribe profile outputs are not present yet. |
| Tribe naming | Config stub exists, but no `src/tribe_namer.py` implementation exists yet. |

Important current metrics:

- Full raw ticket rows: 191,017,715.
- Full unique customers: 1,482,715.
- Customers eligible after `min_tickets_per_customer=3`: about 1.085M.
- Product coverage: 100% of ticket product IDs covered by the product master.
- Promo split: 77.64% no promo, 22.36% promo.
- Dev Word2Vec vocabulary: 55,974 products.
- Dev weighted customer vectors: 43,998 customers.
- Dev HDBSCAN configured result: 4 non-noise clusters, 1,843 noise customers, silhouette 0.6325, Davies-Bouldin 0.4591.
- Dev K-Means baselines: K=8/12/15/18 generated; K=8 has the best silhouette among those baselines at 0.4301.

Do not treat the HDBSCAN result as final simply because the silhouette score is higher. The next selection gate is product-level interpretability: top-product lift, segment distinctiveness, stability, and business actionability.

## Pipeline Architecture

1. Phase 0: Data quality and preprocessing
   - Convert raw CSVs to Parquet.
   - Validate schemas, nulls, anomalies, product coverage, temporal coverage, promo integrity, and store coverage.
   - Build `df_combined.parquet` and customer KPIs.

2. Phase 1: Product embeddings
   - Module: `src/embeddings.py`.
   - Treat each ticket as a sentence and each product ID as a word.
   - Train Word2Vec / Item2Vec.
   - Outputs: `basket_sentences.parquet`, `models/<mode>/word2vec_product.model`, `product_embeddings.parquet`.

3. Phase 2: Customer vectors
   - Module: `src/customer_vectors.py`.
   - Primary vector: recency- and frequency-weighted mean of product embeddings.
   - Baseline vector: simple mean.
   - Additional behavioral fields: `promo_rate` and store spend-share features.
   - Outputs: `customer_product_weights.parquet`, `customer_vectors_weighted.parquet`, `customer_vectors_mean.parquet`, `customer_store_features.parquet`.

4. Phase 3: Dimensionality reduction
   - Module: `src/dimensionality.py`.
   - Primary: UMAP.
   - Baseline: PCA.
   - Current UMAP input is 100 product-vector dims plus promo and store features when weights are enabled.
   - KPI feature weight is currently `0.0`, so KPIs are profiled after clustering rather than driving clusters.

5. Phase 4: Clustering and profiling
   - Module: `src/clustering.py`.
   - Primary: HDBSCAN on UMAP embedding.
   - Baselines: fixed-K MiniBatchKMeans.
   - Evaluation: silhouette, Davies-Bouldin, noise rate, and product-level tribe profiles.
   - Top products are ranked by lift, not raw frequency, inside `profile_tribes()`.

## Repository Map

```text
configs/       Dev/prod hyperparameters
data/          Local raw, processed, and dev artifacts; do not commit data
docs/          Current status, roadmap, and original project context
models/        Local Word2Vec models; do not commit models
notebooks/     Ordered notebooks for exploration, preprocessing, and ML pipeline
outputs/       Local plots; do not commit outputs
src/           Reusable pipeline code
tests/         Reproducibility, loader, and quality tests
```

## Setup and Operating Commands

Create the environment:

```bash
conda env create -f environment.yml
conda activate carrefour
python -m ipykernel install --user --name=carrefour --display-name "Python (carrefour)"
```

Convert raw data:

```python
from src.data_loader import verify_csv_checksums, convert_csv_to_parquet

verify_csv_checksums()
convert_csv_to_parquet()
```

Run the production quality report:

```python
from src.data_quality import build_quality_report

build_quality_report(force=False)
```

Generate the dev subset after production preprocessing exists:

```powershell
$env:CARREFOUR_MODE = "prod"
python -m src.generate_dev_subset
```

Run the ML notebook in dev mode:

```powershell
$env:CARREFOUR_MODE = "dev"
jupyter notebook notebooks/03_ml_pipeline.ipynb
```

Run the ML notebook in prod mode:

```powershell
$env:CARREFOUR_MODE = "prod"
jupyter notebook notebooks/03_ml_pipeline.ipynb
```

Run tests:

```bash
pytest
```

## Configuration Rules

All hyperparameters live in YAML:

- `configs/base.yaml`: production defaults.
- `configs/dev.yaml`: dev overrides.
- `src/config.py`: loader and exported constants only.

Do not hardcode config values inside notebooks or modules. When adding a hyperparameter:

1. Add it to `configs/base.yaml`.
2. Add a dev override only if needed in `configs/dev.yaml`.
3. Export it through `src/config.py`.
4. Import the exported constant in the relevant module.
5. Update the docs if the operating behavior changes.

## Modes and Artifact Paths

`CARREFOUR_MODE` controls where generated artifacts are read and written:

| Mode | `DATA_PROCESSED` | `MODELS` | `OUTPUTS` |
|---|---|---|---|
| `dev` | `data/dev/` | `models/dev/` | `outputs/dev/` |
| `prod` | `data/processed/` | `models/prod/` | `outputs/prod/` |

Prod is the default when `CARREFOUR_MODE` is unset.

## Naming Conventions

- Source modules are lowercase snake_case under `src/`.
- Notebook names stay numeric and ordered: `01_*`, `02_*`, `03_*`.
- Pipeline caches use descriptive snake_case names.
- Mode-specific outputs must go through `src.config.DATA_PROCESSED`, `MODELS`, and `OUTPUTS`.
- Do not write generated artifacts to hardcoded `data/dev` or `data/processed` paths inside pipeline code.

Known naming mismatch:

- `umap_cluster_20d.parquet` and `pca_cluster_20d.parquet` are historical filenames.
- The active config currently uses `cluster_dims: 50`, so these files contain 50 dimensions in current runs.
- If you rename these artifacts, migrate all references in `src/dimensionality.py`, `src/clustering.py`, notebooks, docs, and any cached output assumptions in one deliberate change.

## Cache Invalidation Rules

Pipeline stages are cached aggressively. When changing upstream logic, rebuild affected downstream caches with `force=True` or delete only the relevant generated artifacts.

Use this dependency chain:

```text
df_combined
-> basket_sentences
-> word2vec_product.model
-> product_embeddings
-> customer_product_weights
-> customer_vectors_weighted / customer_vectors_mean / customer_store_features
-> umap_cluster / pca_cluster
-> cluster labels
-> tribe profiles
-> visualizations and naming
```

Examples:

- Change Word2Vec settings: rebuild embeddings, customer vectors, dimensionality reduction, clusters, profiles, plots.
- Change recency half-life: rebuild interaction weights, customer vectors, dimensionality reduction, clusters, profiles, plots.
- Change feature weights for UMAP input: rebuild UMAP/PCA if relevant, clusters, profiles, plots.
- Change HDBSCAN or K-Means parameters: rebuild cluster labels, profiles, plots.

## Reproducibility Contract

The current repo is designed for deterministic dev/prod runs on identical data and code.

| Source of nondeterminism | Guardrail |
|---|---|
| Word2Vec thread race | `W2V_WORKERS=1` |
| Python randomized hash | custom stable hash in `src.embeddings._stable_hash` |
| UMAP nearest-neighbor graph | `n_jobs=1` and fixed `RANDOM_SEED` |
| HDBSCAN / kNN parallel order | `n_jobs=1` |
| Polars group order | explicit `.sort(...)` before cached outputs |
| Sampling | `np.random.default_rng(RANDOM_SEED)` |

Do not weaken these guardrails unless speed matters more than bit-exact reproducibility and the docs/tests are updated to say so.

## Memory and Performance Rules

The raw ticket data is large enough to punish casual full scans.

- Use Polars lazy scans for large Parquet reads.
- Use `.collect(engine="streaming")` when scanning `df_combined.parquet` or raw ticket Parquet.
- Prefer column selection before joins/grouping.
- Avoid `n_unique()` inside large `group_by` aggregations; use `approx_n_unique()` when exact cardinality is not required.
- Cache expensive per-customer and per-product results to Parquet.
- Never load raw `ie_linea_ticket.csv` directly once Parquet conversion exists.

## Product-First Segmentation Rules

The central question is: can we find meaningful customer segments from products bought?

Use these rules when interpreting or changing the pipeline:

- Product embeddings are the core signal.
- KPIs such as spend, visits, and basket size should usually be profiled after clustering, not used to create the clusters.
- Promo and store affinity are behavioral signals, but they can dominate topology. Run product-only ablations before selecting the final segmentation.
- Use product lift to name tribes. Raw top products mostly reveal universal staples.
- Do not choose a final model from silhouette alone. A lower-silhouette model with cleaner product stories can be better for the capstone.

The current next-step plan is documented in [docs/PROJECT_STATUS_AND_ROADMAP.md](docs/PROJECT_STATUS_AND_ROADMAP.md).

## Git and Commit Rules

- Main development branch: `dev`.
- Safe to commit: source code, notebooks, docs, configs, tests, environment files.
- Never commit: raw CSVs, Parquet artifacts, trained models, plots, `.env`, or secrets.
- Before pushing, check:

```bash
git status --short
pytest
```

If tests cannot run because local data is unavailable or the environment is incomplete, state that clearly in the handoff.
