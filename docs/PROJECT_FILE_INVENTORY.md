# Project File Inventory — Branch `perf_normalization_nb_sebas`

> **This is the reference version of the pipeline.** All teammates should treat the files listed here as the canonical set for the capstone deliverable.
>
> Last updated: 2026-06-09 | Author: sebog99

---

## Claude Code Setup Instructions

This section is written for Claude Code (and any AI coding agent) joining this project cold. Read this before touching any file.

### What this project is

A product-first customer segmentation pipeline for Carrefour supermarket checkout data (~191M ticket rows, ~1.48M customers). The goal is to discover commercially interpretable customer "tribes" based entirely on **what products they buy**, not demographics. The final deliverable is a set of named tribes with product-level explanations, KPI profiles, and business narratives.

### Environment setup

```powershell
conda env create -f environment.yml
conda activate carrefour
python -m ipykernel install --user --name=carrefour --display-name "Python (carrefour)"
```

### The two modes — always set this before running anything

| Mode | Env var | Data reads from | Artifacts write to |
|---|---|---|---|
| Dev (44k customers) | `$env:CARREFOUR_MODE = "dev"` | `data/dev/` | `data/dev/`, `models/dev/`, `outputs/dev/` |
| Prod (1.48M customers) | `$env:CARREFOUR_MODE = "prod"` | `data/processed/` | `data/processed/`, `models/prod/`, `outputs/prod/` |

Prod is the default when the variable is unset. **Always set it explicitly.**

### How to run the pipeline

```powershell
# Dev (fast iteration, 44k customers)
$env:CARREFOUR_MODE = "dev"
jupyter notebook notebooks/03_ml_pipeline.ipynb

# Prod (full population, slow)
$env:CARREFOUR_MODE = "prod"
jupyter notebook notebooks/03_ml_pipeline.ipynb
```

### Pipeline stage order and cache chain

Each stage reads the previous stage's Parquet output. If you change upstream logic, delete or force-rebuild every downstream cache.

```
df_combined.parquet
  └─ basket_sentences.parquet
       └─ word2vec_product.model
            └─ product_embeddings.parquet
                 └─ customer_product_weights.parquet
                      ├─ customer_vectors_weighted.parquet
                      ├─ customer_vectors_mean.parquet
                      ├─ customer_vectors_tfidf_svd.parquet
                      ├─ customer_vectors_hybrid_product_only.parquet
                      └─ customer_store_features.parquet
                           ├─ umap_cluster_*.parquet
                           └─ pca_cluster_*.parquet
                                └─ cluster_labels_*.parquet
                                     └─ tribe_profiles_*.parquet
```

### Config rules — never hardcode values

- All hyperparameters live in `configs/base.yaml` (prod defaults) and `configs/dev.yaml` (dev overrides).
- `src/config.py` is only a loader — it exports named constants, it does not store values.
- When adding a hyperparameter: add to YAML first, export from `src/config.py`, then import the constant in the module.

### Current model selection state (dev)

The pipeline ran in `auto` selection mode and chose:

| Component | Winner |
|---|---|
| Vector source | `item2vec` (beat `tfidf_svd` and `hybrid`) |
| Reducer | `umap` |
| Clusterer | `hdbscan_assigned` |

These selections are persisted in `models/dev/*.json` and `outputs/dev/model_selection/*.json`.

### Reproducibility guardrails — do not remove

| Risk | Fix in place |
|---|---|
| Word2Vec thread race | `W2V_WORKERS=1` in config |
| Python hash randomization | `_stable_hash()` in `src/embeddings.py` |
| UMAP nondeterminism | `n_jobs=1`, fixed `RANDOM_SEED=42` |
| HDBSCAN parallel order | `n_jobs=1` |
| Polars group order | explicit `.sort()` before every cached write |
| Sampling | `np.random.default_rng(RANDOM_SEED)` everywhere |

### What NOT to commit

Raw CSVs, Parquet artifacts, trained models (`.model`, `.pkl`), plots, `.env`, secrets. See `.gitignore`.

### Run tests

```bash
pytest
```

---

## Full File Inventory

### Root

| File | Purpose |
|---|---|
| `CLAUDE.md` | Primary instructions for AI coding agents — read this first |
| `README.md` | Human-facing project overview |
| `environment.yml` | Conda environment definition (Python + all dependencies) |
| `pytest.ini` | Pytest configuration; sets test discovery paths |

---

### `configs/`

All hyperparameters. Edit here, never in source modules.

| File | Purpose |
|---|---|
| `configs/base.yaml` | Production defaults for every pipeline stage |
| `configs/dev.yaml` | Dev-mode overrides (faster UMAP, smaller HDBSCAN thresholds, fewer W2V epochs) |

---

### `src/` — Pipeline source modules

| File | Phase | Purpose |
|---|---|---|
| `src/__init__.py` | — | Package init |
| `src/config.py` | — | Loads YAML configs, exports all pipeline constants, handles `CARREFOUR_MODE` path switching |
| `src/data_loader.py` | Phase 0 | CSV checksum verification, CSV→Parquet conversion |
| `src/data_quality.py` | Phase 0 | Builds `quality_report.json` (8 checks: schema, nulls, anomalies, product coverage, temporal, promo, store) |
| `src/generate_dev_subset.py` | Phase 0 | Samples 44k customers from prod Parquet into `data/dev/df_combined.parquet` |
| `src/embeddings.py` | Phase 1 | Builds basket sentences, trains Word2Vec/Item2Vec, saves `product_embeddings.parquet` |
| `src/product_filtering.py` | Phase 1 | Hard-removes ultra-common products (>15% basket share or >40% customer share); computes IDF weights for soft downweighting |
| `src/customer_vectors.py` | Phase 2 | Aggregates product embeddings into per-customer vectors: recency-weighted, mean, TF-IDF/SVD, and hybrid variants |
| `src/dimensionality.py` | Phase 3 | UMAP (primary) and PCA (baseline) dimensionality reduction; fit-on-sample + transform-all strategy for scale |
| `src/clustering.py` | Phase 4 | HDBSCAN (density-based discovery) and MiniBatchKMeans (fixed-K baseline); `profile_tribes()` ranks top products by lift |
| `src/model_selection.py` | Phase 4 | Config-driven auto-selection of best vector source, reducer, and clusterer from benchmark results |
| `src/product_themes.py` | Phase 4 | Strategic product-name regex patterns (baby, pet, organic, fitness, etc.) used as a profiling overlay on tribe outputs |

---

### `notebooks/`

Run in numeric order. All use `src/` modules; notebooks are orchestration, not logic.

| File | Purpose |
|---|---|
| `notebooks/01_exploration.ipynb` | Initial data exploration: schema inspection, distributions, promo split, store coverage |
| `notebooks/02_pre-analysis.ipynb` | Preprocessing pipeline: builds `df_combined.parquet` and `customer_kpis.parquet` from raw Parquet |
| `notebooks/03_ml_pipeline.ipynb` | Full ML pipeline: Phases 1–4 (embeddings → vectors → UMAP/PCA → HDBSCAN/KMeans → tribe profiles). This is the main execution notebook. |

---

### `tests/`

| File | Purpose |
|---|---|
| `tests/__init__.py` | Package init |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/test_data_loader.py` | Tests CSV checksum verification and Parquet conversion |
| `tests/test_data_quality.py` | Tests the 8 quality checks in `src/data_quality.py` |
| `tests/test_reproducibility.py` | Verifies that pipeline outputs are bit-exact across re-runs |

---

### `docs/`

| File | Purpose |
|---|---|
| `docs/Carrefour_Data_Challenge_Project_Context.md` | Original challenge brief and business context from Carrefour |
| `docs/CROSS_MACHINE_REPRODUCIBILITY.md` | Documents the reproducibility problem (float non-determinism across machines) and the guardrails that fix it |
| `docs/notebook_contract.md` | Rules for how notebooks interact with `src/` modules (no logic in cells, cache invalidation protocol) |
| `docs/PROJECT_STATUS_AND_ROADMAP.md` | Current pipeline status, completed milestones, and next steps |
| `docs/PROJECT_FILE_INVENTORY.md` | **This file** — canonical file listing for branch `perf_normalization_nb_sebas` |

---

### `data/` — Local artifacts (not committed)

> These files exist locally but are excluded from git. Listed here for orientation.

#### `data/raw/`

| File | Description |
|---|---|
| `data/raw/csv/ie_linea_ticket.csv` | Raw ticket lines (~191M rows) — source of truth, never modify |
| `data/raw/csv/ie_maestra_articulos.csv` | Product master (descriptions, categories) |
| `data/raw/parquet/linea_tickets.parquet` | Converted ticket Parquet (from `src.data_loader`) |
| `data/raw/parquet/maestra_articulos.parquet` | Converted product master Parquet |

#### `data/processed/`

| File | Description |
|---|---|
| `data/processed/quality_report.json` | Output of `src.data_quality.build_quality_report()` — 8/8 checks pass |
| `data/processed/customer_kpis.parquet` | Per-customer KPIs: total spend, visit count, basket size, recency, promo rate |

#### `data/dev/` — Dev subset artifacts (44k customers)

| File | Pipeline stage | Description |
|---|---|---|
| `data/dev/df_combined.parquet` | Phase 0 | Dev subset: 44,000 customers, 7,490,843 rows |
| `data/dev/subset_metadata.json` | Phase 0 | Records which customers were sampled and the random seed used |
| `data/dev/basket_sentences.parquet` | Phase 1 | One row per ticket; `products` column is list of product IDs |
| `data/dev/product_popularity.parquet` | Phase 1 | Per-product basket share, customer share, and IDF weight |
| `data/dev/product_embeddings.parquet` | Phase 1 | 100-dim Word2Vec vectors for 55,974 products |
| `data/dev/customer_product_weights.parquet` | Phase 2 | Per-(customer, product) interaction weights before aggregation |
| `data/dev/customer_product_share_features.parquet` | Phase 2 | Per-customer category spend-share features |
| `data/dev/customer_vectors_weighted.parquet` | Phase 2 | Primary vectors: recency+frequency-weighted mean of product embeddings |
| `data/dev/customer_vectors_mean.parquet` | Phase 2 | Baseline vectors: simple mean (binary product ownership) |
| `data/dev/customer_vectors_tfidf_svd.parquet` | Phase 2 | TF-IDF basket matrix compressed via SVD |
| `data/dev/customer_vectors_hybrid_product_only.parquet` | Phase 2 | Hybrid: item2vec + TF-IDF SVD + category shares (no store features) |
| `data/dev/customer_store_features.parquet` | Phase 2 | Per-customer store spend-share features |
| `data/dev/umap_cluster_item2vec.parquet` | Phase 3 | 50-dim UMAP embedding of item2vec vectors (primary, full population) |
| `data/dev/umap_cluster_item2vec_bmark.parquet` | Phase 3 | 50-dim UMAP benchmark run for item2vec (sampled) |
| `data/dev/umap_cluster_tfidf_svd_bmark.parquet` | Phase 3 | 50-dim UMAP benchmark run for TF-IDF SVD |
| `data/dev/umap_cluster_hybrid_bmark.parquet` | Phase 3 | 50-dim UMAP benchmark run for hybrid |
| `data/dev/umap_viz_item2vec_umap_2d.parquet` | Phase 3 | 2-dim UMAP for visualization (full population) |
| `data/dev/umap_viz_item2vec_umap_preview_2d.parquet` | Phase 3 | 2-dim UMAP preview (sampled subset for fast plotting) |
| `data/dev/pca_cluster_item2vec.parquet` | Phase 3 | 50-dim PCA baseline embedding |
| `data/dev/pca_model_item2vec.pkl` | Phase 3 | Serialized sklearn PCA model for transform reuse |
| `data/dev/hdbscan_grid_item2vec_bmark.parquet` | Phase 4 | HDBSCAN grid search results over item2vec benchmark UMAP |
| `data/dev/hdbscan_grid_tfidf_svd_bmark.parquet` | Phase 4 | HDBSCAN grid search results over TF-IDF SVD benchmark UMAP |
| `data/dev/hdbscan_grid_hybrid_bmark.parquet` | Phase 4 | HDBSCAN grid search results over hybrid benchmark UMAP |
| `data/dev/vector_source_hdbscan_comparison.parquet` | Phase 4 | Side-by-side benchmark scores for all three vector sources |
| `data/dev/cluster_labels_hdbscan_item2vec_umap_mcs136_ms1_leaf.parquet` | Phase 4 | HDBSCAN labels: item2vec, best grid params (mcs=136, ms=1, leaf) |
| `data/dev/cluster_labels_hdbscan_assigned_item2vec_bmark.parquet` | Phase 4 | HDBSCAN assigned labels for item2vec benchmark |
| `data/dev/cluster_labels_hdbscan_assigned_tfidf_svd_bmark.parquet` | Phase 4 | HDBSCAN assigned labels for TF-IDF SVD benchmark |
| `data/dev/cluster_labels_hdbscan_assigned_hybrid_bmark.parquet` | Phase 4 | HDBSCAN assigned labels for hybrid benchmark |
| `data/dev/cluster_labels_hdbscan_item2vec_bmark.parquet` | Phase 4 | HDBSCAN core labels for item2vec benchmark |
| `data/dev/cluster_labels_hdbscan_tfidf_svd_bmark.parquet` | Phase 4 | HDBSCAN core labels for TF-IDF SVD benchmark |
| `data/dev/cluster_labels_hdbscan_hybrid_bmark.parquet` | Phase 4 | HDBSCAN core labels for hybrid benchmark |
| `data/dev/cluster_labels_kmeans_item2vec_umap_k8.parquet` | Phase 4 | KMeans labels K=8 on item2vec UMAP |
| `data/dev/cluster_labels_kmeans_item2vec_umap_k10.parquet` | Phase 4 | KMeans labels K=10 |
| `data/dev/cluster_labels_kmeans_item2vec_umap_k12.parquet` | Phase 4 | KMeans labels K=12 |
| `data/dev/cluster_labels_kmeans_item2vec_umap_k15.parquet` | Phase 4 | KMeans labels K=15 |
| `data/dev/cluster_labels_kmeans_item2vec_umap_k18.parquet` | Phase 4 | KMeans labels K=18 |
| `data/dev/kmeans_baseline_results_item2vec_umap.parquet` | Phase 4 | Silhouette and Davies-Bouldin scores for all KMeans K values |
| `data/dev/tribe_profiles_hdbscan_assigned.parquet` | Phase 4 | Top-product-by-lift profiles for the primary HDBSCAN segmentation |
| `data/dev/tribe_profiles_kmeans_k8.parquet` | Phase 4 | Tribe profiles for KMeans K=8 |
| `data/dev/tribe_profiles_kmeans_k10.parquet` | Phase 4 | Tribe profiles for KMeans K=10 |
| `data/dev/tribe_profiles_kmeans_k12.parquet` | Phase 4 | Tribe profiles for KMeans K=12 |
| `data/dev/tribe_profiles_kmeans_k15.parquet` | Phase 4 | Tribe profiles for KMeans K=15 |
| `data/dev/tribe_profiles_kmeans_k18.parquet` | Phase 4 | Tribe profiles for KMeans K=18 |

---

### `models/dev/` — Serialized model state (not committed)

| File | Description |
|---|---|
| `models/dev/word2vec_product.model` | Trained Word2Vec/Item2Vec model (55,974 product vocabulary, 100 dims) |
| `models/dev/vector_source_selection.json` | Auto-selection result: item2vec won over tfidf_svd and hybrid |
| `models/dev/reducer_selection.json` | Auto-selection result for dimensionality reducer |
| `models/dev/clusterer_selection.json` | Auto-selection result for clustering algorithm |

---

### `outputs/dev/` — Plots and model selection records (not committed)

| File | Description |
|---|---|
| `outputs/dev/embedding_basket_sizes.png` | Phase 1: basket size distribution used during embedding |
| `outputs/dev/phase2_vector_comparison.png` | Phase 2: weighted vs. mean vector quality comparison |
| `outputs/dev/phase2_store_affinity.png` | Phase 2: store spend-share heatmap across customers |
| `outputs/dev/phase3_pca_scree.png` | Phase 3: PCA explained variance scree plot |
| `outputs/dev/phase3_umap_preservation.png` | Phase 3: UMAP neighbor preservation quality diagnostic |
| `outputs/dev/phase3_umap2d_preview.png` | Phase 3: 2D UMAP scatter preview |
| `outputs/dev/phase4_clustering_comparison.png` | Phase 4: side-by-side HDBSCAN vs. KMeans cluster plots |
| `outputs/dev/model_selection/vector_source_selection.json` | Copy of vector source selection for outputs traceability |
| `outputs/dev/model_selection/reducer_selection.json` | Copy of reducer selection |
| `outputs/dev/model_selection/clusterer_selection.json` | Copy of clusterer selection |
