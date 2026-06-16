# Notebook Contract

Last updated: 2026-06-16

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

`notebooks/03_ml_pipeline.ipynb` is the official run. It should use promoted YAML settings and avoid hardcoded experiment grids, side-by-side recipe trials, or sandbox-style model bake-offs.

Stages:

0. Mode setup, path validation, cache audit, and prepared-data overview.
1. Basket sentence generation.
   Stage 1 basket sentences use product ids as tokens. Under the current official settings, products are not repeated by `unidades`; quantities enter the official customer representation in Stage 4. When common-product downsampling is enabled, Stage 1 must persist the SKU-level keep/exclusion plan and a compact basket/token retention summary.
2. Item2Vec product embedding training/export.
   Stage 2 must write a training-corpus diagnostic CSV showing how many baskets and product tokens are actually used after `min_tokens_per_basket`, `max_tokens_per_basket`, and context-window settings.
3. Product embedding validation.
   Stage 3 must review more than a small random product sample: it includes common and rare products, staple and niche products, and data-selected niche-theme checks from the product text/theme taxonomy. The report must compare common-vs-rare nearest-neighbor quality and flag niche-focus products whose neighbors are mostly generic staples. Guardrail status is computed dynamically in the notebook; when it finds a problem, remediate through Stage 1/2 settings and rebuild downstream stages rather than using a manual acceptance toggle.
4. Customer embedding aggregation.
5. Feature-set construction and dimensionality representations.
6. Hard UMAP-HDBSCAN core tribe discovery.
6.4. Cluster validity and perturbation-stability diagnostics.
7. Deep tribe profiling and interpretation. This is the official final handoff stage.

Official customer embeddings are quantity weighted:

- Uses `cliente`, `idarticu`, `unidades`, and product embedding columns.
- Applies the configured quantity transform (`customer_embeddings.quantity_transform`, currently `log1p`) before aggregating product vectors.
- Applies configured product-purchase recency decay and product-specific basket-count frequency scaling as weights, not as separate behavioral feature columns.
- Writes only compact Stage 4 weight diagnostics by default; detailed top-product exports are opt-in, and review should happen mainly through notebook tables and figures.
- Treats unique-product coverage as informational because rare long-tail SKUs are intentionally excluded by Item2Vec `min_count`; line, unit, and customer coverage are the hard Stage 4 coverage gates.
- Does not use `importe`.
- Capped, equal-weight, or alternate IDF customer embedding variants belong in `notebooks/04_experiment_sandbox.ipynb` unless one is promoted into the official YAML recipe after evidence review.

Official clustering should use the product/quantity `embeddings_only` feature set. Feature sets that include behavior or spend-derived profile columns belong in sandbox evidence, not in the official pipeline notebook.

Stage 5 writes compact diagnostics for the official feature set under `outputs/<mode>/artifacts/stage5/` and displays the key row in-notebook: row/feature counts, finite/null rates, zero-variance columns, customer alignment, and selection-feature status. Behavioral features may still be built as separate profiling inputs for later interpretation, but they are not an alternate modeling feature set in Notebook 03.

Official Stage 6 runs the promoted UMAP-to-HDBSCAN recipe as hard core-tribe discovery. It is split into notebook subsections: Stage 6.1 builds and checks the UMAP representation, Stage 6.2 runs hard HDBSCAN and checks assignment quality/provenance, and Stage 6.3 writes compact model diagnostics only after the prior checks pass. It must not soft-assign HDBSCAN noise customers in the official discovery step; `tribe_id = -1` remains the honest non-core population. The final selected number of tribes is judged after evaluation through metrics, stability, cluster balance, product/sector lift, and commercial interpretability, without forcing the client hypothesis of 10-15 tribes.

UMAP is a dimensionality-reduction aid. It is useful when it improves clustering evidence; it is not automatically selected because it exists.

For production scale, Stage 6 is sample-fit/full-assign. UMAP fits on the configured deterministic sample and transforms the full customer population in batches. HDBSCAN then fits on the configured deterministic sample and assigns the full UMAP population through the supported HDBSCAN prediction path. Stage 6.3 and Stage 6.4 diagnostics use sampled feature matrices while retaining full assignment counts for coverage, noise, and readiness summaries.

Stage 7 can be slow in prod on a cache miss because it scans prepared transactions and computes product, sector, strategic-theme, product-term, KPI, noise-population audit, subsegment, clustering-atlas, evidence-storyline, and aggregate-only LLM prompt evidence for the selected assignment. In the current local workspace, `outputs/prod/profiles/` may be empty until that cold build completes.

The official assignment must keep HDBSCAN noise as `tribe_id = -1`; Stage 7 must still audit that population before handoff. The noise audit is aggregate-only and answers whether noise looks like weak signal, sparse/lapsed customers, broad generalists, rare-product behavior, or a candidate pool for separate second-pass investigation. It does not rename the official core tribes or silently soft-assign customers.

Official visual outputs belong under `outputs/<mode>/figures/`; presentation-ready figures go under `outputs/<mode>/figures/presentation/`, model-selection views under `outputs/<mode>/figures/model_selection/`, and per-tribe lift plots under `outputs/<mode>/figures/tribe_lifts/`.

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
-> evidence storyline
-> deep tribe profiles, noise-population audit, dossiers, subsegment overlays, clustering atlas, and LLM interpretation prompt pack
```

The current quantity plus recency/frequency vectorization recipe requires rerunning from Stage 4 onward before interpreting new clustering results when changed.

