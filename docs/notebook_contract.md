# Notebook Contract

Last updated: 2026-06-18

The notebooks should orchestrate the pipeline, show compact diagnostics, and explain decisions. Reusable logic belongs in `src/`.

## Mode Contract

`CARREFOUR_MODE` controls mode-specific paths. The Stage 0 cell in `notebooks/03_ml_pipeline.ipynb` is authoritative inside a notebook kernel: changing the mode there reloads `src.config` and determines the rest of the run.

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

### Stage Overview

0. Mode setup, path validation, prepared-data overview, and stage report setup.
0.5. Expensive artifact cache audit for Stage 1-6.
1. Basket sentence generation and common-product downsampling diagnostics.
2. Item2Vec product embedding training/export and corpus diagnostics.
3. Product embedding validation with nearest-neighbor, common/rare, niche, and hubness checks.
4. Customer embedding aggregation using the official `quantity_idf` recipe.
5. Feature-set construction and feature-set diagnostics.
6. Hard two-stage UMAP-HDBSCAN core tribe discovery.
6.6. Cluster stability, confidence, and profile-readiness checks.
6.7. Remaining-noise structure probe.
6.8. Tribe evidence assembly.
7. Read-only tribe profiling and communication handoff.

### Stage 0 And 0.5

Stage 0 must validate active mode, paths, and prepared-data readiness. `src.cache_audit.assert_mode_path_audit` should catch accidental dev/prod namespace mixing.

Stage 0.5 audits expensive Stage 1-6 artifacts before they run. A cache hit requires the artifact to exist while caching is enabled and `force=False`; metadata status is reported for diagnostics but does not block reuse. If an existing artifact was produced from the current code/data/config but lacks central metadata, use `adopt_existing_stage_1_6_cache_metadata` deliberately.

### Stage 1: Basket Sentences

Stage 1 basket sentences use product IDs as tokens. Under current official settings, products are not repeated by `unidades`; quantities enter the official customer representation in Stage 4.

The current basket strategy is `common_downsampled`. When common-product downsampling is enabled, Stage 1 must persist:

- SKU-level keep/exclusion plan.
- Product ubiquity diagnostics.
- Basket common-product exposure diagnostics.
- Compact basket/token retention summary.

### Stage 2: Item2Vec

Stage 2 trains or loads Item2Vec product embeddings and exports `product_embeddings.parquet`. It must write a training-corpus diagnostic CSV showing how many baskets and product tokens are actually used after `min_tokens_per_basket`, `max_tokens_per_basket`, and context-window settings.

### Stage 3: Product Embedding Validation

Stage 3 must review more than a small random product sample. It includes common and rare products, staple and niche products, data-selected niche-theme checks from the product text/theme taxonomy, and hubness diagnostics.

The report must compare common-vs-rare nearest-neighbor quality and flag niche-focus products whose neighbors are mostly generic staples. Guardrail status is computed dynamically; when a problem appears, remediate through Stage 1/2 settings and rebuild downstream stages rather than using a manual acceptance toggle.

### Stage 4: Customer Embeddings

Official customer embeddings are product/quantity driven:

- Use `cliente`, `idarticu`, `unidades`, configured date/ticket columns, and product embedding columns.
- Use `customer_embeddings.weight_strategy: quantity_idf`.
- Apply the configured quantity transform, currently `log1p(unidades)`.
- Apply product-purchase recency decay.
- Apply customer-product basket-count frequency scaling.
- Apply product IDF weighting to downweight ubiquitous products.
- Normalize final customer vectors when `customer_embeddings.normalize_vectors: true`.
- Do not use `importe`, total spend, revenue tier, or demographics.

Stage 4 writes compact weight, coverage, and dominance diagnostics under `outputs/<mode>/artifacts/stage4/`. Detailed top-product exports are opt-in. Unique-product coverage is informational because rare long-tail SKUs are intentionally excluded by Item2Vec `min_count`; line, unit, and customer coverage are the hard gates.

Dev mode enables variant diagnostics so alternatives such as plain `quantity`, capped quantity, and other IDF variants can be compared without changing the official output. A variant is official only when it is promoted into YAML.

### Stage 5: Feature Sets

Official clustering uses the product/quantity `embeddings_only` feature set selected by `modeling.feature_set_for_selection`.

Behavioral, spend, frequency-anchor, and product-exposure features may be built as separate diagnostics, challenger evidence, or profiling inputs. They should not become official clustering inputs unless the modeling contract and YAML are deliberately updated.

Stage 5 writes compact diagnostics under `outputs/<mode>/artifacts/stage5/`: row/feature counts, finite/null rates, zero-variance columns, customer alignment, neighbor-overlap/topology checks, and selection-feature status.

### Stage 6: Hard Two-Stage UMAP-HDBSCAN

Stage 6 is the official hard core-tribe discovery flow. It uses product-derived `embeddings_only` features and keeps HDBSCAN noise unassigned.

Current official flow:

1. Stage 6.1 builds the UMAP clustering representation. The promoted recipe PCA-pre-reduces customer embeddings, then builds the UMAP feature table under `outputs/<mode>/features/`.
2. Stage 6.2 runs first-pass hard HDBSCAN on the UMAP representation.
3. Stage 6.3 runs a stricter second hard HDBSCAN pass only on customers labeled noise by Stage 6.2.
4. Stage 6.4 merges first-pass cores and second-pass noise cores, then applies a product-lift filter. Clusters without enough strong significant product-lift evidence are held out as noise/review rather than forced into the official tribe story.
5. Stage 6.5 writes representation and density-quality evidence for the merged official assignment.
6. Stage 6.6 writes cluster stability, confidence, and profile-readiness diagnostics. These readiness statuses are the promotion source for Stage 7.
7. Stage 6.7 builds a remaining-noise UMAP probe for visual review. Candidate-only HDBSCAN can be enabled after review, but any result is exploratory and does not change the official assignment.
8. Stage 6.8 reads the final assignment, prepared transactions, behavioral features, and Stage 6.6 readiness once, then writes the reusable evidence bundle for Stage 7.

UMAP is a dimensionality-reduction aid. It is useful when it improves clustering evidence; it is not automatically selected because it exists. The final selected number of tribes is judged after metrics, stability, cluster balance, product/sector lift, and commercial interpretability, without forcing the client hypothesis of 10-15 tribes.

For production scale, UMAP fits on the configured deterministic sample and transforms the full customer population in batches. HDBSCAN uses the configured deterministic sample/full-assignment path where applicable. Diagnostics sample large matrices when needed while retaining full assignment counts for coverage, noise, and readiness summaries.

### Stage 6.8 Evidence Boundary

Stage 6.8 writes the evidence bundle under:

```text
outputs/<mode>/artifacts/stage6/stage6_8_evidence/
```

It owns the expensive scans over prepared transactions, behavioral features, and assignments. The bundle includes the master `tribe_evidence_<mode>.parquet`, product and sector lift tables, customer-metric tests, noise-vs-core context, per-tribe transaction exports, per-tribe customer exports, and a manifest.

After Stage 6.8, Stage 7 should not reopen global transaction files, behavioral feature files, or assignment files.

### Stage 7: Read-Only Profiling And Communication

Stage 7 accepts Stage 6.6 readiness from the Stage 6.8 evidence parquet as final. It must not rerun clustering checks, recompute readiness, apply extra promotion gates, reopen global raw-data inputs, or modify assignments.

Stage 7 writes the final handoff pack under:

```text
outputs/<mode>/artifacts/stage7/final_handoff/
```

Key outputs:

- `stage7_final_story_<mode>.md` and `.html`
- `stage7_final_index_<mode>.csv`
- `stage7_final_manifest_<mode>.json`
- `support/stage7_final_evidence_storyline_<mode>.*`
- `support/stage7_final_tribe_comparison_<mode>.*`
- `support/stage7_final_product_summary_long_<mode>.csv`
- `support/stage7_llm_evidence_long_<mode>.csv`
- Tribe card PNGs under `outputs/<mode>/figures/stage7_tribe_cards/`

Spend and KPI metrics are allowed here as context and commercial interpretation. They must not be retroactively treated as clustering inputs.

## Notebook 4: Experiment Sandbox

`notebooks/04_experiment_sandbox.ipynb` is dev-only and self-contained. Use it before changing official YAML settings.

Sandbox outputs go to:

```text
outputs/dev/experiments/<experiment_name>/
```

Each experiment folder should stay focused and limited to:

- canonical detailed Parquet diagnostics
- compact `*_summary.csv`
- compact `*_summary.md`
- only the minimum trial artifacts needed to reproduce/inspect the comparison

Do not run sandboxes in prod.

## Cache Contract

Pipeline stages are cached with centralized metadata in `outputs/<mode>/.artifact_metadata.json`.

Current cache policy:

- `cache.use_cached: true` and `force=False` allow an existing artifact to be reused.
- Missing artifacts are rebuilt.
- Metadata matches/mismatches are diagnostic and shown in Stage 0.5; mismatch alone does not force rebuild.
- Use `force=True`, set `cache.use_cached: false`, or delete only the impacted generated files when changing upstream logic.

Dependency chain:

```text
df_combined
-> basket_sentences and Stage 1 diagnostics
-> word2vec_product.model
-> product_embeddings
-> product embedding validation
-> customer_embeddings
-> feature_sets / PCA / UMAP
-> two-stage hard HDBSCAN assignments
-> Stage 6.5 density evidence
-> Stage 6.6 stability/readiness
-> Stage 6.7 remaining-noise probe
-> Stage 6.8 evidence bundle
-> Stage 7 final handoff pack
```

The current `quantity_idf` plus recency/frequency vectorization recipe requires rerunning from Stage 4 onward before interpreting new clustering results when changed.
