# Project File Inventory

Last updated: 2026-06-18

This inventory lists the current source and handoff-relevant files that matter for operating and maintaining the Carrefour segmentation pipeline. Generated artifacts live outside the committed repo state.

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
| `configs/dev.yaml` | Dev-mode overrides for faster local experimentation; enables sandbox/variant diagnostics. |
| `configs/prod.yaml` | Production-scale overrides; same modeling recipe as dev, scaled to larger data. |

Official customer embeddings use `customer_embeddings.weight_strategy: quantity_idf` with `customer_embeddings.quantity_transform: log1p`, configured recency weighting, configured basket-frequency weighting, and vector normalization. Alternative customer-vector recipes belong in the experiment sandbox until promoted deliberately.

## Source Modules

| Module | Purpose |
|---|---|
| `src/config.py` | Loads YAML configuration, mode-specific paths, and exported constants. |
| `src/data_loader.py` | Raw CSV checksum verification and CSV-to-Parquet conversion. |
| `src/data_quality.py` | Production quality report. |
| `src/generate_dev_subset.py` | Builds the stratified dev subset from production processed data. |
| `src/basket_builder.py` | Lower-level basket construction helpers used by embedding stages. |
| `src/cache_audit.py` | Stage 0/0.5 mode-path and cache-audit helpers, including central metadata adoption for existing artifacts. |
| `src/item2vec.py` | Item2Vec training, product embedding export, and training-corpus diagnostics. |
| `src/embeddings.py` | Basket sentence construction and product embedding helpers. |
| `src/embedding_validation.py` | Nearest-neighbor, common/rare, niche, and hubness product embedding validation reports. |
| `src/customer_embeddings.py` | Official `quantity_idf` customer vector aggregation, recency/frequency weighting, vector normalization, Stage 4 gates, and variant diagnostics. |
| `src/customer_vectors.py` | Legacy/customer-vector compatibility helpers retained for older workflows. |
| `src/feature_engineering.py` | Behavioral/profile feature engineering, product-exposure features, feature-set construction, and Stage 5 diagnostics. |
| `src/dimensionality.py` | UMAP and PCA feature representations. |
| `src/dimensionality_experiments.py` | Experimental dimensionality helpers. |
| `src/clustering.py` | GMM, HDBSCAN, PCA-KMeans, assignment, evaluation, and profile support. |
| `src/model_selection.py` | Official Stage 6 candidate suite, two-stage UMAP-HDBSCAN flow, remaining-noise probe, and focused UMAP-HDBSCAN experiments. |
| `src/cluster_validation.py` | Stage 6.6 cluster stability, perturbation, confidence, and profile-readiness report. |
| `src/profiling.py` | Stage 6.8 evidence assembly plus Stage 7 final story, final index, manifest, tribe cards, product summaries, comparison, noise/KPI context, and LLM-evidence tables. |
| `src/llm_analysis.py` | Optional post-Stage-7 Gemini interpretation from aggregate evidence only; does not affect clustering, filtering, or promotion. |
| `src/product_filtering.py` | Product filtering utilities used before embedding/modeling where needed. |
| `src/product_themes.py` | Curated product-theme taxonomy used for validation anchors, optional challenger features, naming context, and supporting profile context. |
| `src/tribe_namer.py` | Deterministic evidence-led tribe naming from lifted product terms, lifted SKUs, and sectors. |
| `src/organic_profiler.py` | Organic profile helper retained for profile/context support. |
| `src/autoencoder.py` | Experimental autoencoder utilities; not part of the official default pipeline. |
| `src/mission_microtribes.py` | Legacy optional shopping-mission helper; not part of the official notebook path. |
| `src/exports.py` | Legacy Stage 9/presentation export helpers retained for reference and tests; the official notebook now ends at Stage 7 final handoff. |
| `src/visualization.py` | Shared Carrefour-style visualization theme plus stage, diagnostic, profile, and experiment figure generation. |
| `src/stage_reports.py` | Compact notebook-facing stage report writer. |
| `src/experiment_reporting.py` | Compact experiment summaries for sandbox outputs. |
| `src/experiment_sandbox.py` | Dev-only experiment helpers for embeddings, vectors, feature sets, and focused clustering. |
| `src/evaluation.py` | Shared cluster metric helpers. |
| `src/progress.py` | Lightweight progress logging. |
| `src/utils.py` | Shared IO, central metadata, cache-status, sampling, schema, and numeric helper utilities. |
| `src/__init__.py` | Source package marker. |

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_exploration.ipynb` | Initial raw-data exploration. |
| `notebooks/02_pre-analysis.ipynb` | Production preprocessing and dev subset generation. |
| `notebooks/03_ml_pipeline.ipynb` | Official YAML-driven ML pipeline through Stage 7 final handoff. |
| `notebooks/04_experiment_sandbox.ipynb` | Dev-only hyperparameter and method experiments. |

## Documentation

| File | Purpose |
|---|---|
| `data/README.md` | Local data layout and rules. |
| `docs/PROJECT_STATUS_AND_ROADMAP.md` | Current modeling contract, pipeline status, and next steps. |
| `docs/notebook_contract.md` | Notebook operating contract and stage outputs. |
| `docs/CROSS_MACHINE_REPRODUCIBILITY.md` | Reproducibility and cache-sharing guardrails. |
| `docs/Carrefour_Data_Challenge_Project_Context.md` | Original project context plus current implementation note. |
| `docs/team_sprint_plan.md` | Historical sprint-planning archive and experiment discipline guide. |

## Tests

The test suite now covers active source behavior, not just package scaffolding.

| Test file | Coverage area |
|---|---|
| `tests/test_config_overrides.py` | Config authority, dev/prod override boundaries, and official Stage 6 config expectations. |
| `tests/test_cache_policy.py` | Central cache-status behavior for missing/existing artifacts, metadata mismatch, force, and disabled cache. |
| `tests/test_basket_staple_diagnostics.py` | Stage 1 common-product diagnostics and downsampled basket output behavior. |
| `tests/test_item2vec_corpus.py` | Item2Vec corpus filtering and training diagnostics. |
| `tests/test_embedding_validation.py` | Product embedding validation reports, hubness, and frequency flags. |
| `tests/test_customer_embeddings.py` | Quantity transforms, recency/frequency/IDF weighting, gates, diagnostics, and variant comparisons. |
| `tests/test_feature_set_diagnostics.py` | Stage 5 health/topology diagnostics and official-selection warnings. |
| `tests/test_product_exposure_features.py` | Product-exposure features and frequency-anchor feature sets. |
| `tests/test_product_themes.py` | Product-theme taxonomy edge cases. |
| `tests/test_dimensionality_pca.py` | PCA representation and dimensionality summary output. |
| `tests/test_model_selection_official_umap.py` | Official PCA pre-reduction before UMAP. |
| `tests/test_stage6_subsection_diagnostics.py` | Stage 6.1-6.7 diagnostics, hard-noise policy, remaining-noise probe, and evidence checks. |
| `tests/test_two_stage_hdbscan_lift_filter.py` | Two-stage HDBSCAN merge and product-lift filter. |
| `tests/test_cluster_validation.py` | Stage 6 readiness threshold behavior. |
| `tests/test_profiling_readiness.py` | Stage 6.8 evidence, Stage 7 final handoff, profile readiness, noise/KPI context, tribe cards, and LLM evidence. |
| `tests/test_tribe_namer.py` | Evidence-led tribe naming guardrails. |
| `tests/test_llm_analysis.py` | Optional Gemini interpretation behavior. |
| `tests/test_stage_reports.py` | Compact notebook stage report writer. |
| `tests/test_stage9_exports.py` | Legacy Stage 9 export helper compatibility. |
| `tests/__init__.py` | Test package marker. |

## Generated Artifacts

Generated files are local and should not be committed.

```text
outputs/<mode>/
  .artifact_metadata.json
  artifacts/
    model_selection/
    stage1/
    stage2/
    stage3/
    stage4/
    stage5/
    stage6/
      stage6_8_evidence/
    stage7/
      final_handoff/
  embeddings/
  features/
  figures/
    tribe_lifts/
    stage7_tribe_cards/
  models/
    model_selection/
  profiles/
  reports/
  experiments/   # dev only
```

`data/raw/`, `data/processed/`, and `data/dev/` hold local raw/prepared data. The official ML pipeline writes generated ML artifacts to `outputs/<mode>/`.
