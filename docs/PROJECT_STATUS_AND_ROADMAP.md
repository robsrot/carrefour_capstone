# Project Status and Roadmap

Last updated: 2026-06-15

## Modeling Contract

This is a product-first customer segmentation pipeline. Clusters should be driven by products purchased and quantities purchased, not demographics and not spend.

Official customer vectorization uses:

- Product identity through Item2Vec product embeddings.
- Product quantity through `unidades` using the configured `log1p` transform.
- Product-purchase recency decay and product-specific basket-count frequency scaling as weights.

Official customer vectorization does not use:

- `importe`
- total spend
- average basket value
- revenue tier
- demographic fields

Spend and business KPIs remain important, but only after clustering: they are used for profiling, interpretation, commercial sizing, and client storytelling.

## Current Pipeline

The clean ML workflow is:

0. Validate mode, paths, cache state, and prepared-data readiness.
1. Build basket sentences from prepared transactions.
2. Train or load Item2Vec product embeddings.
3. Validate product embeddings with nearest-neighbor diagnostics.
4. Aggregate customer embeddings using quantity-weighted product vectors.
5. Build model-ready feature sets.
6. Compare official clustering candidates without forcing a target number of clusters.
6.4. Produce cluster validity and perturbation-stability diagnostics.
7. Deeply profile and interpret tribes by product, sector, theme, term, KPI, noise-population audit, subsegment, readiness, and clustering-atlas evidence. This is the official final handoff stage.

UMAP is a dimensionality-reduction aid. It can improve density clustering, but it must earn its place through metrics, stability, interpretability, and clean product-lift profiles.

## Handoff Status

This repository is ready for colleague experimentation after the source changes are handed over, but it is not yet a final submission package.

Current handoff notes:

- Reconcile the environment before running UMAP-HDBSCAN. The project expects `scikit-learn=1.7.2` with `hdbscan==0.8.40`.
- Local prod embeddings, feature sets, and model-selection caches exist. Prod Stage 7 profile artifacts may still be absent or in progress; if `outputs/prod/profiles/` is empty, the selected tribe profile has not finished building.
- Prod Stage 6 is configured for 1.4M-customer scale by fitting UMAP/HDBSCAN on a deterministic 300k customer sample, transforming/assigning the full population, and keeping diagnostics sampled rather than loading full feature matrices.
- Include local source modules used by the official flow in any handoff. `src/cache_audit.py` supports Stage 0, and `src/profiling.py` now owns the final Stage 7 evidence storyline, noise-population audit, dossier, subsegment, atlas, and LLM-prompt artifacts.
- Use dev sandboxes for experimentation; do not change production YAML based on one unreviewed run.
- Commit or otherwise share all source files, configs, notebooks, and docs. Do not share raw data, Parquet caches, trained models, figures, or `.env` files through git.

Recommended verification commands:

```powershell
python -c "import sklearn, hdbscan; print(sklearn.__version__); print('hdbscan ok')"
pytest
git status --short
```

## Experiments

All broad experimentation belongs in `notebooks/04_experiment_sandbox.ipynb`, not in the official pipeline notebook.

Current sandbox areas:

- Item2Vec hyperparameters.
- Product embedding diagnostics.
- Customer vector aggregation, including capped and alternate weighting variants.
- Feature-set composition.
- Focused UMAP-HDBSCAN trials.

IDF downweighting remains an available sandbox variant, but the current shared Stage 4 YAML recipe is quantity plus recency/frequency weighting. Treat downstream cluster validity, stability, and product-lift interpretability as the evidence check for keeping it official.

## Configuration

Shared pipeline settings live in `configs/base.yaml`.

Dev overrides live in `configs/dev.yaml`.

Production-scale overrides live in `configs/prod.yaml`.

Prod mirrors the same modeling recipe as dev, but scales sample sizes and density thresholds for the larger population. Experiments are disabled in prod.

The active Python environment must match `environment.yml`. If `sklearn.__version__` reports `1.8.x`, recreate the environment or downgrade to `scikit-learn=1.7.2` before running HDBSCAN.

## Artifact Layout

Generated artifacts stay under `outputs/<mode>/`:

- `embeddings/`
- `features/`
- `figures/`
  - `figures/model_selection/`
  - `figures/presentation/`
  - `figures/tribe_lifts/`
- `models/`
  - `models/model_selection/`
- `profiles/`
- `reports/`
  - `reports/evidence/`
  - `reports/model_selection/`
  - `reports/presentation/`
- `experiments/` in dev only

The repo should not commit generated Parquet files, trained model binaries, plots, `.env`, or raw data.

## Next Steps

1. Reconcile the local environment to `environment.yml`.
2. Let prod Stage 7 complete if it is currently building profiles; expect a cold profile build to scan the large prepared transaction table.
3. Inspect Stage 6.1-6.4 and Stage 7 outputs together: metrics, noise, balance, stability/readiness, product/theme/term lift, and the noise-audit recommendation.
4. Review the Stage 7 evidence storyline first, then use the noise-population audit, dossier, all-tribe comparison, subsegment overlays, clustering atlas, and LLM interpretation prompt pack as supporting proof before presenting the selected solution.
5. Use the sandbox to test alternate customer-vector recipes and focused UMAP-HDBSCAN variants only if the official run shows weak product separation or unstable labels.
6. Promote only the strongest evidence-backed settings into YAML.

