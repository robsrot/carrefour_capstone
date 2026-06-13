# Project Status and Roadmap

Last updated: 2026-06-13

## Modeling Contract

This is a product-first customer segmentation pipeline. Clusters should be driven by products purchased and quantities purchased, not demographics and not spend.

Official customer vectorization uses:

- Product identity through Item2Vec product embeddings.
- Product quantity through `unidades`.

Official customer vectorization does not use:

- `importe`
- total spend
- average basket value
- revenue tier
- demographic fields

Spend and business KPIs remain important, but only after clustering: they are used for profiling, interpretation, commercial sizing, and client storytelling.

## Current Pipeline

The clean ML workflow is:

1. Build basket sentences from prepared transactions.
2. Train or load Item2Vec product embeddings.
3. Validate product embeddings with nearest-neighbor diagnostics.
4. Aggregate customer embeddings using quantity-weighted product vectors.
5. Build model-ready feature sets.
6. Compare official clustering candidates without forcing a target number of clusters.
7. Produce cluster validity and perturbation-stability diagnostics.
8. Profile tribes by product, sector, KPI, promo, and business-readability metrics.
9. Export final assignments, profiles, reports, and figures.

UMAP is a dimensionality-reduction aid. It can improve density clustering, but it must earn its place through metrics, stability, interpretability, and clean product-lift profiles.

## Handoff Status

This repository is ready for colleague experimentation after the source changes are handed over, but it is not yet a final submission package.

Current handoff notes:

- Reconcile the environment before running UMAP-HDBSCAN. The project expects `scikit-learn=1.7.2` with `hdbscan==0.8.40`.
- Generated Stage 4+ artifacts should be treated as stale unless rebuilt after the latest quantity-only customer-vector change.
- Production artifacts are not complete after the latest pipeline updates.
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
- Customer vector aggregation, including IDF-downweighted variants.
- Feature-set composition.
- Focused UMAP-HDBSCAN trials.

IDF downweighting is intentionally a sandbox experiment for now. It may help reduce the dominance of products bought by many customers, but it should only be promoted to YAML if it improves downstream cluster validity, stability, and product-lift interpretability.

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
- `models/`
- `profiles/`
- `reports/`
- `experiments/` in dev only

The repo should not commit generated Parquet files, trained model binaries, plots, `.env`, or raw data.

## Next Steps

1. Reconcile the local environment to `environment.yml`.
2. Re-run dev from Stage 4 onward after the quantity-only vectorization change.
3. Inspect Stage 6 and Stage 7 outputs together: metrics, noise, balance, stability, and profile lift.
4. Use the sandbox to test IDF and focused UMAP-HDBSCAN variants only if the official dev run shows weak product separation or unstable labels.
5. Promote only the strongest evidence-backed settings into YAML.
6. Run the full production pipeline once the dev modeling contract is stable.
