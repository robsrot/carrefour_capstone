# Cross-Machine Reproducibility

Last updated: 2026-06-15

This project aims for deterministic source behavior and stable tribe structure across machines. Exact bit-for-bit equality is realistic for prepared-data checksums and many cached tables, but not guaranteed for UMAP/HDBSCAN boundary assignments across different CPUs.

## Environment Guardrail

Use the project Conda environment:

```powershell
conda env create -f environment.yml
conda activate carrefour
python -m ipykernel install --user --name=carrefour --display-name "Python (carrefour)"
```

Verify the HDBSCAN-compatible dependency set:

```powershell
python -c "import sklearn, hdbscan; print(sklearn.__version__); print('hdbscan ok')"
```

Expected pairing:

- `scikit-learn=1.7.2`
- `hdbscan==0.8.40`

If `sklearn.__version__` reports `1.8.x`, reconcile the environment before running or interpreting UMAP-HDBSCAN. The external `hdbscan` package version used here is not compatible with the sklearn 1.8 API change in sampled production HDBSCAN.

## Deterministic Inputs

The prepared-data stage should be reproducible when the same raw files and source code are used.

Key files to compare after Notebook 02 and dev-subset generation:

| File | Purpose |
|---|---|
| `data/processed/quality_report.json` | Full data quality gate output |
| `data/dev/subset_metadata.json` | Dev subset parameters and validation hashes |

Important hash fields in `subset_metadata.json`:

| Hash field | Meaning |
|---|---|
| `selected_customer_sha256` | Identifies the dev customers selected into `data/dev/df_combined.parquet` |
| `strata_assignment_sha256` | Identifies the stratification assignment used during sampling |

If these hashes differ across machines, do not compare downstream model results until the prepared data mismatch is resolved.

## Current Modeling Reproducibility Contract

Official customer vectors are quantity-only aggregations of product embeddings:

- Include product identity from Item2Vec.
- Include product quantities through `unidades`.
- Exclude `importe`, total spend, average basket value, and revenue tier.

Spend and KPIs are still used after clustering for profiling and business interpretation.

The latest quantity-only vectorization change invalidates generated Stage 4+ artifacts from older runs. Rebuild from Stage 4 onward before interpreting Stage 6+ comparisons.

## Known Sources of Variation

| Source | Guardrail | Residual risk |
|---|---|---|
| Word2Vec training | Fixed seed and deterministic worker settings | Small numeric drift if worker/thread settings change |
| Polars streaming aggregation | Stable sorting and integer-style accumulation where needed | Rebuild prepared data if source code changes |
| UMAP nearest-neighbor graph | Fixed seed and single-job settings | CPU-specific floating-point paths can shift boundary points |
| HDBSCAN density assignment | Fixed seed where applicable and deterministic input order | Noise and low-confidence boundary labels may vary slightly |
| Sampling | YAML seed and centralized config | Any changed sampling config invalidates downstream caches |

For this capstone, evaluate reproducibility at the level of cluster structure, stability diagnostics, and product-lift profiles, not only exact row-level labels.

## Artifact Sharing Between Machines

Use git for source-controlled files only:

- source code
- notebooks
- configs
- tests
- docs
- `environment.yml`

Share local data artifacts outside git:

```text
data/dev/df_combined.parquet
data/dev/subset_metadata.json
data/processed/customer_kpis.parquet
```

Generated ML artifacts under `outputs/<mode>/` should usually be rebuilt locally. If they are shared for speed, treat them as caches tied to the exact code, config, environment, and prepared data used to create them.

When sharing generated outputs outside git, keep the mode-scoped layout intact:

```text
outputs/<mode>/
  .artifact_metadata.json
  embeddings/
  features/
  figures/
  models/
  profiles/
  reports/
```

Stage 8 profile artifacts are especially sensitive to the exact selected assignment file, prepared transaction table, product-theme rules, product-term rules, and profiling thresholds. If any of those change, rebuild profiles and downstream Stage 9/10 reports.

## If Results Differ

1. Check `git status --short` and confirm teammates are using the same source revision.
2. Verify the environment, especially `scikit-learn` and `hdbscan`.
3. Compare `data/dev/subset_metadata.json` hashes.
4. Delete or force-rebuild generated Stage 4+ artifacts after vectorization or feature changes.
5. Compare product-lift profiles and stability diagnostics before concluding that a model is materially different.
