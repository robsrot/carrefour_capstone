# Cross-Machine Reproducibility

Last updated: 2026-06-18

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

Official customer vectors are quantity-IDF weighted aggregations of product embeddings:

- Include product identity from Item2Vec.
- Include product quantities through the configured `unidades` transform, currently `log1p`.
- Apply configured product-purchase recency decay.
- Apply customer-product basket-count frequency scaling.
- Apply product IDF weighting through `customer_embeddings.weight_strategy: quantity_idf`.
- Normalize final customer vectors when `customer_embeddings.normalize_vectors: true`.
- Exclude `importe`, total spend, average basket value, revenue tier, and demographic fields.

Spend and KPIs are still used after clustering for profiling and business interpretation.

The current Stage 4 vectorization recipe invalidates generated Stage 4+ artifacts when changed. Rebuild from Stage 4 onward before interpreting Stage 6+ comparisons.

## Known Sources Of Variation

| Source | Guardrail | Residual risk |
|---|---|---|
| Word2Vec training | Fixed seed and deterministic worker settings where configured | Small numeric drift if worker/thread settings change |
| Product IDF and customer-product weighting | Deterministic Polars aggregations from prepared transactions | Changes to product filtering, prepared data, or Stage 4 config invalidate downstream vectors |
| Polars streaming aggregation | Stable sorting and integer-style accumulation where needed | Rebuild prepared data if source code changes |
| PCA pre-reduction | Fixed seed and deterministic inputs | Floating-point differences can slightly change downstream UMAP |
| UMAP nearest-neighbor graph | Fixed seed and single-job settings | CPU-specific floating-point paths can shift boundary points |
| HDBSCAN density assignment | Deterministic input order and hard-noise policy | Noise and low-confidence boundary labels may vary slightly |
| Sampling | YAML seed and centralized config | Any changed sampling config invalidates downstream caches |

For this capstone, evaluate reproducibility at the level of cluster structure, Stage 6.6 readiness, and product-lift profiles, not only exact row-level labels.

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
data/processed/df_combined.parquet
data/processed/quality_report.json
```

Generated ML artifacts under `outputs/<mode>/` should usually be rebuilt locally. If they are shared for speed, treat them as caches tied to the exact code, config, environment, and prepared data used to create them.

When sharing generated outputs outside git, keep the mode-scoped layout intact:

```text
outputs/<mode>/
  .artifact_metadata.json
  artifacts/
    stage6/stage6_8_evidence/
    stage7/final_handoff/
  embeddings/
  features/
  figures/
  models/
  profiles/
  reports/
```

Stage 6.8 and Stage 7 artifacts are especially sensitive to the exact selected assignment file, Stage 6.6 readiness file, prepared transaction table, behavioral feature table, product-theme rules, product-term rules, and profiling thresholds. If any of those change, rebuild Stage 6.8 and then Stage 7.

Production Stage 6 uses deterministic sampling for expensive manifold and density fitting, then transforms/assigns the full customer population in batches where configured. Reproducibility therefore depends on the same `modeling.fit_sample_size`, UMAP `transform_batch_size`, random seed, promoted HDBSCAN parameters, and two-stage/lift-filter settings.

## Cache Reproducibility

Cache metadata is centralized in `outputs/<mode>/.artifact_metadata.json`. Stage 0.5 reports whether artifacts exist, whether cache reuse is enabled, and whether recorded metadata matches the current config/data fingerprints.

Current cache behavior intentionally treats metadata mismatch as diagnostic rather than automatically destructive. To guarantee a rebuild after source/config changes, use one of:

- Pass the relevant stage `force=True`.
- Set `cache.use_cached: false` for the run.
- Delete only the affected generated artifacts under `outputs/<mode>/`.

Do not compare downstream results across machines until both teams agree whether artifacts were reused or rebuilt.

## If Results Differ

1. Check `git status --short` and confirm teammates are using the same source revision.
2. Verify the environment, especially `scikit-learn` and `hdbscan`.
3. Compare `data/dev/subset_metadata.json` hashes and prepared-data quality outputs.
4. Compare the active YAML, especially Stage 1, Stage 4, UMAP, HDBSCAN, two-stage, and profiling settings.
5. Delete or force-rebuild generated Stage 4+ artifacts after vectorization or feature changes.
6. Compare Stage 6.6 readiness and product-lift profiles before concluding that a model is materially different.
