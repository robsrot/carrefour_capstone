# Cross-Machine Reproducibility

## Problem

Running Notebooks 02 and 03 on different laptops produced different results despite
fixed random seeds. The seeds controlled shuffle order but not floating-point
summation order, which varies by machine.

## Root Causes

### Critical — recency weight streaming sum (fixed)

**File:** `src/customer_vectors.py` — `_build_interactions()`

Polars streams `df_combined.parquet` in chunks. Chunk boundaries depend on available
RAM and OS scheduling, so the order in which partial sums are merged differs between
machines. Floating-point addition is not associative, so `(a + b) + c ≠ a + (b + c)`
when values are close in magnitude. This caused `customer_product_weights.parquet` to
differ, which cascaded through every downstream artifact.

**Fix applied (this branch):** multiply `recency_weight` by `_WEIGHT_SCALE = 1e8`,
round to `Int64`, sum as integers (associative and exact), then divide back. The same
pattern was already used for sector spend in `generate_dev_subset.py`.

### Accepted — UMAP and HDBSCAN platform variation (not fixed)

UMAP uses `pynndescent` for nearest-neighbor graph construction. Even with
`random_state` fixed and `n_jobs=1`, distance calculations hit CPU-specific
floating-point paths (AVX2 vs AVX512 vs SSE). Results are stable within a single
machine but not bit-identical across different CPUs. HDBSCAN noise-point assignment
also uses nearest-neighbor distances, so tie-breaking can flip on different hardware.

These are documented limitations of the libraries and cannot be fixed without
replacing UMAP with a deterministic reducer (e.g., PCA-only). For this capstone
the cluster *structure* (tribe profiles, top-product lift) is stable across machines
even when individual cluster assignments shift at boundaries.

### Residual — customer KPI float sums (fixed)

`customer_kpis.parquet` is built from streaming float sums (`avg_basket_size`,
`total_spend_6m`). The same issue as the recency weight. Applied the same
integer-quantization fix in Notebook 02 cell `bd862d2b`: `importe` is multiplied
by `_EUR_SCALE = 100` and cast to `Int64` before any streaming aggregation.
`total_spend_6m` and `avg_basket_size` are accumulated as integer eurocents and
converted back to EUR only at the final `.with_columns()` step.
Both `strata_assignment_sha256` and `selected_customer_sha256` should now be
bit-identical across machines after a clean re-run.

## Verification Checksums

After running Notebook 02 and regenerating the dev subset, compare these two files
against the committed versions:

**`data/processed/quality_report.json`** — depends only on raw parquets, must be
identical across machines.

**`data/dev/subset_metadata.json`** — two hashes to check:

| Hash field | Required | Notes |
|---|---|---|
| `selected_customer_sha256` | must match | determines which rows are in `df_combined.parquet` dev |
| `strata_assignment_sha256` | must match | KPI float fix applied; mismatch means `customer_kpis.parquet` was not rebuilt |

Both files are tracked in git (gitignore exceptions added for `*.json` rule).

## Workflow for Sharing Artifacts Between Machines

Only one machine needs to run Notebook 02. Once it completes:

1. Verify `selected_customer_sha256` matches the committed value.
2. Share these two files with teammates:
   - `data/dev/df_combined.parquet`
   - `data/processed/customer_kpis.parquet`
3. Teammates place them in the correct directories and run Notebook 03 directly.
   Notebook 03 uses `data/dev/df_combined.parquet` as its transaction source and
   `data/processed/customer_kpis.parquet` for tribe profiling.
4. UMAP and clustering results will vary slightly by CPU but tribe structure is stable.

## If the Hashes Don't Match

**`selected_customer_sha256` differs:** the dev subset contains different customers.
All Notebook 03 caches are invalid. Likely cause: the residual KPI float issue grew
large enough to change tertile bin assignments for enough customers to shift the
final selection. Apply the integer-quantization fix to the KPI aggregations in
Notebook 02 Section 2.8 (the `avg_basket_size` and `total_spend_6m` streaming sums).

**`strata_assignment_sha256` differs but `selected_customer_sha256` matches:**
this should no longer happen after the integer-quantization fix. If it does,
check whether `customer_kpis.parquet` was regenerated with the updated cell;
delete the cache and re-run Notebook 02 Section 4.1.

**`quality_report.json` differs:** something changed in production preprocessing.
The raw parquets may differ between machines, or `src/data_quality.py` changed.
Investigate before proceeding.
