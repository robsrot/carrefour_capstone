# Carrefour Capstone

Product-first behavioral customer segmentation for Carrefour checkout data. The project discovers customer tribes from what people buy, how often they buy it, and how distinctive those product missions are. Spend and KPI fields are used only after clustering for interpretation, sizing, and business actionability.

## Final Project Snapshot

The final production run in this workspace covers 1,478,831 customers.

| Segment layer | Customers | Share | Treatment |
|---|---:|---:|---|
| Promoted tribes (15) | 301,041 | 20.4% | Campaign-ready precision tribes |
| Review tribes (7) | 375,800 | 25.4% | Pilot with monitoring; not fully autonomous |
| Centroid-rescued customers | 694,356 | 46.9% | Softer tribe-matched activation coverage |
| Bridge, near-tribe, sparse, and other unassigned customers | 107,634 | 7.3% | Separate remaining-customer strategies |

The official result is intentionally conservative: hard HDBSCAN tribes remain the core discovery layer, review tribes stay clearly labelled, and centroid rescue improves activation coverage without rewriting the core cluster evidence.

## Official Modeling Recipe

The official clustering signal is product-first:

- Build baskets from transaction lines and train product embeddings with Word2Vec.
- Build customer vectors with `quantity_idf`, `log1p(unidades)`, recency decay, product basket-frequency scaling, and vector normalization.
- Use `embeddings_only` as the official feature set for clustering.
- Exclude `importe`, total spend, average basket value, revenue tier, demographics, and downstream KPIs from customer embeddings and clustering.
- Use PCA pre-reduction, UMAP representation, and three-stage HDBSCAN with product-lift filtering to discover dense organic tribes.
- Use Stage 6 readiness, jitter recovery, assignment confidence, product lift, and remaining-customer evidence before profiling.
- Use Stage 7 for business naming, validation, review-tribe treatment, remaining-customer analysis, campaign playbooks, and final handoff tables.
- Use Stage 8 as the final dashboard semantic contract: normalized `rel_*` tables, presentation marts, manifests, SQL schema, readiness checks, and dashboard page contracts.

## Final Artifacts

Generated data and model outputs are local artifacts and are not committed to GitHub.

| Artifact | Location | Purpose |
|---|---|---|
| Stage 7 final handoff | `outputs/prod/artifacts/stage7/final_handoff/` | Final tribe index, story, manifests, cards, and supporting evidence tables |
| Cache metadata | `outputs/<mode>/.artifact_metadata.json` | Cache provenance for generated artifacts |

Dashboards should consume Stage 8 structured files, especially `dashboard_manifest_prod.json`, `relational_manifest_prod.json`, `rel_dim_*`, `rel_fact_*`, `rel_bridge_*`, and `rel_mart_*`. They should not parse Markdown reports or recompute model decisions.

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate carrefour
python -m ipykernel install --user --name=carrefour --display-name "Python (carrefour)"
```

Verify the HDBSCAN-compatible dependency set:

```powershell
python -c "import sklearn, hdbscan, polars; print('sklearn', sklearn.__version__); print('hdbscan ok'); print('polars', polars.__version__)"
```

Expected scikit-learn version: `1.7.2`. If the environment reports `1.8.x`, recreate it or run:

```powershell
conda activate carrefour
conda install -c conda-forge scikit-learn=1.7.2
pip install --force-reinstall --no-deps hdbscan==0.8.40
```

Add the raw source files locally:

```text
data/raw/csv/ie_maestra_articulos.csv
data/raw/csv/ie_linea_ticket.csv
```

Convert raw CSV files to Parquet:

```python
from src.data_loader import verify_csv_checksums, convert_csv_to_parquet

verify_csv_checksums()
convert_csv_to_parquet()
```

Use `verify_csv_checksums(record=True)` only on the machine that establishes the canonical raw files.

Run the notebooks in order:

```bash
jupyter notebook notebooks/01_exploration.ipynb
jupyter notebook notebooks/02_pre-analysis.ipynb
```

After `02_pre-analysis.ipynb` creates `data/processed/df_combined.parquet` and `data/processed/customer_kpis.parquet`, generate the dev subset:

```powershell
$env:CARREFOUR_MODE = "prod"
python -m src.generate_dev_subset
```

Use the sandbox only for experiments:

```powershell
$env:CARREFOUR_MODE = "dev"
jupyter notebook notebooks/04_experiment_sandbox.ipynb
```

Run the official production pipeline:

```powershell
$env:CARREFOUR_MODE = "prod"
jupyter notebook notebooks/03_ml_pipeline.ipynb
```

`prod` is the default when `CARREFOUR_MODE` is unset. Use `dev` for smoke tests and experiment loops only.

## Repository Layout

```text
configs/          YAML configuration for dev/prod modes and modeling parameters
data/             Local raw, processed, and dev data; tracked only for README/.gitkeep files
notebooks/        Ordered exploration, preprocessing, official ML pipeline, and experiment sandbox
outputs/          Local generated artifacts, models, reports, figures, and final handoff packs
src/              Reusable pipeline modules for data, embeddings, clustering, profiling, exports, and Stage 8
tests/            Unit tests for config, caching, embeddings, model selection, profiling, exports, and Stage 8/9 helpers
```

## Data And Git Hygiene

Do not commit raw data, processed data, dev subsets, Parquet caches, generated JSON metadata, model binaries, figures, reports, dashboard outputs, `.env`, or secrets.

Only `.gitkeep` files and documentation should be tracked under `data/`. git commit -m "Finalize README and stop tracking generated data metadata"

Generated project outputs belong under `outputs/<mode>/` and stay local. If a data or output artifact is needed for review, share it outside the Git repo or regenerate it from the pipeline.

## Tests And Handoff Checks

Before handing off the repo or opening a pull request, run:

```powershell
python -c "import sklearn, hdbscan; print(sklearn.__version__); print('hdbscan ok')"
pytest
git status --short
```

For targeted final-output checks:

```powershell
pytest tests/test_stage8_exports.py tests/test_stage9_exports.py tests/test_profiling_readiness.py
```

## Operating Notes

- Keep reusable logic in `src/`; notebooks should orchestrate and explain the official flow.
- Promote experiment settings into YAML only after sandbox evidence supports them.
- Rebuild from the affected upstream stage after changing vectorization, feature, UMAP, HDBSCAN, profiling, or Stage 8 export logic.
- Use `force=True`, disable cache, or delete targeted generated artifacts when cache metadata no longer reflects the code/config state.
- Always join customer-level tables on `cliente`; do not rely on positional row order.
