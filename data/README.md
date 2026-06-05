# Data Directory

Data files are local artifacts and are not committed to this repository.

## Expected Layout

```text
data/
|-- raw/
|   |-- csv/          Original source CSV files
|   `-- parquet/      Generated raw Parquet files
|-- processed/        Production-mode pipeline artifacts
`-- dev/              Stratified dev subset and dev-mode ML artifacts
```

## Required Raw Files

Place these files in `data/raw/csv/` after cloning:

| File | Description |
|---|---|
| `ie_maestra_articulos.csv` | Product master / article catalogue |
| `ie_linea_ticket.csv` | Transactional ticket lines, around 26 GB as CSV |

Then run:

```python
from src.data_loader import verify_csv_checksums, convert_csv_to_parquet

verify_csv_checksums()
convert_csv_to_parquet()
```

## Main Generated Artifacts

Production preprocessing writes to `data/processed/`:

| Artifact | Purpose |
|---|---|
| `quality_report.json` | Full data quality gate results |
| `df_combined.parquet` | Clean joined ticket/product table |
| `customer_kpis.parquet` | Per-customer spend, visit, promo, and basket metrics |
| `product_eda_*.parquet` | Product demand, repeat, pair-lift, and temporal EDA tables |

Dev-mode ML runs write to `data/dev/`:

| Artifact | Purpose |
|---|---|
| `subset_metadata.json` | Stratified 44k-customer subset validation |
| `basket_sentences.parquet` | Ticket-level product lists for Item2Vec |
| `product_embeddings.parquet` | Product vectors from Word2Vec |
| `customer_vectors_weighted.parquet` | Primary recency/frequency-weighted customer vectors |
| `customer_vectors_mean.parquet` | Simple-mean baseline customer vectors |
| `customer_store_features.parquet` | Store spend-share features by customer |
| `umap_cluster_20d.parquet` | Current UMAP clustering embedding; filename is historical, active config uses 50 dims |
| `umap_viz_2d.parquet` | 2D UMAP map for visualization |
| `pca_cluster_20d.parquet` | PCA baseline embedding; filename is historical, active config uses 50 dims |
| `cluster_labels_hdbscan.parquet` | Primary density-based cluster labels |
| `cluster_labels_kmeans_k*.parquet` | Fixed-K K-Means baseline labels |
| `hdbscan_grid_results.parquet` | HDBSCAN hyperparameter sweep results |
| `kmeans_baseline_results.parquet` | K-Means baseline comparison table |

## Rules

- Never commit files under `data/raw/`, `data/processed/`, or `data/dev/` except `.gitkeep` and documentation.
- Regenerate caches with the relevant `force=True` flag when changing upstream feature definitions or hyperparameters.
- Keep prod and dev artifacts separate by setting `CARREFOUR_MODE` before running pipeline code.
