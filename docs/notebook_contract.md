# Notebook Contract

Project root used for full paths: `C:\Users\sebog\proyectos\carrefour_capstone`.

## Notebook 1 — `01_exploration.ipynb`

### Defines
| Artifact | Type | Schema / Shape | Origin (cell or src/ function) | Notes |
|----------|------|----------------|-------------------------------|-------|
| `_parquet_dir` | `pathlib.Path` | `../data/raw/parquet` | Cell 4 | Local pointer to raw parquet folder. |
| `pf` | `pyarrow.parquet.ParquetFile` | Metadata for `linea_tickets.parquet`: 191,017,715 rows x 10 cols | Cell 4 | Reads `data/raw/parquet/linea_tickets.parquet`. |
| `pf2` | `pyarrow.parquet.ParquetFile` | Metadata for `maestra_articulos.parquet`: 893,944 rows x 4 cols | Cell 4 | Reads `data/raw/parquet/maestra_articulos.parquet`. |
| `df_articulos` | Polars `DataFrame` | `idarticu:Int64`; `desc_larga_articulo:String`; `idsector:Int64`; `desc_sector:String`; confirmed 893,944 x 4 | Cell 7 via `src.data_loader.load_maestra_articulos()` | Same physical artifact as Notebook 2 `df_articles` and Notebook 3 fallback `df_articles`, with a different variable name. |
| `df_tickets` | Polars `LazyFrame` | `idempres:Int64`; `fecha:Date`; `hora:Int64`; `ticket:String`; `cliente:String`; `idarticu:Int64`; `unidades:Int64`; `importe:Float64`; `idpromoc:String`; `idtiprod:Int64`; confirmed 191,017,715 x 10 | Cell 7 via `pl.scan_parquet("../data/raw/parquet/linea_tickets.parquet")` | Shared raw ticket contract with Notebook 2. |
| `schema` | Polars schema object | First: raw ticket schema; overwritten in Cell 19 with the same raw ticket schema | Cells 7 and 19 | Local metadata only. |
| `null_counts` | Polars `DataFrame` | One row with null counts for all 10 raw ticket columns | Cell 19 | Saved output confirms zero nulls. |
| `stats` | Polars `DataFrame` | 1 x 6: `unique_customers`, `unique_products`, `unique_tickets`, `fecha_min`, `fecha_max`, `unique_stores` | Cell 21 | ⚠️ Schema mismatch - same name, different shape than Notebook 2 `stats`, which is a Python dict from clean-join metrics. |
| `orphaned` | Python `set[int]` | Product ids in tickets not present in product master; inferred empty from Notebook 2 `quality_report.json` product coverage | Cell 27 | Inferred because the code treats `df_tickets` as subscriptable. The quality report confirms `orphaned_products=0`. |
| `df` | pandas-like merged transaction `DataFrame`, inferred | Expected 13 columns: raw ticket schema plus `desc_larga_articulo`, `idsector`, `desc_sector`; `fecha` and `hora` reformatted as strings if execution succeeds | Cell 29 | Inferred; not persisted. Code path treats `df_tickets` as a pandas DataFrame even though Cell 7 defines it as a Polars `LazyFrame`. |
| `convert_csv_to_parquet()` side effect | File conversion | Writes `maestra_articulos.parquet` and `linea_tickets.parquet` under `data/raw/parquet` | Cell 3 via `src.data_loader.convert_csv_to_parquet()` | File-mediated source for Notebooks 1 and 2. |

### Consumes
| Artifact | Expected Type | Expected Schema / Shape | Source |
|----------|---------------|------------------------|--------|
| `ie_maestra_articulos.csv` | CSV, semicolon-delimited, Latin-1 | Converted to 893,944 x 4 parquet product master | `src.data_loader._SOURCES["maestra_articulos"]`; `data/raw/csv/ie_maestra_articulos.csv` |
| `ie_linea_ticket.csv` | CSV, semicolon-delimited, Latin-1 | Converted to 191,017,715 x 10 parquet transaction lines | `src.data_loader._SOURCES["linea_tickets"]`; `data/raw/csv/ie_linea_ticket.csv` |
| `data/raw/parquet/maestra_articulos.parquet` | Parquet | `idarticu:Int64`; `desc_larga_articulo:String`; `idsector:Int64`; `desc_sector:String` | Produced by `convert_csv_to_parquet()` or pre-existing raw parquet |
| `data/raw/parquet/linea_tickets.parquet` | Parquet | `idempres:Int64`; `fecha:Date`; `hora:Int64`; `ticket:String`; `cliente:String`; `idarticu:Int64`; `unidades:Int64`; `importe:Float64`; `idpromoc:String`; `idtiprod:Int64` | Produced by `convert_csv_to_parquet()` or pre-existing raw parquet |
| `src.data_loader.load_maestra_articulos()` | src function | Returns full Polars `DataFrame` product master | `src/data_loader.py` |
| `src.data_loader.convert_csv_to_parquet()` | src function | Returns `None`; writes raw parquet files | `src/data_loader.py` |

## Notebook 2 — `02_pre-analysis.ipynb`

### Defines
| Artifact | Type | Schema / Shape | Origin (cell or src/ function) | Notes |
|----------|------|----------------|-------------------------------|-------|
| `MODE` | config scalar | Saved output: `prod` | Cell 3 via `src.config.MODE` | Notebook 2 builds production artifacts. |
| `DATA_PROCESSED` | config path | `data/processed` when `MODE=prod` | Cell 3 via `src.config` | Shared path contract with many downstream files. |
| `OUTPUTS` | config path | `outputs/prod` when `MODE=prod` | Cell 3 via `src.config` | Plot outputs are not consumed by Notebook 3. |
| `SAMPLE_SIZE` | config int | `100000` | Cell 3 via `configs/base.yaml` | Used by quality feasibility checks. |
| `df_articles` | Polars `DataFrame` | `idarticu:Int64`; `desc_larga_articulo:String`; `idsector:Int64`; `desc_sector:String`; confirmed 893,944 x 4 | Cell 5 via `load_maestra_articulos()` | Same physical product master as Notebook 1 `df_articulos`. |
| `df_tickets` | Polars `LazyFrame` | Raw ticket schema: 191,017,715 x 10 | Cell 6 via `load_linea_tickets()` | Same raw ticket contract as Notebook 1. |
| `n_rows` | int | `191017715` | Cell 6 | Derived by streaming count over `df_tickets`. |
| `cols` | list[str] | `idempres`, `fecha`, `hora`, `ticket`, `cliente`, `idarticu`, `unidades`, `importe`, `idpromoc`, `idtiprod` | Cell 6 | Raw ticket column list. |
| `report` | dict metadata | Keys: `null_audit`, `anomaly_audit`, `schema_validation`, `product_coverage`, `temporal_completeness`, `customer_activity`, `promotional_integrity`, `store_integrity`, `_summary` | Cell 8 via `src.data_quality.build_quality_report()` | Saved to `data/processed/quality_report.json`; consumed by Notebook 3. |
| `PROD_PROCESSED` | `pathlib.Path` | `ROOT / "data" / "processed"` | Cells 9, 10, 18, 24, 26, 28, 32, 34, 36 | Rebound repeatedly to the same prod directory. |
| `quality_report_path` | `pathlib.Path` | `data/processed/quality_report.json` | Cells 9, 10 | Fallback read path for `report`. |
| `anom` | dict metadata | `report["anomaly_audit"]`: total rows, anomaly counts/pcts, retained/dropped rows, cleaning rule | Cell 9 | Confirms clean rule `DROP unidades <= 0 OR importe <= 0`. |
| `ca` | dict metadata | `report["customer_activity"]`: customer counts, threshold, quantiles | Cell 10 | Confirms eligible customer population for sampling. |
| `pre` | dict metadata | `n_raw`, `returns`, `credit_notes`, `zero_price`, `n_drop`, `n_keep` | Cell 12 | `n_keep` becomes the expected row count for `df_combined.parquet`. |
| `w` | int | `42` | Cell 12 | Formatting width only. |
| `df_combined` | Polars `LazyFrame` | Clean transaction schema: `idempres:Int64`; `fecha:Date`; `hora:Int64`; `ticket:String`; `cliente:String`; `idarticu:Int64`; `unidades:Int64`; `importe:Float64`; `idpromoc:String`; `idtiprod:Int64`; `desc_larga_articulo:String`; `idsector:Int64`; `desc_sector:String`; confirmed prod cache 190,362,519 x 13 | Cell 13 builds from filtered tickets joined to articles; Cell 14 rebinds to `pl.scan_parquet(PROD_COMBINED)` | Primary shared downstream artifact. |
| `stats` | dict metadata | `n_rows`, `n_customers`, `n_products`, `n_tickets`, `promo_rate`, `orphaned_lines` | Cell 13 | ⚠️ Schema mismatch - same name, different shape than Notebook 1 `stats`. |
| `n_customers` | int | Clean unique customers from `stats` | Cell 13 | Derived scalar. |
| `n_products` | int | In Cell 13 clean unique products; later reused as product-demand row count | Cells 13, 32, 34, 36 | Same scalar name, different meaning across sections. |
| `n_tickets_total` | int | Clean unique tickets from `stats` | Cell 13 | Derived scalar. |
| `overall_promo_rate` | float | Clean line promo rate, later recalculated from promo-sector cache or quality report | Cells 13, 28, 34, 36 | Same semantic metric, multiple derivations. |
| `n_clean` | int | Clean row count from `stats["n_rows"]` | Cell 13 | Should equal `pre["n_keep"]`. |
| `PROD_COMBINED` | `pathlib.Path` | `data/processed/df_combined.parquet` | Cell 14 | Production clean transaction output. |
| `post` | dict metadata | `n_rows`, approx `n_customers`, approx `n_products`, approx `n_tickets`, `null_sectors`, `null_names` | Cell 14 | Verifies persisted `df_combined` has no product-name/sector nulls. |
| `product_lf` | Polars `LazyFrame` | Scan of prod `df_combined.parquet`; same 13-column clean schema | Cell 18 | Source for product EDA caches. |
| `PRODUCT_DEMAND_CACHE` | `pathlib.Path` | `data/processed/product_eda_demand.parquet` | Cells 18, 24, 32, 34, 36 | Product EDA cache. |
| `product_demand` | Polars `DataFrame` | `idarticu:Int64`; `desc_larga_articulo:String`; `desc_sector:String`; `purchase_lines:UInt32`; `units:Int64`; `revenue:Float64`; `approx_customers:UInt32`; `approx_baskets:UInt32`; confirmed 117,621 x 8 | Cell 18 | File-mediated within Notebook 2; not consumed by Notebook 3. |
| `sector_demand` | Polars `DataFrame` | `desc_sector`, `n_products`, `purchase_lines`, `units`, `revenue`, `summed_product_customer_reach`, `line_share_pct`, `revenue_share_pct` | Cell 18 | Derived from `product_demand`. |
| `total_revenue`, `total_lines`, `n_product_skus` | scalar metrics | Float total revenue, int line count, int product count | Cell 18 | Derived from `product_demand`. |
| `product_concentration` | Polars `DataFrame` | `product_demand` plus `rank`, `product_pct`, `cum_revenue_pct`, `cum_line_pct` | Cell 18 | Used for concentration summary. |
| `concentration_rows` | list[dict] | Rows for top 1, 5, 10, 20 percent product concentration | Cells 18, 32 | Metadata for `product_concentration_summary`. |
| `product_concentration_summary` | pandas `DataFrame` | `top_product_pct`, `cum_revenue_pct`, `cum_line_pct` | Cells 18, 32 | Derived display table. |
| `sector_pd`, `conc_pd` | pandas `DataFrame` | Plot-ready slices of `sector_demand` and `product_concentration` | Cell 18 | Local plotting inputs. |
| `TOP_N_PAIR_PRODUCTS` | int constant | `250` | Cell 20 | Pair-lift top-product limit. |
| `PAIR_BASKET_SAMPLE` | int constant | `500000` | Cell 20 | Basket sample cap for pair lift. |
| `MIN_PAIR_BASKETS` | int constant | `75` | Cell 20 | Minimum pair support. |
| `PAIR_LIFT_CACHE` | `pathlib.Path` | `data/processed/product_eda_pair_lift_top250.parquet` | Cells 20, 32 | Pair-lift cache. |
| `top_pair_ids` | list[int] | Top 250 product ids by purchase lines | Cell 20 | Derived from `product_demand`. |
| `basket_products` | Polars `DataFrame`, transient | `ticket:String`; `products:List[Int64]` for sampled multi-product baskets | Cell 20 | Used only when pair cache is rebuilt. |
| `item_counter`, `pair_counter` | `collections.Counter` | Product and product-pair counts | Cell 20 | Local pair-lift computation state. |
| `n_baskets_for_pairs` | int | Number of sampled baskets used for pair lift | Cell 20 | Local scalar. |
| `product_lookup` | dict | `idarticu -> (desc_larga_articulo, desc_sector)` | Cell 20 | Local metadata lookup. |
| `pair_lift` | Polars `DataFrame` | `idarticu_a:Int64`; `product_a:String`; `sector_a:String`; `idarticu_b:Int64`; `product_b:String`; `sector_b:String`; `pair_baskets:Int64`; `lift:Float64`; confirmed 17,902 x 8 | Cell 20 | Cached at `PAIR_LIFT_CACHE`. |
| `rows`, `unique_products`, `support_a`, `support_b`, `pair_support`, `lift` | local variables | Pair-lift intermediate list/scalars | Cell 20 | Local only. |
| `TEMPORAL_CACHE` | `pathlib.Path` | `data/processed/product_eda_temporal_lift.parquet` | Cell 22 | Temporal EDA cache. |
| `TOP_N_TEMPORAL_PRODUCTS` | int constant | `100` | Cell 22 | Product-hour analysis limit. |
| `MIN_PRODUCT_HOUR_LINES` | int constant | `100` | Cell 22 | Minimum product-hour support. |
| `weekday_names` | dict[int, str] | `{1:"Mon", ..., 7:"Sun"}` | Cell 22 | Display metadata. |
| `top_temporal_ids` | list[int] | Top 100 product ids by purchase lines | Cell 22 | Derived from `product_demand`. |
| `temporal_bundle` | Polars `DataFrame` | `analysis:String`; `desc_sector:String`; `weekday:Int64`; `hour:Int64`; `idarticu:Int64`; `desc_larga_articulo:String`; `lines:UInt32`; `lift:Float64`; confirmed 1,773 x 8 | Cell 22 | Cached at `TEMPORAL_CACHE`. |
| `sector_weekday_lift` | Polars `DataFrame` | Same schema as `temporal_bundle` filtered to `analysis="sector_weekday"` | Cell 22 | Derived temporal table. |
| `product_hour_lift` | Polars `DataFrame` | Same schema as `temporal_bundle` filtered to `analysis="product_hour"` | Cell 22 | Derived temporal table. |
| `temporal_base`, `sector_weekday_counts`, `sector_totals`, `weekday_totals`, `product_hour_counts`, `product_totals`, `hour_totals` | Polars frames | Temporal intermediate schemas inferred from group-by columns and count/sum columns | Cell 22 | Local rebuild-only intermediates. |
| `sector_weekday_pd`, `heatmap_source`, `heatmap_data` | pandas `DataFrame` | Plot-ready temporal tables | Cell 22 | Local plotting inputs. |
| `DEV_COMBINED` | `pathlib.Path` | `data/dev/df_combined.parquet` | Cell 24 | Optional recurrence EDA input if dev subset exists. |
| `FULL_COMBINED` | `pathlib.Path` | `data/processed/df_combined.parquet` | Cell 24 | Full recurrence fallback input. |
| `USE_DEV_SUBSET` | bool | `DEV_COMBINED.exists()` | Cell 24 | Controls recurrence EDA source. |
| `SOURCE_COMBINED` | `pathlib.Path` | `DEV_COMBINED` if present else `FULL_COMBINED` | Cell 24 | Recurrence EDA input path. |
| `SOURCE_LABEL` | str | `"dev subset"` or `"bounded full-file prefix"` | Cell 24 | Display label. |
| `REPEAT_PAIR_CACHE` | `pathlib.Path` | `data/processed/product_eda_customer_product_repeats_dev.parquet` or `data/processed/product_eda_customer_product_repeats_full_prefix.parquet` | Cells 24, 32 | Existing cache is `product_eda_customer_product_repeats_full_prefix.parquet` with 239 x 12. |
| `MIN_REPEAT_LINES` | int constant | `3` | Cells 24, 32 | Repeat-pair support threshold. |
| `MAX_FULL_FALLBACK_ROWS` | int constant | `1000000` | Cell 24 | Full-file prefix cap for recurrence EDA. |
| `MAX_REPEAT_PAIRS_TO_KEEP` | int constant | `50000` | Cell 24 | Repeat-pair retained-row cap. |
| `MIN_REPEAT_CUSTOMERS_FOR_RATE` | int constant | `10` | Cells 24, 32 | Repeat product summary threshold. |
| `repeat_pairs` | Polars `DataFrame` | `cliente:String`; `idarticu:Int64`; `desc_larga_articulo:String`; `desc_sector:String`; `purchase_lines:UInt32`; `units:Int64`; `revenue:Float64`; `approx_baskets:UInt32`; `first_date:Date`; `last_date:Date`; `purchase_days:UInt32`; `span_days:Int64`; confirmed full-prefix cache 239 x 12 | Cell 24 | Cached recurrence readout. |
| `repeat_product_summary` | Polars `DataFrame` | `idarticu`, `desc_larga_articulo`, `desc_sector`, `repeat_customers_in_sample`, `avg_lines_per_repeat_customer`, `max_customer_lines`, `repeat_pair_revenue`, optional `approx_customers` | Cells 24, 32 | Derived from `repeat_pairs` and optionally `product_demand`. |
| `recurrence_lf`, `recurrence_rows` | Polars `LazyFrame` / `DataFrame` | Selected recurrence input columns from clean transactions | Cell 24 | Rebuild-only intermediates. |
| `_cache_path` | `pathlib.Path` | `data/processed/customer_kpis.parquet` | Cell 26 | Customer KPI cache path. |
| `cust_stats` | pandas `DataFrame` | `cliente:String`; `visit_count:UInt32`; `avg_basket_size:Float64`; `total_spend_6m:Float64`; `avg_promo_rate:Float64`; `unique_products:UInt32`; confirmed 1,482,715 x 6 | Cells 26, 28, 30, 34, 36 | Consumed by Notebook 3 profiles through `customer_kpis.parquet`. |
| `clean_lf` | Polars `LazyFrame` | Prod clean transaction schema | Cell 26 | Source for KPI computation. |
| `basket_lf` | Polars `LazyFrame` | `cliente`, `ticket`, `n_items`, `total_spend`, `promo_items`, `promo_rate` | Cell 26 | KPI intermediate. |
| `cust_lf` | Polars `LazyFrame` | `cliente`, `visit_count`, `avg_basket_size`, `total_spend_6m`, `avg_promo_rate` | Cell 26 | KPI intermediate. |
| `diversity_lf` | Polars `LazyFrame` | `cliente`, `unique_products` | Cell 26 | KPI intermediate. |
| `cust_df`, `diversity_df` | pandas `DataFrame` | Materialized KPI intermediates | Cell 26 | Merged into `cust_stats`. |
| `s`, `sc` | pandas sample / matplotlib scatter | Sample of up to 5,000 customers; scatter handle | Cell 26 | Plot-only variables. |
| `PROMO_SECTOR_CACHE` | `pathlib.Path` | `data/processed/eda_promo_sector.parquet` | Cell 28 | Promo sector cache. |
| `CUSTOMER_KPI_CACHE` | `pathlib.Path` | `data/processed/customer_kpis.parquet` | Cell 28 | Customer KPI cache. |
| `COMBINED_PATH` | `pathlib.Path` | `data/processed/df_combined.parquet` | Cell 28 | Clean transaction source. |
| `promo_sector_pl` | Polars `DataFrame` | `desc_sector:String`; `total:UInt32`; `promo:UInt32`; `promo_rate_pct:Float64`; confirmed 5 x 4 | Cell 28 | Cached at `PROMO_SECTOR_CACHE`. |
| `promo_sector` | pandas `DataFrame` | Same fields as `promo_sector_pl`, indexed by `desc_sector` | Cell 28 | Used to recalculate `overall_promo_rate`. |
| `cust_stats_sorted` | pandas `DataFrame` | `cust_stats` plus `cumulative_rev_pct`, `customer_pct` | Cell 30 | Revenue concentration local table. |
| `top10_share`, `top20_share` | floats | Share of revenue from top 10 percent and top 20 percent customers | Cell 30 | Derived from `cust_stats_sorted`. |
| `_clip_p99`, `_pct_shown` | floats | Spend p99 clip threshold and percent of customers shown | Cell 30 | Plot/display scalars. |
| `QUALITY_REPORT` | `pathlib.Path` | `data/processed/quality_report.json` | Cells 34, 36 | Metadata input to summary cells. |
| `kpi_labels` | dict[str, str] | Maps KPI column names to display labels | Cells 34, 36 | Metadata object for viability tables. |
| `vals`, `cv`, `signal`, `high_cv` | local scalar metrics | KPI coefficient-of-variation inputs and count | Cells 34, 36 | Local summary variables. |
| `qr`, `qr_tmp` | dict metadata | Parsed `quality_report.json` | Cells 34, 36 | Summary input. |
| `overview` | pandas `DataFrame` | Dataset overview table with `Metric`, `Value` | Cell 36 | Display-only metadata table. |
| `kpis` | dict alias | Same mapping as `kpi_labels` | Cell 36 | Local alias. |
| `behaviour` | pandas `DataFrame` | KPI summary: `Median`, `Mean`, `p25`, `p75`, `CV (std/mean)`, `Clustering signal` | Cell 36 | Display-only metadata table. |
| `_spend_desc`, `_total`, `_n` | pandas Series / scalars | Sorted spend series, total spend, number of customers | Cell 36 | Revenue summary intermediates. |
| `concentration` | pandas `DataFrame` | Revenue concentration display table | Cell 36 | Display-only metadata table. |
| `generate()` side effects | src function output | Writes `data/dev/df_combined.parquet` and `data/dev/subset_metadata.json`; metadata confirms 44,000 customers and 7,511,379 rows | Cell 38 via `src.generate_dev_subset.generate()` | File-mediated handoff into Notebook 3 dev mode. |

### Consumes
| Artifact | Expected Type | Expected Schema / Shape | Source |
|----------|---------------|------------------------|--------|
| `data/raw/parquet/maestra_articulos.parquet` | Parquet | Product master schema, 893,944 x 4 | Produced or verified by Notebook 1 / `src.data_loader.convert_csv_to_parquet()` |
| `data/raw/parquet/linea_tickets.parquet` | Parquet | Raw ticket schema, 191,017,715 x 10 | Produced or verified by Notebook 1 / `src.data_loader.convert_csv_to_parquet()` |
| `src.config` base values | config object | `random_seed=42`; `min_tickets_per_customer=3`; `sample_size=100000`; paths derived from `MODE=prod` | `configs/base.yaml` through `src/config.py` |
| `src.data_loader.load_maestra_articulos()` | src function | Returns Polars product master `DataFrame` | `src/data_loader.py` |
| `src.data_loader.load_linea_tickets()` | src function | Returns Polars raw ticket `LazyFrame`, optionally selected columns | `src/data_loader.py` |
| `src.data_quality.*` functions | src functions | Return dict quality checks; `build_quality_report()` persists combined report | `src/data_quality.py` |
| `data/dev/df_combined.parquet` | Optional Parquet | Dev clean transaction schema, 7,511,379 x 13 if present | Cell 24 recurrence EDA uses this if it already exists |
| Existing Notebook 2 caches | Parquet / JSON | Product demand, pair lift, temporal lift, repeat pairs, customer KPIs, promo sector, quality report | Cells use cache-if-exists behavior to avoid recompute |

## Notebook 3 — `03_ml_pipeline.ipynb`

### Defines
| Artifact | Type | Schema / Shape | Origin (cell or src/ function) | Notes |
|----------|------|----------------|-------------------------------|-------|
| `_root` | `pathlib.Path` | Project root containing `src/config.py` | Cells 4, 5 | Local import-root resolver. |
| `MODE` | config scalar | Forced to `dev` in Cell 4 | Cell 4 via `os.environ["CARREFOUR_MODE"]="dev"` then `importlib.reload(src.config)` | Critical mode divergence from Notebook 2. |
| `_is_dev` | bool | `True` in saved configuration | Cell 4 | Drives run flags. |
| `DATA_PROCESSED` | config path | `data/dev` when `MODE=dev` | Cells 4, 5, 7, 18 | Notebook 3 writes most model artifacts under `data/dev`. |
| `MODELS` | config path | `models/dev` when `MODE=dev` | Cells 4, 7, 18 | Word2Vec model output directory. |
| `OUTPUTS` | config path | `outputs/dev` when `MODE=dev` | Cells 4, 5, 7, 18 | Plot and selection JSON output directory. |
| `PIPELINE_SELECTION_MODE` | config scalar | `auto` | Cell 4 | From `configs/base.yaml`. |
| `PIPELINE_VECTOR_SOURCE` | config scalar | `auto` | Cells 4, 18 | Vector source request. |
| `PIPELINE_REDUCER` | config scalar | `auto` | Cells 4, 18 | Reducer request. |
| `PIPELINE_CLUSTERER` | config scalar | `auto` | Cells 4, 18 | Clusterer request. |
| `PIPELINE_FALLBACK_VECTOR_SOURCE` | config scalar | `hybrid` | Cells 4, 18 | Fallback before benchmark selection. |
| `PIPELINE_FALLBACK_REDUCER` | config scalar | `umap` | Cells 4, 18 | Fallback reducer. |
| `PIPELINE_FALLBACK_CLUSTERER` | config scalar | `hdbscan_assigned` | Cells 4, 18 | Fallback clusterer. |
| `VECTOR_SOURCE_CANDIDATES` | config list[str] | `item2vec`, `tfidf_svd`, `hybrid` | Cells 4, 18 | Benchmark candidates. |
| `REDUCER_CANDIDATES` | config list[str] | `umap`, `pca` | Cells 4, 18 | Reducer candidates. |
| `CLUSTERER_CANDIDATES` | config list[str] | `hdbscan_assigned`, `kmeans` | Cells 4, 18 | Clusterer candidates. |
| `MODEL_SELECTION_TARGET_TRIBE_COUNT` | config int | `15` | Cells 4, 18 | Target segment count. |
| `CONFIG_TARGET_KMEANS_K`, `TARGET_KMEANS_K` | config int / int | `15` | Cells 4, 18 | Fixed-K benchmark target. |
| `FORCE_EMBEDDINGS`, `FORCE_VECTORS`, `FORCE_UMAP`, `FORCE_CLUSTERING` | bool controls | All `False` in Cell 4 | Cells 4, 7, 18 | Cache-control flags. |
| `VECTOR_SOURCE`, `REDUCER`, `CLUSTERER` | scalar controls | Initially `auto`; later `VECTOR_SOURCE=item2vec`, `REDUCER=umap`, `SELECTED_CLUSTERER=kmeans` from saved selection artifacts | Cells 4, 24, 29, 44, 55 | In-memory run controls. |
| `VECTOR_SOURCE_BENCHMARKS`, `REDUCER_BENCHMARKS`, `CLUSTERER_BENCHMARKS` | list[str] | Vector: 3 candidates; reducer: 2; clusterer: 2 | Cells 4, 18, 29 | Candidate lists. |
| `TARGET_TRIBE_COUNT` | int | `15` | Cells 4, 18 | Selection target. |
| `RUN_VECTOR_SOURCE_BENCHMARK`, `RUN_REDUCER_BENCHMARK`, `RUN_N_NEIGHBORS_SENSITIVITY`, `RUN_HDBSCAN_GRID_SEARCH`, `RUN_KMEANS_BASELINES`, `FORCE_VECTOR_SOURCE_BENCHMARK` | bool controls | In dev: vector benchmark `True`, reducer benchmark `True`, sensitivity `False`, grid `False`, kmeans baselines `True`, vector benchmark force `False` | Cells 4, 18, 29, 36, 48, 51 | Execution switches. |
| `PROD_PROCESSED` | `pathlib.Path` | `data/processed` | Cell 5 | Prod-only artifact source for `customer_kpis.parquet` and `quality_report.json`. |
| `required` | dict metadata | Cell 5 maps `Path -> notebook section`; Cell 18 maps filename -> source string | Cells 5, 18 | ⚠️ Schema mismatch - same name reused for different key/value shapes inside Notebook 3. |
| `all_ok`, `size`, `unit`, `size_disp`, `path` | local scalars | Existence and display metadata for required files | Cells 5, 18 | Local validation variables. |
| `df_articles` | Polars `DataFrame` | Product master schema, 893,944 x 4 | Cell 7 fallback via `src.data_loader.load_maestra_articulos()` | Same physical product master as Notebooks 1 and 2. |
| `PRODUCT_EMBEDDING_METHOD` | config scalar | `word2vec` | Cell 7 | Only implemented product embedding method. |
| `PRODUCT_EMBEDDING_CANDIDATES` | config list[str] | `word2vec` | Cell 7 | Product embedding candidate list. |
| `PRODUCT_IDF_WEIGHTING_ENABLED`, `PRODUCT_POPULARITY_ENABLED` | config bools | Both `true` | Cell 7 | Affect product popularity, baskets, and customer vectors. |
| `product_popularity` | Polars `DataFrame` returned by src | In-memory return has `idarticu:Int64`; `n_purchase_lines:UInt32`; `n_baskets:UInt32`; `basket_share:Float32`; `n_customers:UInt32`; `customer_share:Float32`; `raw_idf:Float32`; `idf_weight:Float32`; `is_popularity_removed:Boolean`; dev cache raw file has first 7 columns only, confirmed 78,011 x 7 | Cell 9 via `src.product_filtering.build_product_popularity()` | Important in-memory vs cache schema difference. |
| `_pop_summary` | dict metadata | `enabled`, `n_products`, `n_removed`, `removed_pct`, `max_basket_share`, `max_customer_share` | Cell 9 via `popularity_filter_summary()` | Product filtering summary. |
| `_pop_view` | Polars `DataFrame` | Top 30 products with product names, sector, basket/customer shares, IDF, removal flag | Cell 9 | Display-only view. |
| `basket_path` | `pathlib.Path` | `data/dev/basket_sentences.parquet` | Cell 9 via `build_basket_sentences()` | File path consumed by Word2Vec training. |
| `baskets` | Polars `DataFrame` | `ticket:String`; `products:List[Int64]`; confirmed dev cache 701,897 x 2 | Cell 9 reads `basket_path` | Loaded for basket-size stats only. |
| `basket_sizes` | Polars `Series` | Int list lengths for `baskets["products"]` | Cell 9 | Local distribution input. |
| `p99_size` | int | 99th percentile basket size | Cell 9 | Plot scalar. |
| `size_stats` | dict metadata | `n_baskets`, `median_size`, `mean_size`, `p25`, `p75`, `p99`, `single_item_pct` | Cell 9 | Basket-size summary. |
| `w2v_model` | gensim `Word2Vec` model | Vector size 100; skip-gram; vocabulary equals saved embedding row count when cache aligned | Cell 11 via `src.embeddings.train_word2vec()` | Saved/reloaded at `models/dev/word2vec_product.model`. |
| `product_embeddings` | Polars `DataFrame` | `idarticu:Int64`; `embedding:List[Float32 x 100]`; confirmed dev cache 56,375 x 2 | Cell 13 via `src.embeddings.save_embeddings()` | Consumed by customer vector builders. |
| `n_catalogue`, `n_transacted`, `n_with_embedding`, `n_below_min`, `n_never_bought` | ints | Product coverage counts | Cell 13 | Derived scalars. |
| `_embedded_set` | set[int] | Product ids with embeddings | Cell 15 | Probe selection helper. |
| `_pop` | Polars `DataFrame` | Embedded product basket counts plus product name/sector | Cell 15 | Probe selection helper. |
| `_target_sectors` | list[str] | `P.G.C.`, `PROD. FRESCOS TRADIC`, `BAZAR` | Cell 15 | Probe sector list. |
| `probe_ids`, `row`, `pid`, `name`, `nb` | list/scalars | Word2Vec sanity-check probes | Cell 15 | Local only. |
| `FEATURE_WEIGHT_KPI`, `FEATURE_WEIGHT_PROMO`, `FEATURE_WEIGHT_STORE` | config floats | All `0.0` | Cell 18 | Product-only baseline; optional extra features disabled. |
| `HDBSCAN_*`, `KMEANS_*`, `RECENCY_HALFLIFE_DAYS`, `UMAP_N_NEIGHBORS`, `N_NEIGHBORS_CANDIDATES`, `REDUCTION_BENCHMARK_SAMPLE`, `VECTOR_BENCHMARK_N_SAMPLE`, `RANDOM_SEED` | config values | Dev overrides: HDBSCAN min cluster size 100, min samples 1, method `leaf`, fit sample 12,000, silhouette sample 3,000; KMeans baseline `[8,10,12,15,18]`; recency half-life 60; UMAP n_neighbors 20; vector benchmark sample 12,000; seed 42 | Cells 18, 29, 36, 42 | Imported from `src.config`; values come from base plus dev override. |
| `interactions` | Polars `DataFrame` | `cliente:String`; `idarticu:Int64`; `weight:Float64`; `promo_purchases:Int32`; `total_purchases:Int32`; confirmed dev cache 4,751,838 x 5 | Cell 20 via `src.customer_vectors._build_interactions()` | Cached at `data/dev/customer_product_weights.parquet`. |
| `pairs_per_cust` | Polars `Series` | Number of products per customer | Cell 20 | Derived from `interactions`. |
| `cv_weighted` | Polars `DataFrame` | `cliente:String`; `vector:List[Float32 x 100]`; `promo_rate:Float32`; confirmed 43,999 x 3 | Cell 22 via `build_customer_vectors()` | Cached at `data/dev/customer_vectors_weighted.parquet`; one fewer row than dev subset customer count because rare-only customers lack embedded products. |
| `n_total_cust`, `n_vectorised` | ints | Unique customers in interactions and customers with vectors | Cell 22 | Coverage scalars. |
| `cv_mean` | Polars `DataFrame` | `cliente:String`; `vector:List[Float32 x 100]`; `promo_rate:Float32`; confirmed 43,999 x 3 | Cell 24 via `build_customer_vectors_mean()` | Cached at `data/dev/customer_vectors_mean.parquet`. |
| `cv_tfidf_svd` | Polars `DataFrame` | `cliente:String`; `vector:List[Float32 x 100]`; `promo_rate:Float32`; confirmed 44,000 x 3 | Cell 24 via `build_customer_vectors_tfidf_svd()` | Cached at `data/dev/customer_vectors_tfidf_svd.parquet`. |
| `share_features` | Polars `DataFrame` | `cliente:String` plus 40 Float32 product share features: sector line/spend shares, type line/spend shares, theme line/spend shares; confirmed 44,000 x 41 | Cell 24 via `build_customer_share_features()` | Cached at `data/dev/customer_product_share_features.parquet`. |
| `cv_hybrid` | Polars `DataFrame` | `cliente:String`; `vector:List[Float32 x 240]`; `promo_rate:Float32`; confirmed 43,999 x 3 | Cell 24 via `build_customer_vectors_hybrid()` | Product-only hybrid: 100 item2vec + 100 tfidf_svd + 40 share features. |
| `cv_hybrid_with_store` | Polars `DataFrame` or `None` | If built: `cliente`, `vector`, `promo_rate`; dims inferred 244 with current 4 store shares | Cell 24 via `build_customer_vectors_hybrid_with_store()` | Not built in the existing artifact inventory because `hybrid_with_store` is not in candidates. |
| `cv_candidates` | dict[str, Polars `DataFrame`] | Keys: `item2vec`, `mean`, `tfidf_svd`, `hybrid`; optional `hybrid_with_store` | Cell 24 | Selection input. |
| `REQUESTED_VECTOR_SOURCE` | str | `auto` | Cell 24 | Before benchmark selection. |
| `SELECTED_VECTOR_SOURCE` | str | Initially fallback `hybrid`, then persisted selected `item2vec` after Cell 29 | Cells 24, 29 | Saved vector-source artifact confirms `item2vec`. |
| `cv_for_umap` | Polars `DataFrame` | Selected customer-vector frame; after Cell 29 current contract is `cv_weighted` with 43,999 rows x 100 dims | Cells 24, 29 | Input to reducers. |
| `_common`, `_v_w`, `_v_m`, `_cos_sim`, `_l2_dist`, `_promo`, `never_promo`, `high_promo` | local comparison variables | Sample of up to 10,000 customers; numpy arrays and promo scalars | Cell 26 | Plot-only / diagnostic. |
| `_target_clusters`, `_vector_selection_path`, `out_path`, `_bmark_n`, `_full_n`, `_scale`, `_bmark_mcs`, `_bmark_grid` | scalar/path/dict metadata | Vector benchmark configuration; `_bmark_grid` contains sampled min cluster sizes, min samples `[1,5]`, cluster method `leaf` | Cell 29 | Benchmark metadata. |
| `vector_source_results` | list[dict] | Rows later persisted to `vector_source_comparison` | Cell 29 | Benchmark result accumulator. |
| `vector_source_artifacts` | dict[str, dict] | For each source: `umap`, `grid`, `core_labels`, `assigned_labels` DataFrames | Cell 29 | In-memory artifact bundle. |
| `bmark_umap_df` | Polars `DataFrame` | `cliente`, `u0` through `u49`, `promo_rate`; benchmark sample 12,000 x 52 | Cell 29 via `reduce_umap_cluster()` | Cached per source as `umap_cluster_{source}_bmark.parquet`. |
| `grid_df` | Polars `DataFrame` | `min_cluster_size`, `min_samples`, `cluster_method`, `n_clusters`, `target_gap`, `noise_pct`, `silhouette`, `davies_bouldin`; 10 x 8 per source | Cell 29 via `grid_search_hdbscan()` | Cached per source. |
| `best` | dict metadata | Best sampled HDBSCAN grid row | Cell 29 | Selected params for benchmark labels. |
| `core_labels` | Polars `DataFrame` | `cliente:String`; `cluster:Int32`; `promo_rate:Float32`; benchmark 12,000 x 3 | Cell 29 via `cluster_hdbscan()` | Per-source benchmark labels. |
| `assigned_labels` | Polars `DataFrame` | `cliente:String`; `cluster:Int32`; `hdbscan_cluster:Int32`; `was_hdbscan_noise:Bool`; `assignment_source:String`; `assignment_distance:Float32`; `promo_rate:Float32`; benchmark 12,000 x 7 | Cell 29 via `assign_hdbscan_noise_to_nearest_tribe()` | Per-source benchmark assigned labels. |
| `core_metrics`, `assigned_metrics` | dict metadata | `method`, `n_clusters`, `noise_pct`, `silhouette`, `davies_bouldin` | Cell 29 via `evaluate_clustering()` or fallback | Benchmark metrics. |
| `runtime_min` | float | Per-source benchmark runtime minutes | Cell 29 | Local scalar. |
| `vector_source_comparison` | pandas `DataFrame` and parquet | `vector_source`; `vector_dims`; `benchmark_sample`; HDBSCAN best params; grid/core/assigned metrics; `runtime_min`; `target_gap`; confirmed parquet 3 x 20 | Cell 29 | Saved to `data/dev/vector_source_hdbscan_comparison.parquet`. |
| `_winner` | dict metadata | Selected row plus `selected` | Cells 29, 44, 55 | Reused for vector, reducer, clusterer selection. |
| `_artifact` | dict or `None` | Loaded persisted vector-source selection if benchmark skipped | Cell 29 | Fallback path. |
| `store_feats` | Polars `DataFrame` | `cliente:String`; `spend_share_s6:Float32`; `spend_share_s7:Float32`; `spend_share_s2:Float32`; `spend_share_s11:Float32`; confirmed 44,000 x 5 | Cell 31 via `build_store_features()` | Cached at `data/dev/customer_store_features.parquet`; profile-only under current feature weights. |
| `store_cols`, `store_arr`, `n_stores_active`, `pct_any`, `pct_dominant`, `vals` | local variables | Store feature column names, numpy array, usage counts, display scalars | Cell 31 | Diagnostics. |
| `REQUESTED_REDUCER`, `_reducer_candidates`, `reduction_artifacts` | scalar/list/dict | Requested reducer `auto`; candidates include `umap`, `pca`; dict maps reducer name to embedding `DataFrame` | Cell 34 | Reducer selection inputs. |
| `umap_cluster` | Polars `DataFrame` | `cliente:String`; `u0` through `u49:Float32`; `promo_rate:Float32`; confirmed selected dev cache 43,999 x 52 | Cell 34 via `reduce_umap_cluster()` | Cached at `data/dev/umap_cluster_item2vec.parquet`. |
| `_dim_count` | int | UMAP/PCA dimension count, 50 under current config | Cells 34, 44 | Local display scalar. |
| `RUN_N_NEIGHBORS_SENSITIVITY`, `_SENS_SAMPLE`, `_rng`, `_sidx`, `_X_sens`, `_nn_candidates`, `_sens_rows`, `_mcs_sample`, `_sens_pd`, `_t`, `_reducer`, `_emb`, `_cl`, `_n_organic`, `_noise`, `_secs`, `_selected` | local sensitivity variables | Sample, arrays, labels, metrics for optional n_neighbors sensitivity | Cell 36 | Skipped when `RUN_N_NEIGHBORS_SENSITIVITY=False`. |
| `pca_cluster` | Polars `DataFrame` or `None` | If built: `cliente:String`; `pc0` through `pc49:Float32`; `promo_rate:Float32`; confirmed dev cache 43,999 x 52 | Cell 38 via `reduce_pca()` | Cached at `data/dev/pca_cluster_item2vec.parquet`. |
| `pca_model` | sklearn `PCA` object or `None` | Fitted PCA with 50 components; saved pickle | Cell 38 via `reduce_pca()` | Saved at `data/dev/pca_model_item2vec.pkl`. |
| `explained` | numpy array | Cumulative explained variance ratios | Cell 38 | PCA diagnostic. |
| `umap_viz` | Polars `DataFrame` | `cliente:String`; `x:Float32`; `y:Float32`; `promo_rate:Float32`; confirmed 43,999 x 4 | Cells 40, 44 via `reduce_umap_viz()` | Cached at preview and selected-space parquet paths. |
| `_sample` | Polars `DataFrame` | Sample up to 100,000 rows from `umap_viz` | Cell 40 | Plot-only. |
| `UMAP_EVAL_SAMPLE`, `UMAP_EVAL_K` | ints | `min(5000, len(cv_for_umap))`, `30` | Cell 42 | UMAP preservation diagnostics. |
| `_umap_cols`, `_eval`, `_X_parts`, `_X_original`, `_X_umap`, `_X_viz`, `_k`, `_orig_nn`, `_umap_nn`, `_viz_nn`, `_quality`, `x`, `width`, `_store_df`, `_store_cols`, `_store_eval`, `_kpi_path`, `_kpi_cols`, `_kpi_eval`, `nn`, `idx` | diagnostic variables | Evaluation DataFrames, numpy arrays, nearest-neighbor indices, quality table | Cell 42 | Optional promo/store/KPI inputs are appended only if feature weights are greater than zero. |
| `_REDUCER_SELECTION_PATH` | `pathlib.Path` | `outputs/dev/model_selection/reducer_selection.json` | Cell 44 | Selection JSON path. |
| `_K` | int | Target K for quick silhouette, 15 | Cell 44 | Reducer benchmark scalar. |
| `reduction_rows` | list[dict] | Reducer benchmark rows | Cell 44 | Converted to `reduction_comparison`. |
| `reduction_comparison` | pandas `DataFrame` | `reducer`, `dims`, `silhouette`, `davies_bouldin`; saved artifact winner selected `umap` | Cell 44 | Not persisted as parquet. |
| `SELECTED_REDUCER`, `REDUCER` | str | `umap` | Cell 44 | Saved reducer selection artifact confirms `umap`. |
| `cluster_space` | Polars `DataFrame` | Selected reducer output; current contract is `umap_cluster` 43,999 x 52 | Cell 44 | Input to clustering. |
| `_cluster_dim_count`, `dim_cols`, `X`, `km`, `lbl`, `rng`, `idx`, `_sil` | local reducer-benchmark variables | Numeric arrays/model/labels for quick silhouette | Cell 44 | Local only. |
| `_hdbscan_run_params` | dict metadata | Current selected params: saved run suffix indicates `min_cluster_size=136`, `min_samples=1`, `cluster_method=leaf`; config fallback is dev `100`, `1`, `leaf` | Cell 48 | May come from vector-source sampled benchmark when full grid skipped. |
| `hdb_grid` | Polars `DataFrame` or `None` | Full HDBSCAN grid result if run; same schema as grid caches | Cell 48 | Currently skipped unless flag true. |
| `_hdb_suffix` | str | `{VECTOR_SOURCE}_{SELECTED_REDUCER}_mcs..._ms..._{method}` | Cell 48 | Used in label cache filenames. |
| `hdb_core_labels` | Polars `DataFrame` | `cliente:String`; `cluster:Int32`; `promo_rate:Float32`; confirmed selected dev cache 43,999 x 3 | Cell 48 via `cluster_hdbscan()` | Cached at `data/dev/cluster_labels_hdbscan_item2vec_umap_mcs136_ms1_leaf.parquet`. |
| `hdb_labels` | Polars `DataFrame` alias | Same object as `hdb_core_labels` | Cell 48 | Alias only. |
| `n_tribes`, `noise_n`, `noise_pct` | scalar metrics | HDBSCAN non-noise tribe count and noise volume | Cell 48 | Used in plots and scorecard. |
| `hdb_assigned_labels` | Polars `DataFrame` | `cliente:String`; `cluster:Int32`; `hdbscan_cluster:Int32`; `was_hdbscan_noise:Bool`; `assignment_source:String`; `assignment_distance:Float32`; `promo_rate:Float32`; confirmed selected dev cache 43,999 x 7 | Cell 48 via `assign_hdbscan_noise_to_nearest_tribe()` | Operational all-customer HDBSCAN-derived labels. |
| `dev_n`, `_PROD_N_CUSTOMERS`, `_scale`, `best_row`, `dev_mcs`, `dev_ms`, `dev_cm`, `prod_mcs` | local scalars | Dev-to-prod HDBSCAN parameter translation | Cell 49 | Only active if full grid search ran. `_PROD_N_CUSTOMERS=1082000` is hardcoded and slightly lower than current quality report eligible customers 1,086,761. |
| `kmeans_results` | Polars `DataFrame` | `method:String`; `requested_k:Int64`; `n_clusters:Int64`; `noise_pct:Float64`; `silhouette:Float64`; `davies_bouldin:Float64`; confirmed 5 x 6 | Cell 51 via `run_kmeans_baselines()` | Cached at `data/dev/kmeans_baseline_results_item2vec_umap.parquet`. |
| `km_labels` | Polars `DataFrame` | `cliente:String`; `cluster:Int32`; `promo_rate:Float32`; confirmed selected K=15 cache 43,999 x 3 | Cell 51 via `cluster_kmeans()` | Cached at `data/dev/cluster_labels_kmeans_item2vec_umap_k15.parquet`. |
| `_fmt_lift_list`, `_compact_profile_table`, `_series_or_empty` | local functions | Format profile list columns into compact pandas display | Cell 53 | Display helpers. |
| `hdb_assigned_profile` | Polars `DataFrame` | `cluster:Int32`; `n_customers:UInt32`; `avg_basket:Float32`; `avg_visits:Float32`; `total_revenue:Float64`; `avg_promo_rate:Float32`; `revenue_share:Float32`; `top_sectors:List[String]`; `top_sector_lifts:List[Float64]`; `top_themes:List[String]`; `top_theme_lifts:List[Float64]`; `top_products:List[String]`; `top_lifts:List[Float64]`; confirmed 11 x 13 | Cell 53 via `profile_tribes()` | Cached at `data/dev/tribe_profiles_hdbscan_assigned.parquet`; uses prod `customer_kpis.parquet` fallback if dev KPI cache absent. |
| `kmeans_profile_summaries` | dict[int, Polars `DataFrame`] | Keys from KMeans baseline K values; each value uses profile schema above | Cells 53, 60 | Existing caches for K=8,10,12,15,18. |
| `labels_k`, `profiles_k` | Polars `DataFrame` | K-specific KMeans labels and tribe profiles | Cell 53 | Loop-local but cached by K. |
| `pdf`, `total_customers`, `out`, `items`, `lifts`, `label`, `parts` | local display variables | Compact profile table inputs | Cell 53 | Display-only. |
| `results_hdb_core`, `results_hdb_assigned`, `results_km` | dict metrics | `method`, `n_clusters`, `noise_pct`, `silhouette`, `davies_bouldin` | Cell 55 via `evaluate_clustering()` | Saved output: core 11 clusters, 58.74 percent noise; assigned 11 clusters; KMeans K=15 selected. |
| `_cmp_rows` | list[dict] | Candidate metric rows plus `candidate` | Cell 55 | Converted to `selected_cmp`. |
| `selected_cmp` | pandas `DataFrame` | `method`, `n_clusters`, `noise_pct`, `silhouette`, `davies_bouldin`, `candidate`, `coverage_pct`, `coverage_adj_silhouette` | Cell 55 | Clusterer selection input. |
| `_CLUSTERER_SELECTION_PATH` | `pathlib.Path` | `outputs/dev/model_selection/clusterer_selection.json` | Cell 55 | Selection JSON path. |
| `_cluster_winner` | dict metadata | Winning row plus `selected` | Cell 55 | Persisted in clusterer selection JSON. |
| `SELECTED_CLUSTERER` | str | `kmeans` | Cell 55 | Saved clusterer selection artifact confirms KMeans. |
| `valid_clusterers`, `_match` | set / pandas row frame | Validation helpers | Cell 55 | Local only. |
| `champion_labels` | Polars `DataFrame` | Current selected label schema; because `SELECTED_CLUSTERER="kmeans"`, same schema as `km_labels` | Cell 55 | In-memory final label artifact; not separately persisted. |
| `_viz_pd` | pandas `DataFrame` | `cliente`, `x`, `y`, `promo_rate`, `tribe_hdbscan_core`, `tribe_hdbscan_assigned`, optional `tribe_kmeans`; sampled up to 100,000 rows | Cell 57 | Visualization input. |
| `_n_panels`, `_title` | int / str | Plot layout and title | Cell 57 | Local only. |
| `candidate_rows` | list[dict] | Candidate scorecard row accumulator | Cell 60 | Converted to `candidate_scorecard`. |
| `_hdb_noise`, `_hdb_assigned_noise` | floats | Original HDBSCAN noise and assigned-from-noise share | Cell 60 | Candidate scorecard inputs. |
| `candidate_scorecard` | pandas `DataFrame` | `candidate`, `role`, `n_clusters`, `original_hdbscan_noise_pct`, `assigned_from_noise_pct`, `silhouette`, `coverage_adj_silhouette`, `davies_bouldin`, `min_cluster_share_pct`, `max_cluster_share_pct`, `clusters_with_theme_lift`, `clusters_with_sector_lift`, `review_note` | Cell 60 | Display-only selection summary. |
| `metrics`, `sizes`, `total`, `size_shares`, `theme_lists`, `sector_lists`, `nonempty_themes`, `nonempty_sectors`, `min_share`, `max_share`, `sil`, `noise_for_penalty`, `coverage_adj_sil`, `review_note` | local profile-quality variables | Scalars/lists used by `_profile_quality_row()` | Cell 60 | Local only. |

### Consumes
| Artifact | Expected Type | Expected Schema / Shape | Source |
|----------|---------------|------------------------|--------|
| `data/dev/df_combined.parquet` | Parquet | Clean transaction schema, confirmed 7,511,379 x 13 | Produced by Notebook 2 Section 2.12 through `src.generate_dev_subset.generate()` |
| `data/processed/df_combined.parquet` | Parquet | Clean production transaction schema, confirmed 190,362,519 x 13 | Produced by Notebook 2 Section 2.3; source for dev subset and fallback/prod runs |
| `data/processed/customer_kpis.parquet` | Parquet | `cliente`, `visit_count`, `avg_basket_size`, `total_spend_6m`, `avg_promo_rate`, `unique_products`; confirmed 1,482,715 x 6 | Produced by Notebook 2 Section 2.8; required by Notebook 3 startup and `profile_tribes()` fallback |
| `data/processed/quality_report.json` | JSON metadata | Data quality report with 8 checks passed | Produced by Notebook 2 Section 2.2; required by Notebook 3 startup |
| `data/raw/parquet/maestra_articulos.parquet` | Parquet | Product master schema, 893,944 x 4 | Produced/verified by Notebook 1; loaded in Notebook 3 if `df_articles` absent |
| `src.config` base and dev values | config module | Base values plus `configs/dev.yaml` overrides because Notebook 3 sets `CARREFOUR_MODE=dev` | `src/config.py`, `configs/base.yaml`, `configs/dev.yaml` |
| `src.embeddings.*` functions | src functions | Return basket path, Word2Vec model, product embeddings | `src/embeddings.py` |
| `src.product_filtering.*` functions | src functions | Return product popularity table and filtering summary | `src/product_filtering.py` |
| `src.customer_vectors.*` functions | src functions | Return interactions, customer vectors, share features, store features | `src/customer_vectors.py` |
| `src.dimensionality.*` functions | src functions | Return UMAP/PCA embeddings and PCA model | `src/dimensionality.py` |
| `src.clustering.*` functions | src functions | Return labels, grid results, metrics, and tribe profiles | `src/clustering.py` |
| `src.model_selection.*` functions | src functions | Return or persist selection metadata dicts | `src/model_selection.py` |
| `src.product_themes.STRATEGIC_THEME_PATTERNS` | dict metadata | Theme regex groups: baby, pet, organic_bio, gluten_free, lactose_free, protein_fitness, health_wellness, plant_based, ethnic_international, fresh, ready_meals, premium_indulgence | Used by customer share features and tribe profiles |
| Existing Notebook 3 caches | Parquet/model/JSON | Product embeddings, vectors, reducer outputs, labels, profiles, model-selection JSONs | Notebook 3 cache-if-exists behavior through src functions |

---

## Shared Artifacts — Cross-Notebook Contract

> High-risk variables. Any change here must be validated against every notebook
> in the "Used by" column before merging.

| Artifact | Type | Contract (schema / shape / value) | Defined in | Used by | Link type |
|----------|------|-----------------------------------|------------|---------|-----------|
| `data/raw/parquet/maestra_articulos.parquet` | Parquet | `idarticu:Int64`; `desc_larga_articulo:String`; `idsector:Int64`; `desc_sector:String`; 893,944 x 4 | Notebook 1 / `src.data_loader.convert_csv_to_parquet()` | Notebook 1 `df_articulos`; Notebook 2 `df_articles`; Notebook 3 `df_articles` fallback and sanity checks | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\raw\parquet\maestra_articulos.parquet)` |
| `data/raw/parquet/linea_tickets.parquet` | Parquet | `idempres:Int64`; `fecha:Date`; `hora:Int64`; `ticket:String`; `cliente:String`; `idarticu:Int64`; `unidades:Int64`; `importe:Float64`; `idpromoc:String`; `idtiprod:Int64`; 191,017,715 x 10 | Notebook 1 / `src.data_loader.convert_csv_to_parquet()` | Notebook 1 raw exploration; Notebook 2 quality gates and clean join | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\raw\parquet\linea_tickets.parquet)` |
| `df_articulos` / `df_articles` | Polars `DataFrame` | Product master schema above | Notebook 1 as `df_articulos`; Notebook 2 and 3 as `df_articles` | Notebook 1 merge preview; Notebook 2 clean join; Notebook 3 embedding sanity check | `src/ function (load_maestra_articulos)` |
| `df_tickets` | Polars `LazyFrame` | Raw ticket schema above | Notebook 1 via direct `pl.scan_parquet`; Notebook 2 via `load_linea_tickets()` | Notebook 1 exploration; Notebook 2 quality and clean join | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\raw\parquet\linea_tickets.parquet)` |
| `quality_report.json` / `report` | JSON dict | Keys: `null_audit`, `anomaly_audit`, `schema_validation`, `product_coverage`, `temporal_completeness`, `customer_activity`, `promotional_integrity`, `store_integrity`, `_summary`; `_summary.all_passed=true`; `rows_retained=190362519`; `eligible_customers=1086761` | Notebook 2 Cell 8 via `build_quality_report()` | Notebook 2 summary cells; Notebook 3 startup check and hardcoded prod-population context | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\processed\quality_report.json)` |
| `df_combined.parquet` production | Parquet | Clean transaction schema: raw 10 columns plus `desc_larga_articulo:String`, `idsector:Int64`, `desc_sector:String`; 190,362,519 x 13; no null sectors/names | Notebook 2 Cell 14 | Notebook 2 product/customer EDA caches; `src.generate_dev_subset.generate()`; Notebook 3 prod mode if enabled | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\processed\df_combined.parquet)` |
| `df_combined.parquet` dev subset | Parquet | Same clean transaction schema as prod; 7,511,379 x 13; 44,000 sampled customers | Notebook 2 Cell 38 via `src.generate_dev_subset.generate()` | Notebook 3 dev-mode embeddings, vectors, reducers, clustering | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\dev\df_combined.parquet)` |
| `subset_metadata.json` | JSON dict | `target_size=44000`; `actual_size=44000`; `n_transaction_rows=7511379`; hashes; stratification metadata; KS validation; source and output file stats | Notebook 2 Cell 38 via `generate()` | Notebook 3 implicit validation of dev subset provenance | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\dev\subset_metadata.json)` |
| `customer_kpis.parquet` | Parquet | `cliente:String`; `visit_count:UInt32`; `avg_basket_size:Float64`; `total_spend_6m:Float64`; `avg_promo_rate:Float64`; `unique_products:UInt32`; 1,482,715 x 6 | Notebook 2 Cell 26 | Notebook 2 EDA summaries; Notebook 3 startup check; `profile_tribes()` profile metrics; optional KPI feature appending | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\processed\customer_kpis.parquet)` |
| `MODE` / `DATA_PROCESSED` | config scalars | Notebook 2 saved run: `MODE=prod`, `DATA_PROCESSED=data/processed`; Notebook 3 master cell: `MODE=dev`, `DATA_PROCESSED=data/dev` | `src.config`, Notebook 3 Cell 4 environment override | Notebook 2 artifact location; Notebook 3 artifact location | `src/ function (_load_config)` |
| `product_embeddings.parquet` | Parquet | `idarticu:Int64`; `embedding:List[Float32 x 100]`; dev cache 56,375 x 2 | Notebook 3 Cell 13 via `save_embeddings()` | Notebook 3 customer vector builders | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\dev\product_embeddings.parquet)` |
| `word2vec_product.model` | gensim `Word2Vec` model | Vector size 100; skip-gram; saved/reloaded from disk | Notebook 3 Cell 11 via `train_word2vec()` | Notebook 3 `save_embeddings()` and sanity checks | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\models\dev\word2vec_product.model)` |
| `customer_vectors_weighted.parquet` / `cv_weighted` | Parquet / Polars `DataFrame` | `cliente:String`; `vector:List[Float32 x 100]`; `promo_rate:Float32`; dev cache 43,999 x 3 | Notebook 3 Cell 22 via `build_customer_vectors()` | Notebook 3 vector comparison, candidate dict, selected `item2vec` vector source | `src/ function (build_customer_vectors)` |
| `customer_vectors_tfidf_svd.parquet` / `cv_tfidf_svd` | Parquet / Polars `DataFrame` | `cliente:String`; `vector:List[Float32 x 100]`; `promo_rate:Float32`; dev cache 44,000 x 3 | Notebook 3 Cell 24 via `build_customer_vectors_tfidf_svd()` | Notebook 3 vector benchmark and hybrid vectors | `src/ function (build_customer_vectors_tfidf_svd)` |
| `customer_product_share_features.parquet` / `share_features` | Parquet / Polars `DataFrame` | `cliente:String` plus 40 Float32 sector/type/theme share features; dev cache 44,000 x 41 | Notebook 3 Cell 24 via `build_customer_share_features()` | Notebook 3 hybrid vector construction | `src/ function (build_customer_share_features)` |
| `umap_cluster_item2vec.parquet` / `cluster_space` | Parquet / Polars `DataFrame` | `cliente:String`; `u0` through `u49:Float32`; `promo_rate:Float32`; dev cache 43,999 x 52 | Notebook 3 Cell 34/44 via `reduce_umap_cluster()` | Notebook 3 clustering, labels, 2D visualization, metrics | `src/ function (reduce_umap_cluster)` |
| `umap_viz_item2vec_umap_2d.parquet` / `umap_viz` | Parquet / Polars `DataFrame` | `cliente:String`; `x:Float32`; `y:Float32`; `promo_rate:Float32`; dev cache 43,999 x 4 | Notebook 3 Cell 44 via `reduce_umap_viz()` | Notebook 3 cluster visualization | `src/ function (reduce_umap_viz)` |
| `hdb_core_labels` | Polars `DataFrame` / Parquet | `cliente:String`; `cluster:Int32`; `promo_rate:Float32`; dev selected cache 43,999 x 3 | Notebook 3 Cell 48 via `cluster_hdbscan()` | Notebook 3 assigned labels, metrics, visualization | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\dev\cluster_labels_hdbscan_item2vec_umap_mcs136_ms1_leaf.parquet)` |
| `hdb_assigned_labels` | Polars `DataFrame` / Parquet | `cliente:String`; `cluster:Int32`; `hdbscan_cluster:Int32`; `was_hdbscan_noise:Bool`; `assignment_source:String`; `assignment_distance:Float32`; `promo_rate:Float32`; dev selected cache 43,999 x 7 | Notebook 3 Cell 48 via `assign_hdbscan_noise_to_nearest_tribe()` | Notebook 3 profiles, metrics, visualization, scorecard | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\dev\cluster_labels_hdbscan_assigned_item2vec_umap_mcs136_ms1_leaf.parquet)` |
| `km_labels` / `champion_labels` | Polars `DataFrame` / Parquet | `cliente:String`; `cluster:Int32`; `promo_rate:Float32`; current selected K=15 cache 43,999 x 3 | Notebook 3 Cell 51 and selected in Cell 55 | Notebook 3 final selected labels, profiles, visualization | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\data\dev\cluster_labels_kmeans_item2vec_umap_k15.parquet)` |
| `tribe_profiles_*` | Parquet / Polars `DataFrame` | `cluster`; `n_customers`; `avg_basket`; `avg_visits`; `total_revenue`; `avg_promo_rate`; `revenue_share`; list columns for top sectors/themes/products and lifts | Notebook 3 Cell 53 via `profile_tribes()` | Notebook 3 compact profiles and scorecard | `src/ function (profile_tribes)` |
| `vector_source_selection.json` | JSON dict | `component=vector_source`; `requested=auto`; `selected=item2vec`; winner metrics; comparison path | Notebook 3 Cell 29 via `write_selection_artifact()` | Notebook 3 reruns and downstream `VECTOR_SOURCE` | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\outputs\dev\model_selection\vector_source_selection.json)` |
| `reducer_selection.json` | JSON dict | `component=reducer`; `requested=auto`; `selected=umap`; winner dims 50, silhouette 0.2793 | Notebook 3 Cell 44 via `write_selection_artifact()` | Notebook 3 reruns and downstream `REDUCER` | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\outputs\dev\model_selection\reducer_selection.json)` |
| `clusterer_selection.json` | JSON dict | `component=clusterer`; `requested=auto`; `selected=kmeans`; winner K=15, silhouette 0.2834 | Notebook 3 Cell 55 via `write_selection_artifact()` | Notebook 3 final selected `champion_labels` | `file-mediated (C:\Users\sebog\proyectos\carrefour_capstone\outputs\dev\model_selection\clusterer_selection.json)` |
| `stats` | local variable name collision | Notebook 1: Polars 1 x 6 raw dataset metric table. Notebook 2: dict with clean join metrics. | Notebook 1 Cell 21; Notebook 2 Cell 13 | Local only | `in-memory` |
| `product_popularity` | in-memory vs cache schema collision | In-memory return has 9 columns with `idf_weight` and `is_popularity_removed`; parquet cache stores 7 raw columns and is decorated on read | Notebook 3 Cell 9 via `build_product_popularity()` | Notebook 3 baskets, vectors, popularity view | `src/ function (build_product_popularity)` |

---

## Risk Summary

The riskiest shared artifact is `df_combined.parquet`, especially the pair of production and dev-subset files. It is the contract boundary between data preparation and the full ML pipeline: `cliente`, `ticket`, `idarticu`, `fecha`, `idpromoc`, `importe`, `idtiprod`, `desc_larga_articulo`, `idsector`, and `desc_sector` are all consumed downstream by quality summaries, product demand caches, customer KPIs, dev sampling, basket sentences, recency weights, promo rates, share features, embeddings, vectors, clustering profiles, and final labels. A dtype change in `fecha`, a renamed `cliente` or `idarticu`, missing product descriptors, or a shift from prod to dev paths without regenerating dependent caches would break or silently skew almost every Notebook 3 phase.
