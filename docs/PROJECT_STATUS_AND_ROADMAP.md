# Project Status and Product-First Roadmap

Status date: 2026-06-05.

This document explains where the Carrefour segmentation project stands, what the current pipeline does, what is still missing, and how to improve the pipeline so the final tribes are meaningful because of the products customers buy.

## Executive Summary

The repo is no longer just an EDA scaffold. Production preprocessing is complete, and the full ML pipeline has been exercised in dev mode on a stratified 44k-customer subset.

The main remaining risk is not "can we run clustering?" We can. The risk is whether the resulting clusters are commercially meaningful product segments. The next work should therefore focus on product-led validation: top-product lift, product-only ablations, better customer vectorization, and temporal stability.

## Current State

### Completed

| Area | Current evidence |
|---|---|
| Raw data conversion | CSV-to-Parquet utilities exist in `src/data_loader.py`. |
| Data quality | `quality_report.json` passes 8/8 checks. |
| Production preprocessing | `df_combined.parquet` and `customer_kpis.parquet` exist in `data/processed/`. |
| Product EDA | Product demand, pair-lift, temporal-lift, promo-sector, and customer-product repeat tables exist in `data/processed/`. |
| Dev subset | 44,000 customers, 7,490,843 transaction rows, all KS validation checks passed. |
| Phase 1 | Basket sentences and Word2Vec product embeddings generated in dev mode. |
| Phase 2 | Weighted and mean customer vectors generated in dev mode. |
| Phase 3 | UMAP and PCA embeddings generated in dev mode. |
| Phase 4 | HDBSCAN, HDBSCAN grid, and K-Means baselines generated in dev mode. |
| Visual outputs | Dev plots exist for embeddings, customer vectors, UMAP/PCA, and clustering comparison. |

### Not Yet Complete

| Gap | Why it matters |
|---|---|
| Production ML run | We have dev proof, but not final 1.48M-customer segmentation artifacts. |
| Tribe profiles | `profile_tribes()` exists but cached profile outputs are not present. |
| Tribe naming | No implemented LLM/manual naming layer yet. |
| Product-first model selection | Current metrics are geometric; final choice needs product-lift interpretation. |
| UMAP input ablation | Current UMAP input includes product vectors plus promo and store features. We need to know whether promo/store are helping or dominating. |
| Customer vector diagnostics | Weighted mean may still blur product signals; we need tests against alternatives. |
| Temporal segmentation | Time windows are declared in config but not implemented as a pipeline. |
| Interactive deliverable | Static plots exist; no product-like interactive segmentation explorer yet. |

## Current Pipeline

```text
Raw CSVs
-> raw Parquet conversion
-> data quality report
-> cleaned joined transaction/product table
-> customer KPIs and product EDA
-> stratified dev subset
-> basket sentences
-> Word2Vec product embeddings
-> customer product weights
-> weighted customer vectors + mean baseline + store features
-> UMAP clustering embedding + UMAP 2D visualization + PCA baseline
-> HDBSCAN grid + configured HDBSCAN + fixed-K K-Means baselines
-> clustering plots
-> tribe profiles and tribe names (next)
```

## Dev Run Snapshot

| Artifact | Rows |
|---|---:|
| `basket_sentences.parquet` | 709,420 |
| `product_embeddings.parquet` | 55,974 |
| `customer_product_weights.parquet` | 4,741,680 |
| `customer_vectors_weighted.parquet` | 43,998 |
| `customer_vectors_mean.parquet` | 43,998 |
| `customer_store_features.parquet` | 44,000 |
| `umap_cluster_20d.parquet` | 43,998 |
| `umap_viz_2d.parquet` | 43,998 |
| `pca_cluster_20d.parquet` | 43,998 |
| `cluster_labels_hdbscan.parquet` | 43,998 |

Note: `umap_cluster_20d.parquet` and `pca_cluster_20d.parquet` are historical filenames. The active config currently uses 50 clustering dimensions.

### Dev Clustering Results

Configured HDBSCAN:

| Cluster | Customers |
|---:|---:|
| 0 | 27,708 |
| 2 | 7,490 |
| 1 | 3,544 |
| 3 | 3,413 |
| -1 noise | 1,843 |

Metrics:

| Method | Clusters | Noise | Silhouette | Davies-Bouldin |
|---|---:|---:|---:|---:|
| HDBSCAN configured | 4 | 4.2% | 0.6325 | 0.4591 |
| K-Means K=8 | 8 | 0.0% | 0.4301 | 0.8369 |
| K-Means K=12 | 12 | 0.0% | 0.3561 | 0.9762 |
| K-Means K=15 | 15 | 0.0% | 0.3599 | 0.9751 |
| K-Means K=18 | 18 | 0.0% | 0.3460 | 1.0108 |

Interpretation: HDBSCAN looks geometrically cleaner, but 4 tribes may be too coarse for a product-first business story. K=8 may be worth profiling because it could produce more actionable segments despite lower silhouette.

## Temporal Pipeline

The config already defines three windows:

| Window | Dates |
|---|---|
| `W1_jan_feb` | 2022-01-01 to 2022-02-28 |
| `W2_mar_apr` | 2022-03-01 to 2022-04-30 |
| `W3_may_jun` | 2022-05-01 to 2022-06-30 |

These windows are not yet wired into the ML pipeline.

Recommended temporal approach:

1. Train one global product embedding space on all baskets so product coordinates are comparable across time.
2. Build customer vectors separately for Jan-Feb, Mar-Apr, and May-Jun using the same product embeddings.
3. Keep only customers with enough activity per window for reliable comparison, and separately profile new/lapsed customers.
4. Run two analyses:
   - Window-by-window clustering, then align tribes by top-product lift.
   - One lifecycle vector per customer: `[W1 vector, W2 vector, W3 vector, W2-W1 delta, W3-W2 delta]`.
5. Produce transition matrices showing which product-led tribes are stable, seasonal, or migrating.

This adds a stronger product story: not just "who are the tribes?", but "which product affinities are stable, and which are seasonal or emerging?"

## Missing Work Before Final Push/Pitch

1. Generate tribe profiles for HDBSCAN and the strongest K-Means baselines.
   - Start with `profile_tribes(hdb_labels, method_name="hdbscan")`.
   - Also profile `kmeans_k8` and any other candidate that looks commercially readable.

2. Compare clusters by product lift, not only metrics.
   - Look for top products that are distinctive, not merely common staples.
   - Add sector/category rollups beside product names.
   - Flag clusters where top products are incoherent or dominated by store/promo artifacts.

3. Run product-only UMAP ablations.
   - Current feature weights: `promo=1.0`, `store=1.0`, `kpi=0.0`.
   - Test `promo=0.0`, `store=0.0`, `kpi=0.0` as the product-only baseline.
   - Then test a small ladder such as store weights `0.0`, `0.25`, `1.0` and promo weights `0.0`, `0.25`, `1.0`.

4. Improve customer vectorization if profiles are weak.
   - Weighted mean vectors may still smooth away the products that make a customer distinctive.
   - Test alternative weighting before committing to final segmentation.

5. Run the chosen pipeline in prod mode.
   - Keep dev as the tuning loop.
   - Only run prod once the product-first selection criteria are clear.

6. Build final naming and presentation artifacts.
   - Tribe names.
   - One-paragraph business descriptions.
   - Top products by lift.
   - Segment size, revenue share, promo sensitivity, visit rate, basket size.
   - Static and ideally interactive UMAP map.

## Product-First Improvement Plan

### 1. Tune Hyperparameters Against Product Meaning

Do not optimize only silhouette. Use a model scorecard:

| Criterion | Why it matters |
|---|---|
| Product lift clarity | Top products should explain why the tribe exists. |
| Product coherence | Top products should form a recognizable mission, category, or lifestyle behavior. |
| Cluster size | Tribes should be large enough for business action but not so broad they lose meaning. |
| Stability | Similar segments should appear across seeds, dev/prod, and time windows. |
| Holdout behavior | A tribe should help predict future product mix or basket mission. |
| Business actionability | The segment should suggest a plausible targeting, assortment, or promo action. |

Parameters to tune:

- Word2Vec: `vector_size`, `window`, `min_count`, `epochs`, skip-gram vs CBOW.
- Customer vectors: recency half-life, frequency transform, product weighting.
- UMAP: `n_neighbors`, `min_dist_cluster`, `cluster_dims`, metric, feature weights.
- HDBSCAN: `min_cluster_size`, `min_samples`, `eom` vs `leaf`.
- K-Means: candidate K range for business-readable segmentation.

### 2. Change UMAP Input Deliberately

Current UMAP input is:

```text
100 product-vector dims + promo_rate + 4 store-share dims
```

This is reasonable, but it can pull clusters toward store format or promo behavior. Because the capstone asks for product-first segmentation, run these ablations:

| Experiment | Purpose |
|---|---|
| Product only | Proves what the product vectors alone can recover. |
| Product + promo | Tests whether promo sensitivity adds useful behavioral separation. |
| Product + store | Tests whether store format reveals product assortment behavior or simply dominates. |
| Product + promo + store | Current setting; compare against ablations. |
| Product + KPI | Only as a stress test; KPIs should usually be profiled after clustering. |

Decision rule: keep promo/store in the final UMAP input only if product-lift profiles improve and clusters do not become merely "store 7 shoppers" or "promo-heavy shoppers".

### 3. Check Customer Vectorization

The current primary vector is a recency/frequency-weighted mean of product embeddings. That is a strong baseline, but it can blur niche product signals. Test alternatives:

- TF-IDF or BM25-style product weights so universal staples matter less.
- `log1p(frequency)` instead of raw frequency accumulation to reduce domination by very common repeats.
- Separate short-term and long-term vectors, then concatenate them.
- Basket-level vectors first, then customer-level aggregation, so basket missions are preserved.
- Product-community share vectors from a product co-purchase graph.
- Sector/category share vectors added as a low-dimensional interpretable view.
- Top-N distinctive product embeddings per customer, pooled with attention-like weights instead of one smooth mean.

The most promising quick win is TF-IDF/BM25 weighting on `(customer, product)` before aggregation. It directly supports the product-first mandate by downweighting products everyone buys and emphasizing products that make a customer distinctive.

### 4. Think Beyond One Clustering Algorithm

Additional product-first ideas worth testing:

- Product co-purchase graph: build product communities from pair-lift edges, then represent each customer as a distribution over product communities.
- Basket mission model: classify each basket into missions such as stock-up, fresh top-up, promo stock-up, non-food trip, or convenience trip; cluster customers by mission mix.
- NMF on customer-product matrix: creates interpretable product topics with top products baked in.
- Soft segmentation: assign customers a mix of tribe affinities instead of one hard label, useful when shoppers have multiple missions.
- Future-purchase validation: train on Jan-April, segment, then test whether tribes predict May-June product mix.
- Rare-affinity mining: identify small but commercially valuable tribes around high-lift products, not just large dense groups.

These are not replacements for the required pipeline. They are ways to make the final answer more meaningful and defensible.

## Recommended Next Sequence

1. Generate and inspect tribe profiles for HDBSCAN, K-Means K=8, and K-Means K=12.
2. Pick 2-3 candidate segmentations based on product-lift readability.
3. Run product-only and product+promo/store UMAP ablations in dev mode.
4. Add a vectorization experiment with TF-IDF or BM25 product weighting.
5. Choose the final dev pipeline using the product-first scorecard.
6. Run the chosen settings in prod mode.
7. Create tribe names, business descriptions, and final visuals.

## Definition of Done

The pipeline is ready for final presentation when each selected tribe has:

- A stable cluster label and customer count.
- Revenue share and key customer KPIs.
- Top products ranked by lift.
- Product/category explanation in plain language.
- A commercial name.
- A recommended Carrefour action.
- Evidence that the segmentation is product-led, not demographic, store-only, or KPI-only.
