Project Status Report — Carrefour Customer Segmentation
Last updated: 2026-06-02 (session review of notebooks 02 and 03)

---

## Cross-Notebook Data Inconsistency — Read First

The three notebooks were NOT run on the same dataset. This is the most important context for any future session.

| Machine | Notebook run | linea_tickets rows | df_combined size | Unique customers |
|---|---|---|---|---|
| Local (WSL2/Linux) | 02_pre-analysis.ipynb | 110,375,977 | 5,811 MB | 1,460,582 |
| Windows (C:\Users\rothl\...) | 03_ml_pipeline.ipynb | ~191M (inferred) | 10,055 MB | 1,482,715 |
| Local (current CSV) | 01_exploration.ipynb | 191,017,715 | not yet built | — |

All Phase 1–4 results (Word2Vec, customer vectors, UMAP, K-Means, tribe profiles) were produced on the 1.48M customer dataset from the Windows machine. The EDA and quality report in notebook 02 reflect the 1.46M customer Linux dataset. These are structurally inconsistent.

When the pipeline is run locally end-to-end for the first time, it will produce 191M raw rows, a larger df_combined, and all downstream numbers will differ from the saved outputs. This is expected and correct — the local run will be the first internally consistent end-to-end execution.

---

1. Data & Preprocessing

What was done: ie_linea_ticket (191M rows on local machine) and ie_maestra_articulos (893k rows) were joined on idarticu, filtered for valid dates, cleaned of negative quantities and prices, and converted to Parquet for efficient streaming. All 1.48M customers were retained.

What's correct: The join is sound. Date parsing is handled. The streaming approach (Polars lazy + engine="streaming") correctly prevents the 190M rows from materialising in RAM. The parquet cache strategy is well-designed. Notebook 02 uses load_linea_tickets() and load_maestra_articulos() from src/data_loader.py correctly throughout with proper Polars API (group_by, agg, collect).

What's confirmed from notebook 02 code review:
- All 8 data quality gates pass (schema, nulls, anomalies, product coverage, temporal completeness, customer activity, promo integrity, store integrity)
- 378,936 rows dropped (0.34%): 162 credit notes + 378,774 zero-price lines
- 100% product coverage — zero orphaned products after join
- df_combined.parquet saved correctly; df_combined is rebound to pl.scan_parquet() at end of Section 3.3 for memory safety
- customer_kpis.parquet cache guard works correctly (reads pandas from cache or computes via Polars streaming then converts)
- Segmentation viability confirmed: 5/5 KPIs show CV > 0.5 (visits=1.42, basket=1.47, spend=10.18, promo=0.79, variety=1.26)

Known minor inconsistencies in notebook 02:
- kpi_labels (Section 4.4) and kpis (Section 4.5) are the same column→label mapping defined twice with slightly different display text. Should be unified.
- Code comment numbers in cells (5, 6, 8) do not match section heading numbers (4.2, 4.3, 4.4). Skip from 6 to 8 suggests a cell was removed without renumbering.
- n_raw is defined in Section 1 (bc875ce1) and again captured as pre["n_raw"] in Section 3 — redundant but not a bug.

What's missing or problematic:
- Four stores are treated as one population (partially addressed — see Phase 3 below)
- No held-out validation window. The entire 6-month period is used for all stages.
- The cosine similarity output in Section 6.4 was truncated. We never confirmed whether time-decay weighting adds real signal vs. simple mean.
- Single-item baskets are included in Word2Vec training but contribute zero co-purchase signal.

---

2. Phase 1 — Product Embedding (Item2Vec)

What was done: Skip-gram Word2Vec (100D, window=5, min_count=3, 10 epochs) trained on 20M basket "sentences". 101,540 products received embeddings (86.3% of purchased products); the remaining 13.7% fall back to their sector centroid in Phase 2.

What's correct: Skip-gram over CBOW is the right choice for rare products. The sanity check passed — bread maps to bread variants, bananas to fresh produce, bags to bags. The sector centroid fallback is sensible rather than silently dropping customers who bought rare products.

What's confirmed from notebook 03 code review:
- Word2Vec model cached at models/word2vec_product.model (loaded from Windows path in saved outputs)
- product_embeddings.parquet = 36.3 MB (101,540 products × 100 dims)
- basket_sentences.parquet = 20,268,652 baskets; median 10.5 items per basket; 0.2% single-item
- Coverage: 893,944 in catalogue; 776,323 never purchased; 117,621 purchased; 101,540 embedded (86.3%); 16,081 below min_count (13.7%)
- del baskets, basket_sizes called correctly after stats to free RAM

What's weak:
- Large basket dominance. A 100-item bulk shopping trip generates ~5,000 training pairs; a 3-item top-up generates ~6.
- No temporal separation in training. Seasonal products may have misleading vector relationships.
- The embedding validates commercially but not statistically. Three probes from three sectors is not a rigorous coverage test.

What would strengthen it: Hierarchical softmax over negative sampling for rare products. Minimum basket size threshold to exclude single-item baskets from training.

---

3. Phase 2 — Customer Mathematization

What was done: For each customer, every (product, date) purchase was weighted by quantity × exp(-decay × days_since_purchase). The weighted average of product vectors produces one 100D profile per customer. A simple mean (no weighting) was computed as a baseline. promo_rate was computed separately as the fraction of purchases that were promotional. Store affinity features (spend share per store) were built separately.

Current config: RECENCY_HALFLIFE_DAYS = 60 (was 30, already updated in config.py)

What's confirmed from notebook 03 code review:
- 1,482,715 unique customers in interaction weights (122,335,658 (customer, product) pairs)
- 1,482,643 customers vectorized (100.0% coverage; 72 customers dropped — only bought products below min_count with no sector fallback)
- Promo rate: mean=0.227, median=0.205, p25=0.137, p75=0.286 — consistent with raw data's 22.4% promo lines
- Mean vectors: 1,482,643 customers × 100 dims, 578.5 MB on disk
- Store affinity: 4 stores named Euros_Málaga, Euros_Jaén, Euros_Granada, Euros_Granada-Centro; 63.2% single-store loyalists; 23.1% 2-store; 13.7% 3+ stores

What's correct: Frequency-weighting and time-decay are both implemented. Coverage is 100%. Promo rate distribution matches raw data — good internal consistency check.

What's wrong:
- IMPORTANT CORRECTION from previous report: promo_rate IS included as an input feature to UMAP (see Phase 3). The status "zero influence" from the previous version of this report was incorrect. promo_rate is part of the 106-dimensional UMAP input (100D vectors + promo_rate + 4 store shares). It has indirect influence on tribe assignment via the manifold. However, it is NOT passed as a direct feature to K-Means or HDBSCAN — only the 20D UMAP output is used, with promo_rate attached as a side column. So the influence is indirect (through the manifold) not direct.
- Quantity-based weighting vs. visit-based weighting. Buying 50 units of toilet paper in one trip is weighted identically to buying 1 unit on 50 separate occasions.

---

4. Phase 3 — Dimensionality Reduction

What was done: UMAP primary (100D + promo_rate + 4 store shares = 106 input dims → 20D output, cosine metric, n_neighbors=30, min_dist=0.0). PCA baseline (100D → 20D, captures 83.6% variance in 20 components). Both fit on 300k sample, transform full 1.48M population.

What's confirmed from notebook 03 code review:
- umap_cluster schema: cliente | u0...u19 | promo_rate (22 columns total; promo_rate is carried as metadata, not part of the 20 UMAP dims)
- umap_viz schema: cliente | x | y | promo_rate (4 columns; 2D for visualization)
- The 2D viz is derived from the 20D umap_cluster, NOT from the raw 100D vectors — geometrically consistent with clustering space
- Section 7.4 (UMAP vs PCA silhouette comparison using proxy K=20) has NO saved output — the quantitative justification for choosing UMAP over PCA is not visible in the notebook

Known code errors in notebook 03:
- Cell 04ce084d comment says "# derived from 50D, not raw 100D" — WRONG. Should be "# derived from 20D UMAP, not raw 100D vectors"
- Section 8.3 markdown says "Both metrics are computed on the 50D UMAP space" — WRONG. UMAP output is 20D.

What's correct: cosine metric for high-dim embedding vectors is appropriate. UMAP → HDBSCAN with euclidean on UMAP output is standard practice. Store features as UMAP input dimensions correctly separates format loyalists from cross-shoppers.

What's problematic:
- UMAP fit on 300k sample (20% of 1.48M). Rare customer types may be underrepresented.
- UMAP can manufacture apparent density separation in 20D even when original 100D space had continuous gradation. HDBSCAN then finds UMAP-induced peaks rather than genuine behavioural groups.
- 20 UMAP dimensions have no commercial interpretability — unlike PCA components.

---

5. Phase 4 — Clustering

What was done: HDBSCAN fit on 100k sample (20D UMAP, euclidean). Remaining 1.38M customers assigned via 1-nearest-neighbour. K-Means with K=4 (derived from HDBSCAN) run on full 1.48M customers.

CRITICAL: HDBSCAN was run with min_cluster_size=200. src/config.py now has HDBSCAN_MIN_CLUSTER_SIZE = 5000. The saved outputs reflect the old parameter. The config fix is already in place; notebooks just need to re-run from Section 8.1 to apply it.

HDBSCAN result (min_cluster_size=200 — OUTDATED):
- Tribe 2: 1,425,206 customers (96.1%) — commercially unusable mega-tribe
- Tribes 0/1/3: 6k–8k customers each (micro-niches)
- Noise: 37,649 (2.5%)
- Silhouette: −0.0675 (negative = tribes are geometrically inside each other)

K-Means result (K=4 — VALID):
- Tribe 3: 455,586 | Tribe 0: 439,485 | Tribe 1: 414,103 | Tribe 2: 173,469
- Silhouette: 0.4153 | Davies-Bouldin: 0.8823
- Balanced, commercially viable sizes

CRITICAL: K-Means tribe profiles cell (c4f41884, Section 8.6) has NO saved output in the notebook. This is the primary deliverable — the tribe table with revenue share, avg basket, visit frequency, promo rate, and top products. The profile_tribes() function was called but the output is missing. This must be run and verified first in the next session.

HDBSCAN tribe profiles (Section 8.5) DO have output but are commercially useless (98% revenue in the mega-tribe).

Known code bug — customer_tribes.parquet saved with positional join:
Cell 71d4e1d7 appends tribe labels using pl.Series().to_list() by row position rather than joining on cliente. This assumes umap_viz, hdb_labels, and km_labels are in the same row order, which parquet does not guarantee. The safer pattern (used correctly in cell 2e17b337 for the scatter plot) is:
  umap_viz.join(hdb_labels.select(["cliente","cluster"]).rename({"cluster":"tribe_hdbscan"}), on="cliente")
          .join(km_labels.select(["cliente","cluster"]).rename({"cluster":"tribe_kmeans"}), on="cliente")

---

6. Commercial Sense Check

K-Means gives 4 balanced tribes with silhouette 0.4153 — this is a usable segmentation. The commercial gaps are:

1. K-Means tribe profiles not yet reviewed (missing output in notebook). No named tribes. No revenue ranking. The entire commercial output depends on this.
2. promo_rate influences tribe assignment only indirectly (via UMAP manifold). It is not a first-class direct input to K-Means. A customer who is 80% promo and one who is 5% promo get the same K-Means treatment if their 20D UMAP coordinates are similar.
3. No temporal tribe stability analysis. Without the 3-window evolution (Jan–Feb, Mar–Apr, May–Jun), Carrefour cannot know whether tribes are stable or seasonal.
4. Interactive visualization not yet built. 2D UMAP coordinates exist in umap_viz_2d.parquet and customer_tribes.parquet. Plotly/Bokeh scatter is not implemented.

What would embarrass you in front of a client:
- Presenting HDBSCAN results where 96% of customers are in one tribe without explanation
- Not having named, commercially described tribes ready (the entire point of the brief)
- Silhouette score of −0.0675 without context explaining why it's negative
- Claiming promo sensitivity drives tribe differentiation — it has only indirect influence

---

7. Priority Action Plan (updated)

#1 — Verify and display K-Means tribe profiles (Impact: CRITICAL)
Cell c4f41884 (Section 8.6) has no output. Run profile_tribes(km_labels, method_name="kmeans") and read the output. This is the core deliverable. Everything else is secondary until this exists.
File: notebooks/03_ml_pipeline.ipynb, cell c4f41884

#2 — Fix customer_tribes.parquet positional join (Impact: HIGH)
Replace the pl.Series().to_list() positional assignment in cell 71d4e1d7 with proper joins on cliente. The current approach silently risks mismatching customer IDs with tribe labels across cached parquet files.
File: notebooks/03_ml_pipeline.ipynb, cell 71d4e1d7

#3 — Re-run HDBSCAN with correct config (Impact: MEDIUM)
Delete cached cluster_labels_hdbscan.parquet and re-run Section 8.1 so HDBSCAN uses min_cluster_size=5000 (already set in config.py). This produces a fairer empirical comparison for the presentation narrative.
Command: delete data/processed/cluster_labels_hdbscan.parquet, then rerun cell 7ce2bfb1

#4 — Fix dimension errors in comments/markdown (Impact: LOW)
- Cell 04ce084d: "derived from 50D" → "derived from 20D UMAP"
- Section 8.3 markdown: "50D UMAP space" → "20D UMAP space"
File: notebooks/03_ml_pipeline.ipynb

#5 — Fix "Notebook 1" reference in Section 6 entry cell (Impact: LOW)
Cell d141f47a says df_combined.parquet comes from "Section 3 (Notebook 1)". Should be "Section 3 (Notebook 2 — 02_pre-analysis.ipynb)".
File: notebooks/03_ml_pipeline.ipynb, cell d141f47a

#6 — Implement tribe naming via Claude API (Impact: HIGH for presentation)
Create src/tribe_namer.py. Call claude-haiku-4-5-20251001 (configured in src/config.py) with the top 30–50 products per K-Means tribe. Output: commercial tribe name + 1-paragraph business description per cluster.
Config already has: CLAUDE_MODEL = "claude-haiku-4-5-20251001", max_tokens=300

#7 — Build interactive Plotly visualization (Impact: HIGH for presentation)
2D UMAP coordinates exist in umap_viz_2d.parquet (x, y, promo_rate). customer_tribes.parquet has tribe_kmeans labels. Load both, sample 50k–100k points, build Plotly scatter with color=tribe, size=total_spend (from customer_kpis.parquet), hover=top products. Save as outputs/tribe_map_interactive.html.

#8 — Unify kpi_labels/kpis in notebook 02 (Impact: LOW)
Section 4.4 and 4.5 both define the same column→label dict with different names and different label text. Consolidate into one dict defined once.
File: notebooks/02_pre-analysis.ipynb, cells 9aeb5091 and 505e6e30

---

8. Cached File Inventory (what exists locally after a full run)

data/processed/:
- df_combined.parquet — 109.9M clean merged rows (built from 110M raw on Linux; will be ~190M on next local run)
- customer_kpis.parquet — 1.46M customers, 5 KPIs (visit_count, avg_basket_size, total_spend_6m, avg_promo_rate, unique_products)
- quality_report.json — 8/8 quality gate results
- basket_sentences.parquet — 20.3M baskets (multi-item only)
- product_embeddings.parquet — 101,540 products × 100 dims, 36.3 MB
- customer_product_weights.parquet — 122.3M (customer, product) pairs with recency weights
- customer_vectors_weighted.parquet — 1,482,643 customers × 100 dims + promo_rate
- customer_vectors_mean.parquet — 1,482,643 customers × 100 dims, 578.5 MB
- customer_store_features.parquet — per-customer spend shares across 4 stores
- umap_cluster_20d.parquet — 1,482,643 customers × 20 UMAP dims + promo_rate (22 cols)
- umap_viz_2d.parquet — 1,482,643 customers × x, y, promo_rate (4 cols)
- pca_cluster_20d.parquet — 1,482,643 customers × 20 PCA dims
- cluster_labels_hdbscan.parquet — 1,482,643 customers with HDBSCAN labels (OUTDATED: used min_cluster_size=200)
- cluster_labels_kmeans.parquet — 1,482,643 customers with K-Means labels (VALID)
- tribe_profiles_hdbscan.parquet — 5 tribes (4 real + noise); commercially useless (96% in tribe 2)
- tribe_profiles_kmeans.parquet — 4 tribes (VALID; content unknown — output not shown in notebook)
- customer_tribes.parquet — 1,482,643 rows: cliente, x, y, promo_rate, tribe_hdbscan, tribe_kmeans (WARNING: built with positional join, not key join)

models/:
- word2vec_product.model — gensim Word2Vec, 101,540 products, 100D

Note: All cached files above reflect the Windows machine run (1.48M customers). When the pipeline is re-run locally from scratch with the 191M row parquet, these files will be regenerated. The local data/processed/ directory is currently empty (only .gitkeep) — the above inventory is what a full run produces.
