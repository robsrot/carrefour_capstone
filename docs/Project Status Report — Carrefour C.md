Project Status Report — Carrefour Customer Segmentation



\---

1\. Data \& Preprocessing



What was done: ie\_linea\_ticket (190M rows) and ie\_maestra\_articulos (893k rows) were joined on idarticu, filtered for valid dates, cleaned of negative quantities and prices, and converted to Parquet for efficient streaming. All 1.48M customers were retained.



What's correct: The join is sound. Date parsing is handled. The streaming approach (Polars lazy + engine="streaming") correctly prevents the 190M rows from materialising in RAM. The parquet cache strategy is well-designed.



What's missing or problematic:



\- Four stores are treated as one population. idempres takes values 2, 7, etc. A customer who shops at a suburban hypermarket and one who shops at a city centre express store have structurally different baskets — not because of who they are, but because of what those store formats stock. Pooling them may blur behavioural distinctions.

\- No held-out validation window. The entire 6-month period is used for training Word2Vec, building customer vectors, fitting UMAP, and fitting HDBSCAN. There is no way to assess whether the tribes reflect genuine stable behaviour or are artefacts of this specific time window.

\- The cosine similarity output in Section 6.4 was truncated. We never actually confirmed that time-decay weighting adds real signal vs. simple mean. This is the key Phase 2 validation and it's missing from the record.

\- Single-item baskets are included in Word2Vec training but contribute zero co-purchase signal. Their proportion is unknown from the current output.



\---

2\. Phase 1 — Product Embedding (Item2Vec)



What was done: Skip-gram Word2Vec (100D, window=5, min\_count=3, 10 epochs) trained on 20M basket "sentences". 101,540 products received embeddings (86.3% of purchased products); the remaining 13.7% fall back to their sector centroid in Phase 2.



What's correct: Skip-gram over CBOW is the right choice for rare products. The sanity check passed — bread maps to bread variants, bananas to fresh produce, bags to bags. The sector centroid fallback is sensible rather than silently dropping customers who bought rare products.



What's weak:



\- Large basket dominance. A 100-item bulk shopping trip generates \~5,000 training pairs; a 3-item top-up generates \~6. The model over-learns from large-basket co-purchase patterns. Frequent small "forgotten" items in big baskets get artificially strong associations with unrelated products.

\- No temporal separation in training. Products with seasonal demand (summer barbecue, Christmas confectionery) may have misleading vector relationships if their co-purchase context changes across the 6 months. The model pools everything.

\- The embedding validates commercially but not statistically. Three probe products from three sectors all returning sensible neighbours is not a rigorous coverage test. It confirms the model isn't broken — it doesn't confirm the vectors are strong enough to carry meaningful signal through 4 more pipeline stages.



What would strengthen it: Hierarchical softmax over negative sampling for rare products. Minimum basket size threshold to exclude single-item baskets from training. Subword product name context if available.



\---

3\. Phase 2 — Customer Mathematization



What was done: For each customer, every (product, date) purchase was weighted by quantity × exp(-decay × days\_since\_purchase) with a 30-day half-life. The weighted average of product vectors produces one 100D profile per customer. A simple mean (no weighting) was computed as a baseline. promo\_rate was computed separately as the fraction of purchases that were promotional.



What's correct: Frequency-weighting and time-decay are both implemented. Coverage is 100% (only 72 customers dropped). The promo rate distribution (mean 22.7%, median 20.5%) matches the raw data's 22.4% promotional lines — a good internal consistency check.



What's wrong:



\- 30-day half-life is too aggressive for 6 months of data. A customer who shopped every week January–March and then stopped gets weight ≈ 0.016 on their January behaviour by June. That customer's profile is almost entirely determined by their last few visits. If those last visits are atypical (holiday season, illness, travel), the vector misrepresents their stable behaviour. A 60–90 day half-life would be more appropriate for a 6-month window.

\- promo\_rate is computed but never used in the clustering. This is the most significant structural flaw in the pipeline. The feature is passed through UMAP as a side-column and explicitly excluded from HDBSCAN's feature matrix by \_embedding\_to\_numpy(). Every part of the notebook describes it as a key differentiating axis — "promo surfers vs brand loyalists" — but it has zero influence on which tribe any customer lands in. It only appears in the post-hoc tribe profiles. This is commercially misleading: you cannot claim to have discovered promo sensitivity tribes if promo sensitivity wasn't an input to the clustering.

\- Quantity-based weighting vs. visit-based weighting. Buying 50 units of toilet paper in one trip is weighted identically to buying 1 unit on 50 separate occasions. These represent very different customer behaviours (bulk buyer vs. habitual shopper) but the weighting scheme treats them the same.



\---

4\. Phase 3 — Dimensionality Reduction



Clarification on what was actually built: UMAP is the primary method (100D → 20D, cosine metric, n\_neighbors=30). PCA was run as a baseline. UMAP is implemented and is the input to HDBSCAN.



What's correct: Using cosine metric in UMAP is appropriate for embedding vectors where direction matters more than magnitude. The UMAP → HDBSCAN pipeline with euclidean distance on the UMAP output is standard practice.



What's problematic:



\- Why UMAP over PCA is justified: Customer purchase behaviour is genuinely non-linear. Two customers who are both "health-conscious" might buy completely different products (one buys organic packaged goods, the other buys fresh produce exclusively) but land in the same behavioural region. PCA's linear projections cannot capture this folded geometry — it would place them far apart. UMAP's manifold learning finds that both points live on the same "health-conscious" surface. The proxy silhouette score confirms UMAP produces better cluster structure than PCA on this data.

\- 20D is an aggressive compression. PCA captures 83.6% variance in 20 components — meaning 16.4% of variance is discarded before HDBSCAN sees the data. Some of that 16.4% likely contains the signal separating behavioural sub-groups. The standard recommendation for HDBSCAN input is 10–50D; 20D is within range but on the low end.

\- UMAP fit on only 100k customers (6.7% of population). If rare but commercially interesting customer types (ultra-high spenders, niche product buyers) are underrepresented in the 100k fit sample, UMAP doesn't learn their manifold region correctly. Their 20D coordinates are interpolated rather than learned.

\- UMAP introduces artificial density separation. This is the most technically serious issue: UMAP's objective function repels distant points and attracts nearby ones. This can manufacture the appearance of distinct clusters in the 20D embedding even when the original 100D space had a continuous gradation. HDBSCAN then finds these UMAP-induced density peaks rather than genuine behavioural groups.

\- The commercial interpretation of the 20 UMAP dimensions is zero. Unlike PCA (where PC1 might be interpretable as "basket size" or "category breadth"), UMAP dimensions are non-linear combinations of all 100 original features. You cannot say "this customer is in this tribe because they score high on dimension 7." The only interpretability comes from the post-hoc tribe profiles.



\---

5\. Phase 4 — Clustering (HDBSCAN)



What was done: HDBSCAN fit on 100k sample (20D UMAP, euclidean, min\_cluster\_size=200, min\_samples=10). Remaining 1.38M customers assigned via 1-nearest-neighbour to the fit sample.



The result is commercially unusable:



┌───────┬───────────┬───────┐

│ Tribe │ Customers │ Share │

├───────┼───────────┼───────┤

│ 2     │ 1,425,206 │ 96.1% │

├───────┼───────────┼───────┤

│ 1     │ 7,846     │ 0.5%  │

├───────┼───────────┼───────┤

│ 0     │ 6,155     │ 0.4%  │

├───────┼───────────┼───────┤

│ 3     │ 5,787     │ 0.4%  │

├───────┼───────────┼───────┤

│ Noise │ 37,649    │ 2.5%  │

└───────┴───────────┴───────┘



Silhouette: −0.0675. A negative silhouette means the three small tribes are geometrically inside the large cluster, not separated from it. HDBSCAN found three tiny high-density islands within one enormous density plateau.



Root cause diagnosis:



The 20D UMAP space has one dominant density region containing \~96% of customers. This is likely real — most Carrefour shoppers buy broadly similar categories (fresh produce, dairy, packaged goods) with moderate promo sensitivity. The truly distinct behavioral patterns are extreme cases. HDBSCAN is correctly reporting the density structure: one big hill, three anthills. The problem is that this density structure, while geometrically honest, is commercially useless.



K-Means result for comparison:



┌───────┬───────────┐

│ Tribe │ Customers │

├───────┼───────────┤

│ 3     │ 455,586   │

├───────┼───────────┤

│ 0     │ 439,485   │

├───────┼───────────┤

│ 1     │ 414,103   │

├───────┼───────────┤

│ 2     │ 173,469   │

└───────┴───────────┘



K-Means silhouette: 0.4153. Balanced, commercially viable tribe sizes. K-Means won on every metric.



Why K-Means is still architecturally the wrong choice (even though it performed better here): K-Means assumes spherical, equally-sized clusters and forces every customer into one — including outliers who genuinely don't belong. It imposes structure rather than discovering it. A K-Means tribe profile is an average across a forced partition, not a description of a natural behavioral group. For a Carrefour jury, the distinction matters: K-Means gives you 4 administrative segments; HDBSCAN (if it worked) would give you 4 real tribes. The practical answer for the presentation, however, is that K-Means produced the only commercially usable result.



Can the clusters be named? Not yet — tribe profiles from 8.5 haven't been reviewed — but based on the K-Means distribution (roughly 30%/30%/28%/12%), plausible commercial names would emerge from the top product analysis.



\---

6\. Commercial Sense Check



What Carrefour can actually do with the current output: very little.



The HDBSCAN result gives you one mega-tribe and three micro-niches. You cannot run a CRM campaign for 1.4M undifferentiated customers labelled "Tribe 2." The three small tribes (6–8k customers each) are interesting as fringe segments but insufficient as a primary segmentation.



Structural gaps that limit commercial applicability regardless of clustering quality:



1\. No spend tier differentiation. Spend level and visit frequency are captured in the KPI profiles but not in the clustering inputs. A high-frequency, high-basket customer and a low-frequency, low-basket customer who buy the same product mix will land in the same tribe. From Carrefour's revenue perspective, these are completely different customers.

2\. Promo sensitivity is absent from clustering (as noted above). The commercial opportunity — identifying customers who only buy on promotion vs. those who pay full price — was positioned as a key differentiator and is not delivered.

3\. No tribe stability metric. Without the temporal window analysis, Carrefour cannot know whether "The Fresh Produce Loyalist" tribe is stable year-round or a January phenomenon. Unstable tribes cannot be acted on with confidence.

4\. No revenue contribution ranking. Without 8.5 running cleanly, there is no answer to "which tribe should Carrefour target first?" The entire commercial value of the segmentation depends on this output.



\---

7\. Priority Action Plan



\#1 — Fix promo\_rate as a clustering input (Impact: HIGH)

Scale promo\_rate to \[0,1] range and append it as a 21st feature before UMAP. Currently it has zero influence on tribe assignment despite being the most commercially differentiating behavioural axis in the data. One line change in reduce\_umap\_cluster(). Re-run UMAP and HDBSCAN (caches will need to be deleted). This is the single change most likely to produce commercially interpretable tribes.



\#2 — Switch to K-Means as primary for the final presentation (Impact: HIGH)

The empirical comparison showed K-Means (silhouette 0.4153) outperformed HDBSCAN (silhouette −0.0675) on this data. Reframe the narrative honestly: you ran both, let the data decide, K-Means won. This is a stronger story than forcing HDBSCAN to be "primary" and presenting a clustering where 96% of customers are one tribe. Use K-Means labels for 8.5 tribe profiles.



\#3 — Increase HDBSCAN min\_cluster\_size to 10,000–50,000 (Impact: MEDIUM)

If you want to keep HDBSCAN in the pipeline, min\_cluster\_size=200 on a 100k fit sample is far too small — it finds micro-niches. At 50,000, HDBSCAN would require tribes to contain at least \~750k customers in the full population, guaranteeing commercially meaningful group sizes. This is an alternative to #2, not a complement.



\#4 — Get 8.5 tribe profiles running on K-Means labels (Impact: HIGH)

The entire commercial output of the project — the "named tribes" — depends on this. Without it, you have a clustering exercise with no deliverable. Run profile\_tribes(km\_labels, method\_name="kmeans") and read the top products. This is the most important single thing to have for the presentation.



\#5 — Extend time-decay half-life to 60–90 days (Impact: MEDIUM)

The current 30-day half-life discards too much of the 6-month behavioural history. A customer's January shopping tells you something about who they are. A 60-day halflife preserves the recency signal while not erasing the history. Requires re-running Phase 2 onward (customer vectors are cached and would need to be regenerated).



\#6 — Run tribe profiles on the 2D UMAP scatter (Impact: MEDIUM)

The 2D visualisation exists but tribe colours haven't been applied yet. This is the single most impressive visual for a jury — an explorable map of 1.48M customers colour-coded by tribe. Even with K-Means results, this is presentation-ready material. Run cell 45 (now fixed), save the plot, and add it to the presentation deck.



\#7 — Address the "one store vs. four stores" issue in the narrative (Impact: LOW)

You don't need to re-run anything — just acknowledge in the presentation that the four stores are pooled and flag it as a known limitation. A Carrefour jury will ask about this. Having a prepared answer ("we pooled stores because customer IDs are anonymised and we cannot track cross-store shopping; a production system would segment by format") is better than being caught without one.



What would embarrass you in front of a client:

\- Presenting HDBSCAN results where 96% of customers are in one tribe without explanation

\- Claiming promo sensitivity drives tribe differentiation when it isn't in the clustering

\- Not having named, commercially described tribes ready (the entire point of the brief)

\- Silhouette score of −0.0675 without context explaining why it's negative



