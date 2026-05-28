# Carrefour MVP — Team Sprint Guide
**May 28 – June 3, 2026**

---

## Steps Overview

| # | Task | Phase | Day |
|---|---|---|---|
| 1 | Data Setup & Conversion | Foundation | Day 1 |
| 2 | Data Quality Check | Foundation | Day 1 |
| 3 | Data Cleaning & Merge | Foundation | Day 1 |
| 4 | Exploratory Analysis (EDA) | Foundation | Day 2 |
| 5 | Product Embeddings (Word2Vec) | Phase 1 | Day 3 |
| 6 | Customer Vectors | Phase 2 | Day 4 |
| 7 | Dimensionality Reduction | Phase 3 | Day 5 |
| 8 | Clustering | Phase 4 | Day 5 |
| 9 | Tribe Profiling & Naming | Output | Day 6 |
| 10 | Interactive Map | Output | Day 6 |

---

## Steps in Detail

---

### Step 1 — Data Setup & Conversion
**Phase:** Foundation | **Day 1**

**Why:** Raw CSV files are too slow to query at 191M rows. Converting to Parquet makes every subsequent step 10–20× faster.

**Subtasks:**
- Place the two raw CSV files in `data/raw/csv/`
- Run the conversion script → generates Parquet files in `data/raw/parquet/`

**Output:** `data/raw/parquet/linea_tickets.parquet` + `maestra_articulos.parquet`

---

### Step 2 — Data Quality Check
**Phase:** Foundation | **Day 1**

**Why:** Before modeling, confirm the data is trustworthy — no missing fields, no broken joins, all 6 months present. Building on bad data means any result we get is meaningless.

**Subtasks:**
- Run 8 automated checks: schema, nulls, anomalies, product coverage, temporal completeness, customer activity, promos, stores
- All 8 must pass before proceeding

**Output:** `data/processed/quality_report.json`

---

### Step 3 — Data Cleaning & Merge
**Phase:** Foundation | **Day 1**

**Why:** Returns and zero-price lines corrupt behavior signals. We also need to attach product category names to every transaction so we can interpret results later.

**Subtasks:**
- Drop rows where quantity ≤ 0 or price ≤ 0
- Join transactions with product master on product ID
- Verify no orphaned rows after join
- Save the cleaned, merged dataset

**Output:** `data/processed/df_combined.parquet` — clean transactions with product info attached

---

### Step 4 — Exploratory Analysis (EDA)
**Phase:** Foundation | **Day 2**

**Why:** Confirm the data has enough behavioral variety to produce meaningful customer groups. If all customers look the same, clustering will fail and we need to know that before investing time in the model.

**Output:** Charts saved to `outputs/eda_*.png` + written confirmation that behavioral variance is sufficient to proceed

---

### Step 5 — Product Embeddings (Word2Vec)
**Phase:** Phase 1 | **Day 3**

**Why:** The model needs to "understand" which products are behaviorally similar — i.e. bought together. This creates a numeric fingerprint for each of the 117K products that captures co-purchase context.

**Subtasks:**
- Group transactions by ticket → list of product IDs per basket (each basket = a "sentence")
- Train Word2Vec on those baskets (vector_size=100, window=5, min_count=10)
- Sanity check: top 5 similar products to a known item should make business sense (e.g. beer → wine, chips)
- Save the product-to-vector lookup table

**Output:** `models/word2vec_product.model` + `data/processed/product_embeddings.parquet`

---

### Step 6 — Customer Vectors
**Phase:** Phase 2 | **Day 4**

**Why:** Each customer bought hundreds of different products over 6 months. We need to compress that history into one single profile vector per customer so we can mathematically compare customers to each other.

**Subtasks:**
- For each customer: look up the vector for every product they bought
- Average them — weighted by purchase frequency and recency (recent = higher weight)
- Add promo sensitivity as an extra feature (% of their purchases that were on promotion)
- Also compute a simple unweighted average as a baseline for comparison
- Save both versions

**Output:** `data/processed/customer_vectors_weighted.parquet` + `customer_vectors_baseline.parquet` — 1 row per customer

---

### Step 7 — Dimensionality Reduction
**Phase:** Phase 3 | **Day 5**

**Why:** Each customer currently has ~101 numbers describing them. We can't visualize or cluster 101 dimensions directly. We compress to 2 numbers so customers can be plotted on a map and similar customers land near each other.

**Subtasks:**
- Sample 200K customers for speed
- Run UMAP (primary method) on weighted customer vectors → 2D coordinates
- Run PCA (baseline method) on same data → 2D coordinates
- Plot both side by side as scatter plots
- Measure silhouette score for each — higher score = better-separated groups

**Output:** `data/processed/umap_coords.parquet` + `pca_coords.parquet` | UMAP score should beat PCA

---

### Step 8 — Clustering
**Phase:** Phase 4 | **Day 5**

**Why:** Now that customers are on a 2D map, we draw circles around the dense regions — those are the tribes. We use two methods and compare them to prove ours is better.

**Subtasks:**
- Run HDBSCAN on UMAP coordinates → automatically finds number of tribes, no K needed
- Run K-Means with K = the number HDBSCAN found → fair comparison baseline
- Measure both: silhouette score + Davies-Bouldin index
- Color-code the map by tribe label
- Extract top 10 most bought products per tribe

**Output:** `data/processed/customer_tribes.parquet` — tribe label per customer + colored cluster map

---

### Step 9 — Tribe Profiling & Naming
**Phase:** Output | **Day 6**

**Why:** Raw cluster numbers mean nothing to a client. This step turns "Cluster 4" into *"The Promo Surfer — 187K customers who only buy on discount."* This is what the client actually responds to.

**Subtasks:**
- Compute per-tribe KPIs: avg basket €, visit frequency, total 6-month revenue, promo rate, top product categories
- Build `src/tribe_namer.py` — sends top 30 products per tribe to Claude API, receives a commercial name + 1-paragraph business description
- Save tribe profiles and names

**Output:** `data/processed/tribe_profiles.parquet` + `outputs/tribe_names.json`

---

### Step 10 — Interactive Map & Final Assembly
**Phase:** Output | **Day 6**

**Why:** A static chart looks like a student project. An interactive map where the client can explore the tribes looks like a real product. This is the deliverable we walk into the meeting with.

**Subtasks:**
- Build Plotly scatter on UMAP coordinates: color = tribe, dot size = total spend, hover = top 5 products + spend
- Export as standalone HTML (`outputs/tribe_map.html` — opens in any browser)
- Assemble final results notebook (`notebooks/03_results.ipynb`) with: tribe map, tribe cards (name + KPIs + top products), method comparison table, 3 commercial recommendations
- Run full pipeline end-to-end once to confirm no errors
- Push all code to repo

**Output:** `outputs/tribe_map.html` + `notebooks/03_results.ipynb` — ready to present to client
