# Project Context: Carrefour Data Challenge
**Discovering Organic Tribes from Massive Purchase Tickets**
*Source: The Data Refinery — Analytical Manifesto v4.01*

---

## 1. The Core Objective

Build an end-to-end machine learning pipeline that segments Carrefour customers into **behavioral clusters ("organic tribes")** derived purely from their purchase history — with no reliance on demographic attributes such as age, gender, or location.

The goal is to move away from traditional demographic profiling (e.g., "Women 30–40") and toward behavior-driven segmentation (e.g., "Weekend shoppers focused on private label, organic fresh produce, and bulk products").

---

## 2. The Dataset

- **Scale:** Over 1 million unique customers (slide notes elsewhere reference 11 million — plan for that upper bound)
- **Input:** Raw transactional history — customer IDs mapped directly to purchased product lines
- **Format:** Pure checkout ticket records; no pre-processed features or pre-built algorithms

---

## 3. The Required Pipeline — 4 Phases

### Phase 1 — Product Embedding: "The Language of Shopping"

**What to do:**
Apply a **Word2Vec / Item2Vec** architecture trained on the full ticket history.

**How it works:**
- Treat every shopping cart as a "sentence"
- Treat every purchased product as a "word"
- Train the model so that products frequently bought together are geometrically close in vector space

**Expected output:** A dense vector representation for every product in the catalogue, encoding its semantic purchase context (e.g., Beer and Chips will be close; Diapers will be far away).

---

### Phase 2 — Customer Mathematization: "From Products to People"

**What to do:**
Aggregate each customer's full purchase history into a single behavioral vector.

**Critical constraints:**
- **DO NOT use simple arithmetic mean** — averaging product vectors destroys signal and amplifies noise permanently
- **Apply frequency-weighted aggregation** — habitual/recurring purchases must outweigh one-off anomalies
- **Apply time-decay curves** — recent purchases must carry significantly more weight than older ones (a purchase from yesterday defines the customer better than one from two years ago)

**Expected output:** One high-dimensional behavioral vector per customer.

---

### Phase 3 — Dimensionality Reduction: "Taming the Vector Space"

**What to do:**
Compress the high-dimensional customer vectors into a manageable latent space using non-linear techniques.

**Approved approaches:**
- **Autoencoders / VAE (Variational Autoencoders):** Use deep learning to compress thousands of raw dimensions into a continuous, hyper-dense latent space of approximately 100 dimensions
- **UMAP / t-SNE:** Advanced manifold learning techniques that preserve the global topological structure of organic groups when moving from ultra-high to low dimensionality

**Explicitly forbidden:**
- **PCA** — assumes linear dependencies, which is invalid for consumer behaviour data (e.g., the combination of buying soy milk AND baby diapers creates non-linear, folded geometries that PCA cannot capture)

---

### Phase 4 — Customer Clustering: "Finding Organic Shapes"

**What to do:**
Run density-based clustering on the reduced vector space to discover natural customer groupings.

**Mandated algorithm: HDBSCAN**

- Groups customers by topological density
- Discovers irregular, organic cluster shapes without requiring a predefined number of clusters (K)
- Naturally isolates outliers and anomalous transactions as mathematical "noise" — leaving them unassigned

**Explicitly forbidden:**
- **K-Means** — assumes clusters are perfect spheres and forces the practitioner to artificially guess K (the number of tribes). Strictly prohibited.

---

## 4. Infrastructure & Technical Directives

| Requirement | Detail |
|---|---|
| **Distributed Processing** | The full transactional history of millions of customers requires a robust distributed computing architecture (e.g., Spark, Dask, or equivalent) |
| **GPU Acceleration** | HDBSCAN and UMAP spatial distance calculations across millions of dense vectors will collapse standard CPUs. Hardware acceleration (GPU) is **mandatory** |
| **Centroid Profiling** | After clustering, cross the resulting mathematical clusters back against the product master catalogue to commercially "name" each tribe — this is the step that transforms math into actionable business intelligence |

---

## 5. Final Deliverable

An interpretable map of **named customer tribes**, each commercially described by their dominant purchase behaviors — ready to power targeting, personalisation, and retail strategy at scale.

---

## 6. What Is Explicitly Out of Scope

- Demographic segmentation of any kind (age, gender, zip code)
- Pre-chewed or pre-engineered feature sets — input must be raw ticket data
- K-Means clustering
- PCA for dimensionality reduction
- Simple mean aggregation of product vectors

---
