# Project Context: Carrefour Data Challenge

Discovering organic customer tribes from massive purchase-ticket data.

Source: The Data Refinery analytical manifesto v4.01.

This document preserves the original project brief. The short implementation note below summarizes how the current repository applies that brief.

## Current Implementation Note

As of 2026-06-22, the active repository implements the original brief through a YAML-driven product-first pipeline:

- Stage 1 builds product-token basket sentences with common-product downsampling.
- Stage 2 trains Item2Vec product embeddings.
- Stage 4 aggregates customer vectors with `quantity_idf`, `log1p(unidades)`, recency decay, customer-product basket-frequency weighting, and vector normalization.
- Official clustering uses `embeddings_only`; spend and KPIs are reserved for post-clustering profiling.
- Stage 6 uses hard three-stage UMAP-HDBSCAN with product-lift filtering and honest noise retention.
- Stage 6.6 marks tribes `strong` when they have no blockers and jitter recovery >= 0.80; `usable` means no blockers but below the strong target; `review` means a stability, confidence, or size blocker remains.
- Stage 6.8 assembles raw-data evidence once, and Stage 7 reads that evidence to produce the final business-facing handoff pack.

## 1. Core Objective

Build an end-to-end machine learning pipeline that segments Carrefour customers into behavioral clusters, or "organic tribes", derived from purchase history alone.

The project must move away from demographic profiling such as "women 30-40" and toward behavior-driven segmentation such as "weekly fresh-produce shoppers with high private-label dairy affinity".

## 2. Dataset

- Scale: over 1 million unique customers in the provided extract.
- Input: raw transactional history mapping anonymized customer IDs to purchased product lines.
- Format: checkout ticket records, not pre-engineered customer features.
- Exclusions: no age, gender, ZIP code, or other demographic attributes.

## 3. Required Four-Phase Pipeline

### Phase 1: Product Embedding

Apply a Word2Vec / Item2Vec architecture trained on the ticket history.

- Treat each shopping basket as a sentence.
- Treat every purchased product as a word.
- Train embeddings so products frequently bought together are geometrically close.

Expected output: one dense vector representation per product, encoding co-purchase context.

### Phase 2: Customer Mathematization

Aggregate each customer's purchase history into a behavioral vector.

Critical constraints:

- Do not use simple arithmetic mean as the final method; it is only acceptable as a baseline.
- Apply frequency weighting so habitual purchases outweigh one-off anomalies.
- Apply time decay so recent purchases carry more weight than older purchases.

Expected output: one high-dimensional behavioral vector per customer.

### Phase 3: Dimensionality Reduction

Compress customer vectors into a manageable latent space using non-linear techniques.

Approved approaches:

- Autoencoders or variational autoencoders.
- UMAP or t-SNE.

The original brief discourages PCA because PCA is linear. The current project still runs PCA as an empirical baseline so we can prove what structure is lost, rather than simply asserting it.

### Phase 4: Customer Clustering

Run density-based clustering on the reduced vector space to discover natural groups.

Primary algorithm: HDBSCAN.

- Groups customers by topological density.
- Finds irregular cluster shapes without requiring a predefined K.
- Isolates outliers as noise.

The original brief discourages K-Means because it imposes spherical clusters and a fixed K. The current project still runs K-Means as a business-readable baseline and compares it against HDBSCAN.

## 4. Infrastructure Directives

| Requirement | Detail |
|---|---|
| Distributed processing | The full transaction history is large enough to require careful streaming or distributed processing. |
| GPU acceleration | Production-scale UMAP/HDBSCAN may need GPU acceleration or stronger infrastructure. |
| Centroid/profile interpretation | Mathematical clusters must be crossed back to product data so tribes can be named commercially. |

## 5. Final Deliverable

An interpretable map of named customer tribes, each described by distinctive product behaviors and commercial relevance.

The final answer should not be "Cluster 4". It should be a product-led segment that Carrefour can understand and act on.

## 6. Out of Scope

- Demographic segmentation.
- Pre-built demographic features.
- Treating simple mean vectors as the final customer representation.
- Selecting clusters from metrics alone without product-level interpretation.
