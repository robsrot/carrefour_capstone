# 3-Day Clustering Improvement Sprint

**Date**: 2026-06-08 | **Mode**: `CARREFOUR_MODE=dev` | **Branch**: one per member

---

## Safety Protocol — Read Before Running Anything

These rules ensure that running all experiments — including by an AI agent — leaves the project in a state that is equal to or better than the baseline. The project must never deteriorate.

### Rule 1 — Preserve the baseline artifacts first

Before touching anything, run this once:

```python
import shutil, os
os.environ["CARREFOUR_MODE"] = "dev"

from src.config import DATA_PROCESSED

BASELINE_ARTIFACTS = [
    "cluster_labels_hdbscan_assigned.parquet",
    "tribe_profiles_hdbscan_assigned.parquet",
    "umap_cluster_item2vec.parquet",
    "customer_vectors_weighted.parquet",
    "product_embeddings.parquet",
]

for name in BASELINE_ARTIFACTS:
    src  = DATA_PROCESSED / name
    dest = DATA_PROCESSED / name.replace(".parquet", "_baseline.parquet")
    if src.exists() and not dest.exists():
        shutil.copy2(src, dest)
        print(f"Backed up: {name}")
    elif dest.exists():
        print(f"Already backed up: {name}")
    else:
        print(f"WARNING — missing: {name}")
```

The `_baseline.parquet` files are never overwritten by any experiment. They are the fallback if everything else breaks.

### Rule 2 — Every experiment writes to its own isolated path

Never run an experiment that writes to the default cache path without an explicit `cache_path=` argument. The default paths (`umap_cluster_item2vec.parquet`, `customer_vectors_weighted.parquet`, etc.) are the live state. Experiments write to versioned paths like `umap_cluster_window5.parquet`, `cluster_labels_gmm_k15.parquet`.

Convention: always include the experiment identifier in the artifact name.

```python
# WRONG — overwrites the live baseline
reduce_umap_cluster(cv, force=True)

# CORRECT — isolated artifact
reduce_umap_cluster(cv, force=True, cache_path=DATA_PROCESSED / "umap_cluster_window5.parquet")
```

### Rule 3 — Every experiment result is written to a JSON scorecard file

The `experiment_scorecard()` function (defined below) writes its result to `outputs/dev/experiment_results.json`. This file accumulates across all runs and is the single source of truth for which experiment won.

### Rule 4 — The project ends with the best configuration applied

After all experiments, run the winner-selection code block (see "Selecting and Applying the Winner" at the end). This reads `experiment_results.json`, finds the experiment with the highest `avg_top5_lift`, updates `configs/dev.yaml`, does one clean final rebuild with `force=True` on the affected stages only, and runs `experiment_scorecard()` one last time to verify. The final scorecard must show improvement over baseline (9.32× lift, 11 tribes).

### Rule 5 — After changing `dev.yaml`, reload ALL pipeline modules

The notebook master cell (Cell 3) runs `importlib.reload(src.config)`, but `src/dimensionality.py`, `src/clustering.py`, `src/embeddings.py`, and `src/customer_vectors.py` each import constants from `src.config` at module load time (`from src.config import UMAP_N_NEIGHBORS, ...`). Reloading `src.config` does NOT update those already-bound names. If you just re-run the master cell, `reduce_umap_cluster()` still uses the old `UMAP_N_NEIGHBORS`.

**After editing `dev.yaml` and re-running the master cell, run this reload cell before calling any pipeline function:**

```python
import importlib
import src.config, src.embeddings, src.customer_vectors, src.dimensionality, src.clustering
importlib.reload(src.config)
importlib.reload(src.embeddings)
importlib.reload(src.customer_vectors)
importlib.reload(src.dimensionality)
importlib.reload(src.clustering)

# Re-import the public functions after reload so the notebook sees the updated versions
from src.embeddings   import train_word2vec, save_embeddings, load_product_embeddings
from src.customer_vectors import build_customer_vectors, build_customer_vectors_mean, \
    build_customer_vectors_tfidf_svd, build_customer_vectors_hybrid
from src.dimensionality   import reduce_umap_cluster, reduce_umap_viz, reduce_pca
from src.clustering       import cluster_hdbscan, assign_hdbscan_noise_to_nearest_tribe, \
    cluster_kmeans, run_kmeans_baselines, grid_search_hdbscan, evaluate_clustering, profile_tribes
print("All pipeline modules reloaded.")
```

Alternatively, restart the kernel (safest option, slowest). Either way is fine — the important thing is that you do one of them before every experiment that changes `dev.yaml`.

### Rule 6 — Force-rebuild only what changed

| If you changed... | Force rebuild... |
|---|---|
| `configs/dev.yaml` word2vec params | `FORCE_EMBEDDINGS`, `FORCE_VECTORS`, `FORCE_UMAP`, `FORCE_CLUSTERING` |
| `configs/dev.yaml` customer vector params | `FORCE_VECTORS`, `FORCE_UMAP`, `FORCE_CLUSTERING` |
| `configs/dev.yaml` UMAP params | `FORCE_UMAP`, `FORCE_CLUSTERING` |
| `configs/dev.yaml` HDBSCAN params | `FORCE_CLUSTERING` only |
| Custom `cache_path=` experiment | Only the specific function you call with `force=True` |

Never set all four force flags to True unless you changed Word2Vec. Each unnecessary rebuild wastes 20–40 minutes.

### Rule 7 — Notebook variable names

The notebook uses `cluster_space` (not `umap_cluster`) as the name for the UMAP embedding passed to clustering functions. When copying code from this document into the notebook, substitute `cluster_space` wherever the sprint plan writes `umap_cluster` as the clustering input.

---

## Repo Handover Status

The pipeline is complete end-to-end in dev mode. The current segmentation produces 11 tribes with avg top-5 lift of 9.32×. The goal for this sprint is to push that number higher across the board — sharper product signals in every tribe, finer granularity, zero noise, and stronger behavioral distinctiveness throughout the entire customer space. Some tribes already have strong lift (C0–C4, C10); others are weaker (C5–C7, C9). Both are targets for improvement.

Setup:
```powershell
git checkout -b experiment/<your-name>
conda activate carrefour
$env:CARREFOUR_MODE = "dev"
```

---

## Baseline to Beat

| Metric | Baseline |
|---|---|
| Tribe count | 11 |
| Avg top-5 product lift | 9.32× |
| Min tribe size | 1,894 |
| Tribes with avg lift ≥ 3× | 11 / 11 |
| Silhouette | 0.28 |
| Davies-Bouldin | 0.46 |

Per-tribe breakdown:

| Tribe | n | Rev% | Avg top-5 lift |
|---|---:|---:|---:|
| C0 — Everyday Staples | 3,750 | 4.8% | 13.3× |
| C1 — Pet + Sweets | 1,894 | 3.4% | 11.8× |
| C2 — Traditional Meat | 3,217 | 4.5% | 11.8× |
| C3 — Health & Protein | 2,132 | 3.3% | 12.3× |
| C4 — Premium Deli | 2,901 | 6.0% | 10.3× |
| C5 — Baby Families | 10,394 | 20.8% | 6.4× |
| C6 — Organic Lifestyle | 7,636 | 25.6% | 4.2× |
| C7 — Premium Alcohol | 4,235 | 12.8% | 6.3× |
| C8 — Spirits & Gourmet | 2,235 | 6.1% | 8.3× |
| C9 — Pet + Beer | 2,845 | 7.0% | 7.4× |
| C10 — Latin Community | 2,760 | 5.6% | 10.4× |

An experiment is an **improvement** if:
- avg top-5 lift > 9.32× **or** tribe count > 11
- **and** min tribe size ≥ 440 (1% of dev set)
- **and** all tribes have avg lift ≥ 3×
- Silhouette drop of more than 0.08 should be flagged as a warning.

The goal is not to fix specific tribes — it is to improve the segmentation globally. Higher lift across all tribes, more tribes with distinct product stories, and fewer customers lumped into large catch-all segments are all improvements.

---

## Shared Evaluation Function

Every experiment ends with this. Paste into a notebook cell near the top — run it once, then call `experiment_scorecard()` after every clustering result.

```python
import polars as pl, numpy as np, json, os
from pathlib import Path
from src.clustering import evaluate_clustering, profile_tribes
from src.config import OUTPUTS

os.environ["CARREFOUR_MODE"] = "dev"

BASELINE = {
    "tribe_count": 11, "min_tribe_size": 1894,
    "avg_top5_lift": 9.32, "silhouette": 0.28, "davies_bouldin": 0.46,
}

RESULTS_FILE = OUTPUTS / "experiment_results.json"

def _load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return []

def _save_result(result: dict):
    results = _load_results()
    # Replace existing entry for same method_name if present
    results = [r for r in results if r.get("method") != result["method"]]
    results.append(result)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

def experiment_scorecard(cluster_labels, umap_cluster, method_name, force_profile=True):
    metrics  = evaluate_clustering(umap_cluster, cluster_labels, method_name)
    profiles = profile_tribes(cluster_labels, method_name=method_name, force=force_profile)
    rows = profiles.to_dicts()

    lift_scores = [
        sum(r["top_lifts"][:5]) / max(1, min(5, len(r["top_lifts"])))
        for r in rows
    ]
    tribe_count            = len(rows)
    min_size               = min(r["n_customers"] for r in rows)
    avg_top5_lift          = sum(lift_scores) / max(1, len(lift_scores))
    tribes_above_threshold = sum(1 for s in lift_scores if s >= 3.0)

    print(f"\n{'='*62}")
    print(f"  SCORECARD — {method_name}")
    print(f"{'='*62}")
    print(f"  {'Metric':<22} {'Result':>10}  {'Baseline':>10}  {'Delta':>8}")
    print(f"  {'-'*57}")

    def row(label, val, base, fmt="{:.2f}", higher_is_better=True):
        v = fmt.format(val) if not isinstance(val, int) else str(val)
        b = fmt.format(base) if not isinstance(base, int) else str(base)
        delta = val - base
        sign  = "+" if delta > 0 else ""
        arrow = ("▲" if delta > 0 else "▼") if delta != 0 else "="
        flag  = "✓" if (delta > 0) == higher_is_better else "✗"
        print(f"  {label:<22} {v:>10}  {b:>10}  {arrow}{sign}{fmt.format(abs(delta)):>6} {flag}")

    row("Tribe count",      tribe_count,    BASELINE["tribe_count"],    fmt="{:d}")
    row("Min tribe size",   min_size,       BASELINE["min_tribe_size"], fmt="{:d}")
    row("Avg top-5 lift",   avg_top5_lift,  BASELINE["avg_top5_lift"],  fmt="{:.2f}")
    row("Tribes lift ≥ 3×", tribes_above_threshold, tribe_count, fmt="{:d}")
    row("Silhouette",       metrics["silhouette"],     BASELINE["silhouette"],     fmt="{:.4f}")
    row("Davies-Bouldin",   metrics["davies_bouldin"], BASELINE["davies_bouldin"], fmt="{:.4f}", higher_is_better=False)

    print(f"\n  Per-tribe lift:")
    for r, s in sorted(zip(rows, lift_scores), key=lambda x: -x[1]):
        top = (r.get("top_products") or ["?"])[0]
        print(f"    C{r['cluster']:>2}: avg5={s:.1f}× | n={r['n_customers']:>5,} | {top[:45]}")

    improved = (
        avg_top5_lift > BASELINE["avg_top5_lift"] or tribe_count > BASELINE["tribe_count"]
    ) and min_size >= 440 and tribes_above_threshold >= tribe_count
    print(f"\n  {'✓  IMPROVEMENT' if improved else '✗  No improvement'}")
    print(f"{'='*62}\n")

    result = {
        "method": method_name,
        "tribe_count": tribe_count,
        "min_size": min_size,
        "avg_top5_lift": round(avg_top5_lift, 3),
        "silhouette": round(metrics["silhouette"], 4),
        "davies_bouldin": round(metrics["davies_bouldin"], 4),
        "tribes_above_threshold": tribes_above_threshold,
        "improved": improved,
    }
    _save_result(result)   # persists to outputs/dev/experiment_results.json
    return result
```

Cache invalidation chain — when you change a stage, force-rebuild everything downstream:
```
df_combined → basket_sentences → word2vec model → product_embeddings
→ customer_product_weights → customer_vectors → umap_cluster → cluster_labels → tribe_profiles
```
Use `force=True` flags or delete the relevant `data/dev/*.parquet` files. Never modify `df_combined.parquet`.

---

## Member 1 — Product Embeddings

**Files**: `src/embeddings.py`, `configs/dev.yaml`  
**Rebuild flags**: `FORCE_EMBEDDINGS=True`, `FORCE_VECTORS=True`, `FORCE_UMAP=True`, `FORCE_CLUSTERING=True`  
**Artifact naming**: use `method_name` of the form `m1_<descriptor>`, e.g. `m1_window5_ep10`. The cluster label file must be saved to `DATA_PROCESSED / f"cluster_labels_{method_name}.parquet"` using `cache_path=`.  
**Run order**: start with the lowest-cost change (window size only), measure, then compound if it improved.

The product embedding is the signal everything else builds on. The current Word2Vec with `window=10` treats products at opposite ends of a large trolley as co-occurring, blurring the embedding space. Run these experiments in order of difficulty, run `experiment_scorecard()` after each.

---

### Classic Word2Vec Tuning

1. **Window size.** Try `window: 3`, `5`, `7` (current: `10`). Smaller windows enforce stricter co-purchase proximity. `window=5` is the standard Item2Vec recommendation.

2. **Training epochs.** Try `epochs: 10`, `15` (current: `5`).

3. **Embedding dimension.** Try `vector_size: 50`, `150`, `200` (current: `100`).

4. **CBOW vs skip-gram.** The `sg=1` flag in `src/embeddings.py` enables skip-gram. Try `sg=0`. CBOW averages context; skip-gram predicts context. Skip-gram is usually better for rare products; CBOW for frequent ones.

5. **Popularity filter tightening.** Reduce `product_popularity.max_basket_share` from `0.15` → `0.10` and `max_customer_share` from `0.40` → `0.25`. Removes more universal staples from basket sentences, forcing the model to learn from distinctive products.

6. **Negative sampling rate.** The `Word2Vec()` call in `src/embeddings.py` currently does not pass a `negative=` argument (gensim default is 5). To control it, make three changes:
   - Add `negative: 5` under `word2vec:` in `configs/base.yaml`
   - Add `W2V_NEGATIVE = _cfg["word2vec"]["negative"]` in `src/config.py` (next to the other W2V constants)
   - Add `negative=W2V_NEGATIVE,` to the `Word2Vec()` call in `src/embeddings.py`
   
   Then set `word2vec.negative: 10` or `15` in `configs/dev.yaml` and rebuild. Higher negative sampling produces better-separated embeddings for rare products.

---

### Node2Vec — Product Co-Purchase Graph Embeddings

Instead of treating baskets as sentences, build a graph where nodes are products and edges are co-purchase pair-lift scores. Node2Vec learns embeddings by running biased random walks on this graph.

```python
# pip install node2vec
import networkx as nx
from node2vec import Node2Vec
import polars as pl

pair_lift = pl.read_parquet("data/processed/product_eda_pair_lift_top250.parquet")

G = nx.Graph()
for row in pair_lift.filter(pl.col("lift") > 2.0).to_dicts():
    G.add_edge(str(row["idarticu_a"]), str(row["idarticu_b"]), weight=float(row["lift"]))

node2vec = Node2Vec(
    G,
    dimensions=100,
    walk_length=30,
    num_walks=200,
    p=1,        # return parameter — controls backtracking
    q=0.5,      # in-out parameter — q<1 favors DFS (community structure)
    workers=1,  # reproducibility
    seed=42,
)
model = node2vec.fit(window=5, min_count=1, workers=1, epochs=10, seed=42)

# Export to same format as Word2Vec product embeddings
product_ids   = list(model.wv.key_to_index.keys())
product_vecs  = [model.wv[pid] for pid in product_ids]
df_embeddings = pl.DataFrame({
    "idarticu": pl.Series(product_ids).cast(pl.Int64),
    "vector":   pl.Series(product_vecs, dtype=pl.List(pl.Float32)),
})
df_embeddings.write_parquet("data/dev/product_embeddings_node2vec.parquet", compression="zstd")
```

Tune `p` and `q`: `p=1, q=0.5` favors structural equivalence (products in similar roles across categories); `p=1, q=2` favors BFS/homophily (products closely co-purchased). Try both and compare.

Pass `product_embeddings_node2vec.parquet` as the embedding input to `build_customer_vectors_weighted()` by setting the `embeddings_path` argument. Then run the full pipeline and call `experiment_scorecard()`.

---

### Sanity Check (run after every experiment)

Use the embedding sanity check cell in the notebook. Pick these probe products:
- A baby food product → top neighbors should be other baby food, not generic dairy
- An Iberian charcuterie product → top neighbors should be other Iberian charcuterie, not mass-market ham
- An organic produce product → top neighbors should be other organic products

If neighbors degrade (random, or incoherent), the change hurt the embedding quality — revert.

---

## Member 2 — Customer Vector Representations

**Files**: `src/customer_vectors.py`, `configs/dev.yaml`  
**Rebuild flags**: `FORCE_VECTORS=True`, `FORCE_UMAP=True`, `FORCE_CLUSTERING=True`  
**Artifact naming**: `method_name` of the form `m2_<descriptor>`, e.g. `m2_bm25_k1_1.5`.  
**Run order**: BM25 first (config-only change, lowest risk), then NMF/LDA (new function, medium risk), then neural network approaches (highest risk, do last).

The current customer vector is a recency/frequency-weighted mean of product embeddings. Averaging all products a customer has bought blurs their identity. Customers who buy 30 different products end up near the centroid of all 30 products' vectors. Multiple alternatives can fix this.

---

### Classical Improvements

1. **BM25 saturation weighting.** Add to `_apply_weight_transform()` in `src/customer_vectors.py`:
   ```python
   if transform == "bm25":
       k1 = 1.5
       v = np.maximum(values, 0.0).astype(np.float64)
       return (v / (v + k1)).astype(np.float32)
   ```
   Set `customer_vectors.weight_transform: bm25` in `configs/dev.yaml`. Also try `k1=1.2` and `k1=2.0`.

2. **Recency half-life sweep.** Try `halflife_days: 30`, `45`, `60`, `90`.

3. **Top-N product pooling.** Restrict each customer's vector to their top-N products by interaction weight before averaging. Try N=20, 50, 100. Prevents one-off purchases from diluting the core product signal.

4. **Separate short-term / long-term vectors.** Split interactions at month 3. Build two weighted vectors per customer and concatenate them before UMAP. The delta may separate stable from shifting customers.

---

### NMF — Latent Product Topic Decomposition

NMF factorizes the customer-product interaction matrix into K non-negative components. Each component is a latent "product topic" with interpretable top products (e.g. Topic 0 = baby food, Topic 1 = Iberian charcuterie). A customer's vector is their mix of these topics. Unlike embedding + averaging, NMF is directly interpretable.

Add to `src/customer_vectors.py`:

```python
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

_NMF_CACHE = DATA_PROCESSED / "customer_vectors_nmf.parquet"

def build_customer_vectors_nmf(
    df_combined_path=None, *, n_components=30, force=False
) -> pl.DataFrame:
    if _NMF_CACHE.exists() and not force:
        return pl.read_parquet(_NMF_CACHE)

    if df_combined_path is None:
        df_combined_path = DATA_PROCESSED / "df_combined.parquet"

    interactions = _build_interactions(df_combined_path, RECENCY_HALFLIFE_DAYS, force=False)

    # Build sparse matrix
    customers = interactions["cliente"].unique().sort().to_list()
    products  = interactions["idarticu"].unique().sort().to_list()
    cust_idx  = {c: i for i, c in enumerate(customers)}
    prod_idx  = {p: i for i, p in enumerate(products)}

    rows = [cust_idx[r["cliente"]] for r in interactions.to_dicts()]
    cols = [prod_idx[r["idarticu"]] for r in interactions.to_dicts()]
    data = interactions["weight"].to_list()
    matrix = csr_matrix((data, (rows, cols)), shape=(len(customers), len(products)))

    nmf = NMF(n_components=n_components, init="nndsvda", random_state=RANDOM_SEED, max_iter=300)
    vectors = nmf.fit_transform(matrix).astype(np.float32)

    # L2-normalise rows
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)

    # Log top products per topic for interpretability check
    from src.data_loader import load_maestra_articulos
    articles = load_maestra_articulos()
    id_to_name = dict(zip(articles["idarticu"].to_list(), articles["desc_larga_articulo"].to_list()))
    for t in range(min(5, n_components)):
        top_idx  = nmf.components_[t].argsort()[-5:][::-1]
        top_prod = [id_to_name.get(products[i], str(products[i])) for i in top_idx]
        _log.info("  NMF Topic %d: %s", t, top_prod)

    df = pl.DataFrame({
        "cliente": pl.Series(customers),
        "vector":  pl.Series(vectors.tolist(), dtype=pl.List(pl.Float32)),
    }).sort("cliente")
    _NMF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_NMF_CACHE, compression="zstd")
    return df
```

Try `n_components: 20`, `30`, `40`. After each run, check the logged topic-product printouts — if topics are coherent (topic 0 = baby food, topic 3 = organic produce), NMF is working. Then call `experiment_scorecard()`.

---

### LDA — Probabilistic Topic Modeling

Latent Dirichlet Allocation treats each customer's purchase history as a "document" and products as "words". Each customer gets a distribution over K topics, and each topic has a distribution over products. Similar to NMF but with a probabilistic generative model — often produces cleaner, more interpretable topics than NMF for sparse data.

```python
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csr_matrix

# Reuse the same sparse matrix as NMF above
# LDA requires non-negative integer counts — use raw purchase frequency, not weights
lda = LatentDirichletAllocation(
    n_components=30,
    random_state=RANDOM_SEED,
    learning_method="online",
    max_iter=20,
)
vectors = lda.fit_transform(matrix).astype(np.float32)
# vectors shape: (n_customers, n_components) — each row sums to 1 (topic proportions)
```

Try `n_components: 20`, `30`, `40`. Check topic coherence (top products per topic). Pass to UMAP + HDBSCAN and call `experiment_scorecard()`.

---

### Autoencoder Customer Embeddings (Neural Network)

Train a feed-forward autoencoder on the customer-product interaction matrix. The bottleneck layer is the customer embedding. Unlike NMF/LDA, the autoencoder can learn non-linear interactions.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Use the dense matrix for small n_components; for 44k customers × 55k products
# use sparse input + mini-batches
# Simpler: use the NMF/TFIDF 100D vectors as input to the autoencoder

# Input: existing customer_vectors_weighted or tfidf_svd vectors (100D)
X = np.array(cv_weighted["vector"].to_list(), dtype=np.float32)
X_tensor = torch.FloatTensor(X)
loader   = DataLoader(TensorDataset(X_tensor), batch_size=512, shuffle=True)

class CustomerAutoencoder(nn.Module):
    def __init__(self, input_dim=100, bottleneck=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

torch.manual_seed(42)
model     = CustomerAutoencoder(input_dim=X.shape[1], bottleneck=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(50):
    total_loss = 0
    for (batch,) in loader:
        optimizer.zero_grad()
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

# Extract bottleneck embeddings
model.eval()
with torch.no_grad():
    _, embeddings = model(X_tensor)
    ae_vectors = embeddings.numpy()

# Save as customer vectors
cv_ae = pl.DataFrame({
    "cliente": cv_weighted["cliente"],
    "vector":  pl.Series(ae_vectors.tolist(), dtype=pl.List(pl.Float32)),
})
cv_ae.write_parquet("data/dev/customer_vectors_autoencoder.parquet", compression="zstd")
```

Try `bottleneck=16`, `32`, `64`. Pass to UMAP + HDBSCAN. Call `experiment_scorecard()`.

---

### Variational Autoencoder (VAE)

A VAE forces the latent space to be smooth and approximately Gaussian, which tends to produce better cluster separation than a plain autoencoder. The reparameterization trick makes the gradients flow through the stochastic bottleneck.

```python
class CustomerVAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=32):
        super().__init__()
        self.fc1    = nn.Linear(input_dim, 64)
        self.fc_mu  = nn.Linear(64, latent_dim)
        self.fc_var = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        h   = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z    = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

def vae_loss(recon, x, mu, log_var, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kld        = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kld

torch.manual_seed(42)
vae       = CustomerVAE(input_dim=X.shape[1], latent_dim=32)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(100):
    for (batch,) in loader:
        optimizer.zero_grad()
        recon, mu, log_var = vae(batch)
        loss = vae_loss(recon, batch, mu, log_var, beta=1.0)
        loss.backward()
        optimizer.step()

vae.eval()
with torch.no_grad():
    mu_vecs, _ = vae.encode(X_tensor)
    vae_vectors = mu_vecs.numpy()   # use mu (mean) as the embedding, not sampled z

cv_vae = pl.DataFrame({
    "cliente": cv_weighted["cliente"],
    "vector":  pl.Series(vae_vectors.tolist(), dtype=pl.List(pl.Float32)),
})
cv_vae.write_parquet("data/dev/customer_vectors_vae.parquet", compression="zstd")
```

Try `latent_dim=16`, `32`, `64`. Try `beta=0.5`, `1.0`, `4.0` (beta-VAE with higher beta forces more disentangled representations). Pass to UMAP + HDBSCAN. Call `experiment_scorecard()`.

---

## Member 3 — Dimensionality Reduction & Clustering

**Files**: `src/dimensionality.py`, `src/clustering.py`, `configs/dev.yaml`  
**Rebuild flags**: UMAP changes → `FORCE_UMAP=True`, `FORCE_CLUSTERING=True`. Clustering changes only → `FORCE_CLUSTERING=True`.  
**Artifact naming**: `method_name` of the form `m3_<descriptor>`, e.g. `m3_hdbscan_mcs50_ms3_leaf` or `m3_gmm_k15`.  
**Run order**: HDBSCAN grid search first (no code changes, just config), then GMM/Bisecting (small additions), then DEC (neural network, highest risk).

---

### UMAP Hyperparameter Sweep

1. **`n_neighbors`**: Try `10`, `15`, `30` (current: `20`). Smaller = more local topology = more tribes.
2. **`cluster_dims`**: Try `20`, `30`, `75` (current: `50`).
3. **`min_dist_cluster`**: Try `0.0`, `0.05`, `0.1`. Lower values pack embeddings tighter, sharpening density peaks.
4. **Metric**: Try `metric="euclidean"` on L2-normalized vectors alongside current `cosine`.

Run every combination. Each UMAP output should be followed by the full HDBSCAN pipeline and `experiment_scorecard()`.

---

### HDBSCAN Full Grid Search

Set `RUN_HDBSCAN_GRID_SEARCH = True` in the notebook master cell. Expand `configs/dev.yaml`:

```yaml
hdbscan_grid:
  min_cluster_size: [15, 25, 50, 75, 100, 150, 200]
  min_samples: [1, 3, 5, 10]
  cluster_method: [leaf, eom]
```

After the grid runs, load results and filter to configs producing 12–20 tribes. The grid result file is named by the notebook as `hdbscan_grid_{VECTOR_SOURCE}_{SELECTED_REDUCER}_full.parquet` — in the current dev baseline that is `data/dev/hdbscan_grid_item2vec_umap_full.parquet`.

```python
# Read the grid results (run grid_search_hdbscan with RUN_HDBSCAN_GRID_SEARCH=True first)
hdb_grid = pl.read_parquet(DATA_PROCESSED / f"hdbscan_grid_{VECTOR_SOURCE}_{SELECTED_REDUCER}_full.parquet")
candidates = hdb_grid.filter(
    (pl.col("n_clusters") >= 12) & (pl.col("n_clusters") <= 20) & (pl.col("noise_pct") < 15)
).sort("noise_pct")
print(candidates.select(["min_cluster_size","min_samples","cluster_method","n_clusters","noise_pct","silhouette"]))
```

For each candidate, run full `cluster_hdbscan()` + `assign_hdbscan_noise_to_nearest_tribe()` with explicit `cache_path` arguments, then `experiment_scorecard()`.

`cluster_method: leaf` finds more, smaller, tighter clusters. `eom` merges sub-clusters more aggressively. Test both across all `min_cluster_size` values.

---

### Gaussian Mixture Models (GMM)

GMM produces soft probabilistic assignments instead of hard labels. Unlike HDBSCAN, it has no noise points — every customer gets assigned. It also handles elliptical clusters that HDBSCAN misses.

Add to `src/clustering.py`:

```python
from sklearn.mixture import GaussianMixture

def cluster_gmm(umap_cluster: pl.DataFrame, n_components: int, *, force=False,
                cache_path=None) -> pl.DataFrame:
    path = cache_path or DATA_PROCESSED / f"cluster_labels_gmm_k{n_components}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)

    dim_cols = [c for c in umap_cluster.columns if c not in ("cliente", "promo_rate")]
    X = umap_cluster.select(dim_cols).to_numpy().astype(np.float32)

    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        random_state=RANDOM_SEED, max_iter=200, n_init=3,
    )
    labels = gmm.fit_predict(X).astype(np.int32)
    bic    = gmm.bic(X)
    _log.info("GMM k=%d: BIC=%.1f", n_components, bic)

    df = umap_cluster.select("cliente").with_columns(
        pl.Series("cluster", labels),
        pl.lit(0.0).alias("promo_rate"),
    ).sort("cliente")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")
    return df
```

Try `n_components: 10`, `12`, `15`, `18`, `20`. Compare BIC scores (lower = better model fit). Call `experiment_scorecard()` on each. Also try `covariance_type="tied"` and `"diag"` — different covariance structures reveal different cluster shapes.

---

### Bisecting K-Means — Surgical Tribe Splitting

Apply bisecting K-Means to any tribe whose avg top-5 lift is below the global average (9.32×). Currently that includes C5, C6, C7, and C9 — but after running other experiments the list may change. This lets you re-split whatever weak tribes the global algorithms leave behind.

```python
from sklearn.cluster import BisectingKMeans
import numpy as np

baseline = pl.read_parquet("data/dev/cluster_labels_hdbscan_assigned.parquet")
umap     = pl.read_parquet("data/dev/umap_cluster_item2vec.parquet")
dim_cols = [c for c in umap.columns if c not in ("cliente", "promo_rate")]

# Target: any tribe with n > 3000 OR avg top-5 lift < 9.32 (fill this in from your scorecard)
weak_tribes = [5, 6, 7, 9]   # update based on your actual scorecard output

new_label_parts = [baseline.filter(~pl.col("cluster").is_in(weak_tribes))]

for tribe_id in weak_tribes:
    for n_splits in [2, 3, 4]:
        tribe_customers = baseline.filter(pl.col("cluster") == tribe_id)["cliente"]
        tribe_umap      = umap.filter(pl.col("cliente").is_in(tribe_customers))
        X_tribe         = tribe_umap.select(dim_cols).to_numpy().astype(np.float32)

        bkm = BisectingKMeans(
            n_clusters=n_splits, random_state=RANDOM_SEED, bisecting_strategy="largest_cluster"
        )
        sub_labels = bkm.fit_predict(X_tribe)
        sub_df = tribe_umap.select("cliente").with_columns(
            pl.Series("cluster", (tribe_id * 10 + sub_labels).astype(np.int32)),
            pl.lit(0.0).alias("promo_rate"),
        )
        merged = pl.concat(new_label_parts[:1] + [sub_df]).sort("cliente")
        experiment_scorecard(merged, umap, method_name=f"bisect_C{tribe_id}_k{n_splits}")
```

Run on each weak tribe independently first, then combine all the winning splits into one final label set and run `experiment_scorecard()` on the combined result.

---

### Agglomerative Hierarchical Clustering

Hierarchical clustering builds a dendrogram — you can cut it at any level to get any number of tribes. Unlike HDBSCAN, it has no noise. Unlike K-Means, it doesn't assume spherical clusters. Does not scale to 44k customers, but scales fine on the 12k HDBSCAN fit sample.

```python
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Use the HDBSCAN fit sample (same 12k customers used to fit the baseline)
fit_sample_customers = pl.read_parquet("data/dev/hdbscan_fit_sample_customers.parquet")  # if saved
# If not saved, resample using RANDOM_SEED:
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(umap), size=12000, replace=False)
X_sample   = umap.select(dim_cols).to_numpy().astype(np.float32)[sample_idx]

# Plot dendrogram to choose the cut
Z = sch.linkage(X_sample, method="ward")
plt.figure(figsize=(12, 5))
sch.dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90)
plt.title("Hierarchical clustering dendrogram")
plt.tight_layout()
plt.savefig("outputs/dev/dendrogram.png")
plt.show()

# Cut at chosen level
for n_tribes in [12, 15, 18]:
    agg = AgglomerativeClustering(n_clusters=n_tribes, linkage="ward")
    sample_labels = agg.fit_predict(X_sample)
    # Extend to full population via nearest-neighbour (same as HDBSCAN noise assignment)
    from sklearn.neighbors import NearestNeighbors
    X_full = umap.select(dim_cols).to_numpy().astype(np.float32)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=1).fit(X_sample)
    _, idx = nn.kneighbors(X_full)
    full_labels = sample_labels[idx.ravel()].astype(np.int32)
    labels_df   = umap.select("cliente").with_columns(
        pl.Series("cluster", full_labels),
        pl.lit(0.0).alias("promo_rate"),
    ).sort("cliente")
    experiment_scorecard(labels_df, umap, method_name=f"hierarchical_ward_k{n_tribes}")
```

The dendrogram is the most useful output — it shows you visually at what distance the natural breaks occur.

---

### OPTICS

OPTICS is a generalization of DBSCAN/HDBSCAN that uses a reachability plot to allow variable-density clusters. It is more robust to clusters with unequal density, which is common in real customer data where niche tribes (tight, high-lift) and broad behavioral groups (diffuse, lower-lift) coexist in the same embedding space.

```python
from sklearn.cluster import OPTICS

dim_cols = [c for c in umap.columns if c not in ("cliente", "promo_rate")]
X = umap.select(dim_cols).to_numpy().astype(np.float32)

for min_samples in [5, 10, 20]:
    for xi in [0.05, 0.1, 0.2]:
        clust = OPTICS(min_samples=min_samples, xi=xi, metric="cosine", n_jobs=1)
        clust.fit(X)
        labels = clust.labels_.astype(np.int32)
        n_noise = (labels == -1).sum()
        n_tribes = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  min_samples={min_samples}, xi={xi}: {n_tribes} tribes, {n_noise} noise ({n_noise/len(labels):.1%})")

        if n_tribes >= 8 and n_noise / len(labels) < 0.15:
            # Assign noise via nearest-neighbour same as HDBSCAN
            noise_mask = labels == -1
            non_noise_X = X[~noise_mask]
            non_noise_labels = labels[~noise_mask]
            nn = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=1).fit(non_noise_X)
            _, idx = nn.kneighbors(X[noise_mask])
            labels[noise_mask] = non_noise_labels[idx.ravel()]

            labels_df = umap.select("cliente").with_columns(
                pl.Series("cluster", labels),
                pl.lit(0.0).alias("promo_rate"),
            ).sort("cliente")
            experiment_scorecard(labels_df, umap, method_name=f"optics_ms{min_samples}_xi{xi}")
```

---

### Deep Embedded Clustering (DEC)

DEC is a neural network that jointly optimizes a cluster assignment and a deep embedding. It starts from a pre-trained autoencoder (use Member 2's autoencoder) and then fine-tunes the encoder so that the embedding clusters cleanly. Purpose-built for unsupervised clustering.

```python
import torch, torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

# Step 1: Pre-train autoencoder (use Member 2's code)
# Assume 'ae_model' is the trained CustomerAutoencoder and X is the input matrix

# Step 2: Initialize cluster centers with K-Means on the bottleneck embeddings
ae_model.eval()
with torch.no_grad():
    _, Z = ae_model(torch.FloatTensor(X))
    Z_np = Z.numpy()

n_clusters = 15
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
km.fit(Z_np)
cluster_centers = torch.FloatTensor(km.cluster_centers_)

# Step 3: DEC fine-tuning — Student's t-distribution soft assignment
def soft_assignment(Z, centers, alpha=1.0):
    q = 1.0 / (1.0 + torch.sum((Z.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2) / alpha)
    q = q ** ((alpha + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True)

def target_distribution(q):
    p = q ** 2 / q.sum(dim=0)
    return p / p.sum(dim=1, keepdim=True)

centers = nn.Parameter(cluster_centers)
optimizer = torch.optim.Adam(list(ae_model.encoder.parameters()) + [centers], lr=1e-4)

X_tensor = torch.FloatTensor(X)
for iteration in range(200):
    _, Z = ae_model(X_tensor)
    q = soft_assignment(Z, centers)
    if iteration % 50 == 0:
        p    = target_distribution(q.detach())
        loss = nn.functional.kl_div(q.log(), p, reduction="batchmean")
        print(f"  DEC iter {iteration}: KL loss={loss.item():.4f}")
    p    = target_distribution(q.detach())
    loss = nn.functional.kl_div(q.log(), p, reduction="batchmean")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Final hard assignments
ae_model.eval()
with torch.no_grad():
    _, Z_final = ae_model(X_tensor)
    q_final    = soft_assignment(Z_final, centers)
    dec_labels = q_final.argmax(dim=1).numpy().astype(np.int32)

labels_df = cv_weighted.select("cliente").with_columns(
    pl.Series("cluster", dec_labels),
    pl.lit(0.0).alias("promo_rate"),
).sort("cliente")

# Note: DEC produces its own embedding — pass the Z_final embeddings as 'umap_cluster'
Z_df = cv_weighted.select("cliente").hstack(
    pl.DataFrame(Z_final.numpy(), schema=[f"dim_{i}" for i in range(Z_final.shape[1])])
)
experiment_scorecard(labels_df, Z_df, method_name=f"dec_k{n_clusters}")
```

Try `n_clusters=10`, `12`, `15`, `18`. DEC is particularly good at producing compact, well-separated clusters because it explicitly optimizes the cluster assignment objective.

---

### Feature Weight Ablations

7. **Promo weight**: `feature_weights.promo` → `0.25`, `0.5`. Useful if promo sensitivity is orthogonal to product preferences; harmful if tribes become purely "promo hunters" with no product story.
8. **Store weight**: `feature_weights.store` → `0.25`. Useful if it reveals format-driven product behavior (e.g. hypermarket vs convenience); harmful if it just separates customers by geography.

Run each, call `experiment_scorecard()`, revert if lift degrades.

---

## Member 4 — Temporal Validation & Tribe Presentation

**Files**: `src/customer_vectors.py`, `src/tribe_namer.py`  
**Artifact naming**: `method_name` of the form `m4_<descriptor>`, e.g. `m4_lifecycle_delta` or `m4_consensus_5runs`.  
**Run order**: temporal stability first (read-only analysis, zero risk), then lifecycle vectors (new function, medium risk), then consensus clustering and BERT4Rec (high risk, do last).

---

### Temporal Stability Experiments

1. **Build time-windowed customer vectors.** Add a date-filtered version of `_build_interactions()` to `src/customer_vectors.py` that accepts `date_from` and `date_to` parameters and uses `date_to` as the recency decay reference. Build:
   - W1+W2 vectors: 2022-01-01 – 2022-04-30 → `data/dev/customer_vectors_w12.parquet`
   - W3 vectors: 2022-05-01 – 2022-06-30 → `data/dev/customer_vectors_w3.parquet`

2. **Cluster W1+W2. Assign W3 via nearest-neighbour.** Run the standard UMAP + HDBSCAN on W1+W2 vectors. Assign W3 customers to the nearest W1+W2 tribe centroid using `sklearn.neighbors.NearestNeighbors(metric="cosine", n_jobs=1)` in raw vector space.

3. **Measure top-product stability per tribe.** For each tribe, compute Jaccard overlap between the tribe's top-10 products in W1+W2 and in W3. ≥ 60% = stable. ≤ 30% = seasonal or noise. Report per-tribe.

4. **Lifestyle vector experiment.** Build a per-customer concatenated vector: `[W1+W2_vector, W3_vector, (W3 − W1W2)_delta]`. Pass this to UMAP + HDBSCAN. The delta dimensions may expose customers transitioning between behavioral patterns and produce finer tribes.

5. **Transition matrix.** For each customer active in both windows, record their tribe assignment in W1+W2 and in W3. Build a tribe × tribe transition matrix. High off-diagonal counts indicate either seasonal tribes or a poor clustering.

---

### Sequential Pattern Model — BERT4Rec (Stretch Goal)

BERT4Rec treats each customer's purchase history as a sequence and trains a bidirectional transformer to predict masked products. The [CLS] token output is a rich customer representation that captures sequential behavioral patterns, not just basket co-occurrence.

```python
# pip install recbole  (includes BERT4Rec, SASRec, and other sequential models)
# OR implement a minimal version with HuggingFace transformers

from transformers import BertConfig, BertModel
import torch

# Build input sequences: for each customer, sort their purchases by date
# and convert to a sequence of product IDs (treat as token IDs)
# Truncate to max_seq_len=50 (most recent 50 products)

# Pseudocode — full implementation requires the interaction log sorted by date:
# sequences = build_purchase_sequences(df_combined, max_len=50)  # (n_customers, 50)
# token_ids = torch.LongTensor(sequences)

config = BertConfig(
    vocab_size=60000,     # number of unique product IDs
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=128,
    max_position_embeddings=52,
)
bert = BertModel(config)

# Train with masked product prediction (mask random items, predict original)
# Extract [CLS] embedding as customer vector
# This captures "what product comes next given this sequence" — rich behavioral signal
```

BERT4Rec is a 2-day implementation. It's most valuable if sequential patterns in purchase order (e.g. customers who systematically expand from baby food to organic) are not captured by the basket co-occurrence approach.

---

### Tribe Profiles and Naming

5. **Profile the best result from other members.** Once you know which configuration has the highest scorecard, run `profile_tribes()` on those labels with `force=True`.

6. **Name all tribes via Claude API.** `src/tribe_namer.py` is implemented:
   ```python
   from src.tribe_namer import name_all_tribes
   profiles = pl.read_parquet("data/dev/tribe_profiles_<best_method>.parquet")
   named    = name_all_tribes(profiles)
   named.write_parquet("data/dev/tribe_profiles_final_named.parquet", compression="zstd")
   ```

7. **Produce one tribe card per tribe.** For the final presentation: tribe name, n customers, revenue share, avg basket, avg promo rate, top 5 products by lift, recommended Carrefour action.

8. **Generate the final 2D UMAP visualization colored by tribe.** This is the primary presentation asset. The notebook already has a `reduce_umap_viz()` call that produces 2D coordinates. Re-run it on the final winning labels and produce the scatter plot:

```python
from src.dimensionality import reduce_umap_viz

umap_2d = reduce_umap_viz(
    cluster_space,
    force=True,
    cache_path=DATA_PROCESSED / "umap_viz_final_2d.parquet",
)

# Join 2D coords with tribe names for the colored scatter
named_profiles = pl.read_parquet(DATA_PROCESSED / "tribe_profiles_final_named.parquet")
id_to_name = dict(zip(named_profiles["cluster"].to_list(), named_profiles["tribe_name"].to_list()))

final_labels = pl.read_parquet(DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet")
plot_df = umap_2d.join(final_labels.select(["cliente", "cluster"]), on="cliente")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

clusters = sorted(plot_df["cluster"].unique().to_list())
colors   = cm.tab20(np.linspace(0, 1, len(clusters)))

fig, ax = plt.subplots(figsize=(14, 10))
for cid, color in zip(clusters, colors):
    sub = plot_df.filter(pl.col("cluster") == cid)
    ax.scatter(sub["x"].to_numpy(), sub["y"].to_numpy(),
               s=0.8, alpha=0.4, color=color,
               label=f"C{cid}: {id_to_name.get(cid, '')[:25]}")

ax.legend(markerscale=8, fontsize=8, loc="upper right")
ax.set_title("Customer Tribes — Final Segmentation (UMAP 2D)")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(OUTPUTS / "tribe_map_final.png", dpi=150)
plt.show()
```

---

## Member 5 — Feature Engineering, Ensemble Methods & Production

**Files**: `src/customer_vectors.py`, `src/clustering.py`, `src/dimensionality.py`  
**Artifact naming**: `method_name` of the form `m5_<descriptor>`, e.g. `m5_price_mission_concat` or `m5_consensus_5runs`.  
**Run order**: feature engineering first (additive, low risk), then ensemble/two-stage methods (medium risk), then production run last (Day 3 only, after winner is selected).

This member's work is orthogonal to Members 1–4. Feature engineering adds new behavioral dimensions to the customer vectors. Ensemble methods improve stability across the whole segmentation. The production run is the final deliverable — the full 1.48M-customer segmentation that the other four members' dev experiments feed into.

---

---

### Consensus / Ensemble Clustering

Run the same clustering algorithm multiple times on different UMAP projections (or different random seeds), then assign each customer to their most consistent tribe across runs. This removes instability from the final labels — a customer only gets firmly assigned to a tribe if they reliably land there.

```python
from collections import Counter
import numpy as np, polars as pl
from src.dimensionality import reduce_umap_cluster
from src.clustering import cluster_hdbscan, assign_hdbscan_noise_to_nearest_tribe
from src.config import DATA_PROCESSED, RANDOM_SEED

N_RUNS = 5
all_labels = []  # list of np.int32 arrays, one per run

# reduce_umap_cluster() uses RANDOM_SEED from config — it does not accept random_state=.
# To get different UMAP projections for consensus, temporarily patch the module constant.
import src.dimensionality as _dim_mod

for run in range(N_RUNS):
    seed = RANDOM_SEED + run * 1000
    _dim_mod.RANDOM_SEED = seed   # patch before calling — restore afterwards
    umap_run = reduce_umap_cluster(
        cv_for_umap, force=True,
        cache_path=DATA_PROCESSED / f"umap_cluster_consensus_run{run}.parquet",
    )
    _dim_mod.RANDOM_SEED = RANDOM_SEED   # restore
    core  = cluster_hdbscan(umap_run, force=True,
                             cache_path=DATA_PROCESSED / f"cluster_core_run{run}.parquet")
    full  = assign_hdbscan_noise_to_nearest_tribe(
                umap_run, core, force=True,
                cache_path=DATA_PROCESSED / f"cluster_full_run{run}.parquet")
    all_labels.append(full.sort("cliente")["cluster"].to_numpy())

# Pairwise label alignment (Hungarian algorithm to match cluster IDs across runs)
# Simplest approach: use majority vote after aligning with run 0 as reference
from scipy.optimize import linear_sum_assignment

def align_labels(ref, new_labels, max_clusters=30):
    cost = np.zeros((max_clusters, max_clusters), dtype=np.int32)
    for r, n in zip(ref, new_labels):
        if r >= 0 and n >= 0 and r < max_clusters and n < max_clusters:
            cost[r, n] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    return np.array([mapping.get(x, x) for x in new_labels], dtype=np.int32)

ref_labels = all_labels[0]
aligned    = [ref_labels] + [align_labels(ref_labels, lab) for lab in all_labels[1:]]

# Assign each customer to the majority-vote tribe across N_RUNS
stacked = np.stack(aligned, axis=1)   # (n_customers, N_RUNS)
consensus_labels = np.array([
    Counter(row).most_common(1)[0][0] for row in stacked
], dtype=np.int32)

customers = pl.read_parquet(DATA_PROCESSED / "umap_cluster_item2vec.parquet").sort("cliente")["cliente"]
consensus_df = pl.DataFrame({
    "cliente":    customers,
    "cluster":    pl.Series(consensus_labels),
    "promo_rate": pl.lit(0.0, dtype=pl.Float32).cast(pl.Float32).repeat_by(len(customers)).explode(),
}).sort("cliente")

umap_base = pl.read_parquet(DATA_PROCESSED / "umap_cluster_item2vec.parquet")
experiment_scorecard(consensus_df, umap_base, method_name="consensus_hdbscan_5runs")
```

Consensus clustering typically improves ALL tribes simultaneously because unstable borderline customers get pulled to whichever tribe is their most consistent home.

---

### Two-Stage Hierarchical Clustering

Instead of running one global clustering, first divide the customer space into 5–8 macro groups, then run a fine-grained clustering within each macro group. This avoids HDBSCAN's tendency to merge dense sub-groups into one large cluster.

```python
from sklearn.cluster import MiniBatchKMeans
from src.clustering import cluster_hdbscan, assign_hdbscan_noise_to_nearest_tribe
import polars as pl, numpy as np

umap = pl.read_parquet("data/dev/umap_cluster_item2vec.parquet")
dim_cols = [c for c in umap.columns if c not in ("cliente", "promo_rate")]
X = umap.select(dim_cols).to_numpy().astype(np.float32)

# Stage 1: coarse macro-groups
km_macro = MiniBatchKMeans(n_clusters=6, random_state=42, n_init=5)
macro_labels = km_macro.fit_predict(X)

# Stage 2: fine-grained HDBSCAN within each macro group
all_sub_labels = np.full(len(X), -1, dtype=np.int32)
global_cluster_id = 0

for macro_id in range(6):
    mask      = macro_labels == macro_id
    idx       = np.where(mask)[0]
    X_sub     = X[mask]
    sub_umap  = umap.filter(pl.Series(mask))

    sub_core  = cluster_hdbscan(
        sub_umap,
        min_cluster_size=50,   # smaller — each macro group is ~7k customers
        force=True,
        cache_path=DATA_PROCESSED / f"cluster_core_macro{macro_id}.parquet",
    )
    sub_full  = assign_hdbscan_noise_to_nearest_tribe(
        sub_umap, sub_core, force=True,
        cache_path=DATA_PROCESSED / f"cluster_full_macro{macro_id}.parquet",
    )
    sub_raw   = sub_full.sort("cliente")["cluster"].to_numpy()

    # Remap to globally unique IDs
    for local_id in np.unique(sub_raw):
        mask_sub = sub_raw == local_id
        all_sub_labels[idx[mask_sub]] = global_cluster_id
        global_cluster_id += 1

labels_df = umap.select("cliente").with_columns(
    pl.Series("cluster", all_sub_labels),
    pl.lit(0.0).alias("promo_rate"),
).sort("cliente")
experiment_scorecard(labels_df, umap, method_name="two_stage_macro6_hdbscan")
```

Try `n_clusters=5`, `6`, `8` for the macro stage. The key insight: within a macro group the density structure is more uniform, so HDBSCAN can use a tighter `min_cluster_size` and still produce clean results.

---

### Contrastive Learning — Behavioral Distinctiveness Embeddings

Contrastive learning explicitly trains customer embeddings so that customers with similar purchase behavior are close in vector space and customers with different behavior are far apart. This directly optimizes what the segmentation needs, rather than hoping that Word2Vec basket co-occurrence produces it indirectly.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Input: existing 100D customer vectors (item2vec weighted mean)
X  = np.array(cv_weighted["vector"].to_list(), dtype=np.float32)
X_t = torch.FloatTensor(X)

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim=100, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, proj_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

def nt_xent_loss(z, temperature=0.5):
    """NT-Xent contrastive loss (SimCLR). z has shape (2N, D)."""
    N  = z.shape[0] // 2
    z1, z2 = z[:N], z[N:]
    z_all  = torch.cat([z1, z2], dim=0)
    sim    = torch.mm(z_all, z_all.T) / temperature
    # Mask out self-similarity
    mask   = torch.eye(2 * N, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)])
    return F.cross_entropy(sim, labels)

# Augmentation: add small Gaussian noise to create a "positive pair"
def augment(x, noise_std=0.05):
    return x + torch.randn_like(x) * noise_std

encoder   = ContrastiveEncoder(input_dim=X.shape[1], proj_dim=64)
optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-4)
loader    = DataLoader(TensorDataset(X_t), batch_size=512, shuffle=True)

torch.manual_seed(42)
for epoch in range(50):
    total = 0
    for (batch,) in loader:
        z1   = encoder(augment(batch))
        z2   = encoder(augment(batch))
        loss = nt_xent_loss(torch.cat([z1, z2], dim=0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={total/len(loader):.4f}")

encoder.eval()
with torch.no_grad():
    contrastive_vecs = encoder(X_t).numpy()

cv_contrastive = pl.DataFrame({
    "cliente": cv_weighted["cliente"],
    "vector":  pl.Series(contrastive_vecs.tolist(), dtype=pl.List(pl.Float32)),
})
cv_contrastive.write_parquet("data/dev/customer_vectors_contrastive.parquet", compression="zstd")
```

Pass to UMAP + HDBSCAN and call `experiment_scorecard()`. Tune `noise_std` (augmentation strength) and `temperature`. Higher `noise_std` forces the encoder to learn a more robust representation of behavioral identity.

---

### Price Tier & Basket Mission Feature Engineering

The current customer vectors are entirely product-embedding-based. Adding explicit behavioral features can separate tribes that are product-similar but behaviorally distinct.

**Price tier affinity** — for each customer, compute the fraction of spend in each price tier (budget / mid-range / premium) and add as 3 extra dimensions:

```python
import polars as pl
from src.config import DATA_PROCESSED

df = pl.scan_parquet(DATA_PROCESSED / "df_combined.parquet").collect(engine="streaming")

# Classify products by price tier using per-product median price
product_price = (
    df.group_by("idarticu")
    .agg(pl.col("importe").median().alias("median_price"))
)
q33 = product_price["median_price"].quantile(0.33)
q67 = product_price["median_price"].quantile(0.67)

product_price = product_price.with_columns(
    pl.when(pl.col("median_price") <= q33).then(pl.lit("budget"))
    .when(pl.col("median_price") <= q67).then(pl.lit("mid"))
    .otherwise(pl.lit("premium"))
    .alias("price_tier")
)

customer_price = (
    df.join(product_price.select(["idarticu", "price_tier"]), on="idarticu")
    .group_by(["cliente", "price_tier"])
    .agg(pl.col("importe").sum().alias("spend"))
    .pivot(values="spend", index="cliente", columns="price_tier", aggregate_function="sum")
    .fill_null(0.0)
)
# Normalise to fractions
tier_cols = [c for c in customer_price.columns if c != "cliente"]
total     = customer_price.select(tier_cols).sum_horizontal()
for col in tier_cols:
    customer_price = customer_price.with_columns((pl.col(col) / total).alias(f"tier_{col}"))
customer_price = customer_price.select(["cliente"] + [f"tier_{c}" for c in tier_cols])
customer_price.write_parquet(DATA_PROCESSED / "customer_price_tier_features.parquet", compression="zstd")
```

**Basket mission distribution** — classify each basket by mission (stock-up, fresh top-up, convenience, non-food), then represent each customer by their basket mission distribution:

```python
basket_stats = (
    df.group_by(["cliente", "idtransac"])
    .agg([
        pl.len().alias("n_items"),
        pl.col("importe").sum().alias("basket_value"),
        pl.col("categoria1").n_unique().alias("n_categories"),
    ])
)

# Simple heuristic classification
basket_stats = basket_stats.with_columns(
    pl.when((pl.col("n_items") >= 20) & (pl.col("basket_value") >= 50))
       .then(pl.lit("stock_up"))
    .when((pl.col("n_categories") <= 3) & (pl.col("n_items") <= 8))
       .then(pl.lit("fresh_topup"))
    .when((pl.col("n_items") <= 5) & (pl.col("basket_value") <= 15))
       .then(pl.lit("convenience"))
    .otherwise(pl.lit("mixed"))
    .alias("mission")
)

customer_missions = (
    basket_stats.group_by(["cliente", "mission"]).len()
    .pivot(values="len", index="cliente", columns="mission", aggregate_function="sum")
    .fill_null(0)
)
# Normalise
mission_cols = [c for c in customer_missions.columns if c != "cliente"]
total = customer_missions.select(mission_cols).sum_horizontal()
for col in mission_cols:
    customer_missions = customer_missions.with_columns((pl.col(col) / total).alias(f"mission_{col}"))
```

Concatenate price tier and mission features to your best customer vector before UMAP:
```python
cv_augmented = cv_best.join(customer_price, on="cliente").join(customer_missions.select(["cliente"] + [f"mission_{c}" for c in mission_cols]), on="cliente")
# Expand the vector column with the extra scalar columns before passing to UMAP
```

These features add orthogonal behavioral dimensions — a customer can be "organic product buyer" in one dimension and "premium price tier, stock-up mission" in another.

---

### Category Breadth & Shopping Diversity Features

Customers who buy many distinct categories are generalists; customers who concentrate spend in 2–3 categories are specialists. The current vectors do not capture this directly — two customers with identical product embeddings but very different category concentration will look the same to UMAP.

```python
import polars as pl
from src.config import DATA_PROCESSED

df = pl.scan_parquet(DATA_PROCESSED / "df_combined.parquet").collect(engine="streaming")

customer_diversity = (
    df.group_by("cliente")
    .agg([
        pl.col("categoria1").n_unique().alias("n_categories"),
        pl.col("idarticu").n_unique().alias("n_distinct_products"),
        pl.col("idtransac").n_unique().alias("n_baskets"),
        (pl.col("importe").sum() / pl.col("idtransac").n_unique()).alias("avg_basket_value"),
        (pl.col("importe") / pl.col("importe").sum()).pow(2).sum().alias("spend_hhi"),
        # Herfindahl-Hirschman Index of category spend — high HHI = category specialist
    ])
)

# Normalise each feature to [0, 1] before concatenating to the customer vector
for col in ["n_categories", "n_distinct_products", "avg_basket_value", "spend_hhi"]:
    min_v = customer_diversity[col].min()
    max_v = customer_diversity[col].max()
    customer_diversity = customer_diversity.with_columns(
        ((pl.col(col) - min_v) / (max_v - min_v + 1e-8)).alias(f"{col}_norm")
    )

customer_diversity.write_parquet(DATA_PROCESSED / "customer_diversity_features.parquet", compression="zstd")
```

Concatenate these 4 normalised scalar features to the product vector before UMAP. The `spend_hhi` (Herfindahl index) is particularly useful — it separates category specialists (high HHI, typically high-lift niche tribes) from generalists (low HHI, typically large catch-all tribes).

---

### Production Pipeline Run (Day 3 — after winner is selected)

This is the most important output of the sprint. The dev experiments run on 44k customers. The production run applies the winning configuration to the full 1.48M eligible customers.

**Prerequisites before starting the prod run:**
1. The winner-selection code has run and identified a winning `method_name`.
2. `configs/dev.yaml` has been updated to match the winning configuration.
3. All final state checklist assertions pass on the dev result.
4. `git status` is clean on your branch.

**Run the production pipeline:**

```powershell
# Switch to prod mode
$env:CARREFOUR_MODE = "prod"

# Verify prod data is present
python -c "import polars as pl; df = pl.scan_parquet('data/processed/df_combined.parquet'); print(df.collect(engine='streaming').shape)"
# Expected: (191017715, 13) — 191M rows

# Open the ML notebook in prod mode
jupyter notebook notebooks/03_ml_pipeline.ipynb
```

In the notebook master cell, set **only** the force flags corresponding to what the winning experiment changed:

```python
# Example: if the winner changed word2vec window + BM25 weighting
FORCE_EMBEDDINGS = True    # word2vec changed
FORCE_VECTORS    = True    # vector transform changed
FORCE_UMAP       = True    # vectors changed, UMAP must refit
FORCE_CLUSTERING = True    # always rebuild final labels in prod
```

The production UMAP fits on a 300k-customer sample (configured in `configs/base.yaml` under `umap.fit_sample`) and transforms all 1.48M customers. HDBSCAN fits on a 300k sample and assigns the rest via nearest-neighbour. This run will take **2–4 hours** depending on the hardware.

**After the prod run completes, run the production scorecard:**

```python
import os
os.environ["CARREFOUR_MODE"] = "prod"
import importlib, src.config; importlib.reload(src.config)
from src.config import DATA_PROCESSED

# Load prod labels and UMAP embedding
prod_labels = pl.read_parquet(DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet")
prod_umap   = pl.read_parquet(DATA_PROCESSED / "umap_cluster_item2vec.parquet")

experiment_scorecard(prod_labels, prod_umap, method_name="prod_final")
```

The production scorecard must show:
- Avg top-5 lift > 9.32× (same bar as dev, adjusted for larger population)
- All tribes ≥ 440 customers (0.04% of 1.48M — a much softer gate than dev)
- Silhouette ≥ 0.20 (production silhouette is typically lower than dev due to population size)

**Finally, run `name_all_tribes()` on the production profiles:**

```python
from src.tribe_namer import name_all_tribes

prod_profiles = pl.read_parquet(DATA_PROCESSED / "tribe_profiles_hdbscan_assigned.parquet")
named = name_all_tribes(prod_profiles)
named.write_parquet(DATA_PROCESSED / "tribe_profiles_final_named.parquet", compression="zstd")
```

This is the final deliverable: `data/processed/tribe_profiles_final_named.parquet` contains the full production segmentation with commercial names, descriptions, and recommended Carrefour actions.

---

## Selecting and Applying the Winner

Run this after all experiments are complete (or at the end of each day to see progress). It reads all accumulated scorecard results, ranks them, and applies the best configuration.

```python
import json, shutil
import polars as pl
from pathlib import Path
from src.config import OUTPUTS, DATA_PROCESSED

RESULTS_FILE = OUTPUTS / "experiment_results.json"
results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))

# Rank: primary = avg_top5_lift, secondary = tribe_count, gate = min_size >= 440 and improved
valid = [r for r in results if r["improved"] and r["min_size"] >= 440]

if not valid:
    print("No experiment beat the baseline yet. Keep experimenting.")
else:
    ranked = sorted(
        valid,
        key=lambda r: (r["avg_top5_lift"], r["tribe_count"]),
        reverse=True,
    )
    winner = ranked[0]

    print(f"\n{'='*62}")
    print(f"  WINNER: {winner['method']}")
    print(f"  Avg top-5 lift: {winner['avg_top5_lift']:.3f}× (baseline: 9.32×)")
    print(f"  Tribe count:    {winner['tribe_count']} (baseline: 11)")
    print(f"  Min tribe size: {winner['min_size']:,} (baseline: 1,894)")
    print(f"{'='*62}")

    print("\nAll improved experiments (ranked):")
    for i, r in enumerate(ranked[:10]):
        print(f"  {i+1:2}. {r['method']:<45} lift={r['avg_top5_lift']:.3f}× tribes={r['tribe_count']}")

# After identifying the winner, promote its label file to the live path:
winner_labels_path = DATA_PROCESSED / f"cluster_labels_{winner['method']}.parquet"
live_labels_path   = DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet"

if winner_labels_path.exists():
    shutil.copy2(live_labels_path, DATA_PROCESSED / "cluster_labels_hdbscan_assigned_pre_sprint.parquet")
    shutil.copy2(winner_labels_path, live_labels_path)
    print(f"\nPromoted {winner_labels_path.name} → cluster_labels_hdbscan_assigned.parquet")
    print("Previous labels saved as cluster_labels_hdbscan_assigned_pre_sprint.parquet")
else:
    print(f"\nWARNING: winner artifact not found at {winner_labels_path}")
    print("Run the final pipeline manually with the winning config and save with cache_path set to the winner method name.")
```

**Important**: the winner-selection code renames artifact files but does not change `configs/dev.yaml` automatically — update the config manually to match the winning experiment's parameters before the final pipeline run. The file name tells you which experiment won; trace it back to the experiment section to find the config values.

---

## Day 3 Integration

Each member shares their best `experiment_scorecard()` output. Agree on a combined config and run the final pipeline.

**Selection order:**
1. Highest avg top-5 lift across all tribes
2. Highest tribe count with all tribes ≥ 3× lift
3. Min tribe size ≥ 440
4. Lowest number of tribes with lift below the current avg (9.32×)

**Compatibility:**
- M1 (embeddings) + M2 (vectors): always compatible — different pipeline stages
- M1/M2 + M5 (features): compatible — M5 concatenates new scalar features to whatever vector M1/M2 produce
- M2 (vectors) + M3 (clustering): always compatible
- M3 (clustering) + M4 (temporal): compatible — temporal is post-hoc validation
- M5 (prod run): runs last, after all other members have identified their winning config

Agree on one `configs/dev.yaml`. One person runs the final pipeline with all four force flags set correctly for the stages that changed. Run `experiment_scorecard()` one final time on the combined result. This result must appear in `outputs/dev/experiment_results.json` with `improved: true`.

Member 4 runs `name_all_tribes()` on the final profiles only after the scorecard confirms improvement.

### Final State Checklist

Before closing the sprint, verify all of the following:

```python
import json, polars as pl
from src.config import DATA_PROCESSED, OUTPUTS

# 1. Winning labels exist and are the live file
assert (DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet").exists()

# 2. Baseline is preserved
assert (DATA_PROCESSED / "cluster_labels_hdbscan_assigned_baseline.parquet").exists()

# 3. Final scorecard beats baseline
results  = json.loads((OUTPUTS / "experiment_results.json").read_text())
final    = sorted([r for r in results if r["improved"]], key=lambda x: x["avg_top5_lift"], reverse=True)
assert final, "No improved result found — sprint did not improve the segmentation"
winner   = final[0]
assert winner["avg_top5_lift"] > 9.32, f"Lift {winner['avg_top5_lift']} not above baseline 9.32"
assert winner["min_size"] >= 440,     f"Min tribe size {winner['min_size']} below threshold"
print(f"Sprint complete. Winner: {winner['method']} | lift={winner['avg_top5_lift']}× | tribes={winner['tribe_count']}")

# 4. Named tribe profiles exist (dev)
assert (DATA_PROCESSED / "tribe_profiles_final_named.parquet").exists(), \
    "Run name_all_tribes() on the final tribe profiles"

# 5. Production run complete (Member 5)
import os; os.environ["CARREFOUR_MODE"] = "prod"
import importlib, src.config; importlib.reload(src.config)
from src.config import DATA_PROCESSED as PROD_DATA
assert (PROD_DATA / "tribe_profiles_final_named.parquet").exists(), \
    "Production run not complete — Member 5 must run the prod pipeline and name_all_tribes()"
```

If any assertion fails, the sprint is not done. Do not merge to the main branch until all assertions pass.

---

## Quick Reference

| File | Purpose |
|---|---|
| `configs/dev.yaml` | All tunable hyperparameters |
| `configs/base.yaml` | Production defaults (read before changing) |
| `src/embeddings.py` | Word2Vec training + basket construction |
| `src/customer_vectors.py` | Per-customer aggregation, `_apply_weight_transform()` |
| `src/dimensionality.py` | UMAP + PCA reduction |
| `src/clustering.py` | HDBSCAN, KMeans, `profile_tribes()` |
| `src/tribe_namer.py` | Claude API tribe naming |
| `data/dev/df_combined.parquet` | Source of truth — never modify or delete |
| `data/dev/tribe_profiles_hdbscan_assigned.parquet` | Current baseline profiles |

## Experiment Summary Matrix

| Member | Experiment | Effort | Expected lift improvement |
|---|---|---|---|
| 1 | Word2Vec window/epoch tuning | Low | Medium — sharper product neighborhoods |
| 1 | Popularity filter tightening | Low | Medium — removes noise from staples |
| 1 | Negative sampling rate | Low | Low-Medium — better rare product embeddings |
| 1 | Embedding dimension sweep | Low | Low-Medium |
| 1 | CBOW vs skip-gram | Low | Unknown — try both |
| 1 | Node2Vec graph embeddings | Medium | High — graph structure captures community membership |
| 2 | BM25 weighting | Low | High — reduces staple domination in all vectors |
| 2 | Recency half-life sweep | Low | Medium — affects all tribes proportionally |
| 2 | Top-N product pooling | Low | Medium — sharpens niche signal in all tribes |
| 2 | NMF topic decomposition | Medium | High — inherently interpretable topics |
| 2 | LDA topic modeling | Medium | High — probabilistic topics, often cleaner than NMF |
| 2 | Autoencoder embeddings | Medium | Medium — non-linear compression |
| 2 | VAE embeddings | Medium-High | High — smooth latent space, better cluster separation |
| 2 | Short/long-term vector split | Medium | Medium — may reveal transitioning customers |
| 3 | UMAP n_neighbors/dims sweep | Low | Medium — changes topology, affects all tribes |
| 3 | HDBSCAN full grid search | Low | High — finer density resolution across the whole space |
| 3 | GMM | Low | Medium — handles elliptical clusters HDBSCAN misses |
| 3 | Bisecting K-Means (surgical) | Low | High — targeted split of any weak tribe |
| 3 | Agglomerative hierarchical | Medium | High — dendrogram reveals natural cut points |
| 3 | OPTICS | Medium | Medium — variable-density clusters |
| 3 | DEC (deep clustering) | High | High — jointly optimizes embedding and cluster assignment |
| 3 | Promo/store feature ablations | Low | Unknown — test both directions |
| 4 | Temporal stability analysis | Medium | N/A — validation only |
| 4 | Lifecycle delta vectors | Medium | Medium — surfaces transitioning behavioral patterns |
| 4 | BERT4Rec sequential | High | Unknown — captures sequential purchase patterns |
| 5 | Price tier affinity features | Low | Medium — separates premium vs budget segments |
| 5 | Basket mission distribution | Low-Medium | Medium — adds orthogonal behavioral dimensions |
| 5 | Category breadth / HHI features | Low | Medium — separates specialists from generalists |
| 5 | Consensus / ensemble clustering | Medium | High — stabilizes all tribes simultaneously |
| 5 | Two-stage hierarchical clustering | Medium | High — resolves dense sub-groups that HDBSCAN merges |
| 5 | Contrastive learning (SimCLR) | Medium-High | High — explicitly optimizes behavioral distinctiveness |
| 5 | Production pipeline run | High (compute) | Final deliverable — 1.48M customers |
