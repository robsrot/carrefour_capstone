# 2-Day Clustering Improvement Sprint

**Date**: 2026-06-08 | **Mode**: `CARREFOUR_MODE=dev` | **Branch**: one per member

---

## Safety Protocol — Read Before Running Anything

### Rule 1 — Preserve the baseline artifacts first

**Do this once, before any experiment. All members wait until confirmed.**

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

Never run an experiment that writes to the default cache path without an explicit `cache_path=` argument. The default paths (`umap_cluster_item2vec.parquet`, `customer_vectors_weighted.parquet`, etc.) are the live state. Experiments write to versioned paths.

```python
# WRONG — overwrites the live baseline
reduce_umap_cluster(cv, force=True)

# CORRECT — isolated artifact
reduce_umap_cluster(cv, force=True, cache_path=DATA_PROCESSED / "umap_cluster_window5.parquet")
```

Convention: always include the experiment identifier in the artifact name.

### Rule 3 — Every experiment result is written to a JSON scorecard file

The `experiment_scorecard()` function (defined below) writes to `outputs/dev/experiment_results.json`. This file accumulates across all runs and is the single source of truth for which experiment won.

### Rule 4 — The project ends with the best configuration applied

After all experiments, run the winner-selection code block (see "Selecting and Applying the Winner"). This reads `experiment_results.json`, finds the experiment with the highest `avg_top5_lift`, updates `configs/dev.yaml`, does one clean final rebuild with `force=True` on the affected stages only, and runs `experiment_scorecard()` one last time to verify. The final scorecard must show improvement over baseline (9.32× lift, 11 tribes).

### Rule 5 — After changing `dev.yaml`, reload ALL pipeline modules

The notebook master cell runs `importlib.reload(src.config)`, but `src/dimensionality.py`, `src/clustering.py`, `src/embeddings.py`, and `src/customer_vectors.py` each import constants from `src.config` at module load time. Reloading `src.config` does NOT update those already-bound names.

**After editing `dev.yaml` and re-running the master cell, run this reload cell before calling any pipeline function:**

```python
import importlib
import src.config, src.embeddings, src.customer_vectors, src.dimensionality, src.clustering
importlib.reload(src.config)
importlib.reload(src.embeddings)
importlib.reload(src.customer_vectors)
importlib.reload(src.dimensionality)
importlib.reload(src.clustering)

from src.embeddings   import train_word2vec, save_embeddings, load_product_embeddings
from src.customer_vectors import build_customer_vectors, build_customer_vectors_mean, \
    build_customer_vectors_tfidf_svd, build_customer_vectors_hybrid
from src.dimensionality   import reduce_umap_cluster, reduce_umap_viz, reduce_pca
from src.clustering       import cluster_hdbscan, assign_hdbscan_noise_to_nearest_tribe, \
    cluster_kmeans, run_kmeans_baselines, grid_search_hdbscan, evaluate_clustering, profile_tribes
print("All pipeline modules reloaded.")
```

### Rule 6 — Force-rebuild only what changed

| If you changed... | Force rebuild... |
|---|---|
| `configs/dev.yaml` word2vec params | `FORCE_EMBEDDINGS`, `FORCE_VECTORS`, `FORCE_UMAP`, `FORCE_CLUSTERING` |
| `configs/dev.yaml` customer vector params | `FORCE_VECTORS`, `FORCE_UMAP`, `FORCE_CLUSTERING` |
| `configs/dev.yaml` UMAP params | `FORCE_UMAP`, `FORCE_CLUSTERING` |
| `configs/dev.yaml` HDBSCAN params | `FORCE_CLUSTERING` only |
| Custom `cache_path=` experiment | Only the specific function you call with `force=True` |

Each unnecessary full rebuild wastes 40–60 minutes. Never set all four force flags unless you changed Word2Vec.

### Rule 7 — Notebook variable names

The notebook uses `cluster_space` (not `umap_cluster`) as the name for the UMAP embedding passed to clustering functions. When copying code from this document into the notebook, substitute `cluster_space` wherever the sprint plan writes `umap_cluster` as the clustering input.

---

## Repo Handover Status

The pipeline is complete end-to-end in dev mode. The current segmentation produces 11 tribes with avg top-5 lift of 9.32×. The goal for this sprint is to push that number higher — sharper product signals, finer granularity, and stronger behavioral distinctiveness throughout the entire customer space. C5–C7 and C9 are the weakest tribes; C0–C4 and C10 are already strong.

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

---

## 2-Day Schedule and Dependency Map

The dependency chain controls what can run in parallel. M2, M3, M4, and M5 can all start Day 1 on the existing baseline artifacts — none of them wait for M1 to finish.

```
pipeline dependency chain:
  Word2Vec model  (M1 changes here)
    └─> product_embeddings
          └─> customer_vectors  (M2 changes here)
                └─> umap_cluster  (M3 UMAP changes here)
                      └─> cluster_labels  (M3 HDBSCAN/GMM/Bisect changes here)
                            └─> tribe_profiles
                                  └─> tribe_names + cards  (M4 lands here)

M5 feature engineering reads df_combined directly — no upstream dependency.
M5 production run starts Day 2 after the dev winner is confirmed.
```

```
DAY 1 — all parallel after backup (Step 0)
─────────────────────────────────────────────────────────────────
M1  window=5 rebuild (~50min) → scorecard
    if improved: window=3 or popularity filter

M2  BM25 on baseline embeddings (~40min) → scorecard
    Top-N=50 (~40min) → scorecard
    recency half-life sweep (if time)
    ↳ if M1 finishes with improvement: re-run BM25 on M1 embeddings

M3  HDBSCAN grid on baseline UMAP (~20min, no rebuild) → pick candidates
    GMM on baseline UMAP (~10min/k, no rebuild)
    Bisecting K-Means on weak tribes (~15min/tribe, no rebuild)
    UMAP sweep on M2's best vectors (~35min/config)  ← wait for M2 BM25

M4  qualitative review of M3 bisecting results
    tribe card template prep
    baseline 2D UMAP verification

M5  HHI + diversity features from df_combined (~60min)
    basket mission features (~45min)
    price tier affinity features (~60min)

END OF DAY 1: share all scorecards → agree on combined winner config
─────────────────────────────────────────────────────────────────
DAY 2 MORNING — sequential
  one person: combined clean rebuild with winner config (~60–90min)
  confirm dev scorecard beats baseline before proceeding

DAY 2 AFTERNOON — parallel
  M5: START PRODUCTION RUN (2–4 hours) ← must start by early afternoon
  M4: name_all_tribes() → tribe cards → 2D UMAP scatter (dev)

DAY 2 EVENING
  prod run completes → prod scorecard → name_all_tribes() on prod profiles
  final state checklist → commit
─────────────────────────────────────────────────────────────────
```

**The production run is the final deliverable.** It cannot start until the dev winner is confirmed. Do not let Day 2 morning turn into more experiments — it is a single clean rebuild and scorecard check, then hand off to M5.

---

## Shared Evaluation Function

Paste into a notebook cell near the top and run it once. Call `experiment_scorecard()` after every clustering result.

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
    _save_result(result)
    return result
```

---

## Member 1 — Product Embeddings

**Files**: `src/embeddings.py`, `configs/dev.yaml`  
**Rebuild flags**: `FORCE_EMBEDDINGS=True`, `FORCE_VECTORS=True`, `FORCE_UMAP=True`, `FORCE_CLUSTERING=True`  
**Artifact naming**: `method_name` of the form `m1_<descriptor>`, e.g. `m1_window5`.  
**Day 1 start**: immediately after Step 0 backup.  
**Blocks**: M2's optional cross-track re-run — only if M1 produces a clear lift improvement.  
**Does NOT block**: M2's initial BM25/Top-N, M3, M4, M5 — all start on baseline embeddings.  
**Run order**: window size first (window=5), measure, then window=3 if needed. Popularity filter only after window is settled and improved. Each full rebuild is ~50 min — do not chain experiments without measuring first.  
**Estimated time per experiment**: ~50 min (Word2Vec ~10 min + vectors ~5 min + UMAP ~25 min + clustering + profiling ~10 min).

The product embedding is the signal everything else builds on. The current Word2Vec uses `window=10` (base.yaml). Smaller windows enforce stricter co-purchase proximity, so products must appear together in the same section of a basket rather than anywhere in the same trolley.

---

### 1. Window Size

Try `window: 5` first — this is the standard Item2Vec recommendation and the most likely single improvement.

Edit `configs/dev.yaml`:
```yaml
word2vec:
  window: 5    # current: 10 (base.yaml)
  epochs: 5    # keep dev override
```

Reload modules (Rule 5), then run with all four force flags. Save the cluster result to an isolated path:

```python
core = cluster_hdbscan(cluster_space, force=True,
           cache_path=DATA_PROCESSED / "cluster_core_m1_window5.parquet")
full = assign_hdbscan_noise_to_nearest_tribe(cluster_space, core, force=True,
           cache_path=DATA_PROCESSED / "cluster_labels_m1_window5.parquet")
experiment_scorecard(full, cluster_space, method_name="m1_window5")
```

If `window=5` does not improve lift, try `window=3`. If `window=5` improved, also try `window=7` to check whether the optimum is between 5 and 10.

---

### 2. Popularity Filter Tightening

**Prerequisite**: window sweep complete and confirmed improvement. Only run this if the window change improved lift — otherwise the two experiments will compound and you won't know which helped.

Edit `configs/dev.yaml`:
```yaml
product_popularity:
  max_basket_share: 0.10    # was 0.15
  max_customer_share: 0.25  # was 0.40
```

This removes more universal staples from basket sentences, forcing the model to learn from distinctive products. Combine with the best window value from step 1. Use artifact name `m1_window5_filter` (or whichever window won).

---

### 3. Quick Follow-ups (only if steps 1–2 finish before end of Day 1)

These are all config-only changes and each requires a full rebuild (~50 min). Only run one if time clearly allows.

- **Epochs**: try `epochs: 10` (current dev override is `5`, base is `10`). More training passes on the same data.
- **CBOW vs skip-gram**: the `sg=1` flag in `src/embeddings.py` enables skip-gram. Try `sg=0` in `base.yaml` (or patch it directly in the Word2Vec call). CBOW averages context; skip-gram is usually better for rare products.

---

### Sanity Check (run after every experiment)

Use the embedding sanity check cell in the notebook. Pick these probe products:
- A baby food product → top neighbors should be other baby food, not generic dairy
- An Iberian charcuterie product → top neighbors should be other Iberian charcuterie, not mass-market ham
- An organic produce product → top neighbors should be other organic products

If neighbors degrade (random or incoherent), the change hurt the embedding quality — revert.

---

## Member 2 — Customer Vector Representations

**Files**: `src/customer_vectors.py`, `configs/dev.yaml`  
**Rebuild flags**: `FORCE_VECTORS=True`, `FORCE_UMAP=True`, `FORCE_CLUSTERING=True`  
**Artifact naming**: `method_name` of the form `m2_<descriptor>`, e.g. `m2_bm25_k1_1.5`.  
**Day 1 start**: immediately after Step 0 — use the existing `product_embeddings.parquet` baseline. Do NOT wait for M1.  
**Blocks**: M3's UMAP sweep (Part B) — M3 should run their UMAP parameter sweep on M2's best customer vectors, not the baseline.  
**Does NOT block**: M3's Part A (HDBSCAN grid, GMM, Bisecting K-Means on baseline UMAP).  
**Run order**: BM25 first (highest expected single lift improvement), then Top-N, then recency half-life sweep if time allows. Cross-track: if M1 finishes with a clear improvement, re-run your best experiment on M1's new embeddings.  
**Estimated time per experiment**: ~40 min (vectors ~5 min + UMAP ~25 min + clustering + profiling ~10 min).

The current customer vector is a recency/frequency-weighted mean of product embeddings with `log1p` transform. Averaging all products a customer has bought blurs their identity — customers who buy 30 different products end up near the centroid of all 30 products' vectors. The experiments below address this directly.

---

### 1. BM25 Saturation Weighting

**Run this first. Highest expected single improvement in the sprint.**

BM25 applies a saturation curve that dampens staple products more aggressively than `log1p`, while preserving the signal from distinctive products.

Add to `_apply_weight_transform()` in `src/customer_vectors.py`:
```python
if transform == "bm25":
    k1 = 1.5
    v = np.maximum(values, 0.0).astype(np.float64)
    return (v / (v + k1)).astype(np.float32)
```

Edit `configs/dev.yaml`:
```yaml
customer_vectors:
  weight_transform: bm25
```

Reload modules, rebuild with `FORCE_VECTORS=True`, `FORCE_UMAP=True`, `FORCE_CLUSTERING=True`.

```python
core = cluster_hdbscan(cluster_space, force=True,
           cache_path=DATA_PROCESSED / "cluster_core_m2_bm25_k1_1.5.parquet")
full = assign_hdbscan_noise_to_nearest_tribe(cluster_space, core, force=True,
           cache_path=DATA_PROCESSED / "cluster_labels_m2_bm25_k1_1.5.parquet")
experiment_scorecard(full, cluster_space, method_name="m2_bm25_k1_1.5")
```

If improved, also try `k1=1.2` and `k1=2.0` — change the constant in the code and re-run with separate artifact names (`m2_bm25_k1_1.2`, `m2_bm25_k1_2.0`).

---

### 2. Top-N Product Pooling

Restrict each customer's vector to their top-N products by interaction weight before averaging. Prevents one-off purchases from diluting the core product signal.

Add a `top_n` parameter to `build_customer_vectors` in `src/customer_vectors.py` that filters to the top-N rows per customer (by weight) before the embedding mean. Default to `None` to preserve existing behavior.

Try `N=50` first, then `N=20` if time allows.

```python
core = cluster_hdbscan(cluster_space, force=True,
           cache_path=DATA_PROCESSED / "cluster_core_m2_topn50.parquet")
full = assign_hdbscan_noise_to_nearest_tribe(cluster_space, core, force=True,
           cache_path=DATA_PROCESSED / "cluster_labels_m2_topn50.parquet")
experiment_scorecard(full, cluster_space, method_name="m2_topn50")
```

---

### 3. Recency Half-Life Sweep (only if M2-1 and M2-2 are done)

Edit `configs/dev.yaml`:
```yaml
customer_vectors:
  recency_halflife_days: 30    # current: 60
```

Try `30` and `90`. A shorter half-life amplifies recent purchase behavior; a longer one treats the full 6-month window more evenly. Run `experiment_scorecard()` after each.

---

### Cross-track: Re-run on M1 embeddings (Day 1 afternoon, if M1 improved)

If M1's window sweep produced a confirmed improvement, re-run your best M2 configuration (whichever of BM25/Top-N gave the highest lift) pointing at M1's new `product_embeddings.parquet`. Name the combined run `m1m2_window5_bm25` (substitute actual experiment IDs). Call `experiment_scorecard()`. This is the most valuable combined test of the sprint.

---

## Member 3 — Dimensionality Reduction & Clustering

**Files**: `src/dimensionality.py`, `src/clustering.py`, `configs/dev.yaml`  
**Rebuild flags**: UMAP changes → `FORCE_UMAP=True`, `FORCE_CLUSTERING=True`. Clustering-only changes → `FORCE_CLUSTERING=True`.  
**Artifact naming**: `method_name` of the form `m3_<descriptor>`, e.g. `m3_hdbscan_mcs50_ms3_leaf` or `m3_gmm_k15`.  
**Day 1 Part A** (HDBSCAN grid, GMM, Bisecting K-Means): start immediately after Step 0 — all run on the existing `umap_cluster_item2vec.parquet`. No rebuild needed.  
**Day 1 Part B** (UMAP sweep): start after M2's BM25 vectors are ready. Run the UMAP sweep on M2's best customer vectors rather than the baseline — the combination of better vectors and tuned UMAP is more valuable than tuning UMAP alone.  
**Run order**: HDBSCAN grid first (fastest, no code changes), then GMM, then Bisecting K-Means, then UMAP sweep (after M2).

---

### 1. HDBSCAN Full Grid Search

**Run this first. No rebuild needed — operates on the existing baseline UMAP embedding.**

Expand `configs/dev.yaml`:
```yaml
hdbscan_grid:
  min_cluster_size: [15, 25, 50, 75, 100, 150, 200]
  min_samples: [1, 3, 5, 10]
  cluster_method: [leaf, eom]
```

Set `RUN_HDBSCAN_GRID_SEARCH = True` in the notebook master cell. After the grid runs, filter for good candidates:

```python
hdb_grid = pl.read_parquet(DATA_PROCESSED / f"hdbscan_grid_{VECTOR_SOURCE}_{SELECTED_REDUCER}_full.parquet")
candidates = hdb_grid.filter(
    (pl.col("n_clusters") >= 12) & (pl.col("n_clusters") <= 20) & (pl.col("noise_pct") < 15)
).sort("noise_pct")
print(candidates.select(["min_cluster_size","min_samples","cluster_method","n_clusters","noise_pct","silhouette"]))
```

For each promising candidate (pick 3–4 configs), run full clustering and score:

```python
core = cluster_hdbscan(cluster_space,
           min_cluster_size=50, min_samples=3, cluster_selection_method="leaf",
           force=True, cache_path=DATA_PROCESSED / "cluster_core_m3_mcs50_ms3_leaf.parquet")
full = assign_hdbscan_noise_to_nearest_tribe(cluster_space, core, force=True,
           cache_path=DATA_PROCESSED / "cluster_labels_m3_mcs50_ms3_leaf.parquet")
experiment_scorecard(full, cluster_space, method_name="m3_mcs50_ms3_leaf")
```

`cluster_method=leaf` finds more, smaller, tighter clusters. `eom` merges sub-clusters more aggressively. Test both across `min_cluster_size` values. The grid output shows which density resolution produces the cleanest product stories without excessive noise.

---

### 2. Gaussian Mixture Models (GMM)

**No UMAP rebuild needed. Runs on the existing baseline UMAP embedding.**

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

Try `n_components=12`, `15`, `18`. Compare BIC scores (lower = better model fit). Call `experiment_scorecard()` on each.

```python
for k in [12, 15, 18]:
    labels = cluster_gmm(cluster_space, n_components=k, force=True,
                 cache_path=DATA_PROCESSED / f"cluster_labels_m3_gmm_k{k}.parquet")
    experiment_scorecard(labels, cluster_space, method_name=f"m3_gmm_k{k}")
```

---

### 3. Bisecting K-Means — Surgical Tribe Splitting

**No UMAP rebuild needed. Works on baseline cluster labels and baseline UMAP.**

Apply bisecting K-Means to the four weakest tribes (C5, C6, C7, C9 — lift 4.2×–7.4×). Run each independently first, then combine the winning splits into one final merged label set.

M4 provides qualitative review of each split result in parallel — check their feedback before deciding whether to combine splits.

```python
from sklearn.cluster import BisectingKMeans
import numpy as np

baseline = pl.read_parquet(DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet")
umap     = pl.read_parquet(DATA_PROCESSED / "umap_cluster_item2vec.parquet")
dim_cols = [c for c in umap.columns if c not in ("cliente", "promo_rate")]

weak_tribes = [5, 6, 7, 9]

for tribe_id in weak_tribes:
    for n_splits in [2, 3]:
        tribe_customers = baseline.filter(pl.col("cluster") == tribe_id)["cliente"]
        tribe_umap      = umap.filter(pl.col("cliente").is_in(tribe_customers))
        X_tribe         = tribe_umap.select(dim_cols).to_numpy().astype(np.float32)

        bkm = BisectingKMeans(
            n_clusters=n_splits, random_state=RANDOM_SEED,
            bisecting_strategy="largest_cluster"
        )
        sub_labels = bkm.fit_predict(X_tribe)
        sub_df = tribe_umap.select("cliente").with_columns(
            pl.Series("cluster", (tribe_id * 10 + sub_labels).astype(np.int32)),
            pl.lit(0.0).alias("promo_rate"),
        )
        others = baseline.filter(~pl.col("cluster").is_in([tribe_id]))
        merged = pl.concat([others, sub_df]).sort("cliente")
        experiment_scorecard(merged, umap, method_name=f"m3_bisect_C{tribe_id}_k{n_splits}")
```

After identifying the best split for each weak tribe, combine all winning splits into one final label set and score:

```python
experiment_scorecard(combined_merged, umap, method_name="m3_bisect_all_combined")
```

---

### 4. UMAP Hyperparameter Sweep

**Prerequisite: M2's BM25 vectors are ready. Run on M2's best customer vector output, not the baseline.**

If M2's vectors are not ready yet, start on the baseline and re-run on M2's vectors once they arrive.

Edit `configs/dev.yaml` (one change at a time, one rebuild each):

```yaml
# n_neighbors — current: 20 in dev.yaml; try these
umap:
  n_neighbors: 15    # more local structure, potentially more distinct tribes

# cluster_dims — current: 50; try these
umap:
  cluster_dims: 30   # tighter manifold, may separate dense sub-groups

# min_dist_cluster — current: 0.05; try this
umap:
  min_dist_cluster: 0.0   # packs embeddings tighter, sharpens density peaks
```

Reload modules between each change. For each config, rebuild UMAP and re-run HDBSCAN with the best parameters found in step 1:

```python
umap_nb15 = reduce_umap_cluster(cv_for_umap, force=True,
                cache_path=DATA_PROCESSED / "umap_cluster_m3_nb15.parquet")
core = cluster_hdbscan(umap_nb15, force=True,
           cache_path=DATA_PROCESSED / "cluster_core_m3_nb15.parquet")
full = assign_hdbscan_noise_to_nearest_tribe(umap_nb15, core, force=True,
           cache_path=DATA_PROCESSED / "cluster_labels_m3_nb15.parquet")
experiment_scorecard(full, umap_nb15, method_name="m3_umap_nb15")
```

Run at most 3–4 configs. Each is ~35 min.

---

### 5. Feature Weight Ablations (only if steps 1–4 are done and time allows)

The current dev config has `promo=0.0` and `store=0.0`. Small non-zero weights add a behavioral dimension — but if they dominate topology the tribes become "promo hunters" or "geography clusters" rather than product tribes. Test carefully.

```yaml
feature_weights:
  promo: 0.25    # try this first; revert if lift degrades
```

Only `FORCE_UMAP=True` and `FORCE_CLUSTERING=True` are needed. Call `experiment_scorecard()`. Revert the config change if lift does not improve.

---

## Member 4 — Tribe Profiles and Presentation

**Files**: `src/tribe_namer.py`, notebook  
**Artifact naming**: `method_name` of the form `m4_<descriptor>`.  
**Day 1**: preparation and qualitative support for M3's bisecting K-Means experiments.  
**Day 2**: all the substantive output — profiling, naming, tribe cards, 2D scatter, prod naming.  
**Run order**: qualitative bisecting review (Day 1) → template prep → combined rebuild complete → profile → name → cards → 2D scatter (all Day 2). Re-run naming on prod profiles after M5's prod run finishes.

M4 does not block any other member on Day 1. On Day 2, M4's tribe cards are the final presentation artifact — they should be ready before the sprint closes.

---

### Day 1 — Qualitative Review of Bisecting K-Means

As M3 runs each bisecting split, inspect the scorecard output. For each sub-tribe, look at the top-product lift and check whether the split produces a coherent product story or merely a mechanical improvement in the metric.

Flag splits that should be rejected:
- Sub-tribes that share the same top products — the split is arbitrary
- Sub-tribes where lift improves but the top products are incoherent (random products with no thematic link)
- Sub-tribes with n < 440 customers (below the minimum gate)

Communicate directly to M3: "C6 split into k=2 produces clean organic vs conventional sub-tribes — keep" vs "C9 split into k=3 is arbitrary — reject k=3, use k=2".

---

### Day 1 — Tribe Card Template

Prepare a reusable Python function that formats a tribe row into a presentation card. This runs on Day 2 with the final named profiles.

```python
def print_tribe_card(row: dict) -> None:
    top5 = list(zip(
        (row.get("top_products") or [])[:5],
        (row.get("top_lifts")    or [])[:5],
    ))
    print(f"\n{'─'*60}")
    print(f"C{row['cluster']:>2} | {row.get('tribe_name', '?')}")
    print(f"{'─'*60}")
    print(f"  Customers : {row['n_customers']:,}")
    print(f"  Revenue   : {row.get('revenue_share', 0):.1%}")
    print(f"  Avg basket: €{row.get('avg_basket_value', 0):.1f}  |  "
          f"Promo rate: {row.get('avg_promo_rate', 0):.1%}")
    print(f"  Top products by lift:")
    for prod, lift in top5:
        print(f"    {lift:.1f}×  {prod[:52]}")
    desc = row.get("tribe_description") or row.get("description") or ""
    if desc:
        print(f"  Description: {desc}")
    action = row.get("carrefour_action") or ""
    if action:
        print(f"  Action: {action}")
```

---

### Day 1 — Baseline 2D UMAP Verification

Confirm the 2D scatter plot pipeline works before Day 2. Run it on the baseline labels now so Day 2 only requires a single re-run with final labels.

```python
from src.dimensionality import reduce_umap_viz
import matplotlib.pyplot as plt, matplotlib.cm as cm, numpy as np

umap_2d        = reduce_umap_viz(cluster_space, force=False)
baseline_labels = pl.read_parquet(DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet")
plot_df         = umap_2d.join(baseline_labels.select(["cliente", "cluster"]), on="cliente")

clusters = sorted(plot_df["cluster"].unique().to_list())
colors   = cm.tab20(np.linspace(0, 1, len(clusters)))

fig, ax = plt.subplots(figsize=(14, 10))
for cid, color in zip(clusters, colors):
    sub = plot_df.filter(pl.col("cluster") == cid)
    ax.scatter(sub["x"].to_numpy(), sub["y"].to_numpy(),
               s=0.8, alpha=0.4, color=color, label=f"C{cid}")
ax.legend(markerscale=8, fontsize=8, loc="upper right")
ax.set_title("Customer Tribes — Baseline (UMAP 2D)")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(OUTPUTS / "tribe_map_baseline.png", dpi=150)
plt.show()
```

---

### Day 2 — Profile Winning Tribes

**Prerequisite**: Day 2 combined rebuild complete and scorecard confirmed.

```python
from src.clustering import profile_tribes

winner_labels = pl.read_parquet(DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet")
profiles      = profile_tribes(winner_labels, method_name="combined_final_dev", force=True)
profiles.write_parquet(DATA_PROCESSED / "tribe_profiles_combined_final_dev.parquet",
                       compression="zstd")
```

---

### Day 2 — Name All Tribes via Claude API

`src/tribe_namer.py` is already implemented:

```python
from src.tribe_namer import name_all_tribes

profiles = pl.read_parquet(DATA_PROCESSED / "tribe_profiles_combined_final_dev.parquet")
named    = name_all_tribes(profiles)
named.write_parquet(DATA_PROCESSED / "tribe_profiles_final_named.parquet", compression="zstd")
print(named.select(["cluster", "tribe_name", "n_customers"]))
```

---

### Day 2 — Tribe Cards

```python
named_profiles = pl.read_parquet(DATA_PROCESSED / "tribe_profiles_final_named.parquet")
for row in named_profiles.sort("cluster").to_dicts():
    print_tribe_card(row)
```

---

### Day 2 — Final 2D UMAP Scatter

```python
from src.dimensionality import reduce_umap_viz
import matplotlib.pyplot as plt, matplotlib.cm as cm, numpy as np

umap_2d = reduce_umap_viz(
    cluster_space,
    force=True,
    cache_path=DATA_PROCESSED / "umap_viz_final_2d.parquet",
)

named_profiles = pl.read_parquet(DATA_PROCESSED / "tribe_profiles_final_named.parquet")
id_to_name     = dict(zip(named_profiles["cluster"].to_list(),
                          named_profiles["tribe_name"].to_list()))

final_labels = pl.read_parquet(DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet")
plot_df      = umap_2d.join(final_labels.select(["cliente", "cluster"]), on="cliente")

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

### Day 2 — Re-run Tribe Naming on Production Profiles

**Prerequisite**: M5's production run complete.

```python
import os; os.environ["CARREFOUR_MODE"] = "prod"
import importlib, src.config; importlib.reload(src.config)
from src.config import DATA_PROCESSED as PROD_DATA
from src.tribe_namer import name_all_tribes

prod_profiles = pl.read_parquet(PROD_DATA / "tribe_profiles_hdbscan_assigned.parquet")
named_prod    = name_all_tribes(prod_profiles)
named_prod.write_parquet(PROD_DATA / "tribe_profiles_final_named.parquet", compression="zstd")
```

---

## Member 5 — Feature Engineering & Production

**Files**: `src/customer_vectors.py`, notebook  
**Artifact naming**: `method_name` of the form `m5_<descriptor>`.  
**Day 1 start**: immediately after Step 0 — all feature engineering reads `df_combined.parquet` directly and has no upstream dependency on M1/M2/M3.  
**Day 1 output**: three feature parquet files ready to concatenate to the winning vector on Day 2 (only if the Day 1 sync decides to include them).  
**Day 2**: production pipeline run — start as early as possible after the dev winner is confirmed. This is the final deliverable and takes 2–4 hours.  
**Run order**: HHI features first (fastest), then basket mission, then price tier. Start prod run immediately on Day 2 after dev is confirmed — do not wait.

---

### 1. Category Breadth and Shopping Diversity Features

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
    ])
)

for col in ["n_categories", "n_distinct_products", "n_baskets", "avg_basket_value"]:
    min_v = customer_diversity[col].min()
    max_v = customer_diversity[col].max()
    customer_diversity = customer_diversity.with_columns(
        ((pl.col(col) - min_v) / (max_v - min_v + 1e-8)).alias(f"{col}_norm")
    )

customer_diversity.write_parquet(
    DATA_PROCESSED / "customer_diversity_features.parquet", compression="zstd"
)
print(f"Written: {len(customer_diversity):,} customers")
```

`n_categories` and `n_distinct_products` separate category specialists (niche tribes, typically high-lift) from generalists (large catch-all tribes). These 4 normalised scalar features can be concatenated to any customer vector before UMAP.

---

### 2. Basket Mission Distribution

Classify each basket by mission, then represent each customer by their basket mission distribution.

```python
basket_stats = (
    df.group_by(["cliente", "idtransac"])
    .agg([
        pl.len().alias("n_items"),
        pl.col("importe").sum().alias("basket_value"),
        pl.col("categoria1").n_unique().alias("n_categories"),
    ])
)

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
mission_cols = [c for c in customer_missions.columns if c != "cliente"]
total = customer_missions.select(mission_cols).sum_horizontal()
for col in mission_cols:
    customer_missions = customer_missions.with_columns(
        (pl.col(col) / total).alias(f"mission_{col}")
    )
customer_missions = customer_missions.select(
    ["cliente"] + [f"mission_{c}" for c in mission_cols]
)
customer_missions.write_parquet(
    DATA_PROCESSED / "customer_mission_features.parquet", compression="zstd"
)
```

---

### 3. Price Tier Affinity

For each customer, compute the fraction of spend in each price tier (budget / mid / premium).

```python
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
tier_cols = [c for c in customer_price.columns if c != "cliente"]
total     = customer_price.select(tier_cols).sum_horizontal()
for col in tier_cols:
    customer_price = customer_price.with_columns(
        (pl.col(col) / total).alias(f"tier_{col}")
    )
customer_price = customer_price.select(["cliente"] + [f"tier_{c}" for c in tier_cols])
customer_price.write_parquet(
    DATA_PROCESSED / "customer_price_tier_features.parquet", compression="zstd"
)
```

---

### 4. Feature Concatenation Helper

Prepare the helper to augment the winning customer vector with M5 features before UMAP. Used on Day 2 if the Day 1 sync decides to include M5 features.

```python
import polars as pl, numpy as np

def augment_vectors_with_features(cv: pl.DataFrame, feature_paths: list) -> pl.DataFrame:
    """Concatenate scalar feature columns onto a customer vector DataFrame."""
    result = cv
    for path in feature_paths:
        feats = pl.read_parquet(path)
        scalar_cols = [c for c in feats.columns if c != "cliente"]
        result = result.join(feats.select(["cliente"] + scalar_cols), on="cliente", how="left")
        for col in scalar_cols:
            result = result.with_columns(pl.col(col).fill_null(0.0))

    vec_array   = np.array(result["vector"].to_list(), dtype=np.float32)
    extra_cols  = [c for c in result.columns if c not in ("cliente", "vector")]
    extra_array = result.select(extra_cols).to_numpy().astype(np.float32)
    combined    = np.hstack([vec_array, extra_array])

    return result.select("cliente").with_columns(
        pl.Series("vector", combined.tolist(), dtype=pl.List(pl.Float32))
    )
```

---

### 5. Production Pipeline Run (Day 2 — start immediately after dev winner confirmed)

**This is the most important output of the sprint. Start it before working on anything else on Day 2 afternoon.**

**Prerequisites before starting:**
1. Day 2 combined dev rebuild complete and scorecard confirms improvement.
2. `configs/dev.yaml` matches the winning configuration.
3. `git status` is clean on your branch.

**Verify prod data is present:**
```python
import polars as pl
df = pl.scan_parquet("data/processed/df_combined.parquet")
print(df.collect(engine="streaming").shape)
# Expected: (191017715, 13)
```

**Switch to prod mode and run:**
```powershell
$env:CARREFOUR_MODE = "prod"
jupyter notebook notebooks/03_ml_pipeline.ipynb
```

In the notebook master cell, set only the force flags corresponding to what the winning experiment changed:

```python
# Example: if the winner changed word2vec window + BM25 weighting
FORCE_EMBEDDINGS = True    # word2vec changed
FORCE_VECTORS    = True    # vector transform changed
FORCE_UMAP       = True    # vectors changed, UMAP must refit
FORCE_CLUSTERING = True    # always rebuild final labels in prod
```

The production UMAP fits on a 300k-customer sample and transforms all 1.48M customers. HDBSCAN fits on 300k and assigns the rest via nearest-neighbour. This run will take **2–4 hours**.

**After the prod run completes, run the production scorecard:**

```python
import os
os.environ["CARREFOUR_MODE"] = "prod"
import importlib, src.config; importlib.reload(src.config)
from src.config import DATA_PROCESSED

prod_labels = pl.read_parquet(DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet")
prod_umap   = pl.read_parquet(DATA_PROCESSED / "umap_cluster_item2vec.parquet")
experiment_scorecard(prod_labels, prod_umap, method_name="prod_final")
```

The production scorecard must show:
- Avg top-5 lift > 9.32×
- All tribes ≥ 440 customers (0.04% of 1.48M — a much softer gate than dev)
- Silhouette ≥ 0.20 (production silhouette is typically lower than dev due to population size)

---

## Day 2 Integration

### Morning: Combined Clean Rebuild

**One designated person runs this.** Everyone else prepares tribe card templates and verifies prod data is accessible.

1. Update `configs/dev.yaml` with the combined winning configuration from the Day 1 sync.
2. Reload all modules (Rule 5).
3. Set force flags only for the stages that changed (Rule 6).
4. Run the full pipeline through to `assign_hdbscan_noise_to_nearest_tribe()`.
5. Call `experiment_scorecard()` with `method_name="combined_final_dev"`.
6. Confirm `improved: true` before proceeding.

If the combined result does not beat the baseline: check that `dev.yaml` was actually updated, modules were reloaded, and force flags hit the right stages. If still no improvement, fall back to the single best individual experiment as the winning config.

### Afternoon: Parallel Deliverables

Once the dev scorecard is confirmed:
- **M5 starts the production run immediately** — this is the longest-running task and blocks the final deliverable.
- **M4 starts tribe profiling, naming, and cards** — use the dev results. The prod run will finish later and M4 re-runs the naming step on prod profiles once M5 is done.

### Selecting the Winner Config

**Compatibility rules**:
- M1 (embeddings) + M2 (vectors): always compatible — sequential pipeline stages.
- M2 (vectors) + M3 (UMAP/clustering): always compatible.
- M5 features + any vector: compatible, but only include M5 features if they improved lift in an isolated test on Day 1. Do not add them speculatively — each additional UMAP dimension can hurt cluster separation.
- M3's best HDBSCAN params should be applied on top of whatever vectors M1+M2 produce.

**Selection order:**
1. Highest avg top-5 lift across all tribes
2. Highest tribe count with all tribes ≥ 3× lift
3. Min tribe size ≥ 440
4. Fewest tribes with lift below the current average

---

## Selecting and Applying the Winner

Run this at the Day 1 sync and again after the Day 2 combined rebuild.

```python
import json, shutil
import polars as pl
from pathlib import Path
from src.config import OUTPUTS, DATA_PROCESSED

RESULTS_FILE = OUTPUTS / "experiment_results.json"
results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))

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

winner_labels_path = DATA_PROCESSED / f"cluster_labels_{winner['method']}.parquet"
live_labels_path   = DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet"

if winner_labels_path.exists():
    shutil.copy2(live_labels_path,
                 DATA_PROCESSED / "cluster_labels_hdbscan_assigned_pre_sprint.parquet")
    shutil.copy2(winner_labels_path, live_labels_path)
    print(f"\nPromoted {winner_labels_path.name} → cluster_labels_hdbscan_assigned.parquet")
    print("Previous labels saved as cluster_labels_hdbscan_assigned_pre_sprint.parquet")
else:
    print(f"\nWARNING: winner artifact not found at {winner_labels_path}")
    print("Run the final pipeline manually with the winning config and save with cache_path set.")
```

**Important**: the winner-selection code renames artifact files but does not change `configs/dev.yaml` automatically. Update the config manually to match the winning experiment's parameters before the prod pipeline run.

---

## Final State Checklist

```python
import json, polars as pl
from src.config import DATA_PROCESSED, OUTPUTS

# 1. Winning labels exist and are the live file
assert (DATA_PROCESSED / "cluster_labels_hdbscan_assigned.parquet").exists()

# 2. Baseline is preserved
assert (DATA_PROCESSED / "cluster_labels_hdbscan_assigned_baseline.parquet").exists()

# 3. Final dev scorecard beats baseline
results  = json.loads((OUTPUTS / "experiment_results.json").read_text())
final    = sorted([r for r in results if r["improved"]], key=lambda x: x["avg_top5_lift"], reverse=True)
assert final, "No improved result found — sprint did not improve the segmentation"
winner   = final[0]
assert winner["avg_top5_lift"] > 9.32, f"Lift {winner['avg_top5_lift']} not above baseline 9.32"
assert winner["min_size"] >= 440,      f"Min tribe size {winner['min_size']} below threshold"
print(f"Sprint complete. Winner: {winner['method']} | lift={winner['avg_top5_lift']}× | tribes={winner['tribe_count']}")

# 4. Named dev tribe profiles exist
assert (DATA_PROCESSED / "tribe_profiles_final_named.parquet").exists(), \
    "Run name_all_tribes() on the final dev tribe profiles"

# 5. Production run complete
import os; os.environ["CARREFOUR_MODE"] = "prod"
import importlib, src.config; importlib.reload(src.config)
from src.config import DATA_PROCESSED as PROD_DATA
assert (PROD_DATA / "tribe_profiles_final_named.parquet").exists(), \
    "Production run not complete — M5 must run the prod pipeline and name_all_tribes()"
```

If any assertion fails, the sprint is not done. Do not merge until all assertions pass.

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

---

## Experiment Summary Matrix

| Member | Experiment | Effort | Est. time | Expected lift | Prerequisite |
|---|---|---|---|---|---|
| 1 | Word2Vec window=5 (try 3 if needed) | Low | ~50 min | Medium-High | Step 0 |
| 1 | Popularity filter tightening | Low | ~50 min | Medium | M1 window improved |
| 1 | Epochs / CBOW / dimension (optional) | Low | ~50 min each | Low-Medium | M1 window done |
| 2 | BM25 weighting (k1=1.5) | Low | ~40 min | High | Step 0 |
| 2 | Top-N product pooling (N=50) | Low | ~40 min | Medium | Step 0 |
| 2 | Recency half-life sweep (30, 90) | Low | ~40 min each | Medium | Step 0 (if time) |
| 2 | BM25 re-run on M1 embeddings | Low | ~40 min | High | M1 improved |
| 3 | HDBSCAN full grid search | Low | ~20 min | High | Step 0 |
| 3 | GMM (k=12, 15, 18) | Low | ~10 min each | Medium | Step 0 |
| 3 | Bisecting K-Means (weak tribes) | Low | ~15 min each | High | Step 0 |
| 3 | UMAP n_neighbors + dims sweep | Low | ~35 min each | Medium | M2 BM25 done |
| 3 | Feature weight ablations (promo/store) | Low | ~35 min each | Unknown | M3-4 done (if time) |
| 4 | Bisecting qualitative review | — | ongoing | N/A (supports M3) | Step 0 |
| 4 | Tribe card template | — | ~1 hr | N/A (deliverable) | Step 0 |
| 4 | Baseline 2D UMAP verification | — | ~30 min | N/A (deliverable) | Step 0 |
| 4 | Profile + name + cards + 2D scatter | — | ~2 hr | N/A (deliverable) | Dev rebuild confirmed |
| 4 | Re-run naming on prod profiles | — | ~30 min | N/A (deliverable) | Prod run complete |
| 5 | HHI + diversity features | Low | ~60 min | Medium | Step 0 |
| 5 | Basket mission features | Low | ~45 min | Medium | Step 0 |
| 5 | Price tier affinity | Low | ~60 min | Medium | Step 0 |
| 5 | **Production pipeline run** | High (compute) | **2–4 hours** | **Final deliverable** | Dev confirmed |

**Out of scope for this sprint**: Node2Vec, NMF, LDA, autoencoder, VAE, DEC, contrastive learning, BERT4Rec, consensus clustering (5 full runs), two-stage hierarchical clustering, OPTICS, agglomerative hierarchical, temporal stability, lifecycle delta vectors, short/long-term vector split.
