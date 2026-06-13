# Team Experiment Sprint Guide

Last updated: 2026-06-13

This guide explains how teammates should experiment without turning the project back into a messy cache maze. The goal is to think creatively, test ideas in a disciplined way, and promote only evidence-backed settings into the official pipeline.

Use these as the current source of truth:

- [README.md](../README.md)
- [PROJECT_STATUS_AND_ROADMAP.md](PROJECT_STATUS_AND_ROADMAP.md)
- [PROJECT_FILE_INVENTORY.md](PROJECT_FILE_INVENTORY.md)
- [notebook_contract.md](notebook_contract.md)

## Non-Negotiables

- Experiments run in `dev` first.
- Generated artifacts stay out of git.
- Official clustering must remain product-first: products plus quantities, not spend.
- The 10-15 tribe range is a business hypothesis, not a hard clustering target.
- Do not promote a setting because one metric looks good. It must also make product sense.
- Keep experiment folders flat and limited under `outputs/dev/experiments/<experiment_name>/`.

## Before Anyone Experiments

Activate and verify the environment:

```powershell
conda activate carrefour
python -c "import sklearn, hdbscan; print(sklearn.__version__); print('hdbscan ok')"
```

Expected pairing:

- `scikit-learn=1.7.2`
- `hdbscan==0.8.40`

For dev work, teammates need these local files outside git:

```text
data/dev/df_combined.parquet
data/dev/subset_metadata.json
data/processed/customer_kpis.parquet
```

Before running a sandbox, run the official pipeline up to the upstream artifact that the sandbox needs. For example, UMAP-HDBSCAN experiments need Stage 5 feature sets to exist.

## The Experiment Loop

Every experiment should follow the same loop.

1. Write the hypothesis in plain English.

Example: "IDF downweighting may reduce universal staple dominance and improve product-lift distinctiveness without creating too much noise."

2. Pick the sandbox section in `notebooks/04_experiment_sandbox.ipynb`.

Use the section closest to the thing being tested:

| Question | Sandbox section |
|---|---|
| Are product embeddings better with different Word2Vec settings? | Item2Vec training sandbox |
| Do product-neighbor diagnostics improve? | Product embedding validation sandbox |
| Does IDF or vector normalization improve customer vectors? | Customer embedding sandbox |
| Does a feature set look separable before full Stage 6? | Feature-set sandbox |
| Can UMAP-HDBSCAN find cleaner density structure? | Focused UMAP-HDBSCAN sandbox |

3. Edit the trial list inside the notebook cell.

Do not put sandbox grids into YAML. YAML is for the selected official pipeline settings only.

4. Run the sandbox in `dev`.

Sandbox outputs go here:

```text
outputs/dev/experiments/<experiment_name>/
```

Each sandbox writes a canonical Parquet plus compact summaries:

```text
<experiment_name>_summary.csv
<experiment_name>_summary.md
```

5. Inspect the summary first, then the detailed Parquet only if needed.

The summary ranks trials using a lightweight heuristic. It is a screening tool, not the final decision.

6. Decide: reject, iterate, or promote.

Use this rule:

| Decision | Meaning | Next action |
|---|---|---|
| Reject | Fails metrics, creates obvious noise, or has weak product logic | Leave it in the sandbox and document why |
| Iterate | Interesting but incomplete | Run a smaller follow-up grid around the promising range |
| Promote | Clearly improves evidence and product story | Move the setting into YAML and rerun the official pipeline |

7. If promoted, rerun the official pipeline from the affected upstream stage.

| Changed area | Rerun official pipeline from |
|---|---|
| Item2Vec settings | Stage 2 |
| Product embedding table | Stage 3 |
| Customer vector aggregation or IDF | Stage 4 |
| Feature-set composition | Stage 5 |
| UMAP or clustering hyperparameters | Stage 6 |

8. Judge the promoted setting with official evidence.

A promoted setting is not accepted until Stage 6, Stage 7, and tribe profiles are reviewed together.

Look at:

- Stage 6 candidate summary.
- Stage 7 validity and stability diagnostics.
- Cluster count, noise rate, balance, and tiny-cluster risk.
- Product and sector lift profiles.
- Whether tribes can be named commercially without forcing a story.
- Whether the result is reproducible enough to explain to the client.

9. Only then update `configs/base.yaml`, `configs/dev.yaml`, or `configs/prod.yaml`.

Prod should inherit the same modeling recipe as dev, with scale-aware sample sizes and thresholds. Do not copy a tiny dev-only threshold into prod blindly.

## How To Read Sandbox Scores

Sandbox scores are helpful shortcuts, not final truth.

| Sandbox | What the score rewards | What still needs human review |
|---|---|---|
| Product embedding validation | Vocabulary coverage, same-sector neighbor coherence, vector norm stability | Cross-sector complements can be valid, so inspect nearest-neighbor examples |
| Customer embedding | Finite values, nonzero vectors, live dimensions, effective dimensionality, norm stability, nearest-neighbor structure | Whether the vector actually creates interpretable tribes downstream |
| Feature set | Quick KMeans separability, balance, finite-value health, dimensional health | Whether the added features are allowed to drive clustering |
| UMAP-HDBSCAN | Quality-gated Stage 6 metrics: silhouette, coverage-adjusted silhouette, Davies-Bouldin, noise, balance | Whether the clusters have clean lift profiles and are stable |

## Must-Do Experiments

These should be run before the team makes a final modeling decision.

| Area | Experiment | Why |
|---|---|---|
| Customer vectors | Compare `quantity` vs `quantity_idf` vs `equal_idf`, with and without vector normalization | Tests whether universal products are dominating the customer space |
| Feature sets | Compare `embeddings_only` against any behavior-augmented variant | Confirms whether product vectors are enough and whether behavior features distort topology |
| UMAP-HDBSCAN | Try a small focused grid over UMAP dimensions/neighbors and HDBSCAN density settings | Verifies whether the current HDBSCAN result is a hyperparameter issue or an upstream representation issue |
| Stability | Review Stage 7 validity/stability after any promoted candidate | Prevents selecting a pretty but fragile segmentation |
| Profiles | Inspect product and sector lift for the selected candidate | The final tribes must be commercially interpretable |

## Should-Do Experiments

These are strongly recommended if time allows.

| Area | Ideas |
|---|---|
| Item2Vec window | Try smaller windows for tight basket complements and larger windows for broad basket missions |
| Item2Vec dimensions | Compare compact vs richer product vectors, such as 64, 100, and 128 dimensions |
| Item2Vec training style | Compare skip-gram vs CBOW if runtime allows |
| Product filtering | Test whether removing ultra-rare products or extreme universal staples improves downstream tribes |
| Quantity handling | Test capped quantities so bulk purchases do not dominate a customer vector |
| UMAP metric | Keep cosine as the main candidate for embedding vectors, but compare euclidean only as an ablation |
| HDBSCAN conservativeness | Vary `min_cluster_size`, `min_samples`, and `cluster_selection_method` to understand noise vs cluster granularity |
| Tribe naming readiness | For top candidates, check whether top-lift products form a believable business story |

## Creative Can-Do Experiments

These are out-of-the-box ideas. They are welcome, but they need stronger evidence before promotion.

| Idea | Why it might help | Guardrail |
|---|---|---|
| Basket mission tokens | Add product-category or basket-mission tokens to Item2Vec training | Must not replace product identity as the core signal |
| Two-view customer vectors | Compare all-history vector vs recent-history vector | Do not use spend to create the cluster |
| Per-basket then per-customer pooling | Average product vectors within basket first, then aggregate baskets per customer | Check that frequent shoppers do not become over-smoothed |
| Product rarity caps | Downweight both extremely common and extremely rare items | Validate with product-lift profiles |
| Category-aware product filters | Remove noisy product IDs within sectors with poor catalog quality | Must document exactly what was removed |
| Ensemble sanity check | Compare whether multiple reasonable settings reveal similar tribes | Use for confidence, not as a black-box final model |
| Manual product-neighbor review | Inspect nearest neighbors for key products by sector and mission | Qualitative review should explain metrics, not replace them |

## Suggested Team Split

| Focus | Main files | Main responsibility |
|---|---|---|
| M1 Product embeddings | `src/item2vec.py`, `src/embeddings.py`, sandbox notebook | Run Item2Vec and product-neighbor experiments |
| M2 Customer vectors | `src/customer_embeddings.py`, `src/feature_engineering.py`, sandbox notebook | Run quantity, IDF, normalization, and feature-set experiments |
| M3 Dimensionality and clustering | `src/dimensionality.py`, `src/clustering.py`, `src/model_selection.py` | Run focused UMAP-HDBSCAN and review Stage 6 diagnostics |
| M4 Profiling and story | `src/profiling.py`, `src/exports.py`, `src/visualization.py` | Inspect lift profiles, figures, tribe readability, and naming readiness |
| M5 Reproducibility | configs, tests, docs | Keep configs clean, verify environment, run tests, and prepare handoff notes |

## What Each Teammate Should Report

Each experiment owner should write a short result note in the team chat or PR description.

Use this template:

```text
Experiment:
Hypothesis:
Trials:
Output folder:
Best trial:
Why it looked better:
What got worse:
Promote / iterate / reject:
YAML changes needed if promoted:
Official stages to rerun:
```

Do not send only a screenshot. Include the summary path and the reasoning.

## Promotion Checklist

Before a sandbox winner becomes official:

- The winning summary row is better than the baseline on relevant metrics.
- The detailed Parquet does not reveal hidden problems.
- Product-lift profiles make sense.
- Noise and cluster-size balance are acceptable.
- Stage 7 stability does not contradict the result.
- The change is explainable to a business audience.
- YAML is updated only after the evidence review.
- Official pipeline has been rerun from the affected stage.

## Definition of Ready for Final Production Run

- Dev run is rebuilt from the latest promoted settings.
- Stage 6 and Stage 7 agree on a defensible candidate.
- Tribe profiles show distinctive products and sectors by lift, not only raw frequency.
- The selected configuration is committed in YAML.
- `pytest` passes or any test gap is explicitly documented.
- Generated artifacts are kept out of git.
