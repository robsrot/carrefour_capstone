# src/config.py — single source of truth for all pipeline constants.
#
# Edit hyperparameters in configs/base.yaml (prod values) or configs/dev.yaml (dev overrides).
# Do NOT hardcode values here — this file is a loader, not a config store.
#
# Mode switching:
#   $env:CARREFOUR_MODE = "dev"   → uses data/dev/, models/dev/, outputs/dev/
#   $env:CARREFOUR_MODE = "prod"  → uses data/processed/, models/prod/, outputs/prod/ (default)

import os
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_MODE = os.environ.get("CARREFOUR_MODE", "prod").lower()
if _MODE not in ("dev", "prod"):
    raise ValueError(f"CARREFOUR_MODE must be 'dev' or 'prod', got '{_MODE!r}'")


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override values into base config."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_config() -> dict:
    base_path = ROOT / "configs" / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(
            f"configs/base.yaml not found at {base_path}. "
            "Ensure you are running from the project root and the configs/ directory exists."
        )
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    override_path = ROOT / "configs" / f"{_MODE}.yaml"
    if override_path.exists():
        with open(override_path) as f:
            override = yaml.safe_load(f) or {}
        _deep_update(cfg, override)

    return cfg


_cfg = _load_config()

# ── Mode & Paths ───────────────────────────────────────────────────────────────
MODE           = _MODE
DATA_RAW       = ROOT / "data" / "raw" / "parquet"
DATA_PROCESSED = ROOT / ("data/dev"       if MODE == "dev" else "data/processed")
MODELS         = ROOT / ("models/dev"     if MODE == "dev" else "models/prod")
OUTPUTS        = ROOT / ("outputs/dev"    if MODE == "dev" else "outputs/prod")

# ── Global ─────────────────────────────────────────────────────────────────────
RANDOM_SEED              = _cfg["global"]["random_seed"]
MIN_TICKETS_PER_CUSTOMER = _cfg["global"]["min_tickets_per_customer"]
SAMPLE_SIZE              = _cfg["global"]["sample_size"]

# ── Word2Vec ───────────────────────────────────────────────────────────────────
W2V_VECTOR_SIZE = _cfg["word2vec"]["vector_size"]
W2V_WINDOW      = _cfg["word2vec"]["window"]
W2V_MIN_COUNT   = _cfg["word2vec"]["min_count"]
W2V_EPOCHS      = _cfg["word2vec"]["epochs"]
W2V_SG          = _cfg["word2vec"]["sg"]
W2V_WORKERS     = _cfg["word2vec"]["workers"]
PRODUCT_EMBEDDING_METHOD = str(_cfg.get("product_embeddings", {}).get("method", "word2vec"))
PRODUCT_EMBEDDING_CANDIDATES = list(
    _cfg.get("product_embeddings", {}).get("candidates", ["word2vec"])
)

# ── Customer vectors ───────────────────────────────────────────────────────────
RECENCY_HALFLIFE_DAYS = _cfg["customer_vectors"]["recency_halflife_days"]
RECENCY_REFERENCE_DATE = _cfg["customer_vectors"]["recency_reference_date"]  # "YYYY-MM-DD"
CUSTOMER_VECTOR_WEIGHT_TRANSFORM = _cfg["customer_vectors"].get("weight_transform", "none")
NORMALIZE_PRODUCT_EMBEDDINGS = bool(_cfg["customer_vectors"].get("normalize_product_embeddings", False))
TFIDF_SVD_DIMS = int(_cfg["customer_vectors"].get("tfidf_svd_dims", W2V_VECTOR_SIZE))
HYBRID_WEIGHT_ITEM2VEC = float(_cfg["customer_vectors"].get("hybrid_weights", {}).get("item2vec", 1.0))
HYBRID_WEIGHT_TFIDF_SVD = float(_cfg["customer_vectors"].get("hybrid_weights", {}).get("tfidf_svd", 1.0))
HYBRID_WEIGHT_CATEGORY_SHARES = float(_cfg["customer_vectors"].get("hybrid_weights", {}).get("category_shares", 1.0))
HYBRID_INCLUDE_STORE_SHARES = bool(_cfg["customer_vectors"].get("hybrid_include_store_shares", False))
FEATURE_WEIGHT_PROMO  = float(_cfg["feature_weights"]["promo"])
FEATURE_WEIGHT_STORE  = float(_cfg["feature_weights"]["store"])
FEATURE_WEIGHT_KPI    = float(_cfg["feature_weights"]["kpi"])

# Product popularity filter / IDF weighting
PRODUCT_POPULARITY_ENABLED    = bool(_cfg["product_popularity"]["enabled"])
PRODUCT_MAX_BASKET_SHARE      = float(_cfg["product_popularity"]["max_basket_share"])
PRODUCT_MAX_CUSTOMER_SHARE    = float(_cfg["product_popularity"]["max_customer_share"])
PRODUCT_IDF_WEIGHTING_ENABLED = bool(_cfg["product_popularity"]["idf_weighting"])
PRODUCT_IDF_MIN_WEIGHT        = float(_cfg["product_popularity"]["idf_min_weight"])
PRODUCT_IDF_MAX_WEIGHT        = float(_cfg["product_popularity"]["idf_max_weight"])

# ── UMAP ───────────────────────────────────────────────────────────────────────
UMAP_CLUSTER_DIMS     = _cfg["umap"]["cluster_dims"]
UMAP_VIZ_DIMS         = _cfg["umap"]["viz_dims"]
UMAP_N_NEIGHBORS      = _cfg["umap"]["n_neighbors"]
UMAP_MIN_DIST_CLUSTER = _cfg["umap"]["min_dist_cluster"]
UMAP_MIN_DIST_VIZ     = _cfg["umap"]["min_dist_viz"]
UMAP_METRIC           = _cfg["umap"]["metric"]
UMAP_FIT_SAMPLE       = _cfg["umap"]["fit_sample"]

# ── HDBSCAN ────────────────────────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = _cfg["hdbscan"]["min_cluster_size"]
HDBSCAN_MIN_SAMPLES      = _cfg["hdbscan"]["min_samples"]
HDBSCAN_METRIC           = _cfg["hdbscan"]["metric"]
HDBSCAN_CLUSTER_METHOD   = _cfg["hdbscan"]["cluster_method"]
HDBSCAN_FIT_SAMPLE       = _cfg["hdbscan"]["fit_sample"]
SILHOUETTE_SAMPLE        = _cfg["hdbscan"]["silhouette_sample"]
HDBSCAN_GRID_MIN_CLUSTER_SIZE = _cfg["hdbscan_grid"]["min_cluster_size"]
HDBSCAN_GRID_MIN_SAMPLES      = _cfg["hdbscan_grid"]["min_samples"]
HDBSCAN_GRID_CLUSTER_METHOD   = _cfg["hdbscan_grid"]["cluster_method"]
KMEANS_BASELINE_CLUSTERS      = _cfg["kmeans"]["baseline_clusters"]
KMEANS_BATCH_SIZE             = _cfg["kmeans"]["batch_size"]
KMEANS_N_INIT                 = _cfg["kmeans"]["n_init"]
KMEANS_MAX_ITER               = _cfg["kmeans"]["max_iter"]

# ── Temporal windows (ISO dates, inclusive) ────────────────────────────────────
TIME_WINDOWS = {
    "W1_jan_feb": ("2022-01-01", "2022-02-28"),
    "W2_mar_apr": ("2022-03-01", "2022-04-30"),
    "W3_may_jun": ("2022-05-01", "2022-06-30"),
}

# Pipeline model selection. Components set to "auto" are selected from configured candidates.
PIPELINE_SELECTION_MODE = str(_cfg["pipeline"].get("selection_mode", "auto"))
PIPELINE_VECTOR_SOURCE = str(_cfg["pipeline"]["vector_source"])
PIPELINE_REDUCER = str(_cfg["pipeline"].get("reducer", "umap"))
PIPELINE_CLUSTERER = str(_cfg["pipeline"].get("clusterer", "hdbscan_assigned"))
PIPELINE_FALLBACK_VECTOR_SOURCE = str(_cfg["pipeline"].get("fallback_vector_source", "hybrid"))
PIPELINE_FALLBACK_REDUCER = str(_cfg["pipeline"].get("fallback_reducer", "umap"))
PIPELINE_FALLBACK_CLUSTERER = str(_cfg["pipeline"].get("fallback_clusterer", "hdbscan_assigned"))

MODEL_SELECTION_TARGET_TRIBE_COUNT = int(
    _cfg.get("model_selection", {}).get("target_tribe_count", 15)
)
VECTOR_SOURCE_CANDIDATES = list(
    _cfg.get("model_selection", {}).get("vector_sources", ["item2vec", "tfidf_svd", "hybrid"])
)
REDUCER_CANDIDATES = list(
    _cfg.get("model_selection", {}).get("reducers", ["umap", "pca"])
)
CLUSTERER_CANDIDATES = list(
    _cfg.get("model_selection", {}).get("clusterers", ["hdbscan_assigned", "kmeans"])
)
TARGET_KMEANS_K = int(_cfg.get("model_selection", {}).get("target_kmeans_k", 15))
REDUCTION_BENCHMARK_SAMPLE = int(
    _cfg.get("model_selection", {}).get("reduction_benchmark_sample", 50_000)
)
N_NEIGHBORS_CANDIDATES = list(
    _cfg.get("model_selection", {}).get("n_neighbors_candidates", [10, 15, 20, 30, 50])
)

# ── Vector source benchmark ────────────────────────────────────────────────────
# Subsample size for the Section 3.2.6 comparison mini-pipeline.
# The full-population UMAP still runs in Section 3.3 for the winning source.
VECTOR_BENCHMARK_N_SAMPLE = int(_cfg["vector_source_benchmark"]["n_sample"])

# ── Claude API (tribe naming) ──────────────────────────────────────────────────
CLAUDE_MODEL      = _cfg["tribe_namer"]["model"]
CLAUDE_MAX_TOKENS = _cfg["tribe_namer"]["max_tokens"]
