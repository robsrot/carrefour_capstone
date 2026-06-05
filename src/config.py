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
        for section, values in override.items():
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section].update(values)
            else:
                cfg[section] = values

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

# ── Customer vectors ───────────────────────────────────────────────────────────
RECENCY_HALFLIFE_DAYS = _cfg["customer_vectors"]["recency_halflife_days"]
FEATURE_WEIGHT_PROMO  = float(_cfg["feature_weights"]["promo"])
FEATURE_WEIGHT_STORE  = float(_cfg["feature_weights"]["store"])
FEATURE_WEIGHT_KPI    = float(_cfg["feature_weights"]["kpi"])

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

# ── Claude API (tribe naming) ──────────────────────────────────────────────────
CLAUDE_MODEL      = _cfg["tribe_namer"]["model"]
CLAUDE_MAX_TOKENS = _cfg["tribe_namer"]["max_tokens"]
