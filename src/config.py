# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "parquet"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

# sampling
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42
MIN_TICKETS_PER_CUSTOMER = 3

# Word2Vec
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 3
W2V_EPOCHS = 10
W2V_SG = 1   # skip-gram (better than CBOW for rare products)

# customer vector aggregation
RECENCY_HALFLIFE_DAYS = 30

# UMAP
UMAP_CLUSTER_DIMS = 50
UMAP_VIZ_DIMS = 2
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST_CLUSTER = 0.0   # 0.0 preserves tighter local structure for clustering
UMAP_MIN_DIST_VIZ = 0.1
UMAP_METRIC = "cosine"

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 200
HDBSCAN_MIN_SAMPLES = 10
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_METHOD = "eom"

# temporal windows (ISO dates, inclusive)
TIME_WINDOWS = {
    "W1_jan_feb": ("2022-01-01", "2022-02-28"),
    "W2_mar_apr": ("2022-03-01", "2022-04-30"),
    "W3_may_jun": ("2022-05-01", "2022-06-30"),
}

# Claude API (tribe naming)
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 300
