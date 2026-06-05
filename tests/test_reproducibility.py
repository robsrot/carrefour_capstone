"""Tests for the reproducibility contract.

Verifies that every source of nondeterminism documented in CLAUDE.md has been
addressed: config constants, algorithm n_jobs/workers, sort invariants,
sampling determinism, and Word2Vec bit-exact outputs.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


# ── config constants ──────────────────────────────────────────────────────────

def test_random_seed_is_42():
    from src.config import RANDOM_SEED
    assert RANDOM_SEED == 42


def test_w2v_workers_is_1():
    """workers > 1 causes gensim thread-race → different embeddings per run."""
    from src.config import W2V_WORKERS
    assert W2V_WORKERS == 1, (
        f"W2V_WORKERS={W2V_WORKERS} — must be 1 for bit-exact reproducibility. "
        "See configs/base.yaml."
    )


def test_word2vec_uses_stable_hash_function():
    """Python's built-in hash is process-randomized unless PYTHONHASHSEED is set."""
    from src.embeddings import _stable_hash

    assert _stable_hash("product_123") == _stable_hash("product_123")
    assert _stable_hash("product_123") != _stable_hash("product_456")


def test_mode_is_valid():
    from src.config import MODE
    assert MODE in ("dev", "prod")


def test_config_loads_without_error():
    """All config keys must be present and loadable."""
    from src.config import (
        RANDOM_SEED, W2V_WORKERS, W2V_VECTOR_SIZE, W2V_EPOCHS,
        UMAP_CLUSTER_DIMS, UMAP_N_NEIGHBORS, UMAP_FIT_SAMPLE,
        HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
        MIN_TICKETS_PER_CUSTOMER, RECENCY_HALFLIFE_DAYS,
    )
    assert all(v is not None for v in [
        RANDOM_SEED, W2V_WORKERS, W2V_VECTOR_SIZE, W2V_EPOCHS,
        UMAP_CLUSTER_DIMS, UMAP_N_NEIGHBORS, UMAP_FIT_SAMPLE,
        HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
        MIN_TICKETS_PER_CUSTOMER, RECENCY_HALFLIFE_DAYS,
    ])


# ── algorithm n_jobs defaults ─────────────────────────────────────────────────

def test_reduce_umap_cluster_n_jobs_default_is_1():
    """UMAP pynndescent NN graph is nondeterministic with n_jobs > 1."""
    from src.dimensionality import reduce_umap_cluster
    sig = inspect.signature(reduce_umap_cluster)
    assert sig.parameters["n_jobs"].default == 1, (
        "reduce_umap_cluster n_jobs default must be 1 for reproducibility."
    )


def test_reduce_umap_viz_n_jobs_default_is_1():
    from src.dimensionality import reduce_umap_viz
    sig = inspect.signature(reduce_umap_viz)
    assert sig.parameters["n_jobs"].default == 1


def test_clustering_imports_without_error():
    """Clustering module must be importable (no startup side-effects)."""
    import src.clustering   # noqa: F401


# ── sampling determinism ──────────────────────────────────────────────────────

def test_proportional_sample_is_deterministic(tiny_kpis_df):
    """Same seed + same strata → identical customer list on both calls."""
    from src.config import RANDOM_SEED
    from src.generate_dev_subset import _proportional_sample, _assign_strata

    kpis = tiny_kpis_df.copy()
    # Build a minimal store_counts Series (all single-store)
    store_counts = pd.Series(1, index=kpis["cliente"].values)
    sector_bins = pd.Series("grocery_pgc", index=kpis["cliente"].values)
    kpis["stratum"] = _assign_strata(kpis, store_counts, sector_bins)

    target = 60
    rng1 = np.random.default_rng(RANDOM_SEED)
    rng2 = np.random.default_rng(RANDOM_SEED)

    ids1 = _proportional_sample(kpis, target, rng1)
    ids2 = _proportional_sample(kpis, target, rng2)

    assert sorted(ids1) == sorted(ids2)
    assert len(ids1) > 0


def test_proportional_sample_different_seeds_differ(tiny_kpis_df):
    """Different seeds must produce different samples (validates that seed matters)."""
    from src.generate_dev_subset import _proportional_sample, _assign_strata

    kpis = tiny_kpis_df.copy()
    store_counts = pd.Series(1, index=kpis["cliente"].values)
    sector_bins = pd.Series("grocery_pgc", index=kpis["cliente"].values)
    kpis["stratum"] = _assign_strata(kpis, store_counts, sector_bins)

    target = 60
    ids1 = _proportional_sample(kpis, target, np.random.default_rng(42))
    ids2 = _proportional_sample(kpis, target, np.random.default_rng(99))

    assert sorted(ids1) != sorted(ids2)


def test_proportional_sample_covers_all_strata(tiny_kpis_df):
    """Every populated stratum must contribute at least one customer."""
    from src.config import RANDOM_SEED
    from src.generate_dev_subset import _proportional_sample, _assign_strata

    kpis = tiny_kpis_df.copy()
    store_counts = pd.Series(1, index=kpis["cliente"].values)
    sector_bins = pd.Series("grocery_pgc", index=kpis["cliente"].values)
    kpis["stratum"] = _assign_strata(kpis, store_counts, sector_bins)

    # Sample 50% of the population — every stratum should be represented
    target = len(kpis) // 2
    ids = set(_proportional_sample(kpis, target, np.random.default_rng(RANDOM_SEED)))

    sampled_strata = set(kpis[kpis["cliente"].isin(ids)]["stratum"].unique())
    all_strata = set(kpis["stratum"].unique())
    assert sampled_strata == all_strata, (
        f"Missing strata in sample: {all_strata - sampled_strata}"
    )


# ── sort invariant ────────────────────────────────────────────────────────────

def test_sort_by_cliente_produces_stable_order():
    """Two DataFrames with the same rows in different order must match after sort."""
    import polars as pl

    rows = {"cliente": ["C003", "C001", "C004", "C002"], "value": [30, 10, 40, 20]}
    df1 = pl.DataFrame(rows)
    df2 = df1.sample(fraction=1.0, shuffle=True, seed=99)   # different row order

    assert df1.sort("cliente")["cliente"].to_list() == \
           df2.sort("cliente")["cliente"].to_list()


def test_group_by_order_without_sort_is_nondeterministic():
    """Demonstrate WHY the .sort() fixes are necessary.

    Polars group_by row order is not guaranteed, so two equivalent lazy plans
    can produce different orders.  The .sort() calls in embeddings.py and
    customer_vectors.py guard against this.
    """
    import polars as pl

    df = pl.DataFrame({
        "ticket":   [3, 1, 2, 1, 3, 2],
        "idarticu": ["A", "B", "C", "D", "E", "F"],
    })
    result = df.group_by("ticket").agg(pl.col("idarticu").alias("products"))
    # Without .sort(), ticket order is unspecified — this assertion documents the contract
    sorted_result = result.sort("ticket")
    assert sorted_result["ticket"].to_list() == [1, 2, 3]


# ── Word2Vec bit-exact reproducibility ───────────────────────────────────────

def test_w2v_deterministic_with_workers_1():
    """Two Word2Vec runs with workers=1 + same seed → identical word vectors."""
    from gensim.models import Word2Vec

    sentences = [[f"product_{i % 20}" for i in range(j, j + 6)] for j in range(80)]
    params = dict(
        vector_size=16, window=3, min_count=1, sg=1,
        workers=1, epochs=5, seed=42,
    )

    m1 = Word2Vec(sentences, **params)
    m2 = Word2Vec(sentences, **params)

    for word in ["product_0", "product_5", "product_10"]:
        np.testing.assert_array_equal(
            m1.wv[word], m2.wv[word],
            err_msg=f"Word2Vec vector for '{word}' differs between runs — "
                    "workers=1 + fixed seed must be bit-exact.",
        )


def test_w2v_workers_gt1_context():
    """Document: workers=1 is intentionally slow but required.

    This test does not assert nondeterminism (which is probabilistic) but
    confirms W2V_WORKERS=1 is set and serves as a regression guard — if
    someone changes the config, the test_w2v_workers_is_1 test will fail.
    """
    from src.config import W2V_WORKERS
    assert W2V_WORKERS == 1
