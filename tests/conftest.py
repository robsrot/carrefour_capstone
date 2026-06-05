"""Shared fixtures for the Carrefour test suite.

All fixtures use synthetic data — no production parquet files required.
The session-scoped `raw_data_dir` fixture writes tiny parquets to a temp
directory so quality-check tests can call the real Polars scan/read paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

_SEED = 42


# ── tiny synthetic DataFrames ─────────────────────────────────────────────────

def _make_tickets() -> pl.DataFrame:
    """6-month, 6-customer synthetic ticket dataset with all expected columns."""
    return pl.DataFrame({
        "idempres": [2, 2, 7, 7, 2, 7, 2, 7, 2, 7, 2, 7],
        "fecha": pl.Series([
            "2022-01-15", "2022-01-20",
            "2022-02-10", "2022-02-15",
            "2022-03-05", "2022-03-10",
            "2022-04-20", "2022-04-25",
            "2022-05-01", "2022-05-05",
            "2022-06-10", "2022-06-15",
        ]).str.to_date(),
        "hora":     ["10:00"] * 12,
        "ticket":   list(range(1001, 1013)),
        "cliente":  ["C001", "C001", "C002", "C002", "C003", "C003",
                     "C004", "C004", "C005", "C005", "C006", "C006"],
        "idarticu": ["A001", "A002", "A001", "A003", "A002", "A003",
                     "A001", "A002", "A003", "A001", "A002", "A003"],
        "unidades": [2, 1, 3, 1, 2, 1, 1, 2, 3, 1, 2, 1],
        "importe":  [5.0, 3.0, 8.0, 2.0, 6.0, 4.0,
                     5.0, 3.0, 7.0, 5.0, 6.0, 4.0],
        "idpromoc": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "idtiprod": [1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2],
    })


def _make_articles() -> pl.DataFrame:
    return pl.DataFrame({
        "idarticu":             ["A001", "A002", "A003"],
        "desc_larga_articulo":  ["Whole Milk 1L", "Sourdough Bread", "Sparkling Water"],
        "idsector":             [1, 2, 3],
        "desc_sector":          ["Dairy", "Bakery", "Beverages"],
    })


# ── session-scoped parquet fixtures ───────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_data_dir(tmp_path_factory):
    """Write synthetic tickets + articles as parquet files; return the directory path."""
    d = tmp_path_factory.mktemp("raw_parquet")
    _make_tickets().write_parquet(d / "linea_tickets.parquet")
    _make_articles().write_parquet(d / "maestra_articulos.parquet")
    return d


@pytest.fixture(scope="session")
def orphan_data_dir(tmp_path_factory):
    """Variant where tickets contain product IDs not present in articles (< 90% coverage)."""
    d = tmp_path_factory.mktemp("orphan_parquet")
    tickets = _make_tickets().with_columns(
        pl.when(pl.col("idarticu") == "A003")
        .then(pl.lit("UNKNOWN_1"))
        .otherwise(pl.col("idarticu"))
        .alias("idarticu")
    ).vstack(pl.DataFrame({
        "idempres": [2] * 8,
        "fecha":    pl.Series(["2022-01-01"] * 8).str.to_date(),
        "hora":     ["10:00"] * 8,
        "ticket":   list(range(2001, 2009)),
        "cliente":  ["C007"] * 8,
        "idarticu": [f"UNKNOWN_{i}" for i in range(2, 10)],
        "unidades": [1] * 8,
        "importe":  [5.0] * 8,
        "idpromoc": [0] * 8,
        "idtiprod": [1] * 8,
    }))
    tickets.write_parquet(d / "linea_tickets.parquet")
    _make_articles().write_parquet(d / "maestra_articulos.parquet")
    return d


# ── per-test DataFrame fixtures ───────────────────────────────────────────────

@pytest.fixture
def tiny_tickets_df() -> pl.DataFrame:
    return _make_tickets()


@pytest.fixture
def tiny_articles_df() -> pl.DataFrame:
    return _make_articles()


@pytest.fixture
def tiny_kpis_df() -> pd.DataFrame:
    """120-row synthetic per-customer KPI DataFrame covering all 5 KPI columns."""
    rng = np.random.default_rng(_SEED)
    n = 120
    return pd.DataFrame({
        "cliente":         [f"C{i:04d}" for i in range(n)],
        "visit_count":     rng.integers(3, 50, n),
        "total_spend_6m":  rng.uniform(20.0, 1000.0, n),
        "avg_basket_size": rng.uniform(5.0, 80.0, n),
        "avg_promo_rate":  rng.uniform(0.0, 1.0, n),
        "unique_products": rng.integers(1, 40, n),
    })
