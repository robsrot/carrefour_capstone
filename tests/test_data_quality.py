"""Tests for src/data_quality.py — quality-gate functions against synthetic data.

Each test monkeypatches src.data_quality.DATA_RAW to point at the session-scoped
temp directory (raw_data_dir fixture from conftest.py) so the real Polars scan/read
paths are exercised without requiring the 191M-row production files.
"""
from __future__ import annotations

import polars as pl
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _patch_raw(monkeypatch, path):
    monkeypatch.setattr("src.data_quality.DATA_RAW", path)


# ── validate_schema ───────────────────────────────────────────────────────────

def test_validate_schema_passes(raw_data_dir, monkeypatch):
    _patch_raw(monkeypatch, raw_data_dir)
    from src.data_quality import validate_schema
    result = validate_schema()
    assert "articles_columns" in result
    assert "ticket_columns" in result
    assert result["ticket_n_cols"] == 10
    assert result["articles_shape"][1] == 4


def test_validate_schema_fails_missing_ticket_column(raw_data_dir, tmp_path, monkeypatch):
    """Tickets parquet missing 'idpromoc' → AssertionError naming the column."""
    tickets = pl.read_parquet(raw_data_dir / "linea_tickets.parquet").drop("idpromoc")
    tickets.write_parquet(tmp_path / "linea_tickets.parquet")
    pl.read_parquet(raw_data_dir / "maestra_articulos.parquet").write_parquet(
        tmp_path / "maestra_articulos.parquet"
    )
    _patch_raw(monkeypatch, tmp_path)
    from src.data_quality import validate_schema
    with pytest.raises(AssertionError, match="idpromoc"):
        validate_schema()


def test_validate_schema_fails_missing_article_column(raw_data_dir, tmp_path, monkeypatch):
    pl.read_parquet(raw_data_dir / "linea_tickets.parquet").write_parquet(
        tmp_path / "linea_tickets.parquet"
    )
    articles = pl.read_parquet(raw_data_dir / "maestra_articulos.parquet").drop("idsector")
    articles.write_parquet(tmp_path / "maestra_articulos.parquet")
    _patch_raw(monkeypatch, tmp_path)
    from src.data_quality import validate_schema
    with pytest.raises(AssertionError, match="idsector"):
        validate_schema()


# ── audit_nulls ───────────────────────────────────────────────────────────────

def test_audit_nulls_passes_on_clean_data(raw_data_dir, monkeypatch):
    _patch_raw(monkeypatch, raw_data_dir)
    from src.data_quality import audit_nulls
    result = audit_nulls()
    assert result["articles_total_nulls"] == 0
    assert result["ticket_total_nulls"] == 0


def test_audit_nulls_fails_on_null_in_tickets(raw_data_dir, tmp_path, monkeypatch):
    tickets = pl.read_parquet(raw_data_dir / "linea_tickets.parquet").with_columns(
        pl.when(pl.col("ticket") == 1001)
        .then(None)
        .otherwise(pl.col("cliente"))
        .alias("cliente")
    )
    tickets.write_parquet(tmp_path / "linea_tickets.parquet")
    pl.read_parquet(raw_data_dir / "maestra_articulos.parquet").write_parquet(
        tmp_path / "maestra_articulos.parquet"
    )
    _patch_raw(monkeypatch, tmp_path)
    from src.data_quality import audit_nulls
    with pytest.raises(AssertionError):
        audit_nulls()


# ── audit_anomalies ───────────────────────────────────────────────────────────

def test_audit_anomalies_no_anomalies(raw_data_dir, monkeypatch):
    _patch_raw(monkeypatch, raw_data_dir)
    from src.data_quality import audit_anomalies
    result = audit_anomalies()
    assert result["anomaly_counts"]["negative_unidades"] == 0
    assert result["anomaly_counts"]["negative_importe"] == 0
    assert result["rows_dropped_by_cleaning_rule"] == 0


def test_audit_anomalies_counts_negative_unidades(raw_data_dir, tmp_path, monkeypatch):
    extra = pl.DataFrame({
        "idempres": [2], "fecha": pl.Series(["2022-01-01"]).str.to_date(),
        "hora": ["10:00"], "ticket": [9001], "cliente": ["C999"],
        "idarticu": ["A001"], "unidades": [-1], "importe": [5.0],
        "idpromoc": [0], "idtiprod": [1],
    })
    tickets = pl.read_parquet(raw_data_dir / "linea_tickets.parquet").vstack(extra)
    tickets.write_parquet(tmp_path / "linea_tickets.parquet")
    pl.read_parquet(raw_data_dir / "maestra_articulos.parquet").write_parquet(
        tmp_path / "maestra_articulos.parquet"
    )
    _patch_raw(monkeypatch, tmp_path)
    from src.data_quality import audit_anomalies
    result = audit_anomalies()
    assert result["anomaly_counts"]["negative_unidades"] == 1
    assert result["rows_dropped_by_cleaning_rule"] == 1


def test_audit_anomalies_counts_zero_importe(raw_data_dir, tmp_path, monkeypatch):
    extra = pl.DataFrame({
        "idempres": [2], "fecha": pl.Series(["2022-02-01"]).str.to_date(),
        "hora": ["10:00"], "ticket": [9002], "cliente": ["C999"],
        "idarticu": ["A002"], "unidades": [1], "importe": [0.0],
        "idpromoc": [0], "idtiprod": [1],
    })
    tickets = pl.read_parquet(raw_data_dir / "linea_tickets.parquet").vstack(extra)
    tickets.write_parquet(tmp_path / "linea_tickets.parquet")
    pl.read_parquet(raw_data_dir / "maestra_articulos.parquet").write_parquet(
        tmp_path / "maestra_articulos.parquet"
    )
    _patch_raw(monkeypatch, tmp_path)
    from src.data_quality import audit_anomalies
    result = audit_anomalies()
    assert result["anomaly_counts"]["zero_importe"] == 1
    assert result["rows_dropped_by_cleaning_rule"] == 1


# ── check_product_coverage ────────────────────────────────────────────────────

def test_check_product_coverage_full_coverage(raw_data_dir, monkeypatch):
    """All ticket products exist in articles → 100% coverage, passes."""
    _patch_raw(monkeypatch, raw_data_dir)
    from src.data_quality import check_product_coverage
    result = check_product_coverage()
    assert result["orphaned_products"] == 0
    assert result["coverage_pct"] == 100.0


def test_check_product_coverage_below_threshold(orphan_data_dir, monkeypatch):
    """Tickets with many orphaned product IDs → coverage < 90% → AssertionError."""
    _patch_raw(monkeypatch, orphan_data_dir)
    from src.data_quality import check_product_coverage
    with pytest.raises(AssertionError, match="coverage"):
        check_product_coverage()


# ── check_temporal_completeness ───────────────────────────────────────────────

def test_check_temporal_completeness_wrong_month_count(raw_data_dir, tmp_path, monkeypatch):
    """Drop one month → n_months != 6 → AssertionError."""
    tickets = pl.read_parquet(raw_data_dir / "linea_tickets.parquet").filter(
        pl.col("fecha").dt.month() != 6   # remove June
    )
    tickets.write_parquet(tmp_path / "linea_tickets.parquet")
    pl.read_parquet(raw_data_dir / "maestra_articulos.parquet").write_parquet(
        tmp_path / "maestra_articulos.parquet"
    )
    _patch_raw(monkeypatch, tmp_path)
    from src.data_quality import check_temporal_completeness
    with pytest.raises(AssertionError, match="6 months"):
        check_temporal_completeness()


def test_check_temporal_completeness_too_few_rows(raw_data_dir, monkeypatch):
    """Our synthetic data has 2 rows/month — well below the 10M threshold → fails."""
    _patch_raw(monkeypatch, raw_data_dir)
    from src.data_quality import check_temporal_completeness
    with pytest.raises(AssertionError, match="transactions"):
        check_temporal_completeness()


# ── audit_customer_activity ───────────────────────────────────────────────────

def test_audit_customer_activity_result_structure(raw_data_dir, monkeypatch):
    _patch_raw(monkeypatch, raw_data_dir)
    from src.data_quality import audit_customer_activity
    result = audit_customer_activity()
    assert "total_unique_customers" in result
    assert "eligible_customers" in result
    assert result["eligible_customers"] <= result["total_unique_customers"]
    assert result["min_tickets_threshold"] == 3   # from configs/base.yaml


def test_audit_customer_activity_all_eligible(raw_data_dir, monkeypatch):
    """Each synthetic customer has exactly 2 tickets — below MIN_TICKETS=3 threshold."""
    _patch_raw(monkeypatch, raw_data_dir)
    from src.data_quality import audit_customer_activity
    result = audit_customer_activity()
    # All 6 customers have 2 tickets each; MIN_TICKETS_PER_CUSTOMER=3 → all ineligible
    assert result["eligible_customers"] == 0


# ── audit_promotional_data ────────────────────────────────────────────────────

def test_audit_promotional_data_structure(raw_data_dir, monkeypatch):
    _patch_raw(monkeypatch, raw_data_dir)
    from src.data_quality import audit_promotional_data
    result = audit_promotional_data()
    assert "n_promo_categories" in result
    assert result["n_promo_categories"] >= 1
    assert result["total_rows"] == 12   # our synthetic fixture has 12 rows


def test_audit_promotional_data_categories(raw_data_dir, monkeypatch):
    """Synthetic data has idpromoc values 0 and 1 only."""
    _patch_raw(monkeypatch, raw_data_dir)
    from src.data_quality import audit_promotional_data
    result = audit_promotional_data()
    categories = set(result["promo_categories"])
    assert "0" in categories
    assert "1" in categories


# ── audit_stores ──────────────────────────────────────────────────────────────

def test_audit_stores_structure(raw_data_dir, monkeypatch):
    _patch_raw(monkeypatch, raw_data_dir)
    from src.data_quality import audit_stores
    result = audit_stores()
    assert result["n_stores"] == 2   # synthetic data has stores 2 and 7
    shares = [v["share_pct"] for v in result["store_breakdown"].values()]
    assert abs(sum(shares) - 100.0) < 0.1
