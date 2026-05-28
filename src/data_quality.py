"""
src/data_quality.py

load-bearing data quality checks for the Carrefour segmentation pipeline.
all heavy scans use Polars lazy frames — nothing loads 191M rows into memory.

pipeline-gate functions (raise AssertionError if a hard constraint is violated):
  validate_schema()            — column names + dtypes
  audit_nulls()                — missing values in both complete datasets
  audit_anomalies()            — negative/zero/extreme values; determines cleaning rule
  check_product_coverage()     — >= 90% of ticket products must be in articles
  check_temporal_completeness()— all 6 months present, each > 10M rows
  audit_customer_activity()    — distribution of tickets-per-customer
  audit_promotional_data()     — idpromoc category integrity
  audit_stores()               — store count and volume distribution

orchestrator:
  build_quality_report()       — runs every check, saves quality_report.json
"""

import json
import polars as pl
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

from src.config import (
    DATA_RAW, DATA_PROCESSED, MIN_TICKETS_PER_CUSTOMER, SAMPLE_SIZE
)

# ─── Expected schema ──────────────────────────────────────────────────────────

_ARTICLES_COLS  = {"idarticu", "desc_larga_articulo", "idsector", "desc_sector"}
_TICKET_COLS  = {
    "idempres", "fecha", "hora", "ticket", "cliente",
    "idarticu", "unidades", "importe", "idpromoc", "idtiprod"
}


def _df_tickets() -> pl.LazyFrame:
    return pl.scan_parquet(DATA_RAW / "linea_tickets.parquet")


def _df_articles() -> pl.DataFrame:
    return pl.read_parquet(DATA_RAW / "maestra_articulos.parquet")


# ─── 1. Schema validation ─────────────────────────────────────────────────────

def validate_schema() -> dict:
    """
    Confirm both files have the expected columns.
    Returns a result dict; raises AssertionError on any missing column.
    """
    df_articles = _df_articles()
    articles_cols = set(df_articles.columns)
    missing_articles = _ARTICLES_COLS - articles_cols
    assert not missing_articles, f"Articles missing columns: {missing_articles}"

    sample = _df_tickets().head(1).collect()
    ticket_cols = set(sample.columns)
    missing_ticket = _TICKET_COLS - ticket_cols
    assert not missing_ticket, f"Ticket data missing columns: {missing_ticket}"

    return {
        "articles_columns": sorted(articles_cols),
        "articles_dtypes": {c: str(df_articles[c].dtype) for c in df_articles.columns},
        "articles_shape": list(df_articles.shape),
        "ticket_columns": sorted(ticket_cols),
        "ticket_dtypes": {c: str(sample[c].dtype) for c in sample.columns},
        "ticket_n_cols": len(ticket_cols),
    }


# ─── 2. Null audit ────────────────────────────────────────────────────────────

def audit_nulls() -> dict:
    """
    check for missing values across both complete datasets.
    raises AssertionError if either dataset has any nulls.
    """
    df_articles = _df_articles()
    articles_nulls = {c: int(df_articles[c].null_count()) for c in df_articles.columns}
    total_articles_nulls = sum(articles_nulls.values())
    assert total_articles_nulls == 0, f"nulls found in articles: {articles_nulls}"

    ticket_null_counts = (
        _df_tickets()
        .select([pl.col(c).null_count().alias(c) for c in sorted(_TICKET_COLS)])
        .collect()
        .to_dicts()[0]
    )
    total_ticket_nulls = sum(ticket_null_counts.values())
    assert total_ticket_nulls == 0, f"nulls found in ticket data: {ticket_null_counts}"

    return {
        "articles_nulls": articles_nulls,
        "articles_total_nulls": int(total_articles_nulls),
        "ticket_null_counts": ticket_null_counts,
        "ticket_total_nulls": int(total_ticket_nulls),
    }


# ─── 3. Anomaly audit ────────────────────────────────────────────────────────

def audit_anomalies() -> dict:
    """
    Count anomalous rows across the full 191M-row dataset.

    Cleaning rule applied in basket_builder.py:
        DROP rows where unidades <= 0 OR importe <= 0

    This function documents the scale of each anomaly type so the rule is
    evidence-based, not arbitrary.
    """
    df_tickets = _df_tickets()
    total = df_tickets.select(pl.len()).collect().item()

    counts = df_tickets.select([
        (pl.col("unidades") < 0).sum().alias("negative_unidades"),
        (pl.col("unidades") == 0).sum().alias("zero_unidades"),
        (pl.col("unidades") > 1_000).sum().alias("extreme_unidades_gt1000"),
        (pl.col("importe") < 0).sum().alias("negative_importe"),
        (pl.col("importe") == 0).sum().alias("zero_importe"),
        (pl.col("importe") > 10_000).sum().alias("extreme_importe_gt10k"),
    ]).collect().row(0, named=True)

    dropped = df_tickets.filter(
        (pl.col("unidades") <= 0) | (pl.col("importe") <= 0)
    ).select(pl.len()).collect().item()

    result = {
        "total_rows": int(total),
        "anomaly_counts": {k: int(v) for k, v in counts.items()},
        "anomaly_pcts": {k: round(v / total * 100, 4) for k, v in counts.items()},
        "rows_dropped_by_cleaning_rule": int(dropped),
        "rows_dropped_pct": round(dropped / total * 100, 3),
        "rows_retained": int(total - dropped),
        "rows_retained_pct": round((total - dropped) / total * 100, 3),
        "cleaning_rule": "DROP unidades <= 0 OR importe <= 0",
    }
    return result


# ─── 4. Product coverage ─────────────────────────────────────────────────────

def check_product_coverage() -> dict:
    """
    Assert >= 90% of unique product IDs in tickets also appear in the articles.
    Orphaned products (in tickets but not in articles) will have no embedding and
    will be dropped from the customer vector computation.

    Raises AssertionError if coverage < 90%.
    """
    df_articles = _df_articles()
    articles_ids = set(df_articles["idarticu"].to_list())

    ticket_ids_series = (
        _df_tickets()
        .select(pl.col("idarticu").unique())
        .collect()["idarticu"]
    )
    ticket_ids = set(ticket_ids_series.to_list())

    covered  = ticket_ids & articles_ids
    orphaned = ticket_ids - articles_ids
    coverage_pct = len(covered) / len(ticket_ids) * 100

    result = {
        "unique_products_in_tickets": len(ticket_ids),
        "unique_products_in_articles": len(articles_ids),
        "covered_products": len(covered),
        "orphaned_products": len(orphaned),
        "coverage_pct": round(coverage_pct, 2),
    }

    assert coverage_pct > 90, (
        f"Product coverage {coverage_pct:.1f}% is below 90% threshold. "
        f"{len(orphaned):,} orphaned product IDs — investigate before proceeding."
    )
    return result


# ─── 5. Temporal completeness ─────────────────────────────────────────────────

def check_temporal_completeness() -> dict:
    """
    Assert all 6 calendar months (Jan–Jun 2022) are present and each has
    > 10M transactions. A month below 10M would indicate a data export gap.
    """
    monthly = (
        _df_tickets()
        .with_columns(pl.col("fecha").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(pl.len().alias("n_transactions"))
        .sort("month")
        .collect()
    )

    n_months   = monthly.shape[0]
    min_n      = int(monthly["n_transactions"].min())
    max_n      = int(monthly["n_transactions"].max())
    total      = int(monthly["n_transactions"].sum())
    breakdown  = {
        r["month"]: int(r["n_transactions"])
        for r in monthly.iter_rows(named=True)
    }

    assert n_months == 6, f"Expected 6 months, found {n_months}: {list(breakdown.keys())}"
    assert min_n > 10_000_000, (
        f"Month '{min(breakdown, key=breakdown.get)}' has only {min_n:,} transactions — "
        "possible data gap."
    )

    return {
        "n_months": n_months,
        "monthly_breakdown": breakdown,
        "min_month_transactions": min_n,
        "max_month_transactions": max_n,
        "total_transactions": total,
        "monthly_variance_pct": round(
            (max_n - min_n) / ((max_n + min_n) / 2) * 100, 1
        ),
    }


# ─── 6. Customer activity ─────────────────────────────────────────────────────

def audit_customer_activity() -> dict:
    """
    Distribution of ticket counts per customer.
    Determines how many customers pass the MIN_TICKETS_PER_CUSTOMER threshold.
    """
    activity = (
        _df_tickets()
        .group_by("cliente")
        .agg(pl.col("ticket").n_unique().alias("n_tickets"))
        .collect()
    )

    n_tickets     = activity["n_tickets"]
    total         = len(activity)
    below_min     = int((n_tickets < MIN_TICKETS_PER_CUSTOMER).sum())
    eligible      = total - below_min

    return {
        "total_unique_customers": int(total),
        "eligible_customers": int(eligible),
        "ineligible_customers": int(below_min),
        "ineligible_pct": round(below_min / total * 100, 2),
        "min_tickets_threshold": MIN_TICKETS_PER_CUSTOMER,
        "sample_size_feasible": eligible >= SAMPLE_SIZE,
        "median_tickets": float(n_tickets.median()),
        "mean_tickets": round(float(n_tickets.mean()), 2),
        "p10_tickets": float(n_tickets.quantile(0.10)),
        "p25_tickets": float(n_tickets.quantile(0.25)),
        "p75_tickets": float(n_tickets.quantile(0.75)),
        "p95_tickets": float(n_tickets.quantile(0.95)),
        "p99_tickets": float(n_tickets.quantile(0.99)),
        "max_tickets": int(n_tickets.max()),
    }


# ─── 7. Promotional data integrity ───────────────────────────────────────────

def audit_promotional_data() -> dict:
    """
    Check idpromoc categories and promotional line rate.
    Promotional sensitivity is a key behavioral axis in Phase 2.
    """
    df_tickets = _df_tickets()
    total = df_tickets.select(pl.len()).collect().item()

    promo_counts = (
        df_tickets.group_by("idpromoc")
          .agg(pl.len().alias("n"))
          .sort("n", descending=True)
          .collect()
    )

    breakdown = {
        str(row["idpromoc"]): {
            "count": int(row["n"]),
            "pct": round(row["n"] / total * 100, 2),
        }
        for row in promo_counts.iter_rows(named=True)
    }

    return {
        "total_rows": int(total),
        "n_promo_categories": len(breakdown),
        "promo_categories": list(breakdown.keys()),
        "breakdown": breakdown,
    }


# ─── 8. Store integrity ───────────────────────────────────────────────────────

def audit_stores() -> dict:
    """Count transactions and revenue per store."""
    df_tickets = _df_tickets()
    total = df_tickets.select(pl.len()).collect().item()

    stores = (
        df_tickets.group_by("idempres")
          .agg([
              pl.len().alias("n_transactions"),
              pl.col("importe").sum().alias("total_importe"),
          ])
          .sort("idempres")
          .collect()
    )

    breakdown = {
        str(row["idempres"]): {
            "n_transactions": int(row["n_transactions"]),
            "share_pct": round(row["n_transactions"] / total * 100, 1),
            "total_importe": round(float(row["total_importe"]), 2),
        }
        for row in stores.iter_rows(named=True)
    }

    return {
        "n_stores": len(breakdown),
        "store_breakdown": breakdown,
    }


# ─── 9. Full orchestration ────────────────────────────────────────────────────

def build_quality_report(verbose: bool = True) -> dict:
    """
    Run every check in sequence. Saves a structured JSON report to
    data/processed/quality_report.json.

    Returns the full report dict. Prints PASS/FAIL for each check.
    Does NOT abort on assertion failures — all checks run regardless,
    so the full picture is visible even if one gate fails.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    report   = {}
    results  = []

    checks = [
        ("schema_validation",      validate_schema),
        ("null_audit",             audit_nulls),
        ("anomaly_audit",          audit_anomalies),
        ("product_coverage",       check_product_coverage),
        ("temporal_completeness",  check_temporal_completeness),
        ("customer_activity",      audit_customer_activity),
        ("promotional_integrity",  audit_promotional_data),
        ("store_integrity",        audit_stores),
    ]

    if verbose:
        print("=" * 65)
        print("  DATA QUALITY REPORT — Carrefour Segmentation Pipeline")
        print("=" * 65)

    for name, fn in checks:
        if verbose:
            print(f"\n[{name}]")
        try:
            result = fn()
            report[name] = {**result, "_passed": True}
            results.append(True)
            if verbose:
                _pretty_print(result)
                print(f"  --> PASS")
        except AssertionError as e:
            report[name] = {"_passed": False, "_error": str(e)}
            results.append(False)
            if verbose:
                print(f"  --> FAIL: {e}")
        except Exception as e:
            report[name] = {"_passed": False, "_error": f"UNEXPECTED: {e}"}
            results.append(False)
            if verbose:
                print(f"  --> ERROR: {e}")

    n_passed = sum(results)
    n_total  = len(results)
    report["_summary"] = {
        "checks_passed": n_passed,
        "checks_total":  n_total,
        "all_passed":    n_passed == n_total,
    }

    out = DATA_PROCESSED / "quality_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    if verbose:
        print("\n" + "=" * 65)
        status = "ALL CHECKS PASSED" if n_passed == n_total else f"WARNING: {n_total - n_passed} check(s) failed"
        print(f"  {status}  ({n_passed}/{n_total})")
        print(f"  Report saved → {out}")
        print("=" * 65)

    return report


def _pretty_print(d: dict, indent: int = 2) -> None:
    """Print key result values from a check dict, skipping deeply nested dicts."""
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict) and len(v) > 6:
            print(f"{pad}{k}: ({len(v)} entries)")
        elif isinstance(v, list) and len(v) > 8:
            print(f"{pad}{k}: {v[:5]} ... ({len(v)} total)")
        else:
            print(f"{pad}{k}: {v}")
