"""Generate a stratified dev subset from the full production dataset.

Stratification: 54 strata = 3 visit-count tertiles × 3 spend tertiles × 2 store bins × 3 sector bins
  - visit_count tertiles  (v_lo / v_md / v_hi)
  - total_spend_6m tertiles  (s_lo / s_md / s_hi)
  - store affinity  (st1 = single-store / st2p = multi-store)
  - dominant product sector  (grocery_pgc / fresh / non_food)

Sector bins ensure the subset preserves the product-category mix of the full population —
critical for Word2Vec to learn embeddings across all category types (dairy, fresh produce,
electronics, etc.), not just the majority grocery segment.

Proportional sampling within each stratum preserves the full population's behavioural
distributions — confirmed by KS tests on all five customer KPIs.

Inputs (prod mode must have been run first):
  data/processed/df_combined.parquet     — full 109M-row transaction dataset
  data/processed/customer_kpis.parquet   — per-customer KPIs from notebook 02

Outputs:
  data/dev/df_combined.parquet           — transaction rows for the sampled customers
  data/dev/subset_metadata.json          — sampling params, strata counts, KS results

Usage:
  python -m src.generate_dev_subset
  python -m src.generate_dev_subset --target-size 44000 --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MIN_TICKETS_PER_CUSTOMER, RANDOM_SEED

PROD_DIR     = ROOT / "data" / "processed"
DEV_DIR      = ROOT / "data" / "dev"
DEFAULT_SIZE = 44_000

# Maps idsector → broad behavioural category for stratification.
# Collapsing BAZAR/ELECTROFOTO/TEXTIL/GASOLINERA into one bin keeps strata count
# manageable (54 total) while still ensuring non-food buyers are represented.
_SECTOR_BIN_MAP = {
    1: "grocery_pgc",  # packaged / dry grocery
    2: "fresh",        # perishables / fresh produce
    3: "non_food",     # BAZAR — general merchandise
    4: "non_food",     # ELECTROFOTO
    6: "non_food",     # TEXTIL
    7: "non_food",     # GASOLINERA
}


# ─── stratification helpers ───────────────────────────────────────────────────

def _tertile_bin(series: pd.Series, prefix: str) -> pd.Series:
    """Cut a numeric series into three equal-frequency bins."""
    q33, q67 = series.quantile([1 / 3, 2 / 3])
    return pd.cut(
        series,
        bins=[-np.inf, q33, q67, np.inf],
        labels=[f"{prefix}_lo", f"{prefix}_md", f"{prefix}_hi"],
    ).astype(str)


def _dominant_sector_bins(combined_path: Path) -> pd.Series:
    """Return a Series (index=cliente) with each customer's dominant sector bin.

    Dominant sector = the idsector that accounts for the highest total spend
    for that customer across the full 6-month window.  Ties broken by Polars
    sort stability (first row kept after sort-desc + drop_duplicates).
    """
    sector_spend = (
        pl.scan_parquet(combined_path)
        .select(["cliente", "idsector", "importe"])
        # Convert to integer cents per row before the group_by so every arithmetic
        # operation is exact Int64.  Float64 parallel sums are non-associative and
        # produce different totals depending on chunk ordering between runs.
        .with_columns(
            (pl.col("importe") * 100).round(0).cast(pl.Int64).alias("spend_cents")
        )
        .group_by(["cliente", "idsector"])
        .agg(pl.col("spend_cents").sum().alias("spend_cents"))
        .collect(engine="streaming")
        # Streaming group_by can leave multiple partial rows for the same
        # (cliente, idsector) key when chunk boundaries split a customer's rows
        # across merge batches.  Re-aggregate in memory to collapse them before
        # sorting — the result set is small (~1 M customers × few sectors).
        .group_by(["cliente", "idsector"])
        .agg(pl.col("spend_cents").sum())
        .sort(["cliente", "spend_cents", "idsector"], descending=[False, True, False])
        .to_pandas()
    )
    dominant = (
        sector_spend
        .drop_duplicates(subset="cliente", keep="first")
        .set_index("cliente")["idsector"]
    )
    return dominant.map(_SECTOR_BIN_MAP).fillna("non_food").rename("sector_bin")


def _assign_strata(
    kpis: pd.DataFrame,
    store_counts: pd.Series,
    sector_bins: pd.Series,
) -> pd.Series:
    """Assign each customer a stratum label.

    54 strata = 3 visit-count bins × 3 spend bins × 2 store-affinity bins × 3 sector bins:
      v_lo / v_md / v_hi          — visit count tertiles
      s_lo / s_md / s_hi          — total 6-month spend tertiles
      st1 / st2p                  — shops at exactly 1 store vs 2+ stores
      grocery_pgc / fresh / non_food — dominant product sector by spend
    """
    visit_bin  = _tertile_bin(kpis["visit_count"],    "v")
    spend_bin  = _tertile_bin(kpis["total_spend_6m"], "s")
    store_bin  = (
        store_counts
        .reindex(kpis["cliente"].values)
        .map(lambda n: "st1" if n == 1 else "st2p")
        .values
    )
    sector_bin = (
        sector_bins
        .reindex(kpis["cliente"].values)
        .fillna("non_food")
        .values
    )
    return (
        visit_bin + "__" + spend_bin + "__"
        + pd.Series(store_bin,  index=kpis.index) + "__"
        + pd.Series(sector_bin, index=kpis.index)
    )


# ─── sampling ─────────────────────────────────────────────────────────────────

def _stable_customer_rank(cliente: object, seed: int) -> int:
    """Return a process-stable rank key for a customer ID and seed."""
    payload = f"{seed}\0{cliente}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _customer_id_digest(clientes: list[object]) -> str:
    """Digest the selected customer set for cheap reproducibility checks."""
    h = hashlib.sha256()
    for cliente in sorted(str(c) for c in clientes):
        h.update(cliente.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _strata_assignment_digest(kpis: pd.DataFrame) -> str:
    """Digest cliente -> stratum assignments to isolate pre-sampling drift."""
    h = hashlib.sha256()
    ordered = kpis[["cliente", "stratum"]].copy()
    ordered["_cliente_key"] = ordered["cliente"].map(str)
    ordered = ordered.sort_values("_cliente_key", kind="mergesort")
    for cliente, stratum in ordered[["cliente", "stratum"]].itertuples(index=False):
        h.update(str(cliente).encode("utf-8"))
        h.update(b"\t")
        h.update(str(stratum).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _file_stats(path: Path) -> dict[str, int | str]:
    """Cheap source-file fingerprint for generated metadata."""
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _proportional_sample(
    kpis: pd.DataFrame,
    target: int,
    seed: int,
) -> list[object]:
    """Sample proportionally from each stratum.

    Allocation uses largest-remainder apportionment for an exact target size.
    Within each stratum, customers are ranked by a stable hash of seed + cliente,
    so selection is independent of DataFrame row order and Python hash state.
    """
    if target <= 0 or kpis.empty:
        return []

    n_total = len(kpis)
    strata_counts = kpis.groupby("stratum", sort=True).size()
    quotas = strata_counts.astype(float) / float(n_total) * int(target)
    n_by_stratum = np.floor(quotas).astype(int)

    remainder_slots = int(target) - int(n_by_stratum.sum())
    if remainder_slots > 0:
        remainders = pd.DataFrame({
            "stratum": quotas.index.to_list(),
            "fraction": (quotas - n_by_stratum).to_numpy(),
        }).sort_values(
            ["fraction", "stratum"],
            ascending=[False, True],
            kind="mergesort",
        )
        for stratum in remainders.head(remainder_slots)["stratum"]:
            n_by_stratum.loc[stratum] += 1

    selected: list[object] = []
    for stratum, n_sample in n_by_stratum.items():
        if n_sample <= 0:
            continue
        group = kpis.loc[kpis["stratum"].eq(stratum), ["cliente"]].copy()
        group["_rank"] = group["cliente"].map(lambda c: _stable_customer_rank(c, seed))
        group["_cliente_key"] = group["cliente"].map(str)
        group = group.sort_values(["_rank", "_cliente_key"], kind="mergesort")
        selected.extend(group.head(int(n_sample))["cliente"].tolist())

    return selected


# ─── validation ───────────────────────────────────────────────────────────────

def _ks_tests(full: pd.DataFrame, subset: pd.DataFrame) -> dict[str, dict]:
    """Two-sample KS test on each KPI.  p > 0.05 = distributions are indistinguishable."""
    cols = [
        "visit_count",
        "avg_basket_size",
        "total_spend_6m",
        "avg_promo_rate",
        "unique_products",
    ]
    results: dict[str, dict] = {}
    for col in cols:
        stat, p = ks_2samp(full[col].dropna().values, subset[col].dropna().values)
        results[col] = {
            "ks_stat": round(float(stat), 4),
            "p_value": round(float(p),    4),
            "pass":    bool(p > 0.05),
        }
    return results


# ─── main entry point ─────────────────────────────────────────────────────────

def generate(target_size: int = DEFAULT_SIZE, *, force: bool = False) -> dict:
    out_path  = DEV_DIR / "df_combined.parquet"
    meta_path = DEV_DIR / "subset_metadata.json"

    if out_path.exists() and not force:
        print(f"Dev subset already exists — use --force to regenerate.\n  {out_path}")
        return json.loads(meta_path.read_text()) if meta_path.exists() else {}

    DEV_DIR.mkdir(parents=True, exist_ok=True)

    # Verify inputs exist before doing any work
    combined_path = PROD_DIR / "df_combined.parquet"
    kpis_path     = PROD_DIR / "customer_kpis.parquet"
    missing = [p for p in (combined_path, kpis_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required prod files are missing:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun notebook 02_pre-analysis.ipynb in prod mode first:\n"
            "  $env:CARREFOUR_MODE = 'prod'\n"
            "  jupyter notebook notebooks/02_pre-analysis.ipynb\n"
            "Note: CARREFOUR_MODE does not affect this script — it always reads from data/processed/."
        )

    # ── 1. Load KPIs + apply minimum-activity filter ──────────────────────────
    print("Loading customer KPIs ...")
    kpis_full = pd.read_parquet(kpis_path)
    kpis = (
        kpis_full[kpis_full["visit_count"] >= MIN_TICKETS_PER_CUSTOMER]
        .sort_values("cliente")
        .reset_index(drop=True)
    )
    print(f"  {len(kpis):,} eligible customers  "
          f"(dropped {len(kpis_full) - len(kpis):,} with < {MIN_TICKETS_PER_CUSTOMER} visits)")

    if target_size >= len(kpis):
        print(f"  target_size={target_size:,} >= eligible population — taking all customers")
        target_size = len(kpis)

    # ── 2. Per-customer store count ────────────────────────────────────────────
    print("Computing per-customer store counts (streaming) ...")
    store_counts = (
        pl.scan_parquet(combined_path)
        .select(["cliente", "idempres"])
        .group_by("cliente")
        .agg(pl.col("idempres").n_unique().alias("n_stores"))
        .collect(engine="streaming")
        .to_pandas()
        .set_index("cliente")["n_stores"]
    )
    single = int((store_counts == 1).sum())
    multi  = int((store_counts  > 1).sum())
    print(f"  {single:,} single-store  /  {multi:,} multi-store customers")

    # ── 3. Per-customer dominant sector ───────────────────────────────────────
    print("Computing per-customer dominant sector (streaming) ...")
    sector_bins = _dominant_sector_bins(combined_path)
    sector_dist = sector_bins.value_counts().sort_index()
    for bin_name, count in sector_dist.items():
        print(f"  {bin_name:<20}  {count:>9,}  ({count / len(sector_bins) * 100:.1f}%)")

    # ── 4. Assign strata ──────────────────────────────────────────────────────
    print("Assigning strata ...")
    kpis = kpis.copy()
    kpis["stratum"] = _assign_strata(kpis, store_counts, sector_bins)
    strata_digest = _strata_assignment_digest(kpis)
    strata_summary  = kpis["stratum"].value_counts().sort_index()
    n_strata        = len(strata_summary)
    print(f"  {n_strata} non-empty strata (of 54 possible):")
    for stratum, count in strata_summary.items():
        print(f"    {stratum:<60}  {count:>9,}")
    print(f"  Strata-assignment SHA256: {strata_digest}")

    # ── 5. Proportional stratified sample ─────────────────────────────────────
    print(f"\nSampling ~{target_size:,} customers (proportional by stratum) ...")
    selected_ids = _proportional_sample(kpis, target_size, int(RANDOM_SEED))
    actual_n     = len(selected_ids)
    selected_digest = _customer_id_digest(selected_ids)
    print(f"  Selected {actual_n:,} customers  "
          f"({actual_n / len(kpis) * 100:.1f}% of eligible population)")
    print(f"  Selected-customer SHA256: {selected_digest}")

    # ── 6. Extract transaction rows from df_combined ──────────────────────────
    print("Extracting transaction rows (streaming filter) ...")
    id_series = pl.Series("cliente", sorted(selected_ids))
    subset_df = (
        pl.scan_parquet(combined_path)
        .filter(pl.col("cliente").is_in(id_series))
        .collect(engine="streaming")
    )
    subset_df = subset_df.sort(subset_df.columns)   # canonical row order → reproducible downstream
    n_rows = len(subset_df)
    subset_df.write_parquet(out_path, compression="zstd")
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"  {n_rows:,} rows  ({n_rows / actual_n:.0f} avg per customer)  "
          f"→  {out_path.name}  ({size_mb:.0f} MB)")

    # ── 7. KS distribution validation ─────────────────────────────────────────
    print("\nKS distribution tests — subset vs full eligible population (p > 0.05 = PASS):")
    subset_kpis = kpis[kpis["cliente"].isin(set(selected_ids))].reset_index(drop=True)
    ks_results  = _ks_tests(kpis, subset_kpis)
    all_pass    = all(r["pass"] for r in ks_results.values())

    col_width = max(len(c) for c in ks_results)
    for col, r in ks_results.items():
        flag = "PASS" if r["pass"] else "FAIL ⚠"
        print(f"  {col:<{col_width}}   KS={r['ks_stat']:.4f}   p={r['p_value']:.4f}   [{flag}]")

    if all_pass:
        print("\n  All KS tests passed — subset preserves the full population's distributions.")
    else:
        failed = [c for c, r in ks_results.items() if not r["pass"]]
        print(f"\n  WARNING: {len(failed)} KS test(s) failed: {failed}")
        print("  The subset distributions diverge from the full population for these KPIs.")
        print("  Consider increasing --target-size or re-running (different seed is not an option")
        print("  since RANDOM_SEED is fixed for reproducibility).")

    # ── 8. Write metadata ─────────────────────────────────────────────────────
    metadata = {
        "target_size":         target_size,
        "actual_size":         actual_n,
        "n_transaction_rows":  n_rows,
        "avg_rows_per_customer": round(n_rows / actual_n, 1),
        "random_seed":         int(RANDOM_SEED),
        "selection_method":     "stratified_largest_remainder_stable_hash_v1",
        "strata_assignment_sha256": strata_digest,
        "selected_customer_sha256": selected_digest,
        "min_tickets_filter":  int(MIN_TICKETS_PER_CUSTOMER),
        "eligible_population": len(kpis),
        "stratification": {
            "dimensions": [
                "visit_count — 3 tertile bins  (v_lo / v_md / v_hi)",
                "total_spend_6m — 3 tertile bins  (s_lo / s_md / s_hi)",
                "store_count — binary  (st1 = single-store / st2p = multi-store)",
                "dominant_sector — 3 bins  (grocery_pgc / fresh / non_food)",
            ],
            "sector_bin_map": {str(k): v for k, v in _SECTOR_BIN_MAP.items()},
            "sector_distribution": sector_dist.to_dict(),
            "n_strata_possible":  54,
            "n_strata_populated": int(n_strata),
            "strata_counts":      {k: int(v) for k, v in strata_summary.items()},
        },
        "ks_validation": ks_results,
        "all_ks_passed":  bool(all_pass),
        "source_combined": str(combined_path),
        "source_kpis":     str(kpis_path),
        "source_file_stats": {
            "combined": _file_stats(combined_path),
            "kpis": _file_stats(kpis_path),
        },
        "output":          str(out_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata written → {meta_path}")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target-size", type=int, default=DEFAULT_SIZE,
        help=f"Number of customers to sample (default: {DEFAULT_SIZE:,})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing dev subset",
    )
    args = parser.parse_args()
    generate(target_size=args.target_size, force=args.force)


if __name__ == "__main__":
    main()
