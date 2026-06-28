import polars as pl

from src.config import PipelineConfig
from src.exports import write_stage9_presentation_pack
from src.profiling import PRODUCT_RANKING_BASIS


def _cfg(tmp_path):
    return PipelineConfig(
        values={
            "paths": {"outputs": "outputs"},
            "exports": {"presentation_dir": "presentation", "evidence_dir": "evidence"},
        },
        mode="dev",
        root=tmp_path,
    )


def test_stage9_manifest_discovers_per_tribe_analyst_exports(tmp_path):
    cfg = _cfg(tmp_path)
    core_summary = tmp_path / "core_summary.csv"
    mission_summary = tmp_path / "mission_summary.csv"
    atlas = tmp_path / "atlas.html"
    assignments = tmp_path / "assignments.parquet"
    profiles = tmp_path / "profiles.csv"
    output_dir = tmp_path / "stage9"

    pl.DataFrame(
        {
            "tribe_id": [0],
            "suggested_tribe_name": ["Greek Yogurt Buyers"],
            "n_customers": [2],
            "population_share_pct": [10.0],
            "soft_assigned_share_pct": [0.0],
            "top_products": ["Greek Yogurt; Honey"],
        }
    ).write_csv(core_summary)
    pl.DataFrame(
        {
            "mission_family": ["dairy"],
            "mission_label": ["Dairy restock"],
            "mission_customers": [2],
            "population_share_pct": [10.0],
            "mission_confidence": ["high"],
            "top_core_tribes": ["T0"],
        }
    ).write_csv(mission_summary)
    atlas.write_text("<html></html>", encoding="utf-8")
    pl.DataFrame({"cliente": ["c1"], "tribe_id": [0]}).write_parquet(assignments)
    profiles.write_text("tribe_id,top_products\n0,Greek Yogurt\n", encoding="utf-8")

    raw_dir = cfg.artifacts / "stage7" / "final_handoff" / "supporting_tables" / "tribe_raw_transactions"
    customer_dir = cfg.artifacts / "stage7" / "final_handoff" / "supporting_tables" / "tribe_customer_summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    customer_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"tribe_raw_transactions_manifest_{cfg.mode}.csv").write_text(
        "tribe_id,rows,customers,path\n0,1,1,tribe_00_transactions.parquet\n",
        encoding="utf-8",
    )
    (customer_dir / f"tribe_customer_summaries_manifest_{cfg.mode}.csv").write_text(
        "tribe_id,rows,customers,path\n0,1,1,tribe_00_customer_summary.parquet\n",
        encoding="utf-8",
    )
    stage67_figure = cfg.figures / "stage6_7_remaining_noise_umap_probe.png"
    stage67_figure.parent.mkdir(parents=True, exist_ok=True)
    stage67_figure.write_bytes(b"png")

    outputs = write_stage9_presentation_pack(
        selected={"model_name": "hdbscan", "model_variant": "official"},
        core_summary_path=core_summary,
        mission_summary_path=mission_summary,
        clustering_atlas_path=atlas,
        assignment_path=assignments,
        profile_path=profiles,
        output_dir=output_dir,
        cfg=cfg,
    )

    manifest = pl.read_csv(outputs["manifest"])
    raw_manifest = manifest.filter(pl.col("artifact") == "Per-Tribe Raw Transaction Export Manifest").row(0, named=True)
    customer_manifest = manifest.filter(pl.col("artifact") == "Per-Tribe Customer Summary Export Manifest").row(0, named=True)
    stage67_manifest = manifest.filter(pl.col("artifact") == "Stage 6.7 Remaining Noise UMAP Probe").row(0, named=True)
    markdown = outputs["markdown"].read_text(encoding="utf-8")
    html = outputs["html"].read_text(encoding="utf-8")

    assert raw_manifest["tier"] == "official export"
    assert "transaction-line parquet" in raw_manifest["purpose"]
    assert customer_manifest["tier"] == "official export"
    assert "customer-level KPI parquet" in customer_manifest["purpose"]
    assert stage67_manifest["tier"] == "presentation visual"
    assert "candidate-only third clustering pass" in stage67_manifest["purpose"]
    assert PRODUCT_RANKING_BASIS in markdown
    assert PRODUCT_RANKING_BASIS in html
