import json
from datetime import date

import polars as pl

from src.config import PipelineConfig
from src.profiling import (
    _add_overindex_diagnostics,
    _empty_customer_metric_tests,
    _empty_noise_vs_core_metric_tests,
    build_stage68_tribe_evidence,
    customer_metric_anova_table,
    noise_vs_core_customer_metric_table,
    noise_audit_table,
    profile_quality_summary,
    profile_readiness_evidence_table,
    remaining_customer_affinity_table,
    remaining_customer_segments_table,
    stage7_all_tribe_behavior_differentiation_table,
    stage7_all_tribe_product_identity_table,
    stage7_all_tribe_profiles_table,
    stage7_campaign_playbook_table,
    stage7_final_index_table,
    stage7_promoted_tribe_validation_table,
    stage7_review_tribe_audit_table,
    stage7_soft_audience_opportunities_table,
    stage7_soft_audience_activation_customer_table,
    stage7_stakeholder_readiness_table,
    stage7_storyline_table,
    stage7_tribe_handbook_table,
    stage7_tribe_promotion_report_table,
    stage7_tribe_relationship_atlas_table,
    stage68_artifact_paths,
    tribe_comparison_table,
    tribe_product_summary_tables,
    write_tribe_product_summary_artifacts,
    write_campaign_signal_artifacts,
    write_llm_profile_interpretation_pack,
    write_noise_audit_artifacts,
    write_stage7_final_handoff_pack,
    write_stage7_storyline_artifacts,
)
from src.visualization import plot_stage68_evidence_overview


def _test_config(tmp_path):
    return PipelineConfig(
        values={
            "paths": {"outputs": "outputs"},
            "profiling": {
                "strong_product_lift_threshold": 1.5,
                "strong_sector_lift_threshold": 1.2,
                "strong_theme_lift_threshold": 1.2,
                "strong_term_lift_threshold": 1.25,
                "significance_q_threshold": 0.05,
                "min_ready_evidence_signals": 1,
                "campaign_signal_themes": [],
                "noise_audit_top_n": 10,
                "noise_audit_min_customers": 1,
                "noise_audit_strong_lift_threshold": 1.25,
                "noise_audit_high_noise_pct": 50.0,
                "noise_audit_second_pass_signal_count": 2,
            },
        },
        mode="dev",
        root=tmp_path,
    )


def test_profile_readiness_evidence_combines_stage6_and_significant_lift(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"
    readiness_path = tmp_path / "stage6_4_cluster_readiness.csv"
    output_path = tmp_path / "stage7_profile_readiness.csv"

    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 500,
                "population_share": 0.10,
                "soft_assigned_share": 0.0,
                "top_products": ["Bio Milk"],
                "top_product_lifts": [2.0],
                "top_product_lifts_vs_rest": [2.2],
                "top_product_q_values": [0.001],
                "top_product_customer_counts": [120],
                "top_sectors": ["P.G.C."],
                "top_sector_lifts": [1.3],
                "top_themes": ["organic_bio", "dairy_eggs"],
                "top_theme_lifts": [1.5, 1.3],
                "top_theme_lifts_vs_rest": [1.6, 1.35],
                "top_theme_q_values": [0.002, 0.01],
                "top_theme_customer_counts": [180, 160],
                "top_product_terms": [],
                "top_product_term_lifts": [],
                "top_product_term_lifts_vs_rest": [],
                "top_product_term_q_values": [],
            }
        ]
    ).write_parquet(profile_path)
    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "profile_readiness": "strong",
                "readiness_issues": "pass",
                "customers": 500,
                "mean_assignment_confidence": 0.85,
                "p10_assignment_confidence": 0.72,
                "jitter_label_recovery_accuracy_mean": 0.91,
            }
        ]
    ).write_csv(readiness_path)

    readiness = profile_readiness_evidence_table(
        profile_path,
        cluster_readiness_path=readiness_path,
        output_csv=output_path,
        cfg=cfg,
    )
    row = readiness.row(0, named=True)

    assert output_path.exists()
    assert row["profiling_readiness"] == "ready_strong"
    assert row["stage6_profile_readiness"] == "strong"
    assert row["profile_significant_interpretability_signals"] == 1
    assert row["profiling_readiness_issues"] == "pass"


def test_profile_readiness_marks_usable_stage6_clusters_as_ready(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"
    readiness_path = tmp_path / "stage6_cluster_readiness.csv"

    pl.DataFrame(
        [
            {
                "tribe_id": 2,
                "n_customers": 800,
                "population_share": 0.10,
                "top_products": ["Lifted SKU"],
                "top_product_lifts": [2.0],
                "top_product_lifts_vs_rest": [2.3],
                "top_product_q_values": [0.001],
                "top_product_customer_counts": [120],
                "top_sectors": ["P.G.C."],
                "top_sector_lifts": [1.2],
            }
        ]
    ).write_parquet(profile_path)
    pl.DataFrame(
        [
            {
                "tribe_id": 2,
                "profile_readiness": "usable",
                "readiness_issues": "pass",
                "customers": 800,
                "mean_assignment_confidence": 0.84,
                "p10_assignment_confidence": 0.62,
                "jitter_label_recovery_accuracy_mean": 0.82,
            }
        ]
    ).write_csv(readiness_path)

    readiness = profile_readiness_evidence_table(
        profile_path,
        cluster_readiness_path=readiness_path,
        cfg=cfg,
    )
    row = readiness.row(0, named=True)

    assert row["stage6_profile_readiness"] == "usable"
    assert row["profiling_readiness"] == "ready"
    assert row["profiling_readiness_issues"] == "pass"


def test_profile_readiness_keeps_stage6_review_cases_in_review(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"
    readiness_path = tmp_path / "stage6_cluster_readiness.csv"

    profile_rows = []
    for tribe_id in [4, 10, 14]:
        profile_rows.append(
            {
                "tribe_id": tribe_id,
                "n_customers": 500,
                "population_share": 0.10,
                "top_products": ["Coherent Organic Product"],
                "top_product_lifts": [2.0],
                "top_product_lifts_vs_rest": [2.2],
                "top_product_q_values": [0.001],
                "top_product_customer_counts": [120],
                "top_sectors": ["Grocery"],
                "top_sector_lifts": [1.2],
            }
        )
    pl.DataFrame(profile_rows).write_parquet(profile_path)
    pl.DataFrame(
        [
            {
                "tribe_id": 4,
                "profile_readiness": "review",
                "readiness_issues": "jitter_recovery<0.70",
                "mean_assignment_confidence": 0.978,
                "p10_assignment_confidence": 0.91,
                "jitter_label_recovery_accuracy_mean": 0.688,
            },
            {
                "tribe_id": 10,
                "profile_readiness": "review",
                "readiness_issues": "jitter_recovery<0.70",
                "mean_assignment_confidence": 0.903,
                "p10_assignment_confidence": 0.70,
                "jitter_label_recovery_accuracy_mean": 0.509,
            },
            {
                "tribe_id": 14,
                "profile_readiness": "review",
                "readiness_issues": "jitter_recovery<0.70",
                "mean_assignment_confidence": 0.997,
                "p10_assignment_confidence": 0.95,
                "jitter_label_recovery_accuracy_mean": 0.590,
            },
        ]
    ).write_csv(readiness_path)

    readiness = profile_readiness_evidence_table(profile_path, cluster_readiness_path=readiness_path, cfg=cfg)
    statuses = {row["tribe_id"]: row for row in readiness.iter_rows(named=True)}

    assert statuses[4]["profiling_readiness"] == "review"
    assert statuses[4]["profiling_readiness_issues"] == "stage6_readiness=review"
    assert statuses[10]["profiling_readiness"] == "review"
    assert statuses[14]["profiling_readiness"] == "review"


def test_overindex_diagnostics_compare_cluster_against_rest_population(tmp_path):
    cfg = _test_config(tmp_path)
    frame = pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "idarticu": "sku_a",
                "cluster_customers": 50,
                "population_customers": 60,
                "n_customers": 100,
            }
        ]
    )

    enriched = _add_overindex_diagnostics(
        frame,
        cluster_count_col="cluster_customers",
        population_count_col="population_customers",
        cluster_total_col="n_customers",
        population_total=1000,
        cfg=cfg,
    )
    row = enriched.row(0, named=True)

    assert round(row["cluster_rate"], 3) == 0.5
    assert round(row["population_rate"], 3) == 0.06
    assert round(row["rest_rate"], 3) == 0.011
    assert row["lift_vs_rest"] > row["lift"]
    assert row["lift_q_value"] < 0.05
    assert row["overindex_evidence"] == "strong_significant_overindex"


def test_profile_quality_summary_reports_significant_profile_evidence(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"

    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 500,
                "population_share": 0.10,
                "soft_assigned_share": 0.0,
                "top_product_lifts": [2.0, 1.2],
                "top_product_lifts_vs_rest": [2.1, 1.25],
                "top_product_q_values": [0.01, 0.20],
                "top_sector_lifts": [1.3],
                "top_theme_lifts": [1.1],
                "top_theme_q_values": [0.01],
                "top_product_term_lifts": [],
                "top_product_term_q_values": [],
            },
            {
                "tribe_id": 1,
                "n_customers": 600,
                "population_share": 0.12,
                "soft_assigned_share": 0.0,
                "top_product_lifts": [1.7],
                "top_product_lifts_vs_rest": [1.8],
                "top_product_q_values": [0.20],
                "top_sector_lifts": [],
                "top_theme_lifts": [],
                "top_theme_q_values": [],
                "top_product_term_lifts": [],
                "top_product_term_q_values": [],
            },
        ]
    ).write_parquet(profile_path)

    quality = profile_quality_summary(profile_path, cfg=cfg)

    assert quality["clusters_with_product_lift"] == 2
    assert quality["clusters_with_significant_product_lift"] == 1
    assert quality["avg_significant_product_lifts_per_cluster"] == 0.5
    assert abs(quality["avg_max_product_lift_vs_rest"] - 1.95) < 1e-9


def test_campaign_signal_artifacts_are_inert_without_explicit_rebuild(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"
    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 100,
                "suggested_tribe_name": "Avocado Tortilla Evidence Cluster",
                "top_themes": ["baby", "plant_based"],
                "top_theme_lifts": [2.5, 1.8],
                "top_theme_customer_counts": [25, 20],
            }
        ]
    ).write_parquet(profile_path)

    default_outputs = write_campaign_signal_artifacts(
        profile_path,
        output_csv=tmp_path / "campaign_default.csv",
        output_md=tmp_path / "campaign_default.md",
        output_html=tmp_path / "campaign_default.html",
        cfg=cfg,
    )
    default_campaign = pl.read_csv(default_outputs["csv"])

    assert default_campaign.is_empty()

    opt_in_outputs = write_campaign_signal_artifacts(
        profile_path,
        campaign_themes={"plant_based"},
        output_csv=tmp_path / "campaign_opt_in.csv",
        output_md=tmp_path / "campaign_opt_in.md",
        output_html=tmp_path / "campaign_opt_in.html",
        cfg=cfg,
    )
    opt_in_campaign = pl.read_csv(opt_in_outputs["csv"])

    assert opt_in_campaign.is_empty()
    assert "No opt-in campaign signals" in opt_in_outputs["markdown"].read_text(encoding="utf-8")


def test_llm_profile_interpretation_pack_is_evidence_grounded(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"
    readiness_path = tmp_path / "stage7_profile_readiness.csv"
    json_path = tmp_path / "llm_pack.json"
    md_path = tmp_path / "llm_pack.md"

    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 500,
                "population_share": 0.10,
                "core_customers": 500,
                "soft_assigned_customers": 0,
                "soft_assigned_share": 0.0,
                "profile_population_customers": 5000,
                "profile_noise_customers": 1000,
                "suggested_tribe_name": "Greek Yogurt Evidence Cluster",
                "suggested_tribe_name_source": "lifted_product_terms",
                "suggested_tribe_name_evidence": "greek yogurt",
                "suggested_tribe_name_status": "unique_working_name",
                "profile_evidence_note": "Product-term label backed by significant lift.",
                "top_products": ["Greek Yogurt", "Honey"],
                "top_product_lifts": [2.0, 1.7],
                "top_product_lifts_vs_rest": [2.4, 1.9],
                "top_product_q_values": [0.001, 0.02],
                "top_product_customer_counts": [120, 80],
                "top_product_reach_pct": [8.0, 30.0],
                "top_product_terms": ["greek yogurt"],
                "top_product_term_lifts": [1.8],
                "top_product_term_lifts_vs_rest": [2.0],
                "top_product_term_q_values": [0.01],
                "top_product_term_customer_counts": [100],
                "top_themes": ["dairy_eggs"],
                "top_theme_lifts": [1.5],
                "top_theme_lifts_vs_rest": [1.6],
                "top_theme_q_values": [0.02],
                "top_theme_customer_counts": [180],
                "top_sectors": ["Dairy"],
                "top_sector_lifts": [1.4],
                "top_sector_lifts_vs_rest": [1.5],
                "top_sector_q_values": [0.03],
                "avg_ticket_count": 5.0,
            }
        ]
    ).write_parquet(profile_path)
    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "stage6_profile_readiness": "strong",
                "profiling_readiness": "ready_strong",
                "profiling_readiness_issues": "pass",
                "mean_assignment_confidence": 0.85,
                "p10_assignment_confidence": 0.70,
                "jitter_label_recovery_accuracy_mean": 0.90,
            }
        ]
    ).write_csv(readiness_path)

    outputs = write_llm_profile_interpretation_pack(
        profile_path,
        readiness_path=readiness_path,
        output_json=json_path,
        output_md=md_path,
        cfg=cfg,
    )
    payload = json_path.read_text(encoding="utf-8")
    markdown = md_path.read_text(encoding="utf-8")

    assert outputs["json"] == json_path
    assert outputs["markdown"] == md_path
    assert "Do not infer age" in payload
    assert "Greek Yogurt" in payload
    assert "supporting_curated_themes" in payload
    assert "cliente" not in payload
    assert "```text" in markdown


def test_stage7_storyline_artifacts_turn_profiles_into_evidence_ladder(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"
    readiness_path = tmp_path / "stage7_profile_readiness.csv"
    subsegment_path = tmp_path / "subsegments.csv"
    metric_path = tmp_path / "customer_metric_tests.csv"

    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 500,
                "population_share": 0.10,
                "core_customers": 500,
                "soft_assigned_customers": 0,
                "soft_assigned_share": 0.0,
                "profile_population_customers": 5000,
                "profile_noise_customers": 1000,
                "suggested_tribe_name": "Greek Yogurt Evidence Cluster",
                "suggested_tribe_name_source": "lifted_product_terms",
                "suggested_tribe_name_evidence": "greek yogurt",
                "suggested_tribe_name_status": "unique_working_name",
                "top_products": ["Greek Yogurt", "Honey"],
                "top_product_lifts": [2.0, 1.7],
                "top_product_lifts_vs_rest": [2.4, 1.9],
                "top_product_q_values": [0.001, 0.02],
                "top_product_customer_counts": [120, 80],
                "top_product_reach_pct": [8.0, 30.0],
                "top_product_terms": ["greek yogurt"],
                "top_product_term_lifts": [1.8],
                "top_product_term_lifts_vs_rest": [2.0],
                "top_product_term_q_values": [0.01],
                "top_product_term_customer_counts": [100],
                "top_themes": ["dairy_eggs"],
                "top_theme_lifts": [1.5],
                "top_theme_lifts_vs_rest": [1.6],
                "top_theme_q_values": [0.02],
                "top_theme_customer_counts": [180],
                "top_sectors": ["Dairy"],
                "top_sector_lifts": [1.4],
                "top_sector_lifts_vs_rest": [1.5],
                "top_sector_q_values": [0.03],
                "avg_ticket_count": 5.0,
            }
        ]
    ).write_parquet(profile_path)
    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "stage6_profile_readiness": "strong",
                "profiling_readiness": "ready_strong",
                "profiling_readiness_issues": "pass",
                "profile_significant_interpretability_signals": 3,
            }
        ]
    ).write_csv(readiness_path)
    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "working_tribe_name": "Greek Yogurt Evidence Cluster",
                "subsegment_type": "data_driven_product_term",
                "subsegment_label": "greek yogurt",
                "subsegment_key": "greek yogurt",
                "subsegment_customers": 100,
                "tribe_customers": 500,
                "subsegment_share_pct": 20.0,
                "subsegment_lift": 1.8,
                "avg_matching_product_count": 1.2,
            }
        ]
    ).write_csv(subsegment_path)
    pl.DataFrame(
        [
            {
                "metric": "ticket_count",
                "anova_q_value": 0.01,
                "anova_effect_eta_squared": 0.4,
                "highest_mean_tribe_id": 0,
                "lowest_mean_tribe_id": 1,
                "statistical_result": "significant",
            }
        ]
    ).write_csv(metric_path)

    outputs = write_stage7_storyline_artifacts(
        profile_path,
        readiness_path=readiness_path,
        subsegment_summary_path=subsegment_path,
        customer_metric_tests_path=metric_path,
        output_csv=tmp_path / "storyline.csv",
        output_md=tmp_path / "storyline.md",
        output_html=tmp_path / "storyline.html",
        cfg=cfg,
    )
    storyline = stage7_storyline_table(
        profile_path,
        readiness_path=readiness_path,
        subsegment_summary_path=subsegment_path,
        cfg=cfg,
    )
    row = storyline.row(0, named=True)
    markdown = outputs["markdown"].read_text(encoding="utf-8")

    assert outputs["csv"].exists()
    assert outputs["html"].exists()
    assert "Stage 7 Evidence Storyline" in markdown
    assert "not a demographic persona" in row["caveat"]
    assert "Greek Yogurt" in row["distinctive_product_evidence"]
    assert "lift_vs_rest x log(customer_count + 1)" in row["product_ranking_basis"]
    assert "Reach caveat" in row["top_product_reach_warning"]
    assert "Reach caveat" in row["caveat"]
    assert "Greek Yogurt" in markdown


def test_tribe_product_summaries_and_comparison_are_product_first(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"
    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 100,
                "population_share": 0.25,
                "core_customers": 100,
                "soft_assigned_customers": 0,
                "soft_assigned_share": 0.0,
                "suggested_tribe_name": "Greek Yogurt Evidence Cluster",
                "suggested_tribe_name_source": "lifted_product_terms",
                "suggested_tribe_name_evidence": "greek yogurt",
                "suggested_tribe_name_status": "unique_working_name",
                "top_product_ids": ["sku_yogurt", "sku_honey"],
                "top_products": ["Greek Yogurt", "Honey"],
                "top_product_sectors": ["Dairy", "Grocery"],
                "top_product_sector_ids": ["1", "2"],
                "top_product_lifts": [2.0, 1.7],
                "top_product_lifts_vs_rest": [2.4, 1.9],
                "top_product_q_values": [0.001, 0.02],
                "top_product_customer_counts": [40, 25],
                "top_sectors": ["Dairy"],
                "top_sector_lifts": [1.4],
                "top_sector_lifts_vs_rest": [1.5],
                "top_sector_q_values": [0.01],
                "top_themes": [],
                "top_theme_lifts": [],
                "top_theme_lifts_vs_rest": [],
                "top_theme_q_values": [],
                "top_product_terms": ["greek yogurt"],
                "top_product_term_lifts": [1.8],
                "top_product_term_lifts_vs_rest": [2.0],
                "top_product_term_q_values": [0.01],
                "top_product_term_customer_counts": [35],
                "behavior_ratio_vs_rest": '{"avg_ticket_count": 2.5, "avg_unique_products": 1.8}',
                "avg_ticket_count": 5.0,
                "avg_total_units": 30.0,
            },
            {
                "tribe_id": 1,
                "n_customers": 300,
                "population_share": 0.75,
                "core_customers": 300,
                "soft_assigned_customers": 0,
                "soft_assigned_share": 0.0,
                "suggested_tribe_name": "Pasta Sauce Evidence Cluster",
                "suggested_tribe_name_source": "lifted_product_terms",
                "suggested_tribe_name_evidence": "pasta sauce",
                "suggested_tribe_name_status": "unique_working_name",
                "top_product_ids": ["sku_pasta"],
                "top_products": ["Pasta Sauce"],
                "top_product_sectors": ["Grocery"],
                "top_product_sector_ids": ["2"],
                "top_product_lifts": [1.8],
                "top_product_lifts_vs_rest": [2.1],
                "top_product_q_values": [0.004],
                "top_product_customer_counts": [90],
                "top_sectors": ["Grocery"],
                "top_sector_lifts": [1.2],
                "top_sector_lifts_vs_rest": [1.3],
                "top_sector_q_values": [0.03],
                "top_themes": [],
                "top_theme_lifts": [],
                "top_theme_lifts_vs_rest": [],
                "top_theme_q_values": [],
                "top_product_terms": ["pasta sauce"],
                "top_product_term_lifts": [1.5],
                "top_product_term_lifts_vs_rest": [1.7],
                "top_product_term_q_values": [0.02],
                "top_product_term_customer_counts": [80],
                "behavior_ratio_vs_rest": '{"avg_ticket_count": 0.4, "avg_unique_products": 0.8}',
                "avg_ticket_count": 2.0,
                "avg_total_units": 10.0,
            },
        ]
    ).write_parquet(profile_path)

    product_tables = tribe_product_summary_tables(profile_path, cfg=cfg)
    first_product = product_tables[0].row(0, named=True)
    comparison = tribe_comparison_table(profile_path, cfg=cfg)
    comparison_row = comparison.filter(pl.col("tribe_id") == 0).row(0, named=True)
    outputs = write_tribe_product_summary_artifacts(profile_path, output_dir=tmp_path / "summaries", cfg=cfg)

    assert first_product["product_description"] == "Greek Yogurt"
    assert first_product["category"] == "Dairy"
    assert first_product["statistical_result"] == "strong_significant_overindex"
    assert first_product["product_rank_score"] > 0
    assert "lift_vs_rest x log(customer_count + 1)" in first_product["product_ranking_basis"]
    assert "Greek Yogurt (Dairy" in comparison_row["top_product_and_category_evidence"]
    assert "lift_vs_rest x log(customer_count + 1)" in comparison_row["product_ranking_basis"]
    assert "Tickets" in comparison_row["customer_behavior_over_under_index"]
    assert outputs["combined_csv"].exists()


def test_stage68_cache_hit_rebases_stale_manifest_paths_to_local_outputs(tmp_path):
    cfg = PipelineConfig(
        values={
            "paths": {"outputs": "outputs", "dev": "data/dev", "processed": "data/processed"},
            "data": {"prepared_transactions": "prepared_transactions.parquet"},
        },
        mode="dev",
        root=tmp_path,
    )
    paths = stage68_artifact_paths(cfg)
    local_transaction_path = paths["transaction_export_dir"] / "tribe_00_transactions_dev.parquet"
    local_customer_path = paths["customer_export_dir"] / "tribe_00_customers_dev.parquet"
    required_outputs = [
        paths["tribe_evidence_path"],
        paths["product_lifts_path"],
        paths["sector_lifts_path"],
        paths["customer_metric_tests_csv"],
        paths["noise_vs_core_customer_metrics_csv"],
        paths["transaction_export_dir"] / "tribe_transactions_manifest_dev.csv",
        paths["customer_export_dir"] / "tribe_customers_manifest_dev.csv",
        local_transaction_path,
        local_customer_path,
    ]
    for output in required_outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("cached", encoding="utf-8")

    old_root = "/home/capstone06/carrefour_capstone/outputs/dev/artifacts/stage6/stage6_8_evidence"
    paths["manifest_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest_json"].write_text(
        json.dumps(
            {
                "stage": "6.8_tribe_evidence_assembly",
                "mode": "dev",
                "outputs": {
                    "tribe_evidence_path": f"{old_root}/tribe_evidence_dev.parquet",
                    "product_lifts_path": f"{old_root}/product_lifts_dev.parquet",
                    "sector_lifts_path": f"{old_root}/sector_lifts_dev.parquet",
                    "customer_metric_tests_csv": f"{old_root}/customer_metric_tests_dev.csv",
                    "noise_vs_core_customer_metrics_csv": f"{old_root}/noise_vs_core_customer_metrics_dev.csv",
                    "transaction_export_dir": f"{old_root}/tribe_transactions",
                    "transaction_export_manifest_csv": f"{old_root}/tribe_transactions/tribe_transactions_manifest_dev.csv",
                    "customer_export_dir": f"{old_root}/tribe_customers",
                    "customer_export_manifest_csv": f"{old_root}/tribe_customers/tribe_customers_manifest_dev.csv",
                },
                "transaction_exports": [
                    {"tribe_id": 0, "path": f"{old_root}/tribe_transactions/tribe_00_transactions_dev.parquet"}
                ],
                "customer_exports": [
                    {"tribe_id": 0, "path": f"{old_root}/tribe_customers/tribe_00_customers_dev.parquet"}
                ],
            }
        ),
        encoding="utf-8",
    )

    stage68 = build_stage68_tribe_evidence(
        tmp_path / "missing_assignments.parquet",
        behavior_path=tmp_path / "missing_behavior.parquet",
        cfg=cfg,
    )

    assert stage68["tribe_evidence_path"] == paths["tribe_evidence_path"]
    assert stage68["product_lifts_path"] == paths["product_lifts_path"]
    assert stage68["transaction_export_manifest_csv"] == paths["transaction_export_dir"] / "tribe_transactions_manifest_dev.csv"
    assert stage68["manifest"]["outputs"]["tribe_evidence_path"] == str(paths["tribe_evidence_path"])
    assert stage68["manifest"]["transaction_exports"][0]["path"] == str(local_transaction_path)
    assert stage68["manifest"]["customer_exports"][0]["path"] == str(local_customer_path)


def test_stage68_evidence_bundle_feeds_stage7_without_raw_inputs(tmp_path):
    cfg = PipelineConfig(
        values={
            "paths": {
                "outputs": "outputs",
                "dev": "data/dev",
                "processed": "data/processed",
                "raw_parquet": "data/raw/parquet",
                "raw_csv": "data/raw/csv",
            },
            "data": {"prepared_transactions": "prepared_transactions.parquet"},
            "profiling": {
                "min_product_customers": 1,
                "strong_product_lift_threshold": 1.2,
                "strong_sector_lift_threshold": 1.1,
                "significance_q_threshold": 0.2,
                "final_handoff_readiness_statuses": ["ready_strong"],
                "final_actionability_allow_theme_proof": False,
            },
        },
        mode="dev",
        root=tmp_path,
    )
    cfg.ensure_directories()
    cfg.prepared_transactions_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_path = tmp_path / "assignments.parquet"
    behavior_path = tmp_path / "behavior.parquet"
    readiness_path = tmp_path / "readiness.csv"
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "c5"],
            "tribe_id": [0, 0, 1, 1, -1],
            "assignment_confidence_score": [0.9, 0.8, 0.85, 0.75, None],
            "assignment_source": ["hdbscan_core", "hdbscan_core", "hdbscan_core", "hdbscan_core", "hdbscan_noise"],
            "jitter_label_recovery_accuracy": [0.95, 0.95, 0.9, 0.9, None],
        }
    ).write_parquet(assignments_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "c5"],
            "ticket_count": [4, 5, 2, 2, 1],
            "total_spend": [40.0, 50.0, 12.0, 15.0, 6.0],
            "avg_basket_value": [10.0, 10.0, 6.0, 7.5, 6.0],
            "promo_share": [0.1, 0.2, 0.05, 0.05, 0.0],
            "recency_days": [3, 4, 20, 25, 40],
            "frequency_per_30d": [3.0, 3.5, 1.0, 1.2, 0.5],
            "unique_products": [3, 3, 2, 2, 1],
            "unique_sectors": [2, 2, 1, 1, 1],
        }
    ).write_parquet(behavior_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "c5"],
            "idarticu": ["p_yogurt", "p_yogurt", "p_hummus", "p_hummus", "p_noise"],
            "ticket": ["t1", "t2", "t3", "t4", "t5"],
            "fecha": [date(2024, 1, 2), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 10), date(2024, 1, 11)],
            "hora": [10, 11, 20, 21, 12],
            "unidades": [1, 2, 1, 1, 1],
            "importe": [4.0, 5.0, 3.0, 3.5, 2.0],
            "idpromoc": [None, "promo", None, None, None],
            "desc_larga_articulo": ["Greek Yogurt", "Greek Yogurt", "Hummus", "Hummus", "Noise Product"],
            "desc_sector": ["Dairy", "Dairy", "Prepared Foods", "Prepared Foods", "Other"],
        }
    ).write_parquet(cfg.prepared_transactions_path)
    pl.DataFrame(
        {
            "tribe_id": [0, 1],
            "profile_readiness": ["ready_strong", "ready_strong"],
            "readiness_issues": ["pass", "pass"],
        }
    ).write_csv(readiness_path)

    stage68 = build_stage68_tribe_evidence(
        assignments_path,
        cluster_readiness_path=readiness_path,
        behavior_path=behavior_path,
        force=True,
        cfg=cfg,
    )
    evidence = pl.read_parquet(stage68["tribe_evidence_path"])
    product_lifts = pl.read_parquet(stage68["product_lifts_path"])
    transaction_manifest = pl.read_csv(stage68["transaction_export_manifest_csv"])
    customer_manifest = pl.read_csv(stage68["customer_export_manifest_csv"])

    assert stage68["manifest_json"].name == "stage68_manifest_dev.json"
    assert stage68["tribe_evidence_path"].parent == (
        tmp_path / "outputs" / "dev" / "artifacts" / "stage6" / "stage6_8_evidence"
    )
    assert evidence.height == 2
    assert evidence["stage6_profile_readiness"].to_list() == ["ready_strong", "ready_strong"]
    assert "unassigned_noise_customers_global" in evidence.columns
    assert "noise_customers" not in evidence.columns
    assert {"lift_vs_rest", "lift_q_value", "reach_pct"}.issubset(set(product_lifts.columns))
    overview = plot_stage68_evidence_overview(
        stage68["tribe_evidence_path"],
        noise_vs_core_path=stage68["noise_vs_core_customer_metrics_csv"],
        output_path=tmp_path / "stage6_8_overview.png",
        cfg=cfg,
    )
    assert overview.exists()
    assert transaction_manifest[0, "path"].endswith("tribe_00_transactions_dev.parquet")
    assert customer_manifest[0, "path"].endswith("tribe_00_customers_dev.parquet")
    assert stage68["manifest"]["readiness_summary"]["promoted_tribes"] == 2
    assert stage68["manifest"]["readiness_summary"]["missing_readiness_tribes"] == 0

    handoff = write_stage7_final_handoff_pack(
        stage68["tribe_evidence_path"],
        readiness_path=readiness_path,
        stage68_manifest_path=stage68["manifest_json"],
        output_dir=tmp_path / "final_handoff",
        write_cards=False,
        cfg=cfg,
    )

    assert handoff["raw_transaction_export_paths"]["manifest_csv"] == stage68["transaction_export_manifest_csv"]
    assert handoff["customer_summary_export_paths"]["manifest_csv"] == stage68["customer_export_manifest_csv"]
    assert pl.read_csv(handoff["customer_metric_tests_csv"]).height > 0
    manifest = handoff["manifest_json"].read_text(encoding="utf-8")
    assert "stage68_fingerprint" in manifest
    assert "never reopens global transactions" in manifest


def test_stage7_final_handoff_pack_creates_curated_profile_first_outputs(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"
    readiness_path = tmp_path / "readiness.csv"

    strong_profile = {
        "tribe_id": 0,
        "n_customers": 100,
        "population_share": 0.25,
        "profile_population_customers": 180,
        "profile_noise_customers": 20,
        "core_customers": 100,
        "soft_assigned_customers": 0,
        "soft_assigned_share": 0.0,
        "stage6_profile_readiness": "strong",
        "suggested_tribe_name": "Greek Yogurt Evidence Cluster",
        "suggested_tribe_name_source": "lifted_product_terms",
        "suggested_tribe_name_evidence": "greek yogurt",
        "suggested_tribe_name_status": "unique_working_name",
        "top_product_ids": ["sku_yogurt", "sku_honey"],
        "top_products": ["Greek Yogurt", "Honey"],
        "top_product_sectors": ["Dairy", "Grocery"],
        "top_product_sector_ids": ["1", "2"],
        "top_product_lifts": [2.0, 1.7],
        "top_product_lifts_vs_rest": [2.4, 1.9],
        "top_product_q_values": [0.001, 0.02],
        "top_product_customer_counts": [40, 25],
        "top_product_reach_pct": [8.0, 30.0],
        "top_sectors": ["Dairy"],
        "top_sector_lifts": [1.4],
        "top_sector_lifts_vs_rest": [1.5],
        "top_sector_q_values": [0.01],
        "top_themes": ["dairy_eggs"],
        "top_theme_lifts": [1.5],
        "top_theme_lifts_vs_rest": [1.6],
        "top_theme_q_values": [0.02],
        "top_theme_customer_counts": [120],
        "top_theme_product_evidence": ["Greek Yogurt (2.4x vs rest; 40.0% reach; q=0.001; n=40); Honey (1.9x vs rest; 25.0% reach; q=0.020; n=25)"],
        "top_theme_tagged_product_counts": [2],
        "top_product_terms": ["greek yogurt"],
        "top_product_term_lifts": [1.8],
        "top_product_term_lifts_vs_rest": [2.0],
        "top_product_term_q_values": [0.01],
        "top_product_term_customer_counts": [35],
        "avg_ticket_count": 5.0,
        "avg_frequency_per_30d": 2.0,
        "avg_avg_basket_value": 18.5,
        "avg_total_spend": 120.0,
        "avg_promo_share": 0.20,
    }
    review_profile = {
        **strong_profile,
        "tribe_id": 1,
        "n_customers": 80,
        "stage6_profile_readiness": "review",
        "suggested_tribe_name": "Honey Review Evidence Cluster",
        "suggested_tribe_name_evidence": "honey",
        "top_product_ids": ["sku_honey"],
        "top_products": ["Honey"],
        "top_product_sectors": ["Grocery"],
        "top_product_sector_ids": ["2"],
        "top_product_lifts": [2.2],
        "top_product_lifts_vs_rest": [2.5],
        "top_product_q_values": [0.001],
        "top_product_customer_counts": [30],
    }
    pl.DataFrame([strong_profile, review_profile]).write_parquet(profile_path)
    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "stage6_profile_readiness": "strong",
                "profiling_readiness": "ready_strong",
                "profiling_readiness_issues": "pass",
                "profile_significant_interpretability_signals": 3,
            },
            {
                "tribe_id": 1,
                "stage6_profile_readiness": "review",
                "profiling_readiness": "review",
                "profiling_readiness_issues": "stage6_readiness=review",
                "profile_significant_interpretability_signals": 3,
            }
        ]
    ).write_csv(readiness_path)
    stage68_dir = tmp_path / "stage68"
    transaction_dir = stage68_dir / "tribe_transactions"
    customer_dir = stage68_dir / "tribe_customers"
    transaction_dir.mkdir(parents=True)
    customer_dir.mkdir(parents=True)
    tribe0_txn = transaction_dir / "tribe_00_transactions_dev.parquet"
    tribe1_txn = transaction_dir / "tribe_01_transactions_dev.parquet"
    tribe0_customers = customer_dir / "tribe_00_customers_dev.parquet"
    tribe1_customers = customer_dir / "tribe_01_customers_dev.parquet"
    pl.DataFrame(
        [
            {"cliente": "c1", "idarticu": "sku_yogurt", "desc_larga_articulo": "Greek Yogurt", "desc_sector": "Dairy", "importe": 6.5},
            {"cliente": "c1", "idarticu": "sku_honey", "desc_larga_articulo": "Honey", "desc_sector": "Grocery", "importe": 4.0},
            {"cliente": "c2", "idarticu": "sku_honey", "desc_larga_articulo": "Honey", "desc_sector": "Grocery", "importe": 5.0},
        ]
    ).write_parquet(tribe0_txn)
    pl.DataFrame(
        [{"cliente": "c3", "idarticu": "sku_honey", "desc_larga_articulo": "Honey", "desc_sector": "Grocery", "importe": 4.2}]
    ).write_parquet(tribe1_txn)
    pl.DataFrame(
        [
            {"cliente": "c1", "total_spend": 80.0, "recency_days": 20, "frequency_per_30d": 2.0, "avg_basket_value": 20.0, "promo_share": 0.10},
            {"cliente": "c2", "total_spend": 120.0, "recency_days": 110, "frequency_per_30d": 3.0, "avg_basket_value": 24.0, "promo_share": 0.20},
        ]
    ).write_parquet(tribe0_customers)
    pl.DataFrame(
        [{"cliente": "c3", "total_spend": 30.0, "recency_days": 45, "frequency_per_30d": 1.0, "avg_basket_value": 15.0, "promo_share": 0.05}]
    ).write_parquet(tribe1_customers)
    transaction_manifest = transaction_dir / "tribe_transactions_manifest_dev.csv"
    customer_manifest = customer_dir / "tribe_customers_manifest_dev.csv"
    pl.DataFrame(
        [
            {"tribe_id": 0, "path": str(tribe0_txn), "rows": 3, "customers": 2},
            {"tribe_id": 1, "path": str(tribe1_txn), "rows": 1, "customers": 1},
        ]
    ).write_csv(transaction_manifest)
    pl.DataFrame(
        [
            {"tribe_id": 0, "path": str(tribe0_customers), "rows": 2, "customers": 2},
            {"tribe_id": 1, "path": str(tribe1_customers), "rows": 1, "customers": 1},
        ]
    ).write_csv(customer_manifest)
    customer_metric_tests = stage68_dir / "customer_metric_tests_dev.csv"
    noise_vs_core_metrics = stage68_dir / "noise_vs_core_customer_metrics_dev.csv"
    _empty_customer_metric_tests().write_csv(customer_metric_tests)
    _empty_noise_vs_core_metric_tests().write_csv(noise_vs_core_metrics)
    manifest_path = stage68_dir / "stage68_manifest_dev.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "6.8_tribe_evidence_assembly",
                "mode": "dev",
                "cache_fingerprint": {"stage68_schema_version": 1},
                "outputs": {
                    "tribe_evidence_path": str(profile_path),
                    "product_lifts_path": str(stage68_dir / "product_lifts_dev.parquet"),
                    "sector_lifts_path": str(stage68_dir / "sector_lifts_dev.parquet"),
                    "customer_metric_tests_csv": str(customer_metric_tests),
                    "noise_vs_core_customer_metrics_csv": str(noise_vs_core_metrics),
                    "transaction_export_dir": str(transaction_dir),
                    "transaction_export_manifest_csv": str(transaction_manifest),
                    "customer_export_dir": str(customer_dir),
                    "customer_export_manifest_csv": str(customer_manifest),
                },
                "transaction_exports": pl.read_csv(transaction_manifest).to_dicts(),
                "customer_exports": pl.read_csv(customer_manifest).to_dicts(),
            }
        ),
        encoding="utf-8",
    )

    stale_card = tmp_path / "final_handoff" / "tribe_cards" / "tribe_99_card.png"
    stale_card.parent.mkdir(parents=True, exist_ok=True)
    stale_card.write_bytes(b"stale")
    outputs = write_stage7_final_handoff_pack(
        profile_path,
        stage68_manifest_path=manifest_path,
        output_dir=tmp_path / "final_handoff",
        write_cards=True,
        cfg=cfg,
    )
    index = pl.read_csv(outputs["index_csv"])
    row = index.row(0, named=True)
    final_index_row = pl.read_csv(outputs["index_csv"]).row(0, named=True)
    review_candidates = outputs["review_candidates"]
    story = outputs["story_markdown"].read_text(encoding="utf-8")
    manifest = outputs["manifest_json"].read_text(encoding="utf-8")

    assert index.height == 1
    assert row["tribe_name"] == "Dairy & Eggs Buyers"
    assert row["name_source"] == "editorial_theme"
    assert row["primary_theme"] == "Dairy & Eggs Buyers"
    assert "Dairy & Eggs Buyers" in row["theme_read"]
    assert row["population_share_pct"] == 100.0
    assert row["population_share_basis"] == "promoted_final_tribes"
    assert row["promoted_population_customers"] == 100
    assert row["assigned_population_share_pct"] == 55.56
    assert row["assigned_population_customers"] == 180
    assert row["review_excluded_customers"] == 80
    assert "tribe_noise_customers" not in index.columns
    assert "unassigned_noise_customers_global" not in index.columns
    assert "noise_customers" not in index.columns
    assert "loyalty_cohort" not in index.columns
    assert "loyalty_context" in index.columns
    assert "Greek Yogurt" in row["primary_theme_product_evidence"]
    assert "lift_vs_rest x log(customer_count + 1)" in row["product_ranking_basis"]
    assert row["actionability_proof_source"] == "data_driven_product_terms"
    assert row["top_product"] == "Greek Yogurt"
    assert row["top_reach_product"] == "Honey"
    assert row["top_reach_product_customers"] == 2
    assert row["spend_p25_eur"] == 80.0
    assert row["spend_p50_eur"] == 120.0
    assert row["active_customer_pct"] == 50.0
    assert row["lapsed_customer_pct"] == 50.0
    assert "Reach caveat" in row["top_product_reach_warning"]
    assert "total spend" in row["spend_and_visit_context"]
    assert sorted(outputs["card_paths"]) == [0, 1]
    assert outputs["card_paths"][0].exists()
    assert outputs["card_paths"][1].exists()
    assert not stale_card.exists()
    assert final_index_row["card_png"].endswith("tribe_00_card.png")
    assert final_index_row["primary_theme"] == "Dairy & Eggs Buyers"
    assert final_index_row["actionability_proof_source"] == "data_driven_product_terms"
    assert final_index_row["population_share_pct"] == 100.0
    assert final_index_row["assigned_population_share_pct"] == 55.56
    assert "tribe_noise_customers" not in final_index_row
    assert "unassigned_noise_customers_global" not in final_index_row
    assert final_index_row["top_reach_product"] == "Honey"
    assert "Reach caveat" in final_index_row["top_product_reach_warning"]
    assert "lift_vs_rest x log(customer_count + 1)" in final_index_row["product_ranking_basis"]
    assert review_candidates["tribe_id"].to_list() == [1]
    assert review_candidates[0, "tribe_status"] == "potential_review"
    assert review_candidates[0, "recommended_use"].startswith("Promising behavioral pocket")
    assert outputs["index_csv"].exists()
    assert outputs["llm_evidence_csv"].exists()
    assert outputs["noise_vs_core_customer_metrics_csv"].exists()
    llm_evidence = pl.read_csv(outputs["llm_evidence_csv"])
    assert "primary_actionability_proof" in llm_evidence["proof_role"].to_list()
    assert "supplemental_curated_theme_context" in llm_evidence["proof_role"].to_list()
    assert outputs["comparison_paths"]["html"].exists()
    assert outputs["substage_paths"]["promoted_tribe_validation"]["csv"].exists()
    assert outputs["substage_paths"]["review_tribe_validation"]["csv"].exists()
    assert outputs["substage_paths"]["all_tribe_product_identity"]["csv"].exists()
    assert outputs["substage_paths"]["all_tribe_behavior_differentiation"]["csv"].exists()
    assert outputs["substage_paths"]["all_tribe_dossiers"]["markdown"].exists()
    assert outputs["substage_paths"]["all_tribe_relationship_atlas"]["csv"].exists()
    redesigned_keys = [
        "stage7_1_promotion_report",
        "stage7_2_identity_dossier",
        "stage7_3_tribe_handbook",
        "stage7_4_customer_coverage_report",
        "stage7_5_segment_action_playbook",
        "stage7_6_segmentation_framework",
        "stage7_7_final_report",
    ]
    for key in redesigned_keys:
        assert outputs["substage_paths"][key]["csv"].exists()
        assert outputs["substage_paths"][key]["markdown"].exists()
    promotion_report = pl.read_csv(outputs["substage_paths"]["stage7_1_promotion_report"]["csv"])
    assert {
        "promotion_decision",
        "validation_tier",
        "technical_name",
        "business_name",
        "legacy_tribe_name",
        "business_confidence",
        "validation_blockers",
        "coverage_group",
        "membership_policy",
        "recommended_use",
    }.issubset(set(promotion_report.columns))
    assert promotion_report.filter(pl.col("promotion_decision") == "promoted").height == 1
    assert promotion_report.filter(pl.col("promotion_decision") == "review").height == 1
    identity_dossier = pl.read_csv(outputs["substage_paths"]["stage7_2_identity_dossier"]["csv"])
    assert identity_dossier[0, "technical_name"] == "Greek Yogurt Buyers"
    assert identity_dossier[0, "business_name"] is None
    handbook = pl.read_csv(outputs["substage_paths"]["stage7_3_tribe_handbook"]["csv"])
    assert handbook[0, "business_name"] == "Dairy & Eggs Buyers"
    assert handbook[0, "technical_name"] == "Greek Yogurt Buyers"
    coverage = pl.read_csv(outputs["substage_paths"]["stage7_4_customer_coverage_report"]["csv"])
    additive = coverage.filter(pl.col("counts_toward_population_total") == True)
    assert int(additive["customers"].sum()) == 200
    action_playbook = pl.read_csv(outputs["substage_paths"]["stage7_5_segment_action_playbook"]["csv"])
    assert "marketing_actions" in action_playbook.columns
    final_report = outputs["substage_paths"]["stage7_7_final_report"]["markdown"].read_text(encoding="utf-8")
    assert "Who are the core customer segments?" in final_report
    assert outputs["campaign_playbook_paths"]["csv"].exists()
    assert outputs["stakeholder_readiness_paths"]["csv"].exists()
    assert outputs["soft_audience_activation_customers_csv"].exists()
    assert outputs["soft_audience_activation_customers_parquet"].exists()
    assert outputs["raw_transaction_export_paths"]["manifest_csv"].exists()
    assert outputs["customer_summary_export_paths"]["manifest_csv"].exists()
    raw_manifest = pl.read_csv(outputs["raw_transaction_export_paths"]["manifest_csv"])
    customer_manifest = pl.read_csv(outputs["customer_summary_export_paths"]["manifest_csv"])
    raw_tribe_0 = pl.read_parquet(outputs["raw_transaction_export_paths"]["tribe_00_parquet"])
    customer_tribe_0 = pl.read_parquet(outputs["customer_summary_export_paths"]["tribe_00_parquet"])
    assert raw_manifest["tribe_id"].to_list() == [0, 1]
    assert customer_manifest["tribe_id"].to_list() == [0, 1]
    assert {"cliente", "desc_larga_articulo", "desc_sector", "importe"}.issubset(set(raw_tribe_0.columns))
    assert set(raw_tribe_0["cliente"].to_list()) == {"c1", "c2"}
    assert {"cliente", "total_spend", "recency_days"}.issubset(set(customer_tribe_0.columns))
    assert customer_tribe_0.height == 2
    assert "Stage 7 Tribe Profiles" in story
    assert "Potential review tribes" in story
    assert "\"promoted_tribes\": [\n    0\n  ]" in manifest
    assert "\"review_tribes\"" in manifest
    assert "all_tribe_product_identity_csv" in manifest
    assert "all_tribe_behavior_differentiation_csv" in manifest
    assert "all_tribe_relationship_atlas_csv" in manifest
    assert "\"llm_enabled\": false" in manifest
    assert "tribe_raw_transaction_export_directory" in manifest
    assert "campaign_playbook_csv" in manifest
    assert "stakeholder_readiness_csv" in manifest
    assert "stage7_1_promotion_report_csv" in manifest
    assert "stage7_7_final_report_markdown" in manifest
    assert "soft_audience_activation_customers_csv" in manifest


def test_stage7_names_from_single_tribe_transaction_theme_evidence(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.values["profiling"].update(
        {
            "theme_label_min_lift": 1.5,
            "theme_label_min_coverage": 0.5,
            "theme_label_min_customers": 1,
            "theme_label_min_tagged_products": 2,
            "theme_label_q_threshold": 0.05,
            "theme_label_require_significant": True,
            "theme_product_example_count": 3,
        }
    )
    profile_path = tmp_path / "profiles.parquet"
    stage68_dir = tmp_path / "stage68"
    transaction_dir = stage68_dir / "tribe_transactions"
    customer_dir = stage68_dir / "tribe_customers"
    transaction_dir.mkdir(parents=True)
    customer_dir.mkdir(parents=True)
    tribe_txn = transaction_dir / "tribe_00_transactions_dev.parquet"
    tribe_customers = customer_dir / "tribe_00_customers_dev.parquet"

    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 2,
                "population_share": 1.0,
                "profile_population_customers": 2,
                "profile_noise_customers": 0,
                "core_customers": 2,
                "soft_assigned_customers": 0,
                "soft_assigned_share": 0.0,
                "stage6_profile_readiness": "strong",
                "top_product_ids": ["p_tarrito", "p_yogolino"],
                "top_products": [
                    "TARRITO HERO RECETAS CASERAS COCIDO TERNERA 2 X 190 GR",
                    "YOGOLINO MELOCOTON PLATANO S/AZUCAR ANADIDO 4X100G",
                ],
                "top_product_sectors": ["Baby", "Baby"],
                "top_product_lifts": [2.6, 2.1],
                "top_product_lifts_vs_rest": [3.0, 2.4],
                "top_product_q_values": [0.001, 0.002],
                "top_product_customer_counts": [1, 1],
                "top_product_reach_pct": [50.0, 50.0],
                "top_sectors": ["Baby"],
                "top_sector_lifts": [2.0],
                "avg_ticket_count": 2.0,
                "avg_frequency_per_30d": 1.0,
                "avg_avg_basket_value": 8.0,
                "avg_total_spend": 16.0,
                "avg_promo_share": 0.0,
            }
        ]
    ).write_parquet(profile_path)
    pl.DataFrame(
        [
            {
                "cliente": "c1",
                "idarticu": "p_tarrito",
                "desc_larga_articulo": "TARRITO HERO RECETAS CASERAS COCIDO TERNERA 2 X 190 GR",
                "desc_sector": "Baby",
                "importe": 3.2,
            },
            {
                "cliente": "c2",
                "idarticu": "p_yogolino",
                "desc_larga_articulo": "YOGOLINO MELOCOTON PLATANO S/AZUCAR ANADIDO 4X100G",
                "desc_sector": "Baby",
                "importe": 4.1,
            },
        ]
    ).write_parquet(tribe_txn)
    pl.DataFrame(
        [
            {"cliente": "c1", "total_spend": 12.0, "recency_days": 10},
            {"cliente": "c2", "total_spend": 20.0, "recency_days": 20},
        ]
    ).write_parquet(tribe_customers)
    transaction_manifest = transaction_dir / "tribe_transactions_manifest_dev.csv"
    customer_manifest = customer_dir / "tribe_customers_manifest_dev.csv"
    pl.DataFrame([{"tribe_id": 0, "path": str(tribe_txn), "rows": 2, "customers": 2}]).write_csv(transaction_manifest)
    pl.DataFrame([{"tribe_id": 0, "path": str(tribe_customers), "rows": 2, "customers": 2}]).write_csv(customer_manifest)
    customer_metric_tests = stage68_dir / "customer_metric_tests_dev.csv"
    noise_vs_core_metrics = stage68_dir / "noise_vs_core_customer_metrics_dev.csv"
    _empty_customer_metric_tests().write_csv(customer_metric_tests)
    _empty_noise_vs_core_metric_tests().write_csv(noise_vs_core_metrics)
    manifest_path = stage68_dir / "stage68_manifest_dev.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "6.8_tribe_evidence_assembly",
                "mode": "dev",
                "cache_fingerprint": {"stage68_schema_version": 2},
                "outputs": {
                    "tribe_evidence_path": str(profile_path),
                    "product_lifts_path": str(stage68_dir / "product_lifts_dev.parquet"),
                    "sector_lifts_path": str(stage68_dir / "sector_lifts_dev.parquet"),
                    "customer_metric_tests_csv": str(customer_metric_tests),
                    "noise_vs_core_customer_metrics_csv": str(noise_vs_core_metrics),
                    "transaction_export_dir": str(transaction_dir),
                    "transaction_export_manifest_csv": str(transaction_manifest),
                    "customer_export_dir": str(customer_dir),
                    "customer_export_manifest_csv": str(customer_manifest),
                },
                "transaction_exports": pl.read_csv(transaction_manifest).to_dicts(),
                "customer_exports": pl.read_csv(customer_manifest).to_dicts(),
            }
        ),
        encoding="utf-8",
    )

    outputs = write_stage7_final_handoff_pack(
        profile_path,
        stage68_manifest_path=manifest_path,
        output_dir=tmp_path / "final_handoff",
        write_cards=False,
        cfg=cfg,
    )
    row = pl.read_csv(outputs["index_csv"]).row(0, named=True)

    assert row["name_source"] == "editorial_theme"
    assert row["tribe_name"] == "Baby Food Buyers"
    assert row["primary_theme_confidence"] == "stage7_transaction_theme_supported"
    assert "2 lifted products support the label" in row["theme_read"]
    assert "YOGOLINO" in row["primary_theme_product_evidence"]


def test_stage7_final_handoff_with_no_promoted_tribes_writes_readable_empty_index(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"
    stage68_dir = tmp_path / "stage68"
    stage68_dir.mkdir()

    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 80,
                "population_share": 0.20,
                "profile_population_customers": 80,
                "profile_noise_customers": 5,
                "core_customers": 80,
                "soft_assigned_customers": 0,
                "soft_assigned_share": 0.0,
                "stage6_profile_readiness": "review",
                "suggested_tribe_name": "Olive Oil Review Cluster",
                "suggested_tribe_name_source": "lifted_product_terms",
                "suggested_tribe_name_evidence": "olive oil",
                "suggested_tribe_name_status": "unique_working_name",
                "top_product_ids": ["sku_oil"],
                "top_products": ["Olive Oil"],
                "top_product_sectors": ["Grocery"],
                "top_product_sector_ids": ["1"],
                "top_product_lifts": [1.9],
                "top_product_lifts_vs_rest": [2.2],
                "top_product_q_values": [0.001],
                "top_product_customer_counts": [30],
                "top_product_reach_pct": [37.5],
                "top_sectors": ["Grocery"],
                "top_sector_lifts": [1.2],
                "top_sector_lifts_vs_rest": [1.3],
                "top_sector_q_values": [0.02],
                "top_sector_line_counts": [45],
                "top_themes": [],
                "top_theme_lifts": [],
                "top_theme_lifts_vs_rest": [],
                "top_theme_q_values": [],
                "top_theme_customer_counts": [],
                "top_theme_product_evidence": [],
                "top_theme_tagged_product_counts": [],
                "top_product_terms": ["olive oil"],
                "top_product_term_lifts": [1.8],
                "top_product_term_lifts_vs_rest": [2.0],
                "top_product_term_q_values": [0.01],
                "top_product_term_customer_counts": [32],
                "avg_ticket_count": 4.0,
                "avg_frequency_per_30d": 1.4,
                "avg_avg_basket_value": 21.0,
                "avg_total_spend": 100.0,
                "avg_promo_share": 0.14,
            }
        ]
    ).write_parquet(profile_path)

    customer_metric_tests = stage68_dir / "customer_metric_tests_dev.csv"
    noise_vs_core_metrics = stage68_dir / "noise_vs_core_customer_metrics_dev.csv"
    _empty_customer_metric_tests().write_csv(customer_metric_tests)
    _empty_noise_vs_core_metric_tests().write_csv(noise_vs_core_metrics)
    manifest_path = stage68_dir / "stage68_manifest_dev.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "6.8_tribe_evidence_assembly",
                "mode": "dev",
                "cache_fingerprint": {"stage68_schema_version": 1},
                "outputs": {
                    "tribe_evidence_path": str(profile_path),
                    "product_lifts_path": str(stage68_dir / "product_lifts_dev.parquet"),
                    "sector_lifts_path": str(stage68_dir / "sector_lifts_dev.parquet"),
                    "customer_metric_tests_csv": str(customer_metric_tests),
                    "noise_vs_core_customer_metrics_csv": str(noise_vs_core_metrics),
                },
            }
        ),
        encoding="utf-8",
    )

    outputs = write_stage7_final_handoff_pack(
        profile_path,
        stage68_manifest_path=manifest_path,
        output_dir=tmp_path / "final_handoff",
        write_cards=True,
        cfg=cfg,
    )

    index = pl.read_csv(outputs["index_csv"])
    llm_evidence = pl.read_csv(outputs["llm_evidence_csv"])
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))

    assert outputs["final_index"].height == 0
    assert index.height == 0
    assert {"tribe_id", "tribe_name", "population_share_pct"}.issubset(set(index.columns))
    assert llm_evidence.height == 0
    assert {"tribe_id", "proof_role", "evidence"}.issubset(set(llm_evidence.columns))
    assert sorted(outputs["card_paths"]) == [0]
    assert outputs["card_paths"][0].exists()
    assert outputs["review_candidates"]["tribe_id"].to_list() == [0]
    assert outputs["review_candidates"][0, "tribe_status"] == "potential_review"
    assert manifest["promoted_tribes"] == []
    assert manifest["review_tribes"] == {"0": "stage6_profile_readiness=review"}


def test_stage7_final_handoff_allows_product_term_proof_without_curated_theme(tmp_path):
    cfg = _test_config(tmp_path)
    profile_path = tmp_path / "profiles.parquet"
    readiness_path = tmp_path / "stage7_profile_readiness.csv"

    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 200,
                "population_share": 0.05,
                "profile_population_customers": 4000,
                "profile_noise_customers": 50,
                "core_customers": 200,
                "soft_assigned_customers": 0,
                "soft_assigned_share": 0.0,
                "stage6_profile_readiness": "strong",
                "suggested_tribe_name": "Hummus Purchase Cluster",
                "suggested_tribe_name_source": "data_driven_product_terms",
                "suggested_tribe_name_evidence": "hummus (lift=2.30, coverage=35.0%, q=0.001)",
                "suggested_tribe_name_status": "unique_working_name",
                "top_product_ids": ["sku_hummus", "sku_falafel"],
                "top_products": ["HUMMUS CLASICO", "FALAFEL VEGETAL"],
                "top_product_sectors": ["P.G.C.", "P.G.C."],
                "top_product_sector_ids": ["1", "1"],
                "top_product_lifts": [2.0, 1.8],
                "top_product_lifts_vs_rest": [2.5, 2.1],
                "top_product_q_values": [0.001, 0.004],
                "top_product_customer_counts": [60, 45],
                "top_sectors": ["P.G.C."],
                "top_sector_lifts": [1.1],
                "top_sector_lifts_vs_rest": [1.1],
                "top_sector_q_values": [0.08],
                "top_themes": [],
                "top_theme_lifts": [],
                "top_theme_lifts_vs_rest": [],
                "top_theme_q_values": [],
                "top_theme_customer_counts": [],
                "top_theme_product_evidence": [],
                "top_theme_tagged_product_counts": [],
                "top_product_terms": ["hummus"],
                "top_product_term_lifts": [2.1],
                "top_product_term_lifts_vs_rest": [2.3],
                "top_product_term_q_values": [0.001],
                "top_product_term_customer_counts": [70],
                "avg_ticket_count": 4.0,
                "avg_frequency_per_30d": 1.7,
                "avg_avg_basket_value": 22.0,
                "avg_total_spend": 130.0,
                "avg_promo_share": 0.18,
            }
        ]
    ).write_parquet(profile_path)
    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "stage6_profile_readiness": "strong",
                "profiling_readiness": "ready_strong",
                "profiling_readiness_issues": "pass",
                "profile_significant_interpretability_signals": 2,
            }
        ]
    ).write_csv(readiness_path)

    index = stage7_final_index_table(profile_path, readiness_path=readiness_path, cfg=cfg)
    row = index.row(0, named=True)

    assert index.height == 1
    assert row["tribe_name"] == "Hummus Clasico Buyers"
    assert row["name_source"] == "sku_fallback"
    assert row["primary_theme_confidence"] == "product_led"
    assert row["primary_theme"] == "Product-led tribe; no broad theme evidence"
    assert row["actionability_proof_source"] == "data_driven_product_terms"
    assert "Product-term proof" in row["actionability_proof"]


def test_customer_metric_anova_table_reports_between_tribe_differences(tmp_path):
    cfg = _test_config(tmp_path)
    assignments_path = tmp_path / "assignments.parquet"
    behavior_path = tmp_path / "behavior.parquet"
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "tribe_id": [0, 0, 0, 1, 1, 1],
        }
    ).write_parquet(assignments_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "ticket_count": [9.0, 10.0, 11.0, 1.0, 2.0, 3.0],
            "promo_share": [0.1, 0.1, 0.2, 0.4, 0.5, 0.5],
        }
    ).write_parquet(behavior_path)

    tests = customer_metric_anova_table(assignments_path, behavior_path, output_csv=tmp_path / "anova.csv", cfg=cfg)
    row = tests.filter(pl.col("metric") == "ticket_count").row(0, named=True)

    assert row["included_tribes"] == 2
    assert row["highest_mean_tribe_id"] == 0
    assert row["lowest_mean_tribe_id"] == 1
    assert row["anova_f_statistic"] > 0
    assert row["anova_p_value"] < 0.01
    assert row["anova_q_value"] < 0.05
    assert row["anova_effect_eta_squared"] > 0.9
    assert row["anova_effect_interpretation"] == "large_effect"
    assert row["statistical_result"] == "large_effect_significant"
    assert (tmp_path / "anova.csv").exists()


def test_noise_vs_core_customer_metric_table_profiles_noise_against_assigned_core(tmp_path):
    cfg = _test_config(tmp_path)
    assignments_path = tmp_path / "assignments.parquet"
    behavior_path = tmp_path / "behavior.parquet"
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4"],
            "tribe_id": [0, 1, -1, -1],
        }
    ).write_parquet(assignments_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4"],
            "ticket_count": [10.0, 8.0, 2.0, 1.0],
            "total_spend": [100.0, 80.0, 12.0, 8.0],
            "promo_share": [0.1, 0.2, 0.5, 0.4],
        }
    ).write_parquet(behavior_path)

    table = noise_vs_core_customer_metric_table(
        assignments_path,
        behavior_path,
        output_csv=tmp_path / "noise_vs_core.csv",
        cfg=cfg,
    )
    ticket = table.filter(pl.col("metric") == "ticket_count").row(0, named=True)

    assert ticket["core_customers"] == 2
    assert ticket["noise_customers"] == 2
    assert ticket["core_mean"] == 9.0
    assert ticket["noise_mean"] == 1.5
    assert ticket["noise_vs_core_ratio"] < 0.2
    assert "noise lower than core" in ticket["interpretation"]
    assert (tmp_path / "noise_vs_core.csv").exists()


def test_noise_audit_profiles_hidden_noise_structure_without_assignment(tmp_path):
    cfg = _test_config(tmp_path)
    assignments_path = tmp_path / "assignments.parquet"
    behavior_path = tmp_path / "behavior.parquet"
    pl.DataFrame(
        {
            "cliente": ["core1", "core2", "core3", "noise1", "noise2", "noise3"],
            "tribe_id": [0, 0, 1, -1, -1, -1],
        }
    ).write_parquet(assignments_path)
    pl.DataFrame(
        {
            "cliente": ["core1", "core2", "core3", "noise1", "noise2", "noise3"],
            "ticket_count": [8.0, 7.0, 6.0, 1.0, 1.0, 2.0],
            "unique_products": [5.0, 4.0, 5.0, 9.0, 10.0, 8.0],
        }
    ).write_parquet(behavior_path)
    transactions = pl.DataFrame(
        {
            "cliente": [
                "core1",
                "core2",
                "core3",
                "core1",
                "noise1",
                "noise2",
                "noise3",
                "noise1",
            ],
            "idarticu": [
                "bread",
                "bread",
                "milk",
                "rice",
                "rare_kefir",
                "rare_kefir",
                "rare_kefir",
                "bread",
            ],
            "desc_larga_articulo": [
                "Pan integral",
                "Pan integral",
                "Leche entera",
                "Arroz largo",
                "Kefir cabra natural",
                "Kefir cabra natural",
                "Kefir cabra natural",
                "Pan integral",
            ],
            "desc_sector": [
                "Bakery",
                "Bakery",
                "Dairy",
                "Grocery",
                "Dairy",
                "Dairy",
                "Dairy",
                "Bakery",
            ],
        }
    ).lazy()

    audit = noise_audit_table(assignments_path, transactions=transactions, behavior_path=behavior_path, cfg=cfg)
    coverage_row = audit.filter(pl.col("evidence_type") == "noise_share").row(0, named=True)

    assert coverage_row["noise_observations"] == 3
    assert coverage_row["recommended_action"] == "keep_unassigned_for_core_profile"
    assert "reported separately" in coverage_row["interpretation"]

    outputs = write_noise_audit_artifacts(
        assignments_path,
        transactions=transactions,
        behavior_path=behavior_path,
        output_csv=tmp_path / "noise_audit.csv",
        output_md=tmp_path / "noise_audit.md",
        output_html=tmp_path / "noise_audit.html",
        cfg=cfg,
    )
    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
    assert "noise_share" in outputs["markdown"].read_text(encoding="utf-8")


def test_remaining_customer_segments_and_soft_audiences_do_not_mutate_hard_assignments(tmp_path):
    cfg = PipelineConfig(
        values={
            "paths": {"outputs": "outputs"},
            "profiling": {
                "soft_audience_min_affinity": 0.20,
                "soft_audience_min_margin": 0.0,
                "remaining_near_tribe_min_affinity": 0.20,
                "remaining_near_tribe_min_margin": 0.0,
                "remaining_affinity_max_features": 4,
            },
        },
        mode="dev",
        root=tmp_path,
    )
    assignments_path = tmp_path / "assignments.parquet"
    behavior_path = tmp_path / "behavior.parquet"
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "n1", "n2", "n3"],
            "tribe_id": [0, 0, 1, 1, -1, -1, -1],
            "assignment_confidence_score": [0.9, 0.8, 0.9, 0.8, None, None, None],
        }
    ).write_parquet(assignments_path)
    before = pl.read_parquet(assignments_path).sort("cliente")
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3", "c4", "n1", "n2", "n3"],
            "total_spend": [100.0, 95.0, 30.0, 35.0, 98.0, 220.0, 5.0],
            "ticket_count": [8.0, 7.0, 3.0, 4.0, 7.0, 9.0, 1.0],
            "unique_products": [20.0, 18.0, 8.0, 9.0, 19.0, 35.0, 1.0],
            "unique_sectors": [6.0, 5.0, 3.0, 3.0, 6.0, 8.0, 1.0],
            "avg_basket_value": [12.5, 13.0, 10.0, 9.0, 12.0, 24.0, 5.0],
            "promo_share": [0.10, 0.12, 0.20, 0.22, 0.11, 0.15, 0.00],
            "recency_days": [10.0, 12.0, 30.0, 28.0, 11.0, 8.0, 90.0],
            "frequency_per_30d": [2.0, 1.8, 1.0, 1.1, 1.9, 2.5, 0.2],
        }
    ).write_parquet(behavior_path)

    affinity = remaining_customer_affinity_table(assignments_path, behavior_path, cfg=cfg)
    segments = remaining_customer_segments_table(assignments_path, behavior_path, cfg=cfg)
    soft = stage7_soft_audience_opportunities_table(assignments_path, behavior_path, cfg=cfg)
    after = pl.read_parquet(assignments_path).sort("cliente")

    assert affinity.height == 3
    assert set(segments["segment_id"].to_list())
    assert {"segment_name", "targetability", "likely_reason_for_no_hard_cluster"}.issubset(set(segments.columns))
    assert not soft.is_empty()
    assert set(soft["audience_label"].to_list()) == {"soft audience opportunity"}
    assert before.equals(after)


def test_stage7_all_profiles_include_review_tribes_and_relationship_interpretation(tmp_path):
    cfg = PipelineConfig(
        values={
            "paths": {"outputs": "outputs"},
            "profiling": {
                "final_handoff_readiness_statuses": ["strong", "usable"],
                "strong_product_lift_threshold": 1.5,
                "significance_q_threshold": 0.05,
                "tribe_name_lookup": {},
            },
        },
        mode="dev",
        root=tmp_path,
    )
    profile_path = tmp_path / "profiles.parquet"
    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 120,
                "population_share": 0.12,
                "stage6_profile_readiness": "strong",
                "top_product_ids": ["p1", "p2"],
                "top_products": ["BIO MILK", "BIO YOGURT"],
                "top_product_lifts": [2.0, 1.8],
                "top_product_lifts_vs_rest": [2.1, 1.9],
                "top_product_q_values": [0.01, 0.02],
                "top_product_customer_counts": [60, 50],
                "top_product_reach_pct": [50.0, 41.7],
                "top_sectors": ["P.G.C."],
                "top_sector_lifts": [1.3],
                "top_sector_line_counts": [200],
                "top_themes": [],
                "behavior_ratio_vs_rest": json.dumps({"avg_total_spend": 1.2, "avg_frequency_per_30d": 1.1}),
                "avg_total_spend": 140.0,
                "avg_frequency_per_30d": 2.0,
                "avg_basket_value": 22.0,
                "avg_promo_share": 0.11,
            },
            {
                "tribe_id": 1,
                "n_customers": 90,
                "population_share": 0.09,
                "stage6_profile_readiness": "usable",
                "top_product_ids": ["p1", "p3"],
                "top_products": ["BIO MILK", "CEREAL"],
                "top_product_lifts": [1.9, 1.7],
                "top_product_lifts_vs_rest": [2.0, 1.8],
                "top_product_q_values": [0.01, 0.03],
                "top_product_customer_counts": [40, 35],
                "top_product_reach_pct": [44.4, 38.9],
                "top_sectors": ["P.G.C."],
                "top_sector_lifts": [1.2],
                "top_sector_line_counts": [150],
                "top_themes": [],
                "behavior_ratio_vs_rest": json.dumps({"avg_total_spend": 1.15, "avg_frequency_per_30d": 1.05}),
                "avg_total_spend": 120.0,
                "avg_frequency_per_30d": 1.8,
                "avg_basket_value": 21.0,
                "avg_promo_share": 0.12,
            },
            {
                "tribe_id": 2,
                "n_customers": 70,
                "population_share": 0.07,
                "stage6_profile_readiness": "review",
                "top_product_ids": ["p9"],
                "top_products": ["NICHE SAUCE"],
                "top_product_lifts": [2.4],
                "top_product_lifts_vs_rest": [2.5],
                "top_product_q_values": [0.01],
                "top_product_customer_counts": [30],
                "top_product_reach_pct": [42.9],
                "top_sectors": ["BAZAR"],
                "top_sector_lifts": [1.5],
                "top_sector_line_counts": [80],
                "top_themes": [],
                "behavior_ratio_vs_rest": json.dumps({"avg_total_spend": 0.7, "avg_frequency_per_30d": 0.8}),
                "avg_total_spend": 60.0,
                "avg_frequency_per_30d": 0.8,
                "avg_basket_value": 15.0,
                "avg_promo_share": 0.30,
            },
        ]
    ).write_parquet(profile_path)

    all_profiles = stage7_all_tribe_profiles_table(profile_path, cfg=cfg)
    promoted = stage7_promoted_tribe_validation_table(profile_path, cfg=cfg)
    product_identity = stage7_all_tribe_product_identity_table(profile_path, cfg=cfg)
    behavior = stage7_all_tribe_behavior_differentiation_table(profile_path, cfg=cfg)
    review = stage7_review_tribe_audit_table(profile_path, cfg=cfg)
    relationships = stage7_tribe_relationship_atlas_table(profile_path, cfg=cfg)

    assert all_profiles.height == 3
    assert set(all_profiles["tribe_status"].to_list()) == {"final_strong", "final_usable", "potential_review"}
    assert all_profiles.filter(pl.col("tribe_id") == 2)[0, "promotion_status"] == "not_promoted_review"
    assert all_profiles.filter(pl.col("tribe_id") == 2)[0, "tribe_status"] == "potential_review"
    assert "stage6_profile_readiness=review" in all_profiles.filter(pl.col("tribe_id") == 2)[0, "promotion_blocker"]
    assert promoted["tribe_id"].to_list() == [0, 1]
    assert product_identity.height == 3
    assert behavior.height == 3
    assert "tribe_status" in product_identity.columns
    assert "tribe_status" in behavior.columns
    assert review["tribe_id"].to_list() == [2]
    assert relationships.height == 3
    assert {
        "tribe_a_status",
        "tribe_b_status",
        "relationship_scope",
        "similarity_evidence",
        "difference_evidence",
        "commercial_interpretation",
    }.issubset(set(relationships.columns))
    assert "includes_potential_review" in relationships["relationship_scope"].to_list()


def test_stage7_promotion_and_handbook_deduplicate_business_names(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.values["profiling"]["tribe_name_lookup"] = {"0": "Pantry Buyers", "1": "Pantry Buyers"}
    profile_path = tmp_path / "profiles.parquet"
    pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "n_customers": 100,
                "population_share": 0.50,
                "stage6_profile_readiness": "strong",
                "top_products": ["Greek Yogurt", "Honey"],
                "top_product_lifts_vs_rest": [2.5, 1.7],
                "top_product_lifts": [2.2, 1.6],
                "top_product_q_values": [0.001, 0.02],
                "top_product_customer_counts": [60, 30],
                "top_product_reach_pct": [60.0, 30.0],
                "top_themes": [],
                "behavior_ratio_vs_rest": json.dumps({"avg_total_spend": 1.2}),
                "avg_total_spend": 120.0,
                "avg_frequency_per_30d": 2.0,
                "avg_basket_value": 18.0,
                "avg_promo_share": 0.1,
            },
            {
                "tribe_id": 1,
                "n_customers": 80,
                "population_share": 0.40,
                "stage6_profile_readiness": "usable",
                "top_products": ["Prepared Salad", "Soup"],
                "top_product_lifts_vs_rest": [2.3, 1.8],
                "top_product_lifts": [2.0, 1.7],
                "top_product_q_values": [0.001, 0.02],
                "top_product_customer_counts": [50, 25],
                "top_product_reach_pct": [62.5, 31.25],
                "top_themes": [],
                "behavior_ratio_vs_rest": json.dumps({"avg_total_spend": 1.1}),
                "avg_total_spend": 100.0,
                "avg_frequency_per_30d": 1.8,
                "avg_basket_value": 16.0,
                "avg_promo_share": 0.12,
            },
        ]
    ).write_parquet(profile_path)

    promotion = stage7_tribe_promotion_report_table(profile_path, cfg=cfg)
    handbook = stage7_tribe_handbook_table(profile_path, cfg=cfg)

    assert promotion["promotion_decision"].to_list() == ["promoted", "promoted"]
    assert handbook["business_name"].n_unique() == 2
    assert "Pantry Buyers" not in handbook["business_name"].to_list()
    assert handbook["technical_name"].to_list() == ["Greek Yogurt Buyers", "Prepared Salad Buyers"]
    assert set(promotion["naming_quality_gate"].to_list()) == {"duplicate_legacy_business_name"}
    assert set(promotion["business_confidence"].to_list()) == {"low"}


def test_stage7_stakeholder_readiness_flags_delivery_blockers(tmp_path):
    cfg = _test_config(tmp_path)
    final_index = pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "tribe_name": "Tribe 1",
                "customers": 100,
                "top_product": None,
                "top_product_lift": 1.1,
                "top_product_reach_pct": 0.4,
                "top_reach_product_customers": 10,
                "actionability_proof": None,
            }
        ]
    )
    all_profiles = pl.DataFrame(
        [
            {
                "tribe_id": 0,
                "tribe_name": "Tribe 1",
                "promotion_status": "promoted",
                "who_is_the_tribe": "Generic shoppers.",
                "defining_behavior": "n/a",
                "shopping_mission": "n/a",
                "targeting_idea": "Use broad basket and lifecycle triggers until stronger product hooks are available.",
                "revenue_lever": "Increase frequency.",
            }
        ]
    )
    remaining = pl.DataFrame(
        [
            {
                "segment_id": "unclear_long_tail_customers",
                "segment_name": "Unclear long-tail customers",
                "customer_count": 600,
                "share_of_remaining_pct": 60.0,
            }
        ]
    )
    soft = pl.DataFrame(
        [
            {
                "target_tribe_id": 0,
                "customer_count": 50,
                "mean_top_affinity": 0.8,
                "p10_top_affinity": 0.74,
                "mean_affinity_margin": 0.15,
                "most_common_second_tribe_id": 1,
                "remaining_customer_share_pct": 5.0,
                "audience_label": "soft audience opportunity",
                "assignment_policy": "campaign-use only",
                "recommended_use": "Test.",
            }
        ]
    )
    readiness = stage7_stakeholder_readiness_table(
        final_index,
        all_profiles,
        remaining,
        soft,
        pl.DataFrame(),
        pl.DataFrame(),
        cfg=cfg,
    )
    statuses = {row["check_id"]: row for row in readiness.iter_rows(named=True)}

    assert statuses["overall_delivery_readiness"]["status"] == "fail"
    assert statuses["tribe_name_quality"]["status"] == "fail"
    assert statuses["promoted_tribe_product_evidence"]["status"] == "fail"
    assert statuses["persona_specificity"]["status"] == "fail"
    assert statuses["remaining_customer_segments"]["status"] == "fail"
    assert statuses["soft_audience_activation_export"]["status"] == "fail"
    assert statuses["campaign_playbook_completeness"]["status"] == "fail"


def test_stage7_stakeholder_readiness_uses_delivery_product_signal(tmp_path):
    cfg = _test_config(tmp_path)
    final_index = pl.DataFrame(
        [
            {
                "tribe_id": 8,
                "tribe_name": "Gluten-Free Breakfast Mission",
                "customers": 1000,
                "top_product": "Very Niche Gluten-Free SKU",
                "top_product_lift": 20.0,
                "top_product_reach_pct": 0.2,
                "delivery_product": "Gluten-Free Breakfast Cereal",
                "delivery_product_lift": 4.2,
                "delivery_product_reach_pct": 6.0,
                "delivery_product_customers": 60,
                "delivery_product_q_value": 0.01,
                "actionability_proof": "Lifted-product proof: Gluten-Free Breakfast Cereal",
            }
        ]
    )
    all_profiles = pl.DataFrame(
        [
            {
                "tribe_id": 8,
                "tribe_name": "Gluten-Free Breakfast Mission",
                "promotion_status": "promoted",
                "who_is_the_tribe": "These customers repeatedly over-index on gluten-free breakfast and bakery products with enough reach to support a clear shopper mission.",
                "defining_behavior": "They combine specialist dietary products with regular breakfast trips and show a clear basket pattern around cereal, bakery, and pantry replenishment.",
                "shopping_mission": "The mission is planned replenishment for gluten-free breakfast occasions, with adjacent bakery and snack products as natural extensions.",
                "targeting_idea": "Use cereal and bakery bundles with measured holdouts and keep creative focused on the dietary breakfast mission.",
                "revenue_lever": "Grow category penetration through cross-sell across breakfast, bakery, and snacks while tracking incremental basket value.",
            }
        ]
    )
    campaign_playbook = pl.DataFrame(
        [
            {
                "tribe_id": 8,
                "offer_idea": "Breakfast bundle anchored on Gluten-Free Breakfast Cereal.",
                "recommended_channel": "App push and loyalty placement.",
                "suppression_rules": "Suppress recent exact-product purchasers and campaign controls.",
                "holdout_control_design": "Hold out 10% of eligible customers.",
                "primary_kpi": "Incremental category penetration.",
                "expected_commercial_lever": "Grow breakfast basket breadth.",
                "risk_caveat": "Dietary-mission segment only.",
            }
        ]
    )

    readiness = stage7_stakeholder_readiness_table(
        final_index,
        all_profiles,
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame(),
        campaign_playbook,
        cfg=cfg,
    )
    statuses = {row["check_id"]: row for row in readiness.iter_rows(named=True)}

    assert statuses["overall_delivery_readiness"]["status"] == "pass"
    assert statuses["promoted_tribe_product_evidence"]["status"] == "pass"


def test_stage7_soft_audience_activation_exports_customer_rows_without_mutating_assignments(tmp_path):
    cfg = _test_config(tmp_path)
    affinity_path = tmp_path / "affinity.parquet"
    soft_path = tmp_path / "soft.csv"
    csv_path = tmp_path / "activation.csv"
    parquet_path = tmp_path / "activation.parquet"
    pl.DataFrame(
        [
            {
                "cliente": "cust-101",
                "official_tribe_id": -1,
                "top_tribe_id": 3,
                "top_affinity_score": 0.82,
                "second_tribe_id": 2,
                "second_affinity_score": 0.60,
                "affinity_margin": 0.22,
                "affinity_confidence_band": "high",
                "recommended_use": "soft audience opportunity",
                "official_assignment_policy": "hard assignments unchanged",
            },
            {
                "cliente": "cust-102",
                "official_tribe_id": 3,
                "top_tribe_id": 3,
                "top_affinity_score": 0.90,
                "second_tribe_id": 2,
                "second_affinity_score": 0.60,
                "affinity_margin": 0.30,
                "affinity_confidence_band": "high",
                "recommended_use": "already assigned",
                "official_assignment_policy": "hard assignments unchanged",
            },
        ]
    ).write_parquet(affinity_path)
    pl.DataFrame(
        [
            {
                "target_tribe_id": 3,
                "customer_count": 1,
                "mean_top_affinity": 0.82,
                "p10_top_affinity": 0.82,
                "mean_affinity_margin": 0.22,
                "most_common_second_tribe_id": 2,
                "remaining_customer_share_pct": 100.0,
                "audience_label": "soft audience opportunity",
                "assignment_policy": "campaign-use only",
                "recommended_use": "Test.",
            }
        ]
    ).write_csv(soft_path)

    activation = stage7_soft_audience_activation_customer_table(
        affinity_path,
        soft_audience_path=soft_path,
        final_index=pl.DataFrame([{"tribe_id": 3, "tribe_name": "Fresh Mission Buyers"}]),
        output_csv=csv_path,
        output_parquet=parquet_path,
        cfg=cfg,
    )

    assert activation["cliente"].to_list() == ["cust-101"]
    assert activation["cliente"].n_unique() == 1
    assert activation[0, "official_tribe_id"] == -1
    assert activation[0, "target_tribe_name"] == "Fresh Mission Buyers"
    assert csv_path.exists()
    assert parquet_path.exists()


def test_stage7_campaign_playbook_contains_campaign_ready_fields(tmp_path):
    cfg = _test_config(tmp_path)
    final_index = pl.DataFrame(
        [
            {
                "tribe_id": 7,
                "tribe_name": "Fresh Meal Builders",
                "customers": 250,
                "actionability_proof": "Lifted-product proof: prepared salad",
                "delivery_product": "Prepared Salad",
                "top_product": "Prepared Salad",
                "top_reach_product": "Chicken",
                "total_spend_ratio_vs_rest": 1.2,
                "visit_frequency_ratio_vs_rest": 1.1,
                "promo_sensitivity_ratio_vs_rest": 0.9,
                "distinctive_products": "Prepared Salad 2.1x",
                "spend_and_visit_context": "Higher total spend",
                "caveat": "Purchase behavior only.",
            }
        ]
    )
    playbook = stage7_campaign_playbook_table(final_index, cfg=cfg)
    row = playbook.row(0, named=True)

    assert row["tribe_id"] == 7
    assert row["offer_idea"]
    assert "Prepared Salad" in row["offer_idea"]
    assert "Chicken" not in row["offer_idea"]
    assert row["recommended_channel"]
    assert row["suppression_rules"]
    assert row["holdout_control_design"]
    assert row["primary_kpi"]
    assert row["expected_commercial_lever"]
    assert row["risk_caveat"]

