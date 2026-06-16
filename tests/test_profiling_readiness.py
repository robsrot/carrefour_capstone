import polars as pl

from src.config import PipelineConfig
from src.profiling import (
    _add_overindex_diagnostics,
    customer_metric_anova_table,
    noise_audit_table,
    profile_quality_summary,
    profile_readiness_evidence_table,
    stage7_storyline_table,
    tribe_comparison_table,
    tribe_product_summary_tables,
    write_tribe_product_summary_artifacts,
    write_campaign_signal_artifacts,
    write_llm_profile_interpretation_pack,
    write_noise_audit_artifacts,
    write_stage7_storyline_artifacts,
)


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
    assert row["profile_significant_interpretability_signals"] == 3
    assert row["profiling_readiness_issues"] == "pass"


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


def test_campaign_signal_artifacts_are_opt_in_not_hardcoded(tmp_path):
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
    row = opt_in_campaign.row(0, named=True)

    assert opt_in_campaign.height == 1
    assert row["theme_key"] == "plant_based"
    assert row["working_tribe_name"] == "Avocado Tortilla Evidence Cluster"


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
    assert "Evidence Ladder" in markdown
    assert "not treated as a black box" in markdown
    assert "not a demographic persona" in row["caveat"]
    assert "Greek Yogurt" in row["distinctive_product_evidence"]
    assert "greek yogurt" in row["subsegment_overlay"]


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
    assert "Greek Yogurt (Dairy" in comparison_row["top_product_and_category_evidence"]
    assert "Tickets" in comparison_row["customer_behavior_over_under_index"]
    assert outputs["combined_csv"].exists()


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
    assert row["anova_effect_eta_squared"] > 0.9
    assert (tmp_path / "anova.csv").exists()


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
    recommendation = audit.filter(pl.col("evidence_type") == "overall_recommendation").row(0, named=True)
    product_row = audit.filter(pl.col("evidence_key") == "rare_kefir").row(0, named=True)
    behavior_row = audit.filter(pl.col("evidence_key") == "ticket_count").row(0, named=True)

    assert recommendation["noise_observations"] == 3
    assert recommendation["recommended_action"] in {
        "run_second_pass_noise_clustering_audit",
        "profile_noise_as_secondary_opportunity",
    }
    assert product_row["evidence_type"] == "product_lift"
    assert product_row["lift_vs_core"] is None or product_row["lift_vs_core"] > 1.0
    assert behavior_row["recommended_action"] == "inspect_sparse_history_noise"

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
    assert "Noise Population Audit" in outputs["markdown"].read_text(encoding="utf-8")

