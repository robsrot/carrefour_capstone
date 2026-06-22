"""Stage 8 dashboard-ready semantic layer and demo data products.

Stage 8 is intentionally a publishing layer. It reads structured Stage 6.8 and
Stage 7 outputs, normalizes them for an interactive presentation app, and avoids
parsing narrative Markdown reports.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.profiling import (
    PRODUCT_RANKING_BASIS,
    _classify_remaining_customer_segments,
    _remaining_segment_definition,
    _stage7_remaining_coverage_group,
)
from src.progress import log_event, stage_timer
from src.utils import collect_streaming


STAGE8_SCHEMA_VERSION = 1
DEFAULT_EMBEDDING_SAMPLE_SIZE = 100_000
SMALL_GROUP_FULL_KEEP = 500


def stage8_artifact_paths(*, cfg: PipelineConfig = CONFIG) -> dict[str, Path]:
    """Return all Stage 8 output paths for the active mode."""

    root = cfg.artifacts / "stage8"
    return {
        "root": root,
        "executive_metrics": root / f"executive_metrics_{cfg.mode}.json",
        "executive_metrics_table": root / f"executive_metrics_{cfg.mode}.parquet",
        "executive_insights": root / f"executive_insights_{cfg.mode}.parquet",
        "presentation_storyline": root / f"presentation_storyline_{cfg.mode}.json",
        "presentation_storyline_table": root / f"presentation_storyline_{cfg.mode}.parquet",
        "tribe_master": root / f"tribe_master_{cfg.mode}.parquet",
        "tribe_profiles": root / f"tribe_profiles_{cfg.mode}.parquet",
        "tribe_products": root / f"tribe_products_{cfg.mode}.parquet",
        "tribe_categories": root / f"tribe_categories_{cfg.mode}.parquet",
        "tribe_similarity": root / f"tribe_similarity_{cfg.mode}.parquet",
        "tribe_deep_dive": root / f"tribe_deep_dive_{cfg.mode}.parquet",
        "tribe_deep_dive_json": root / f"tribe_deep_dive_{cfg.mode}.json",
        "tribe_name_proposals": root / f"tribe_name_proposals_{cfg.mode}.parquet",
        "customer_assignments": root / f"customer_assignments_{cfg.mode}.parquet",
        "customer_coverage": root / f"customer_coverage_{cfg.mode}.parquet",
        "customer_coverage_summary": root / f"customer_coverage_summary_{cfg.mode}.parquet",
        "remaining_customer_segments": root / f"remaining_customer_segments_{cfg.mode}.parquet",
        "remaining_customer_assignment": root / f"remaining_customer_assignment_{cfg.mode}.json",
        "segment_actions": root / f"segment_actions_{cfg.mode}.parquet",
        "embedding_2d": root / f"embedding_2d_{cfg.mode}.parquet",
        "embedding_3d": root / f"embedding_3d_{cfg.mode}.parquet",
        "embedding_3d_sample": root / f"embedding_3d_sample_{cfg.mode}.parquet",
        "embedding_3d_pca": root / f"embedding_3d_pca_{cfg.mode}.parquet",
        "embedding_3d_pca_sample": root / f"embedding_3d_pca_sample_{cfg.mode}.parquet",
        "embedding_centroids": root / f"embedding_centroids_{cfg.mode}.parquet",
        "embedding_sample_metadata": root / f"embedding_sample_metadata_{cfg.mode}.json",
        "product_affinity_network": root / f"product_affinity_network_{cfg.mode}.parquet",
        "tribe_similarity_network": root / f"tribe_similarity_network_{cfg.mode}.parquet",
        "coverage_funnel": root / f"coverage_funnel_{cfg.mode}.parquet",
        "opportunity_matrix": root / f"opportunity_matrix_{cfg.mode}.parquet",
        "segment_radar": root / f"segment_radar_{cfg.mode}.parquet",
        "revenue_value_charts": root / f"revenue_value_charts_{cfg.mode}.parquet",
        "visualization_specs": root / f"visualization_specs_{cfg.mode}.parquet",
        "model_metadata": root / f"model_metadata_{cfg.mode}.json",
        "model_cards": root / f"model_cards_{cfg.mode}.parquet",
        "pipeline_lineage": root / f"pipeline_lineage_{cfg.mode}.json",
        "pipeline_stages": root / f"pipeline_stages_{cfg.mode}.parquet",
        "feature_catalog": root / f"feature_catalog_{cfg.mode}.parquet",
        "validation_metrics": root / f"validation_metrics_{cfg.mode}.parquet",
        "data_quality_metrics": root / f"data_quality_metrics_{cfg.mode}.parquet",
        "relational_manifest": root / f"relational_manifest_{cfg.mode}.json",
        "relational_schema": root / f"relational_schema_{cfg.mode}.sql",
        "relational_integrity_report": root / f"relational_integrity_report_{cfg.mode}.parquet",
        "rel_dim_tribe": root / f"rel_dim_tribe_{cfg.mode}.parquet",
        "rel_dim_tribe_name_proposal": root / f"rel_dim_tribe_name_proposal_{cfg.mode}.parquet",
        "rel_dim_coverage_group": root / f"rel_dim_coverage_group_{cfg.mode}.parquet",
        "rel_dim_remaining_segment": root / f"rel_dim_remaining_segment_{cfg.mode}.parquet",
        "rel_dim_product": root / f"rel_dim_product_{cfg.mode}.parquet",
        "rel_dim_category": root / f"rel_dim_category_{cfg.mode}.parquet",
        "rel_dim_action_target": root / f"rel_dim_action_target_{cfg.mode}.parquet",
        "rel_dim_executive_insight": root / f"rel_dim_executive_insight_{cfg.mode}.parquet",
        "rel_dim_story_chapter": root / f"rel_dim_story_chapter_{cfg.mode}.parquet",
        "rel_dim_model_card": root / f"rel_dim_model_card_{cfg.mode}.parquet",
        "rel_dim_pipeline_stage": root / f"rel_dim_pipeline_stage_{cfg.mode}.parquet",
        "rel_dim_feature_group": root / f"rel_dim_feature_group_{cfg.mode}.parquet",
        "rel_dim_dashboard_page": root / f"rel_dim_dashboard_page_{cfg.mode}.parquet",
        "rel_dim_visualization": root / f"rel_dim_visualization_{cfg.mode}.parquet",
        "rel_dim_optional_gap": root / f"rel_dim_optional_gap_{cfg.mode}.parquet",
        "rel_fact_executive_metric": root / f"rel_fact_executive_metric_{cfg.mode}.parquet",
        "rel_fact_tribe_metrics": root / f"rel_fact_tribe_metrics_{cfg.mode}.parquet",
        "rel_fact_tribe_profile_text": root / f"rel_fact_tribe_profile_text_{cfg.mode}.parquet",
        "rel_fact_coverage_group_metrics": root / f"rel_fact_coverage_group_metrics_{cfg.mode}.parquet",
        "rel_fact_remaining_segment_metrics": root / f"rel_fact_remaining_segment_metrics_{cfg.mode}.parquet",
        "rel_fact_validation_metric": root / f"rel_fact_validation_metric_{cfg.mode}.parquet",
        "rel_fact_data_quality_metric": root / f"rel_fact_data_quality_metric_{cfg.mode}.parquet",
        "rel_fact_customer_assignment": root / f"rel_fact_customer_assignment_{cfg.mode}.parquet",
        "rel_fact_customer_value_behavior": root / f"rel_fact_customer_value_behavior_{cfg.mode}.parquet",
        "rel_fact_customer_nearest_tribe": root / f"rel_fact_customer_nearest_tribe_{cfg.mode}.parquet",
        "rel_fact_tribe_product_affinity": root / f"rel_fact_tribe_product_affinity_{cfg.mode}.parquet",
        "rel_fact_tribe_category_affinity": root / f"rel_fact_tribe_category_affinity_{cfg.mode}.parquet",
        "rel_fact_segment_action": root / f"rel_fact_segment_action_{cfg.mode}.parquet",
        "rel_bridge_tribe_similarity": root / f"rel_bridge_tribe_similarity_{cfg.mode}.parquet",
        "rel_fact_embedding_2d": root / f"rel_fact_embedding_2d_{cfg.mode}.parquet",
        "rel_fact_embedding_3d": root / f"rel_fact_embedding_3d_{cfg.mode}.parquet",
        "rel_fact_embedding_3d_sample": root / f"rel_fact_embedding_3d_sample_{cfg.mode}.parquet",
        "rel_fact_embedding_3d_pca": root / f"rel_fact_embedding_3d_pca_{cfg.mode}.parquet",
        "rel_fact_embedding_3d_pca_sample": root / f"rel_fact_embedding_3d_pca_sample_{cfg.mode}.parquet",
        "rel_fact_embedding_centroid": root / f"rel_fact_embedding_centroid_{cfg.mode}.parquet",
        "artifact_manifest": root / f"artifact_manifest_{cfg.mode}.parquet",
        "dashboard_page_contracts": root / f"dashboard_page_contracts_{cfg.mode}.parquet",
        "missing_file_register": root / f"missing_file_register_{cfg.mode}.parquet",
        "completeness_audit": root / f"stage8_dashboard_data_model_completeness_audit_{cfg.mode}.md",
        "completeness_audit_json": root / f"stage8_dashboard_data_model_completeness_audit_{cfg.mode}.json",
        "readme": root / "README.md",
        "readiness_csv": root / f"stage8_readiness_report_{cfg.mode}.csv",
        "readiness_md": root / f"stage8_readiness_report_{cfg.mode}.md",
        "dashboard_manifest": root / f"dashboard_manifest_{cfg.mode}.json",
    }


def stage8_input_paths(*, cfg: PipelineConfig = CONFIG) -> dict[str, Path]:
    """Return the structured upstream artifacts Stage 8 consumes."""

    stage6 = cfg.artifacts / "stage6"
    evidence = stage6 / "stage6_8_evidence"
    stage7 = cfg.artifacts / "stage7"
    handoff = stage7 / "final_handoff"
    supporting = handoff / "supporting_tables"
    features = cfg.outputs / "features"
    model_cache = cfg.model_selection_cache
    return {
        "stage68_manifest": evidence / f"stage68_manifest_{cfg.mode}.json",
        "stage68_product_lifts": evidence / f"product_lifts_{cfg.mode}.parquet",
        "stage68_sector_lifts": evidence / f"sector_lifts_{cfg.mode}.parquet",
        "stage68_remaining_affinity": evidence / f"stage6_7_remaining_customer_affinity_{cfg.mode}.parquet",
        "stage68_customer_metric_tests": evidence / f"customer_metric_tests_{cfg.mode}.csv",
        "stage68_noise_vs_core_metrics": evidence / f"noise_vs_core_customer_metrics_{cfg.mode}.csv",
        "stage6_pca_summary": stage6 / "stage6_1_pca_for_umap_summary.csv",
        "stage6_umap_checks": stage6 / "stage6_1_umap_checks.csv",
        "stage6_merged_checks": stage6 / "stage6_5_merged_three_stage_hdbscan_checks.csv",
        "stage6_quality": stage6 / "stage6_6_representation_cluster_quality.csv",
        "stage6_readiness": stage6 / f"stage6_6_cluster_stability_readiness_clusters_summary.csv",
        "stage3_embedding_validation": cfg.artifacts / "stage3" / "embedding_validation.csv",
        "stage3_embedding_hubness": cfg.artifacts / "stage3" / "embedding_hubness.csv",
        "assignments": model_cache / "cluster_assignments_model_e_three_stage_hdbscan_lift_core.parquet",
        "umap_features": features / str(cfg.get("official_model_suite.umap_hdbscan.umap.output", "feature_set_umap_pca64_u20_n75.parquet")),
        "pca_features": features / "feature_set_pca_for_umap.parquet",
        "behavior_features": features / str(cfg.get("behavioral_features.output", "customer_behavior_features.parquet")),
        "customer_embeddings": features / str(cfg.get("customer_embeddings.output", "customer_embeddings.parquet")),
        "prepared_transactions": cfg.prepared_transactions_path,
        "customer_kpis": cfg.customer_kpis_path,
        "product_embeddings": cfg.outputs / "embeddings" / str(cfg.get("word2vec.embeddings_output", "product_embeddings.parquet")),
        "stage7_manifest": handoff / f"stage7_final_manifest_{cfg.mode}.json",
        "stage7_final_index": handoff / f"stage7_final_index_{cfg.mode}.csv",
        "stage7_all_profiles": stage7 / f"stage7_all_tribe_profiles_{cfg.mode}.csv",
        "stage7_coverage": stage7 / f"stage7_4_customer_coverage_report_{cfg.mode}.csv",
        "stage7_actions": stage7 / f"stage7_5_segment_action_playbook_{cfg.mode}.csv",
        "stage7_remaining_segments": stage7 / f"stage7_remaining_customer_analysis_{cfg.mode}.csv",
        "stage7_campaign_playbook": stage7 / f"stage7_campaign_playbook_{cfg.mode}.csv",
        "stage7_relationships": stage7 / f"stage7_all_tribe_relationship_atlas_{cfg.mode}.csv",
        "stage7_behavior": stage7 / f"stage7_all_tribe_behavior_differentiation_{cfg.mode}.csv",
        "stage7_product_identity": stage7 / f"stage7_all_tribe_product_identity_{cfg.mode}.csv",
        "stage7_product_summary_long": supporting / f"stage7_all_tribe_product_summary_long_{cfg.mode}.csv",
        "stage7_stakeholder_readiness": stage7 / f"stage7_stakeholder_readiness_{cfg.mode}.csv",
        "stage7_soft_activation_customers": stage7 / f"stage7_soft_audience_activation_customers_{cfg.mode}.parquet",
    }


def write_stage8_dashboard_pack(
    *,
    cfg: PipelineConfig = CONFIG,
    embedding_sample_size: int = DEFAULT_EMBEDDING_SAMPLE_SIZE,
    fail_on_missing: bool = True,
) -> dict[str, Path]:
    """Build the complete Stage 8 dashboard-ready artifact pack."""

    paths = stage8_artifact_paths(cfg=cfg)
    inputs = stage8_input_paths(cfg=cfg)
    paths["root"].mkdir(parents=True, exist_ok=True)

    with stage_timer("Stage 8", "building dashboard-ready semantic layer", cfg=cfg):
        audit = _input_audit(inputs)
        missing_required = [
            row["artifact"]
            for row in audit.iter_rows(named=True)
            if row["required"] and not row["exists"]
        ]
        if missing_required and fail_on_missing:
            missing = ", ".join(missing_required)
            raise FileNotFoundError(f"Stage 8 missing required structured inputs: {missing}")

        stage68_manifest = _read_json(inputs["stage68_manifest"])
        stage7_manifest = _read_json(inputs["stage7_manifest"])
        all_profiles = _read_csv(inputs["stage7_all_profiles"])
        coverage_report = _read_csv(inputs["stage7_coverage"])
        remaining_segments = _read_csv(inputs["stage7_remaining_segments"])
        action_playbook = _read_csv(inputs["stage7_actions"])
        product_summary = _read_csv(inputs["stage7_product_summary_long"])
        relationships = _read_csv(inputs["stage7_relationships"])
        stage6_readiness = _read_csv(inputs["stage6_readiness"])

        tribe_master = _build_tribe_master(all_profiles, stage6_readiness=stage6_readiness)
        tribe_master.write_parquet(paths["tribe_master"])

        tribe_profiles = _build_tribe_profiles(all_profiles)
        tribe_profiles.write_parquet(paths["tribe_profiles"])

        tribe_products = _build_tribe_products(product_summary, tribe_master)
        tribe_products.write_parquet(paths["tribe_products"])

        tribe_categories = _build_tribe_categories(inputs["stage68_sector_lifts"], tribe_products, tribe_master)
        tribe_categories.write_parquet(paths["tribe_categories"])

        tribe_similarity = _build_tribe_similarity(relationships)
        tribe_similarity.write_parquet(paths["tribe_similarity"])

        tribe_name_proposals = _build_tribe_name_proposals(tribe_master)
        tribe_name_proposals.write_parquet(paths["tribe_name_proposals"])

        remaining_customer_segments = _build_remaining_segments(remaining_segments)
        remaining_customer_segments.write_parquet(paths["remaining_customer_segments"])

        segment_actions = _build_segment_actions(action_playbook)
        segment_actions.write_parquet(paths["segment_actions"])

        customer_assignments = _build_customer_assignments(inputs["assignments"], tribe_master)
        customer_assignments.write_parquet(paths["customer_assignments"])

        customer_coverage = _build_customer_coverage(
            inputs["assignments"],
            inputs["behavior_features"],
            inputs["stage68_remaining_affinity"],
            tribe_master,
            cfg=cfg,
        )
        customer_coverage.write_parquet(paths["customer_coverage"])

        customer_coverage_summary = _customer_coverage_summary(customer_coverage)
        customer_coverage_summary.write_parquet(paths["customer_coverage_summary"])

        tribe_deep_dive, tribe_deep_dive_json = _build_tribe_deep_dive(
            tribe_master=tribe_master,
            tribe_profiles=tribe_profiles,
            tribe_products=tribe_products,
            tribe_categories=tribe_categories,
            tribe_similarity=tribe_similarity,
            segment_actions=segment_actions,
            customer_coverage=customer_coverage,
            tribe_name_proposals=tribe_name_proposals,
        )
        tribe_deep_dive.write_parquet(paths["tribe_deep_dive"])
        _write_json(paths["tribe_deep_dive_json"], tribe_deep_dive_json)

        remaining_assignment = _remaining_customer_assignment_explanation(
            customer_coverage=customer_coverage,
            remaining_customer_segments=remaining_customer_segments,
        )
        _write_json(paths["remaining_customer_assignment"], remaining_assignment)

        product_affinity_network = _product_affinity_network(tribe_products, tribe_categories)
        product_affinity_network.write_parquet(paths["product_affinity_network"])

        tribe_similarity_network = _tribe_similarity_network(tribe_similarity)
        tribe_similarity_network.write_parquet(paths["tribe_similarity_network"])

        coverage_funnel = _coverage_funnel(customer_coverage_summary)
        coverage_funnel.write_parquet(paths["coverage_funnel"])

        opportunity_matrix = _opportunity_matrix(tribe_deep_dive, remaining_customer_segments)
        opportunity_matrix.write_parquet(paths["opportunity_matrix"])

        segment_radar = _segment_radar(tribe_deep_dive, remaining_customer_segments)
        segment_radar.write_parquet(paths["segment_radar"])

        revenue_value_charts = _revenue_value_charts(tribe_deep_dive, customer_coverage_summary)
        revenue_value_charts.write_parquet(paths["revenue_value_charts"])

        embedding_2d, embedding_3d = _build_embedding_tables(
            inputs["umap_features"],
            customer_coverage,
            tribe_master,
        )
        embedding_2d.write_parquet(paths["embedding_2d"])
        embedding_3d.write_parquet(paths["embedding_3d"])

        embedding_sample = _stratified_embedding_sample(
            embedding_3d,
            sample_size=embedding_sample_size,
            seed=cfg.random_seed,
        )
        embedding_sample.write_parquet(paths["embedding_3d_sample"])

        embedding_3d_pca = _build_embedding_3d_optional(inputs["pca_features"], customer_coverage, tribe_master)
        embedding_3d_pca.write_parquet(paths["embedding_3d_pca"])
        embedding_3d_pca_sample = _stratified_embedding_sample(
            embedding_3d_pca,
            sample_size=embedding_sample_size,
            seed=cfg.random_seed,
        ) if not embedding_3d_pca.is_empty() else embedding_3d_pca
        embedding_3d_pca_sample.write_parquet(paths["embedding_3d_pca_sample"])

        centroids = _embedding_centroids(embedding_3d)
        centroids.write_parquet(paths["embedding_centroids"])

        embedding_sample_metadata = _embedding_sample_metadata(
            embedding_3d=embedding_3d,
            embedding_sample=embedding_sample,
            embedding_3d_pca=embedding_3d_pca,
            embedding_3d_pca_sample=embedding_3d_pca_sample,
            cfg=cfg,
        )
        _write_json(paths["embedding_sample_metadata"], embedding_sample_metadata)

        executive_metrics = _executive_metrics(
            coverage_report,
            tribe_master,
            remaining_customer_segments,
            stage68_manifest,
            inputs,
            cfg=cfg,
        )
        _write_json(paths["executive_metrics"], executive_metrics)
        executive_metrics_table = _executive_metrics_table(executive_metrics)
        executive_metrics_table.write_parquet(paths["executive_metrics_table"])
        executive_metrics_table.write_parquet(paths["rel_fact_executive_metric"])

        storyline = _presentation_storyline()
        _write_json(paths["presentation_storyline"], storyline)
        presentation_storyline_table = _presentation_storyline_table(storyline)
        presentation_storyline_table.write_parquet(paths["presentation_storyline_table"])
        presentation_storyline_table.write_parquet(paths["rel_dim_story_chapter"])

        executive_insights = _executive_insights(tribe_deep_dive, customer_coverage_summary, remaining_customer_segments)
        executive_insights.write_parquet(paths["executive_insights"])
        executive_insights.write_parquet(paths["rel_dim_executive_insight"])

        model_metadata = _model_metadata(cfg, inputs, stage68_manifest, stage7_manifest)
        _write_json(paths["model_metadata"], model_metadata)
        model_cards = _model_cards(model_metadata)
        model_cards.write_parquet(paths["model_cards"])
        model_cards.write_parquet(paths["rel_dim_model_card"])

        lineage = _pipeline_lineage(inputs, _public_artifact_paths(paths), audit)
        _write_json(paths["pipeline_lineage"], lineage)
        pipeline_stages = _pipeline_stages(lineage)
        pipeline_stages.write_parquet(paths["pipeline_stages"])
        pipeline_stages.write_parquet(paths["rel_dim_pipeline_stage"])

        feature_catalog = _feature_catalog(cfg, inputs, model_metadata)
        feature_catalog.write_parquet(paths["feature_catalog"])
        feature_catalog.write_parquet(paths["rel_dim_feature_group"])

        validation_metrics = _validation_metrics(inputs)
        validation_metrics.write_parquet(paths["validation_metrics"])
        validation_metrics.write_parquet(paths["rel_fact_validation_metric"])

        data_quality_metrics = _data_quality_metrics(audit, validation_metrics)
        data_quality_metrics.write_parquet(paths["data_quality_metrics"])
        data_quality_metrics.write_parquet(paths["rel_fact_data_quality_metric"])

        relational_tables = _build_relational_tables(
            tribe_master=tribe_master,
            tribe_profiles=tribe_profiles,
            tribe_products=tribe_products,
            tribe_categories=tribe_categories,
            tribe_similarity=tribe_similarity,
            tribe_deep_dive=tribe_deep_dive,
            tribe_name_proposals=tribe_name_proposals,
            customer_assignments=customer_assignments,
            customer_coverage=customer_coverage,
            customer_coverage_summary=customer_coverage_summary,
            remaining_customer_segments=remaining_customer_segments,
            segment_actions=segment_actions,
            embedding_2d=embedding_2d,
            embedding_3d=embedding_3d,
            embedding_sample=embedding_sample,
            embedding_3d_pca=embedding_3d_pca,
            embedding_3d_pca_sample=embedding_3d_pca_sample,
            centroids=centroids,
        )
        for table_name, table in relational_tables.items():
            table.write_parquet(paths[table_name])
        relational_integrity_report = _relational_integrity_report(relational_tables)
        relational_integrity_report.write_parquet(paths["relational_integrity_report"])
        _write_json(paths["relational_manifest"], _relational_manifest(relational_tables, paths, cfg=cfg))
        paths["relational_schema"].write_text(_relational_schema_sql(), encoding="utf-8")

        visualization_specs = _visualization_specs()
        visualization_specs.write_parquet(paths["visualization_specs"])
        visualization_specs.write_parquet(paths["rel_dim_visualization"])

        dashboard_page_contracts = _dashboard_page_contracts()
        dashboard_page_contracts.write_parquet(paths["dashboard_page_contracts"])
        dashboard_page_contracts.write_parquet(paths["rel_dim_dashboard_page"])

        missing_file_register = _missing_file_register()
        missing_file_register.write_parquet(paths["missing_file_register"])
        missing_file_register.write_parquet(paths["rel_dim_optional_gap"])

        artifact_manifest = _artifact_manifest(paths)
        artifact_manifest.write_parquet(paths["artifact_manifest"])

        completeness_audit = _completeness_audit_payload(paths, missing_file_register)
        _write_json(paths["completeness_audit_json"], completeness_audit)
        paths["completeness_audit"].write_text(_completeness_audit_markdown(completeness_audit), encoding="utf-8")

        paths["readme"].write_text(_stage8_readme(cfg=cfg), encoding="utf-8")

        readiness = _stage8_readiness_report(
            paths,
            inputs,
            audit,
            coverage_report=coverage_report,
            tribe_master=tribe_master,
            tribe_products=tribe_products,
            tribe_deep_dive=tribe_deep_dive,
            segment_actions=segment_actions,
            customer_coverage=customer_coverage,
            embedding_3d=embedding_3d,
            embedding_sample=embedding_sample,
            relational_integrity_report=relational_integrity_report,
        )
        readiness.write_csv(paths["readiness_csv"])
        paths["readiness_md"].write_text(_readiness_markdown(readiness), encoding="utf-8")

        manifest = _dashboard_manifest(paths, inputs, readiness, cfg=cfg)
        _write_json(paths["dashboard_manifest"], manifest)

        final_lineage = _pipeline_lineage(inputs, _public_artifact_paths(paths), audit)
        _write_json(paths["pipeline_lineage"], final_lineage)
        final_pipeline_stages = _pipeline_stages(final_lineage)
        final_pipeline_stages.write_parquet(paths["pipeline_stages"])
        final_pipeline_stages.write_parquet(paths["rel_dim_pipeline_stage"])
        _artifact_manifest(paths).write_parquet(paths["artifact_manifest"])
        _write_json(paths["relational_manifest"], _relational_manifest_from_paths(paths, cfg=cfg))
        final_manifest = _dashboard_manifest(paths, inputs, readiness, cfg=cfg)
        _write_json(paths["dashboard_manifest"], final_manifest)
        _cleanup_stage8_output_folder(paths)

    log_event(
        "Stage 8",
        "wrote dashboard-ready semantic pack",
        cfg=cfg,
        output_dir=paths["root"],
        manifest=paths["dashboard_manifest"],
    )
    return paths


def _input_audit(inputs: Mapping[str, Path]) -> pl.DataFrame:
    required = {
        "stage68_manifest",
        "assignments",
        "umap_features",
        "behavior_features",
        "stage7_all_profiles",
        "stage7_coverage",
        "stage7_actions",
        "stage7_remaining_segments",
        "stage7_relationships",
        "stage7_product_summary_long",
    }
    rows = []
    for name, path in sorted(inputs.items()):
        exists = path.exists()
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "required": name in required,
                "exists": exists,
                "file_type": path.suffix.lower().lstrip(".") or "directory",
                "size_bytes": path.stat().st_size if exists and path.is_file() else None,
            }
        )
    return pl.DataFrame(rows)


def _build_tribe_master(all_profiles: pl.DataFrame, *, stage6_readiness: pl.DataFrame | None = None) -> pl.DataFrame:
    readiness_by_id = _rows_by_id(stage6_readiness if stage6_readiness is not None else pl.DataFrame(), "tribe_id")
    rows = []
    for row in all_profiles.iter_rows(named=True):
        tribe_id = _to_int(row.get("tribe_id"))
        readiness = readiness_by_id.get(tribe_id, {})
        promotion = _as_text(row.get("promotion_decision") or row.get("promotion_status"))
        tribe_status = _as_text(row.get("tribe_status"))
        stage6_profile_readiness = _as_text(row.get("stage6_profile_readiness") or readiness.get("profile_readiness"))
        is_final = promotion == "promoted" or tribe_status.startswith("final_")
        is_review = promotion == "review" or tribe_status == "potential_review" or stage6_profile_readiness == "review"
        rows.append(
            {
                "tribe_id": tribe_id,
                "tribe_key": f"tribe_{tribe_id:02d}" if tribe_id is not None else None,
                "tribe_name": _as_text(row.get("tribe_name") or row.get("business_name") or row.get("technical_name")),
                "business_name": _as_text(row.get("business_name") or row.get("tribe_name")),
                "technical_name": _as_text(row.get("technical_name")),
                "legacy_tribe_name": _as_text(row.get("legacy_tribe_name")),
                "promotion_decision": "promoted" if is_final else "review" if is_review else promotion,
                "tribe_status": tribe_status or ("final_usable" if is_final else "potential_review"),
                "tribe_status_label": _as_text(row.get("tribe_status_label")),
                "coverage_group": "core_promoted_tribe" if is_final else "review_tribe",
                "membership_policy": _as_text(row.get("membership_policy")),
                "validation_tier": _as_text(row.get("validation_tier")),
                "business_confidence": _as_text(row.get("business_confidence") or row.get("confidence_level")),
                "stage6_profile_readiness": stage6_profile_readiness,
                "validation_blockers": _as_text(row.get("validation_blockers") or row.get("promotion_blocker") or readiness.get("readiness_issues")),
                "readiness_caveat": _as_text(row.get("readiness_caveat")),
                "recommended_use": _as_text(row.get("recommended_use")),
                "customers": _to_int(row.get("customers") or row.get("customers_count") or row.get("n_customers")),
                "population_share_pct": _to_float(row.get("population_share_pct")),
                "mean_assignment_confidence": _to_float(row.get("mean_assignment_confidence") or readiness.get("mean_assignment_confidence")),
                "p10_assignment_confidence": _to_float(row.get("p10_assignment_confidence") or readiness.get("p10_assignment_confidence")),
                "jitter_label_recovery_accuracy": _to_float(
                    row.get("jitter_label_recovery_accuracy")
                    or row.get("jitter_label_recovery_accuracy_mean")
                    or readiness.get("jitter_label_recovery_accuracy_mean")
                ),
                "is_final_tribe": is_final,
                "is_review_tribe": is_review,
            }
        )
    schema = {
        "tribe_id": pl.Int64,
        "tribe_key": pl.Utf8,
        "tribe_name": pl.Utf8,
        "business_name": pl.Utf8,
        "technical_name": pl.Utf8,
        "legacy_tribe_name": pl.Utf8,
        "promotion_decision": pl.Utf8,
        "tribe_status": pl.Utf8,
        "tribe_status_label": pl.Utf8,
        "coverage_group": pl.Utf8,
        "membership_policy": pl.Utf8,
        "validation_tier": pl.Utf8,
        "business_confidence": pl.Utf8,
        "stage6_profile_readiness": pl.Utf8,
        "validation_blockers": pl.Utf8,
        "readiness_caveat": pl.Utf8,
        "recommended_use": pl.Utf8,
        "customers": pl.Int64,
        "population_share_pct": pl.Float64,
        "mean_assignment_confidence": pl.Float64,
        "p10_assignment_confidence": pl.Float64,
        "jitter_label_recovery_accuracy": pl.Float64,
        "is_final_tribe": pl.Boolean,
        "is_review_tribe": pl.Boolean,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row").sort("tribe_id")


def _build_tribe_profiles(all_profiles: pl.DataFrame) -> pl.DataFrame:
    if all_profiles.is_empty():
        return pl.DataFrame()
    return all_profiles.with_columns(
        [
            _cast_existing_or_null(all_profiles, "tribe_id", pl.Int64),
            _cast_existing_or_null(all_profiles, "customers", pl.Int64),
            _cast_existing_or_null(all_profiles, "population_share_pct", pl.Float64),
        ]
    )


def _build_tribe_products(product_summary: pl.DataFrame, tribe_master: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for row in product_summary.iter_rows(named=True):
        q_value = _to_float(row.get("q_value"))
        lift = _to_float(row.get("lift_vs_rest"))
        rows.append(
            {
                "tribe_id": _to_int(row.get("tribe_id")),
                "rank": _to_int(row.get("rank")),
                "product_id": _as_text(row.get("product_id") or row.get("idarticu")),
                "product_description": _as_text(row.get("product_description") or row.get("desc_larga_articulo")),
                "category": _as_text(row.get("category") or row.get("desc_sector")),
                "lift_vs_rest": lift,
                "lift_vs_population": _to_float(row.get("lift_vs_population") or row.get("lift")),
                "reach_pct": _to_float(row.get("reach_pct")),
                "q_value": q_value,
                "customers": _to_int(row.get("customers") or row.get("cluster_customers")),
                "product_rank_score": _to_float(row.get("product_rank_score")),
                "statistical_result": _as_text(row.get("statistical_result")),
                "is_actionable": bool((lift or 0.0) >= 1.5 and (q_value is None or q_value <= 0.05)),
                "product_ranking_basis": _as_text(row.get("product_ranking_basis") or PRODUCT_RANKING_BASIS),
            }
        )
    schema = {
        "tribe_id": pl.Int64,
        "rank": pl.Int64,
        "product_id": pl.Utf8,
        "product_description": pl.Utf8,
        "category": pl.Utf8,
        "lift_vs_rest": pl.Float64,
        "lift_vs_population": pl.Float64,
        "reach_pct": pl.Float64,
        "q_value": pl.Float64,
        "customers": pl.Int64,
        "product_rank_score": pl.Float64,
        "statistical_result": pl.Utf8,
        "is_actionable": pl.Boolean,
        "product_ranking_basis": pl.Utf8,
    }
    frame = pl.DataFrame(rows, schema=schema, orient="row") if rows else pl.DataFrame(schema=schema)
    return _join_tribe_context(frame, tribe_master, select_status=True).sort(["tribe_id", "rank"])


def _build_tribe_categories(
    sector_lifts_path: Path,
    tribe_products: pl.DataFrame,
    tribe_master: pl.DataFrame,
) -> pl.DataFrame:
    if sector_lifts_path.exists():
        try:
            sector = pl.read_parquet(sector_lifts_path)
            rows = []
            for row in sector.iter_rows(named=True):
                rows.append(
                    {
                        "tribe_id": _to_int(row.get("tribe_id")),
                        "category": _as_text(row.get("desc_sector") or row.get("sector") or row.get("category")),
                        "lift_vs_rest": _to_float(row.get("lift_vs_rest") or row.get("lift")),
                        "lift_vs_population": _to_float(row.get("lift_vs_population")),
                        "line_count": _to_int(row.get("line_count") or row.get("cluster_lines")),
                        "customers": _to_int(row.get("customers") or row.get("cluster_customers")),
                        "q_value": _to_float(row.get("q_value") or row.get("lift_q_value")),
                        "rank": None,
                    }
                )
            frame = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
            if not frame.is_empty():
                frame = frame.with_columns(
                    pl.col("lift_vs_rest")
                    .rank("dense", descending=True)
                    .over("tribe_id")
                    .cast(pl.Int64)
                    .alias("rank")
                )
                return _join_tribe_context(frame, tribe_master, select_status=True).sort(["tribe_id", "rank"])
        except Exception:
            pass
    if tribe_products.is_empty():
        return pl.DataFrame()
    derived = (
        tribe_products.filter(pl.col("category").is_not_null())
        .group_by(["tribe_id", "category"])
        .agg(
            [
                pl.max("lift_vs_rest").alias("lift_vs_rest"),
                pl.sum("customers").alias("customers"),
                pl.len().alias("supporting_products"),
                pl.min("q_value").alias("q_value"),
            ]
        )
        .with_columns(
            pl.col("lift_vs_rest")
            .rank("dense", descending=True)
            .over("tribe_id")
            .cast(pl.Int64)
            .alias("rank")
        )
    )
    return _join_tribe_context(derived, tribe_master, select_status=True).sort(["tribe_id", "rank"])


def _build_tribe_similarity(relationships: pl.DataFrame) -> pl.DataFrame:
    if relationships.is_empty():
        return pl.DataFrame()
    typed_columns = {
        "tribe_a_id": pl.Int64,
        "tribe_b_id": pl.Int64,
        "product_overlap_score": pl.Float64,
        "behavior_similarity_score": pl.Float64,
        "relationship_score": pl.Float64,
    }
    cast_cols = [
        (
            pl.col(column).cast(dtype, strict=False).alias(column)
            if column in relationships.columns
            else pl.lit(None, dtype=dtype).alias(column)
        )
        for column, dtype in typed_columns.items()
    ]
    return relationships.with_columns(cast_cols)


def _build_tribe_name_proposals(tribe_master: pl.DataFrame) -> pl.DataFrame:
    """Publish external LLM naming suggestions as proposals, not truth labels."""

    proposals = _external_llm_name_proposals()
    rows = []
    for row in tribe_master.iter_rows(named=True):
        tribe_id = _to_int(row.get("tribe_id"))
        proposal = proposals.get(tribe_id, {})
        has_proposal = bool(proposal)
        technical_name = _as_text(row.get("technical_name") or row.get("business_name") or row.get("tribe_name"))
        proposed_name = _as_text(proposal.get("proposed_business_name"))
        confidence_label = _as_text(proposal.get("proposal_confidence_label") or "missing")
        rows.append(
            {
                "tribe_id": tribe_id,
                "current_tribe_name": _as_text(row.get("tribe_name")),
                "current_business_name": _as_text(row.get("business_name")),
                "current_technical_name": _as_text(row.get("technical_name")),
                "recommended_technical_name": technical_name,
                "recommended_business_name": proposed_name or _as_text(row.get("business_name") or row.get("tribe_name")),
                "promotion_decision": _as_text(row.get("promotion_decision")),
                "tribe_status": _as_text(row.get("tribe_status")),
                "proposed_business_name": proposed_name,
                "proposed_technical_name": _as_text(proposal.get("proposed_technical_name") or technical_name),
                "proposal_confidence_label": confidence_label,
                "proposal_rationale": _as_text(proposal.get("proposal_rationale")),
                "proposal_stat_highlights": _as_text(proposal.get("proposal_stat_highlights")),
                "proposal_source": "user_provided_independent_llm_html" if has_proposal else "",
                "proposal_status": "candidate_needs_business_review" if has_proposal else "no_external_proposal",
                "business_name_readiness": _business_name_readiness(confidence_label, has_proposal),
                "evidence_alignment": "supported_by_stage8_product_and_behavior_evidence" if has_proposal else "missing_external_name_review",
                "recommended_use": (
                    "Use recommended_business_name for executive display after final human sign-off; keep recommended_technical_name for traceability."
                    if has_proposal
                    else "Keep Stage 7 name until a reviewed business name is supplied."
                ),
                "naming_warning": (
                    "Business names are interpretation layers; never overwrite tribe_id, technical labels, or evidence fields."
                ),
            }
        )
    schema = {
        "tribe_id": pl.Int64,
        "current_tribe_name": pl.Utf8,
        "current_business_name": pl.Utf8,
        "current_technical_name": pl.Utf8,
        "recommended_technical_name": pl.Utf8,
        "recommended_business_name": pl.Utf8,
        "promotion_decision": pl.Utf8,
        "tribe_status": pl.Utf8,
        "proposed_business_name": pl.Utf8,
        "proposed_technical_name": pl.Utf8,
        "proposal_confidence_label": pl.Utf8,
        "proposal_rationale": pl.Utf8,
        "proposal_stat_highlights": pl.Utf8,
        "proposal_source": pl.Utf8,
        "proposal_status": pl.Utf8,
        "business_name_readiness": pl.Utf8,
        "evidence_alignment": pl.Utf8,
        "recommended_use": pl.Utf8,
        "naming_warning": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema, orient="row").sort("tribe_id")


def _build_tribe_deep_dive(
    *,
    tribe_master: pl.DataFrame,
    tribe_profiles: pl.DataFrame,
    tribe_products: pl.DataFrame,
    tribe_categories: pl.DataFrame,
    tribe_similarity: pl.DataFrame,
    segment_actions: pl.DataFrame,
    customer_coverage: pl.DataFrame,
    tribe_name_proposals: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Create a dashboard-ready one-row-per-tribe deep-dive table and nested JSON."""

    total_customers = max(customer_coverage.height, 1)
    hard_customers = (
        customer_coverage.filter(pl.col("tribe_id") >= 0).height
        if not customer_coverage.is_empty() and "tribe_id" in customer_coverage.columns
        else 0
    )
    total_revenue = _sum_float_column(customer_coverage, "total_spend") or _sum_float_column(customer_coverage, "customer_value")
    coverage_metrics = _customer_metrics_by_tribe(customer_coverage, total_customers, max(hard_customers, 1), total_revenue)
    profile_by_id = _rows_by_id(tribe_profiles, "tribe_id")
    metrics_by_id = _rows_by_id(coverage_metrics, "tribe_id")
    proposal_by_id = _rows_by_id(tribe_name_proposals, "tribe_id")

    rows: list[dict[str, Any]] = []
    nested: list[dict[str, Any]] = []
    for row in tribe_master.iter_rows(named=True):
        tribe_id = _to_int(row.get("tribe_id"))
        if tribe_id is None:
            continue
        profile = profile_by_id.get(tribe_id, {})
        metrics = metrics_by_id.get(tribe_id, {})
        proposal = proposal_by_id.get(tribe_id, {})
        products = _top_rows_for_tribe(
            tribe_products,
            tribe_id,
            ["rank", "product_id", "product_description", "category", "lift_vs_rest", "reach_pct", "q_value", "customers"],
            limit=10,
        )
        categories = _top_rows_for_tribe(
            tribe_categories,
            tribe_id,
            ["rank", "category", "lift_vs_rest", "customers", "q_value"],
            limit=5,
        )
        actions = _top_rows_for_tribe(
            segment_actions,
            tribe_id,
            [
                "priority",
                "recommended_use",
                "marketing_actions",
                "merchandising_actions",
                "cross_sell_opportunities",
                "retention_actions",
                "suppression_rules",
                "primary_kpi",
                "evidence_basis",
            ],
            limit=3,
        )
        similar = _similar_tribes_for_deep_dive(tribe_similarity, tribe_id, limit=5)

        summary = {
            "tribe_id": tribe_id,
            "tribe_name": _as_text(row.get("tribe_name")),
            "business_name": _as_text(row.get("business_name")),
            "technical_name": _as_text(row.get("technical_name")),
            "recommended_technical_name": _as_text(proposal.get("recommended_technical_name") or row.get("technical_name")),
            "business_ready_name": _as_text(proposal.get("recommended_business_name") or proposal.get("proposed_business_name") or row.get("business_name")),
            "proposed_business_name": _as_text(proposal.get("proposed_business_name")),
            "proposed_technical_name": _as_text(proposal.get("proposed_technical_name")),
            "proposal_confidence_label": _as_text(proposal.get("proposal_confidence_label")),
            "proposal_status": _as_text(proposal.get("proposal_status")),
            "business_name_readiness": _as_text(proposal.get("business_name_readiness")),
            "proposal_rationale": _as_text(proposal.get("proposal_rationale")),
            "proposal_stat_highlights": _as_text(proposal.get("proposal_stat_highlights")),
            "promotion_decision": _as_text(row.get("promotion_decision")),
            "tribe_status": _as_text(row.get("tribe_status")),
            "coverage_group": _as_text(row.get("coverage_group")),
            "assigned_customers": _to_int(metrics.get("assigned_customers")) or _to_int(row.get("customers")) or 0,
            "share_of_total_customers_pct": _to_float(metrics.get("share_of_total_customers_pct"))
            or _to_float(row.get("population_share_pct")),
            "share_of_hard_assigned_customers_pct": _to_float(metrics.get("share_of_hard_assigned_customers_pct")),
            "total_revenue": _to_float(metrics.get("total_revenue")),
            "revenue_share_pct": _to_float(metrics.get("revenue_share_pct")),
            "avg_revenue_per_customer": _to_float(metrics.get("avg_revenue_per_customer")),
            "median_revenue_per_customer": _to_float(metrics.get("median_revenue_per_customer")),
            "avg_ticket_count": _to_float(metrics.get("avg_ticket_count")),
            "avg_basket_value": _to_float(metrics.get("avg_basket_value")),
            "avg_items_per_basket": _to_float(metrics.get("avg_items_per_basket")),
            "avg_promo_share": _to_float(metrics.get("avg_promo_share")),
            "avg_unique_products": _to_float(metrics.get("avg_unique_products")),
            "avg_unique_sectors": _to_float(metrics.get("avg_unique_sectors")),
            "avg_recency_days": _to_float(metrics.get("avg_recency_days")),
            "avg_frequency_per_30d": _to_float(metrics.get("avg_frequency_per_30d")),
            "mean_assignment_confidence": _to_float(row.get("mean_assignment_confidence"))
            or _to_float(metrics.get("avg_assignment_confidence_score")),
            "p10_assignment_confidence": _to_float(row.get("p10_assignment_confidence")),
            "jitter_label_recovery_accuracy": _to_float(row.get("jitter_label_recovery_accuracy")),
            "validation_tier": _as_text(row.get("validation_tier")),
            "business_confidence": _as_text(row.get("business_confidence")),
            "stage6_profile_readiness": _as_text(row.get("stage6_profile_readiness")),
            "readiness_caveat": _as_text(row.get("readiness_caveat")),
            "who_is_the_tribe": _as_text(profile.get("who_is_the_tribe")),
            "defining_behavior": _as_text(profile.get("defining_behavior")),
            "shopping_mission": _as_text(profile.get("shopping_mission")),
            "promo_loyalty_recency": _as_text(profile.get("promo_loyalty_recency")),
            "targeting_idea": _as_text(profile.get("targeting_idea")),
            "revenue_lever": _as_text(profile.get("revenue_lever")),
            "confidence_level": _as_text(profile.get("confidence_level")),
            "evidence_caveat": _as_text(profile.get("evidence_caveat")),
            "actionability_proof": _as_text(profile.get("actionability_proof")),
            "top_products_display": _display_list(products, "product_description", "lift_vs_rest"),
            "top_categories_display": _display_list(categories, "category", "lift_vs_rest"),
            "recommended_action_summary": _as_text(actions[0].get("marketing_actions")) if actions else "",
            "top_products_json": json.dumps(products, ensure_ascii=False, default=str),
            "top_categories_json": json.dumps(categories, ensure_ascii=False, default=str),
            "actions_json": json.dumps(actions, ensure_ascii=False, default=str),
            "similar_tribes_json": json.dumps(similar, ensure_ascii=False, default=str),
        }
        rows.append(summary)
        nested.append(
            {
                **{key: _json_value(value) for key, value in summary.items() if not key.endswith("_json")},
                "top_products": products,
                "top_categories": categories,
                "recommended_actions": actions,
                "similar_tribes": similar,
            }
        )

    frame = pl.DataFrame(rows, infer_schema_length=None).sort("tribe_id") if rows else pl.DataFrame()
    payload = {
        "stage": "8_tribe_deep_dive",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "grain": "one record per retained hard tribe",
        "warning": "Use proposed business names as review candidates only; tribe_id remains the authoritative model key.",
        "tribes": nested,
    }
    return frame, payload


def _customer_metrics_by_tribe(
    customer_coverage: pl.DataFrame,
    total_customers: int,
    hard_customers: int,
    total_revenue: float,
) -> pl.DataFrame:
    if customer_coverage.is_empty() or "tribe_id" not in customer_coverage.columns:
        return pl.DataFrame()
    frame = customer_coverage.filter(pl.col("tribe_id") >= 0)
    if frame.is_empty():
        return pl.DataFrame()

    aggs = [
        pl.len().alias("assigned_customers"),
        (pl.len() * 100.0 / max(total_customers, 1)).alias("share_of_total_customers_pct"),
        (pl.len() * 100.0 / max(hard_customers, 1)).alias("share_of_hard_assigned_customers_pct"),
    ]
    numeric_aggs = {
        "total_spend": [
            pl.col("total_spend").cast(pl.Float64, strict=False).sum().alias("total_revenue"),
            pl.col("total_spend").cast(pl.Float64, strict=False).mean().alias("avg_revenue_per_customer"),
            pl.col("total_spend").cast(pl.Float64, strict=False).median().alias("median_revenue_per_customer"),
        ],
        "ticket_count": [pl.col("ticket_count").cast(pl.Float64, strict=False).mean().alias("avg_ticket_count")],
        "avg_basket_value": [pl.col("avg_basket_value").cast(pl.Float64, strict=False).mean().alias("avg_basket_value")],
        "avg_items_per_basket": [pl.col("avg_items_per_basket").cast(pl.Float64, strict=False).mean().alias("avg_items_per_basket")],
        "promo_share": [pl.col("promo_share").cast(pl.Float64, strict=False).mean().alias("avg_promo_share")],
        "unique_products": [pl.col("unique_products").cast(pl.Float64, strict=False).mean().alias("avg_unique_products")],
        "unique_sectors": [pl.col("unique_sectors").cast(pl.Float64, strict=False).mean().alias("avg_unique_sectors")],
        "recency_days": [pl.col("recency_days").cast(pl.Float64, strict=False).mean().alias("avg_recency_days")],
        "frequency_per_30d": [pl.col("frequency_per_30d").cast(pl.Float64, strict=False).mean().alias("avg_frequency_per_30d")],
        "assignment_confidence_score": [
            pl.col("assignment_confidence_score").cast(pl.Float64, strict=False).mean().alias("avg_assignment_confidence_score")
        ],
    }
    for column, exprs in numeric_aggs.items():
        if column in frame.columns:
            aggs.extend(exprs)

    grouped = frame.group_by("tribe_id").agg(aggs)
    if "total_revenue" in grouped.columns:
        grouped = grouped.with_columns(
            (pl.col("total_revenue").cast(pl.Float64, strict=False) * 100.0 / max(total_revenue, 1e-9)).alias("revenue_share_pct")
        )
    else:
        grouped = grouped.with_columns(pl.lit(None, dtype=pl.Float64).alias("revenue_share_pct"))
    return grouped


def _remaining_customer_assignment_explanation(
    *,
    customer_coverage: pl.DataFrame,
    remaining_customer_segments: pl.DataFrame,
) -> dict[str, Any]:
    total_customers = customer_coverage.height
    remaining = (
        customer_coverage.filter(pl.col("tribe_id") < 0)
        if not customer_coverage.is_empty() and "tribe_id" in customer_coverage.columns
        else pl.DataFrame()
    )
    groups = []
    if not remaining.is_empty() and "coverage_group" in remaining.columns:
        counts = (
            remaining.group_by("coverage_group")
            .agg(
                [
                    pl.len().alias("customers"),
                    (pl.len() * 100.0 / max(total_customers, 1)).alias("share_of_total_pct"),
                    (pl.len() * 100.0 / max(remaining.height, 1)).alias("share_of_remaining_pct"),
                ]
            )
            .sort("customers", descending=True)
        )
        segment_by_name = {row.get("segment_name"): row for row in remaining_customer_segments.iter_rows(named=True)} if not remaining_customer_segments.is_empty() else {}
        segment_by_id = {row.get("segment_id"): row for row in remaining_customer_segments.iter_rows(named=True)} if not remaining_customer_segments.is_empty() else {}
        for row in counts.iter_rows(named=True):
            coverage_group = _as_text(row.get("coverage_group"))
            definition = _remaining_assignment_definition(coverage_group)
            groups.append(
                {
                    "coverage_group": coverage_group,
                    "customers": _json_value(row.get("customers")),
                    "share_of_total_pct": _json_value(row.get("share_of_total_pct")),
                    "share_of_remaining_pct": _json_value(row.get("share_of_remaining_pct")),
                    "meaning": definition["meaning"],
                    "assignment_basis": definition["assignment_basis"],
                    "recommended_dashboard_language": definition["recommended_dashboard_language"],
                    "segment_rows_available": [
                        _json_clean(segment)
                        for segment in segment_by_id.values()
                        if _as_text(segment.get("segment_id")) in definition.get("segment_ids", [])
                    ]
                    or [
                        _json_clean(segment)
                        for segment in segment_by_name.values()
                        if _as_text(segment.get("segment_name")).lower() in definition.get("segment_names", [])
                    ],
                }
            )

    return {
        "stage": "8_remaining_customer_assignment_explanation",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "grain": "policy and coverage-group explanation for customers with negative/noise tribe_id",
        "headline": "Remaining customers were not hard-assigned to tribes. They are retained as honest noise/coverage groups with nearest-tribe affinity fields.",
        "counts": {
            "total_customers": total_customers,
            "remaining_customers": remaining.height,
            "remaining_share_pct": (remaining.height * 100.0 / max(total_customers, 1)) if total_customers else None,
        },
        "hard_assignment_rule": "Customers with tribe_id >= 0 are retained hard HDBSCAN/lift-filter tribe members.",
        "remaining_rule": "Customers with tribe_id < 0 are not forced into final tribes. They are described through coverage_group, segment_id/segment_name, and nearest-tribe affinity fields.",
        "why_not_force_assignment": [
            "HDBSCAN intentionally marks low-density or ambiguous customers as noise.",
            "Stage 6 lift filtering keeps product-first tribes only when the product signal is sufficiently interpretable.",
            "Forcing every customer into a tribe would overstate confidence and blur the meaning of the promoted tribes.",
        ],
        "per_customer_fields_to_use": [
            "coverage_group",
            "segment_id",
            "segment_name",
            "top_tribe_id",
            "top_affinity_score",
            "second_tribe_id",
            "second_affinity_score",
            "affinity_margin",
            "affinity_confidence_band",
            "recommended_use",
        ],
        "remaining_segments": [_json_clean(row) for row in remaining_customer_segments.iter_rows(named=True)]
        if not remaining_customer_segments.is_empty()
        else [],
        "coverage_groups": groups,
        "dashboard_guidance": [
            "Show remaining customers as accounted-for coverage groups, not failed assignments.",
            "Use nearest tribe fields for cautious expansion audiences only.",
            "Use sparse and long-tail groups for onboarding, reactivation, or data-enrichment journeys rather than tribe messaging.",
        ],
    }


def _rows_by_id(frame: pl.DataFrame, column: str) -> dict[int, dict[str, Any]]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    rows = {}
    for row in frame.iter_rows(named=True):
        key = _to_int(row.get(column))
        if key is not None:
            rows[key] = row
    return rows


def _top_rows_for_tribe(frame: pl.DataFrame, tribe_id: int, columns: list[str], *, limit: int) -> list[dict[str, Any]]:
    if frame.is_empty() or "tribe_id" not in frame.columns:
        return []
    scoped = frame.filter(pl.col("tribe_id") == tribe_id)
    if scoped.is_empty():
        return []
    if "rank" in scoped.columns:
        scoped = scoped.sort("rank")
    elif "priority" in scoped.columns:
        scoped = scoped.sort("priority")
    keep = [column for column in columns if column in scoped.columns]
    return [_json_clean(row) for row in scoped.select(keep).head(limit).iter_rows(named=True)]


def _similar_tribes_for_deep_dive(frame: pl.DataFrame, tribe_id: int, *, limit: int) -> list[dict[str, Any]]:
    if frame.is_empty() or not {"tribe_a_id", "tribe_b_id"}.issubset(set(frame.columns)):
        return []
    scoped = frame.filter((pl.col("tribe_a_id") == tribe_id) | (pl.col("tribe_b_id") == tribe_id))
    if scoped.is_empty():
        return []
    if "relationship_score" in scoped.columns:
        scoped = scoped.sort("relationship_score", descending=True, nulls_last=True)
    rows = []
    for row in scoped.head(limit).iter_rows(named=True):
        is_a = _to_int(row.get("tribe_a_id")) == tribe_id
        rows.append(
            {
                "other_tribe_id": _to_int(row.get("tribe_b_id" if is_a else "tribe_a_id")),
                "other_tribe_name": _as_text(row.get("tribe_b_name" if is_a else "tribe_a_name")),
                "relationship_type": _as_text(row.get("relationship_type")),
                "relationship_score": _json_value(row.get("relationship_score")),
                "similarity_evidence": _as_text(row.get("similarity_evidence")),
                "difference_evidence": _as_text(row.get("difference_evidence")),
                "campaign_guidance": _as_text(row.get("campaign_guidance")),
            }
        )
    return rows


def _display_list(rows: list[dict[str, Any]], label_column: str, metric_column: str) -> str:
    parts = []
    for row in rows[:5]:
        label = _as_text(row.get(label_column))
        metric = _to_float(row.get(metric_column))
        parts.append(f"{label} ({metric:.2f}x)" if label and metric is not None else label)
    return "; ".join(part for part in parts if part)


def _json_clean(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in row.items()}


def _customer_coverage_summary(customer_coverage: pl.DataFrame) -> pl.DataFrame:
    if customer_coverage.is_empty() or "coverage_group" not in customer_coverage.columns:
        return pl.DataFrame(schema=_coverage_summary_schema())
    total_customers = max(customer_coverage.height, 1)
    total_revenue = _sum_float_column(customer_coverage, "total_spend")
    aggs = [
        pl.len().alias("customers"),
        (pl.len() * 100.0 / total_customers).alias("customer_share_pct"),
    ]
    for source, alias in [
        ("total_spend", "revenue"),
        ("customer_value", "customer_value"),
        ("assignment_confidence_score", "avg_assignment_confidence"),
        ("ticket_count", "avg_ticket_count"),
        ("avg_basket_value", "avg_basket_value"),
        ("promo_share", "avg_promo_share"),
        ("unique_products", "avg_unique_products"),
        ("recency_days", "avg_recency_days"),
        ("frequency_per_30d", "avg_frequency_per_30d"),
    ]:
        if source in customer_coverage.columns:
            expr = pl.col(source).cast(pl.Float64, strict=False)
            aggs.append((expr.sum() if alias in {"revenue", "customer_value"} else expr.mean()).alias(alias))
    frame = customer_coverage.group_by("coverage_group").agg(aggs)
    if "revenue" in frame.columns:
        frame = frame.with_columns((pl.col("revenue") * 100.0 / max(total_revenue, 1e-9)).alias("revenue_share_pct"))
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("revenue_share_pct"))
    return frame.with_columns(
        [
            pl.col("coverage_group").map_elements(_coverage_group_label, return_dtype=pl.Utf8).alias("coverage_group_label"),
            pl.col("coverage_group").map_elements(_coverage_group_description, return_dtype=pl.Utf8).alias("description"),
            pl.col("coverage_group").map_elements(_coverage_group_recommended_treatment, return_dtype=pl.Utf8).alias("recommended_treatment"),
            pl.col("coverage_group").map_elements(_coverage_group_sort_order, return_dtype=pl.Int64).alias("sort_order"),
        ]
    ).sort("sort_order")


def _coverage_summary_schema() -> dict[str, pl.DataType]:
    return {
        "coverage_group": pl.Utf8,
        "customers": pl.Int64,
        "customer_share_pct": pl.Float64,
        "revenue": pl.Float64,
        "revenue_share_pct": pl.Float64,
        "coverage_group_label": pl.Utf8,
        "description": pl.Utf8,
        "recommended_treatment": pl.Utf8,
        "sort_order": pl.Int64,
    }


def _product_affinity_network(tribe_products: pl.DataFrame, tribe_categories: pl.DataFrame) -> pl.DataFrame:
    rows = []
    if not tribe_products.is_empty():
        for row in tribe_products.filter(pl.col("rank") <= 10 if "rank" in tribe_products.columns else pl.lit(True)).iter_rows(named=True):
            rows.append(
                {
                    "edge_type": "tribe_product",
                    "source_node_id": _tribe_node_id(row.get("tribe_id")),
                    "source_node_label": _as_text(row.get("business_name") or row.get("tribe_name")),
                    "target_node_id": f"product_{_as_text(row.get('product_id'))}",
                    "target_node_label": _as_text(row.get("product_description")),
                    "tribe_id": _to_int(row.get("tribe_id")),
                    "product_id": _as_text(row.get("product_id")),
                    "category": _as_text(row.get("category")),
                    "rank": _to_int(row.get("rank")),
                    "weight": _to_float(row.get("product_rank_score") or row.get("lift_vs_rest")),
                    "lift_vs_rest": _to_float(row.get("lift_vs_rest")),
                    "reach_pct": _to_float(row.get("reach_pct")),
                    "q_value": _to_float(row.get("q_value")),
                    "actionability": _as_text(row.get("is_actionable")),
                }
            )
    if not tribe_categories.is_empty():
        for row in tribe_categories.filter(pl.col("rank") <= 5 if "rank" in tribe_categories.columns else pl.lit(True)).iter_rows(named=True):
            rows.append(
                {
                    "edge_type": "tribe_category",
                    "source_node_id": _tribe_node_id(row.get("tribe_id")),
                    "source_node_label": _as_text(row.get("business_name") or row.get("tribe_name")),
                    "target_node_id": f"category_{_as_text(row.get('category')).lower().replace(' ', '_')}",
                    "target_node_label": _as_text(row.get("category")),
                    "tribe_id": _to_int(row.get("tribe_id")),
                    "product_id": "",
                    "category": _as_text(row.get("category")),
                    "rank": _to_int(row.get("rank")),
                    "weight": _to_float(row.get("lift_vs_rest")),
                    "lift_vs_rest": _to_float(row.get("lift_vs_rest")),
                    "reach_pct": None,
                    "q_value": _to_float(row.get("q_value")),
                    "actionability": "category_context",
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def _tribe_similarity_network(tribe_similarity: pl.DataFrame) -> pl.DataFrame:
    if tribe_similarity.is_empty():
        return pl.DataFrame()
    rows = []
    for row in tribe_similarity.iter_rows(named=True):
        a = _to_int(row.get("tribe_a_id"))
        b = _to_int(row.get("tribe_b_id"))
        rows.append(
            {
                "source_tribe_id": a,
                "source_node_id": f"tribe_{a:02d}" if a is not None else "",
                "source_node_label": _as_text(row.get("tribe_a_name")),
                "target_tribe_id": b,
                "target_node_id": f"tribe_{b:02d}" if b is not None else "",
                "target_node_label": _as_text(row.get("tribe_b_name")),
                "edge_type": _as_text(row.get("relationship_type")),
                "edge_weight": _to_float(row.get("relationship_score")),
                "product_overlap_score": _to_float(row.get("product_overlap_score")),
                "behavior_similarity_score": _to_float(row.get("behavior_similarity_score")),
                "similarity_evidence": _as_text(row.get("similarity_evidence")),
                "difference_evidence": _as_text(row.get("difference_evidence")),
                "campaign_guidance": _as_text(row.get("campaign_guidance")),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _coverage_funnel(customer_coverage_summary: pl.DataFrame) -> pl.DataFrame:
    if customer_coverage_summary.is_empty():
        return pl.DataFrame()
    rows = []
    for index, row in enumerate(customer_coverage_summary.sort("sort_order").iter_rows(named=True), start=1):
        rows.append(
            {
                "step_order": index,
                "coverage_group": _as_text(row.get("coverage_group")),
                "label": _as_text(row.get("coverage_group_label")),
                "customers": _to_int(row.get("customers")),
                "customer_share_pct": _to_float(row.get("customer_share_pct")),
                "revenue": _to_float(row.get("revenue")),
                "revenue_share_pct": _to_float(row.get("revenue_share_pct")),
                "funnel_role": "hard_assigned" if _as_text(row.get("coverage_group")) in {"core_promoted_tribe", "review_tribe"} else "remaining_explained",
                "recommended_treatment": _as_text(row.get("recommended_treatment")),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _opportunity_matrix(tribe_deep_dive: pl.DataFrame, remaining_customer_segments: pl.DataFrame) -> pl.DataFrame:
    rows = []
    if not tribe_deep_dive.is_empty():
        for row in tribe_deep_dive.iter_rows(named=True):
            rows.append(
                {
                    "segment_id": f"tribe_{_to_int(row.get('tribe_id')):02d}",
                    "segment_name": _as_text(row.get("business_ready_name") or row.get("business_name")),
                    "segment_type": "tribe",
                    "tribe_id": _to_int(row.get("tribe_id")),
                    "customers": _to_int(row.get("assigned_customers")),
                    "customer_share_pct": _to_float(row.get("share_of_total_customers_pct")),
                    "revenue": _to_float(row.get("total_revenue")),
                    "revenue_share_pct": _to_float(row.get("revenue_share_pct")),
                    "confidence_score": _to_float(row.get("mean_assignment_confidence")),
                    "jitter_score": _to_float(row.get("jitter_label_recovery_accuracy")),
                    "priority": _as_text(row.get("business_name_readiness")),
                    "matrix_x_reach": _to_float(row.get("share_of_total_customers_pct")),
                    "matrix_y_value": _to_float(row.get("revenue_share_pct")),
                    "matrix_size_customers": _to_int(row.get("assigned_customers")),
                    "recommended_action": _as_text(row.get("recommended_action_summary")),
                }
            )
    if not remaining_customer_segments.is_empty():
        for row in remaining_customer_segments.iter_rows(named=True):
            rows.append(
                {
                    "segment_id": _as_text(row.get("segment_id")),
                    "segment_name": _as_text(row.get("segment_name")),
                    "segment_type": "remaining_customer_segment",
                    "tribe_id": None,
                    "customers": _to_int(row.get("customer_count")),
                    "customer_share_pct": _to_float(row.get("share_of_total_pct")),
                    "revenue": None,
                    "revenue_share_pct": None,
                    "confidence_score": _to_float(row.get("closest_tribe_affinity_mean")),
                    "jitter_score": None,
                    "priority": _as_text(row.get("targetability")),
                    "matrix_x_reach": _to_float(row.get("share_of_total_pct")),
                    "matrix_y_value": _to_float(row.get("avg_total_spend")),
                    "matrix_size_customers": _to_int(row.get("customer_count")),
                    "recommended_action": _as_text(row.get("recommended_action")),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def _segment_radar(tribe_deep_dive: pl.DataFrame, remaining_customer_segments: pl.DataFrame) -> pl.DataFrame:
    metric_map = {
        "avg_revenue_per_customer": "spend",
        "avg_ticket_count": "frequency",
        "avg_basket_value": "basket_value",
        "avg_unique_products": "product_diversity",
        "avg_promo_share": "promo_sensitivity",
        "avg_recency_days": "recency_days",
    }
    rows = []
    if not tribe_deep_dive.is_empty():
        for row in tribe_deep_dive.iter_rows(named=True):
            for source, metric in metric_map.items():
                rows.append(
                    {
                        "segment_id": f"tribe_{_to_int(row.get('tribe_id')):02d}",
                        "segment_name": _as_text(row.get("business_ready_name") or row.get("business_name")),
                        "segment_type": "tribe",
                        "tribe_id": _to_int(row.get("tribe_id")),
                        "metric": metric,
                        "value": _to_float(row.get(source)),
                        "display_value": _fmt_number(row.get(source)),
                    }
                )
    remaining_metric_map = {
        "avg_total_spend": "spend",
        "avg_ticket_count": "frequency",
        "avg_basket_value": "basket_value",
        "avg_unique_products": "product_diversity",
        "avg_promo_share": "promo_sensitivity",
        "avg_recency_days": "recency_days",
    }
    if not remaining_customer_segments.is_empty():
        for row in remaining_customer_segments.iter_rows(named=True):
            for source, metric in remaining_metric_map.items():
                rows.append(
                    {
                        "segment_id": _as_text(row.get("segment_id")),
                        "segment_name": _as_text(row.get("segment_name")),
                        "segment_type": "remaining_customer_segment",
                        "tribe_id": None,
                        "metric": metric,
                        "value": _to_float(row.get(source)),
                        "display_value": _fmt_number(row.get(source)),
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def _revenue_value_charts(tribe_deep_dive: pl.DataFrame, customer_coverage_summary: pl.DataFrame) -> pl.DataFrame:
    rows = []
    if not tribe_deep_dive.is_empty():
        for row in tribe_deep_dive.iter_rows(named=True):
            rows.append(
                {
                    "chart_id": "revenue_by_tribe",
                    "series": "tribe",
                    "x_id": f"tribe_{_to_int(row.get('tribe_id')):02d}",
                    "x_label": _as_text(row.get("business_ready_name") or row.get("business_name")),
                    "value": _to_float(row.get("total_revenue")),
                    "value_unit": "currency",
                    "secondary_value": _to_float(row.get("revenue_share_pct")),
                    "secondary_unit": "pct",
                    "sort_order": _to_int(row.get("assigned_customers")),
                }
            )
    if not customer_coverage_summary.is_empty():
        for row in customer_coverage_summary.iter_rows(named=True):
            rows.append(
                {
                    "chart_id": "revenue_by_coverage_group",
                    "series": "coverage_group",
                    "x_id": _as_text(row.get("coverage_group")),
                    "x_label": _as_text(row.get("coverage_group_label")),
                    "value": _to_float(row.get("revenue")),
                    "value_unit": "currency",
                    "secondary_value": _to_float(row.get("revenue_share_pct")),
                    "secondary_unit": "pct",
                    "sort_order": _to_int(row.get("sort_order")),
                }
            )
            rows.append(
                {
                    "chart_id": "customers_by_coverage_group",
                    "series": "coverage_group",
                    "x_id": _as_text(row.get("coverage_group")),
                    "x_label": _as_text(row.get("coverage_group_label")),
                    "value": _to_float(row.get("customers")),
                    "value_unit": "customers",
                    "secondary_value": _to_float(row.get("customer_share_pct")),
                    "secondary_unit": "pct",
                    "sort_order": _to_int(row.get("sort_order")),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def _build_relational_tables(
    *,
    tribe_master: pl.DataFrame,
    tribe_profiles: pl.DataFrame,
    tribe_products: pl.DataFrame,
    tribe_categories: pl.DataFrame,
    tribe_similarity: pl.DataFrame,
    tribe_deep_dive: pl.DataFrame,
    tribe_name_proposals: pl.DataFrame,
    customer_assignments: pl.DataFrame,
    customer_coverage: pl.DataFrame,
    customer_coverage_summary: pl.DataFrame,
    remaining_customer_segments: pl.DataFrame,
    segment_actions: pl.DataFrame,
    embedding_2d: pl.DataFrame,
    embedding_3d: pl.DataFrame,
    embedding_sample: pl.DataFrame,
    embedding_3d_pca: pl.DataFrame,
    embedding_3d_pca_sample: pl.DataFrame,
    centroids: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Build a normalized star-schema layer beside the wide dashboard tables."""

    customer_assignment_context = _customer_assignment_context(customer_assignments, customer_coverage)
    return {
        "rel_dim_tribe": _rel_dim_tribe(tribe_master, tribe_name_proposals),
        "rel_dim_tribe_name_proposal": _rel_dim_tribe_name_proposal(tribe_name_proposals),
        "rel_dim_coverage_group": _rel_dim_coverage_group(customer_coverage_summary),
        "rel_dim_remaining_segment": _rel_dim_remaining_segment(remaining_customer_segments, customer_coverage),
        "rel_dim_product": _rel_dim_product(tribe_products),
        "rel_dim_category": _rel_dim_category(tribe_products, tribe_categories),
        "rel_dim_action_target": _rel_dim_action_target(segment_actions),
        "rel_fact_tribe_metrics": _rel_fact_tribe_metrics(tribe_master, tribe_deep_dive),
        "rel_fact_tribe_profile_text": _rel_fact_tribe_profile_text(tribe_profiles, tribe_deep_dive),
        "rel_fact_coverage_group_metrics": _rel_fact_coverage_group_metrics(customer_coverage_summary),
        "rel_fact_remaining_segment_metrics": _rel_fact_remaining_segment_metrics(remaining_customer_segments),
        "rel_fact_customer_assignment": _rel_fact_customer_assignment(customer_assignment_context),
        "rel_fact_customer_value_behavior": _rel_fact_customer_value_behavior(customer_coverage),
        "rel_fact_customer_nearest_tribe": _rel_fact_customer_nearest_tribe(customer_coverage),
        "rel_fact_tribe_product_affinity": _rel_fact_tribe_product_affinity(tribe_products),
        "rel_fact_tribe_category_affinity": _rel_fact_tribe_category_affinity(tribe_categories),
        "rel_fact_segment_action": _rel_fact_segment_action(segment_actions),
        "rel_bridge_tribe_similarity": _rel_bridge_tribe_similarity(tribe_similarity),
        "rel_fact_embedding_2d": _rel_fact_embedding(embedding_2d, dimensions=2, sample_scope="full_umap_2d"),
        "rel_fact_embedding_3d": _rel_fact_embedding(embedding_3d, dimensions=3, sample_scope="full_umap_3d"),
        "rel_fact_embedding_3d_sample": _rel_fact_embedding(embedding_sample, dimensions=3, sample_scope="sample_umap_3d"),
        "rel_fact_embedding_3d_pca": _rel_fact_embedding(embedding_3d_pca, dimensions=3, sample_scope="full_pca_3d"),
        "rel_fact_embedding_3d_pca_sample": _rel_fact_embedding(embedding_3d_pca_sample, dimensions=3, sample_scope="sample_pca_3d"),
        "rel_fact_embedding_centroid": _rel_fact_embedding_centroid(centroids),
    }


def _rel_dim_tribe(tribe_master: pl.DataFrame, tribe_name_proposals: pl.DataFrame) -> pl.DataFrame:
    if tribe_master.is_empty():
        return pl.DataFrame()
    cols = [
        _rel_col(tribe_master, "tribe_id", pl.Int64),
        _rel_col(tribe_master, "tribe_key", pl.Utf8),
        _rel_col(tribe_master, "technical_name", pl.Utf8),
        _rel_col(tribe_master, "business_name", pl.Utf8).alias("stage7_business_name"),
        _rel_col(tribe_master, "legacy_tribe_name", pl.Utf8),
        _rel_col(tribe_master, "promotion_decision", pl.Utf8),
        _rel_col(tribe_master, "tribe_status", pl.Utf8),
        _rel_col(tribe_master, "tribe_status_label", pl.Utf8),
        _rel_col(tribe_master, "coverage_group", pl.Utf8),
        _rel_col(tribe_master, "membership_policy", pl.Utf8),
        _rel_col(tribe_master, "validation_tier", pl.Utf8),
        _rel_col(tribe_master, "business_confidence", pl.Utf8),
        _rel_col(tribe_master, "stage6_profile_readiness", pl.Utf8),
        _rel_col(tribe_master, "validation_blockers", pl.Utf8),
        _rel_col(tribe_master, "readiness_caveat", pl.Utf8),
        _rel_col(tribe_master, "recommended_use", pl.Utf8),
        _rel_col(tribe_master, "is_final_tribe", pl.Boolean),
        _rel_col(tribe_master, "is_review_tribe", pl.Boolean),
    ]
    frame = tribe_master.select(cols)
    if not tribe_name_proposals.is_empty() and "tribe_id" in tribe_name_proposals.columns:
        proposal_cols = [
            "tribe_id",
            "recommended_business_name",
            "recommended_technical_name",
            "proposal_status",
            "business_name_readiness",
            "proposal_confidence_label",
        ]
        proposals = tribe_name_proposals.select([col for col in proposal_cols if col in tribe_name_proposals.columns])
        frame = frame.join(proposals, on="tribe_id", how="left")
    for column in [
        "recommended_business_name",
        "recommended_technical_name",
        "proposal_status",
        "business_name_readiness",
        "proposal_confidence_label",
    ]:
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    return frame.with_columns(
        [
            pl.coalesce(["recommended_business_name", "stage7_business_name", "technical_name"]).alias("display_name"),
            pl.coalesce(["recommended_technical_name", "technical_name"]).alias("display_technical_name"),
        ]
    ).select(
        [
            "tribe_id",
            "tribe_key",
            "display_name",
            "display_technical_name",
            "technical_name",
            "stage7_business_name",
            "legacy_tribe_name",
            "recommended_business_name",
            "recommended_technical_name",
            "proposal_status",
            "business_name_readiness",
            "proposal_confidence_label",
            "promotion_decision",
            "tribe_status",
            "tribe_status_label",
            "coverage_group",
            "membership_policy",
            "validation_tier",
            "business_confidence",
            "stage6_profile_readiness",
            "validation_blockers",
            "readiness_caveat",
            "recommended_use",
            "is_final_tribe",
            "is_review_tribe",
        ]
    ).sort("tribe_id")


def _rel_dim_tribe_name_proposal(tribe_name_proposals: pl.DataFrame) -> pl.DataFrame:
    if tribe_name_proposals.is_empty():
        return pl.DataFrame()
    cols = [
        "tribe_id",
        "current_tribe_name",
        "current_business_name",
        "current_technical_name",
        "recommended_technical_name",
        "recommended_business_name",
        "proposed_business_name",
        "proposed_technical_name",
        "proposal_confidence_label",
        "proposal_rationale",
        "proposal_stat_highlights",
        "proposal_source",
        "proposal_status",
        "business_name_readiness",
        "evidence_alignment",
        "naming_warning",
    ]
    frame = tribe_name_proposals.select([col for col in cols if col in tribe_name_proposals.columns])
    return frame.with_columns(_rel_col(frame, "tribe_id", pl.Int64)).sort("tribe_id")


def _rel_dim_coverage_group(customer_coverage_summary: pl.DataFrame) -> pl.DataFrame:
    if customer_coverage_summary.is_empty():
        return pl.DataFrame()
    cols = [
        _rel_col(customer_coverage_summary, "coverage_group", pl.Utf8),
        _rel_col(customer_coverage_summary, "coverage_group_label", pl.Utf8),
        _rel_col(customer_coverage_summary, "description", pl.Utf8),
        _rel_col(customer_coverage_summary, "recommended_treatment", pl.Utf8),
        _rel_col(customer_coverage_summary, "sort_order", pl.Int64),
    ]
    return customer_coverage_summary.select(cols).unique("coverage_group").sort("sort_order")


def _rel_dim_remaining_segment(remaining_customer_segments: pl.DataFrame, customer_coverage: pl.DataFrame) -> pl.DataFrame:
    if remaining_customer_segments.is_empty() and (
        customer_coverage.is_empty()
        or not {"segment_id", "segment_name", "coverage_group"}.issubset(customer_coverage.columns)
    ):
        return pl.DataFrame()
    segment_group = (
        customer_coverage.select(["segment_id", "segment_name", "coverage_group"])
        .filter(pl.col("segment_id").is_not_null())
        .unique("segment_id")
        if not customer_coverage.is_empty() and {"segment_id", "segment_name", "coverage_group"}.issubset(customer_coverage.columns)
        else pl.DataFrame(
            {"segment_id": [], "segment_name": [], "coverage_group": []},
            schema={"segment_id": pl.Utf8, "segment_name": pl.Utf8, "coverage_group": pl.Utf8},
        )
    )
    if remaining_customer_segments.is_empty():
        return segment_group.with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("product_theme_signal"),
                pl.lit(None, dtype=pl.Utf8).alias("targetability"),
                pl.lit(None, dtype=pl.Utf8).alias("likely_reason_for_no_hard_cluster"),
                pl.lit(None, dtype=pl.Utf8).alias("recommended_action"),
                pl.lit("descriptive segment; not hard tribe membership").alias("membership_policy"),
            ]
        ).select(
            [
                "segment_id",
                "segment_name",
                "product_theme_signal",
                "targetability",
                "likely_reason_for_no_hard_cluster",
                "recommended_action",
                "membership_policy",
                "coverage_group",
            ]
        ).sort("segment_id")
    cols = [
        _rel_col(remaining_customer_segments, "segment_id", pl.Utf8),
        _rel_col(remaining_customer_segments, "segment_name", pl.Utf8),
        _rel_col(remaining_customer_segments, "product_theme_signal", pl.Utf8),
        _rel_col(remaining_customer_segments, "targetability", pl.Utf8),
        _rel_col(remaining_customer_segments, "likely_reason_for_no_hard_cluster", pl.Utf8),
        _rel_col(remaining_customer_segments, "recommended_action", pl.Utf8),
        _rel_col(remaining_customer_segments, "membership_policy", pl.Utf8),
    ]
    explicit = remaining_customer_segments.select(cols)
    missing_segments = (
        segment_group
        .join(explicit.select("segment_id"), on="segment_id", how="anti")
        .with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("product_theme_signal"),
                pl.lit(None, dtype=pl.Utf8).alias("targetability"),
                pl.lit(None, dtype=pl.Utf8).alias("likely_reason_for_no_hard_cluster"),
                pl.lit(None, dtype=pl.Utf8).alias("recommended_action"),
                pl.lit("descriptive segment; not hard tribe membership").alias("membership_policy"),
            ]
        )
        .select(explicit.join(segment_group.select(["segment_id", "coverage_group"]), on="segment_id", how="left").columns)
    )
    return (
        pl.concat(
            [explicit.join(segment_group.select(["segment_id", "coverage_group"]), on="segment_id", how="left"), missing_segments],
            how="vertical_relaxed",
        )
        .sort("segment_id")
    )


def _rel_dim_product(tribe_products: pl.DataFrame) -> pl.DataFrame:
    if tribe_products.is_empty() or "product_id" not in tribe_products.columns:
        return pl.DataFrame()
    cols = [
        _rel_col(tribe_products, "product_id", pl.Utf8),
        _rel_col(tribe_products, "product_description", pl.Utf8),
        _rel_col(tribe_products, "category", pl.Utf8),
    ]
    return (
        tribe_products.select(cols)
        .filter(pl.col("product_id").is_not_null() & (pl.col("product_id") != ""))
        .sort(["product_id", "product_description"])
        .unique("product_id", maintain_order=True)
    )


def _rel_dim_category(tribe_products: pl.DataFrame, tribe_categories: pl.DataFrame) -> pl.DataFrame:
    frames = []
    if not tribe_products.is_empty() and "category" in tribe_products.columns:
        frames.append(tribe_products.select(_rel_col(tribe_products, "category", pl.Utf8)))
    if not tribe_categories.is_empty() and "category" in tribe_categories.columns:
        frames.append(tribe_categories.select(_rel_col(tribe_categories, "category", pl.Utf8)))
    if not frames:
        return pl.DataFrame()
    return (
        pl.concat(frames, how="vertical_relaxed")
        .filter(pl.col("category").is_not_null() & (pl.col("category") != ""))
        .unique("category")
        .with_columns(pl.col("category").map_elements(_slug, return_dtype=pl.Utf8).alias("category_slug"))
        .sort("category")
    )


def _rel_dim_action_target(segment_actions: pl.DataFrame) -> pl.DataFrame:
    if segment_actions.is_empty():
        return pl.DataFrame()
    return segment_actions.select(
        [
            _rel_col(segment_actions, "segment_id", pl.Utf8).alias("target_id"),
            _rel_col(segment_actions, "target_type", pl.Utf8),
            _rel_nullable_tribe_id(segment_actions, "tribe_id", "tribe_id"),
            pl.when(_rel_col(segment_actions, "target_type", pl.Utf8) == "remaining_customer_segment")
            .then(_rel_col(segment_actions, "segment_id", pl.Utf8))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("remaining_segment_id"),
            _rel_col(segment_actions, "coverage_group", pl.Utf8),
            _rel_col(segment_actions, "promotion_decision", pl.Utf8),
            _rel_col(segment_actions, "validation_tier", pl.Utf8),
            _rel_col(segment_actions, "business_confidence", pl.Utf8),
            _rel_col(segment_actions, "recommended_use", pl.Utf8),
        ]
    ).unique("target_id").sort("target_id")


def _rel_fact_tribe_metrics(tribe_master: pl.DataFrame, tribe_deep_dive: pl.DataFrame) -> pl.DataFrame:
    if tribe_master.is_empty():
        return pl.DataFrame()
    master = tribe_master.select(
        [
            _rel_col(tribe_master, "tribe_id", pl.Int64),
            _rel_col(tribe_master, "customers", pl.Int64).alias("hard_assigned_customers"),
            _rel_col(tribe_master, "population_share_pct", pl.Float64),
            _rel_col(tribe_master, "mean_assignment_confidence", pl.Float64),
            _rel_col(tribe_master, "p10_assignment_confidence", pl.Float64),
            _rel_col(tribe_master, "jitter_label_recovery_accuracy", pl.Float64),
        ]
    )
    if tribe_deep_dive.is_empty():
        return master.sort("tribe_id")
    metrics = tribe_deep_dive.select(
        [
            _rel_col(tribe_deep_dive, "tribe_id", pl.Int64),
            _rel_col(tribe_deep_dive, "assigned_customers", pl.Int64),
            _rel_col(tribe_deep_dive, "share_of_total_customers_pct", pl.Float64),
            _rel_col(tribe_deep_dive, "share_of_hard_assigned_customers_pct", pl.Float64),
            _rel_col(tribe_deep_dive, "total_revenue", pl.Float64),
            _rel_col(tribe_deep_dive, "revenue_share_pct", pl.Float64),
            _rel_col(tribe_deep_dive, "avg_revenue_per_customer", pl.Float64),
            _rel_col(tribe_deep_dive, "median_revenue_per_customer", pl.Float64),
            _rel_col(tribe_deep_dive, "avg_ticket_count", pl.Float64),
            _rel_col(tribe_deep_dive, "avg_basket_value", pl.Float64),
            _rel_col(tribe_deep_dive, "avg_items_per_basket", pl.Float64),
            _rel_col(tribe_deep_dive, "avg_promo_share", pl.Float64),
            _rel_col(tribe_deep_dive, "avg_unique_products", pl.Float64),
            _rel_col(tribe_deep_dive, "avg_unique_sectors", pl.Float64),
            _rel_col(tribe_deep_dive, "avg_recency_days", pl.Float64),
            _rel_col(tribe_deep_dive, "avg_frequency_per_30d", pl.Float64),
        ]
    )
    return master.join(metrics, on="tribe_id", how="left").sort("tribe_id")


def _rel_fact_tribe_profile_text(tribe_profiles: pl.DataFrame, tribe_deep_dive: pl.DataFrame) -> pl.DataFrame:
    source = tribe_deep_dive if not tribe_deep_dive.is_empty() else tribe_profiles
    if source.is_empty():
        return pl.DataFrame()
    text_cols = [
        "who_is_the_tribe",
        "defining_behavior",
        "shopping_mission",
        "promo_loyalty_recency",
        "targeting_idea",
        "revenue_lever",
        "confidence_level",
        "evidence_caveat",
        "actionability_proof",
        "top_products_display",
        "top_categories_display",
        "recommended_action_summary",
    ]
    return source.select([_rel_col(source, "tribe_id", pl.Int64), *[_rel_col(source, col, pl.Utf8) for col in text_cols]]).sort("tribe_id")


def _rel_fact_coverage_group_metrics(customer_coverage_summary: pl.DataFrame) -> pl.DataFrame:
    if customer_coverage_summary.is_empty():
        return pl.DataFrame()
    cols = [
        "coverage_group",
        "customers",
        "customer_share_pct",
        "revenue",
        "customer_value",
        "avg_assignment_confidence",
        "avg_ticket_count",
        "avg_basket_value",
        "avg_promo_share",
        "avg_unique_products",
        "avg_recency_days",
        "avg_frequency_per_30d",
        "revenue_share_pct",
    ]
    return customer_coverage_summary.select([_rel_col(customer_coverage_summary, col, _rel_dtype(col)) for col in cols]).sort("coverage_group")


def _rel_fact_remaining_segment_metrics(remaining_customer_segments: pl.DataFrame) -> pl.DataFrame:
    if remaining_customer_segments.is_empty():
        return pl.DataFrame()
    cols = [
        "segment_id",
        "customer_count",
        "share_of_remaining_pct",
        "share_of_total_pct",
        "closest_tribe_id",
        "closest_tribe_affinity_mean",
        "second_tribe_id",
        "affinity_margin_mean",
        "avg_ticket_count",
        "avg_total_spend",
        "avg_basket_value",
        "avg_promo_share",
        "avg_unique_products",
        "avg_unique_sectors",
        "avg_recency_days",
        "avg_frequency_per_30d",
    ]
    return remaining_customer_segments.select([_rel_col(remaining_customer_segments, col, _rel_dtype(col)) for col in cols]).sort("segment_id")


def _rel_fact_customer_assignment(customer_context: pl.DataFrame) -> pl.DataFrame:
    if customer_context.is_empty():
        return pl.DataFrame()
    return customer_context.select(
        [
            _rel_col(customer_context, "cliente", pl.Utf8),
            _rel_nullable_tribe_id(customer_context, "tribe_id", "tribe_id"),
            _rel_col(customer_context, "coverage_group", pl.Utf8),
            _rel_remaining_segment_id(customer_context),
            _rel_col(customer_context, "assignment_status", pl.Utf8),
            _rel_col(customer_context, "model_name", pl.Utf8),
            _rel_col(customer_context, "model_variant", pl.Utf8),
            _rel_col(customer_context, "assignment_probability", pl.Float64),
            _rel_col(customer_context, "assignment_confidence_score", pl.Float64),
            _rel_col(customer_context, "assignment_confidence_type", pl.Utf8),
            _rel_col(customer_context, "assignment_source", pl.Utf8),
        ]
    )


def _rel_fact_customer_value_behavior(customer_coverage: pl.DataFrame) -> pl.DataFrame:
    if customer_coverage.is_empty():
        return pl.DataFrame()
    cols = [
        "cliente",
        "ticket_count",
        "total_spend",
        "total_units",
        "avg_basket_value",
        "avg_items_per_basket",
        "promo_share",
        "unique_products",
        "unique_sectors",
        "recency_days",
        "frequency_per_30d",
        "customer_value",
        "revenue_tier",
    ]
    return customer_coverage.select([_rel_col(customer_coverage, col, _rel_dtype(col)) for col in cols])


def _rel_fact_customer_nearest_tribe(customer_coverage: pl.DataFrame) -> pl.DataFrame:
    if customer_coverage.is_empty():
        return pl.DataFrame()
    return customer_coverage.select(
        [
            _rel_col(customer_coverage, "cliente", pl.Utf8),
            _rel_nullable_tribe_id(customer_coverage, "top_tribe_id", "top_tribe_id"),
            _rel_col(customer_coverage, "top_affinity_score", pl.Float64),
            _rel_nullable_tribe_id(customer_coverage, "second_tribe_id", "second_tribe_id"),
            _rel_col(customer_coverage, "second_affinity_score", pl.Float64),
            _rel_col(customer_coverage, "affinity_margin", pl.Float64),
            _rel_col(customer_coverage, "affinity_confidence_band", pl.Utf8),
            _rel_col(customer_coverage, "recommended_use", pl.Utf8).alias("nearest_tribe_recommended_use"),
        ]
    )


def _rel_fact_tribe_product_affinity(tribe_products: pl.DataFrame) -> pl.DataFrame:
    if tribe_products.is_empty():
        return pl.DataFrame()
    cols = [
        "tribe_id",
        "rank",
        "product_id",
        "lift_vs_rest",
        "lift_vs_population",
        "reach_pct",
        "q_value",
        "customers",
        "product_rank_score",
        "statistical_result",
        "is_actionable",
        "product_ranking_basis",
    ]
    return tribe_products.select([_rel_col(tribe_products, col, _rel_dtype(col)) for col in cols]).sort(["tribe_id", "rank"])


def _rel_fact_tribe_category_affinity(tribe_categories: pl.DataFrame) -> pl.DataFrame:
    if tribe_categories.is_empty():
        return pl.DataFrame()
    cols = [
        "tribe_id",
        "rank",
        "category",
        "lift_vs_rest",
        "lift_vs_population",
        "line_count",
        "customers",
        "q_value",
    ]
    return tribe_categories.select([_rel_col(tribe_categories, col, _rel_dtype(col)) for col in cols]).sort(["tribe_id", "rank"])


def _rel_fact_segment_action(segment_actions: pl.DataFrame) -> pl.DataFrame:
    if segment_actions.is_empty():
        return pl.DataFrame()
    return segment_actions.select(
        [
            _rel_col(segment_actions, "segment_id", pl.Utf8).alias("target_id"),
            _rel_col(segment_actions, "target_type", pl.Utf8),
            _rel_col(segment_actions, "priority", pl.Utf8),
            _rel_col(segment_actions, "marketing_actions", pl.Utf8),
            _rel_col(segment_actions, "merchandising_actions", pl.Utf8),
            _rel_col(segment_actions, "cross_sell_opportunities", pl.Utf8),
            _rel_col(segment_actions, "retention_actions", pl.Utf8),
            _rel_col(segment_actions, "suppression_rules", pl.Utf8),
            _rel_col(segment_actions, "primary_kpi", pl.Utf8),
            _rel_col(segment_actions, "expected_impact", pl.Utf8),
            _rel_col(segment_actions, "evidence_basis", pl.Utf8),
        ]
    ).sort("target_id")


def _rel_bridge_tribe_similarity(tribe_similarity: pl.DataFrame) -> pl.DataFrame:
    if tribe_similarity.is_empty():
        return pl.DataFrame()
    cols = [
        "tribe_a_id",
        "tribe_b_id",
        "relationship_scope",
        "relationship_type",
        "product_overlap_score",
        "behavior_similarity_score",
        "relationship_score",
        "similarity_evidence",
        "difference_evidence",
        "commercial_interpretation",
        "campaign_guidance",
    ]
    return tribe_similarity.select([_rel_col(tribe_similarity, col, _rel_dtype(col)) for col in cols]).sort(["tribe_a_id", "tribe_b_id"])


def _rel_fact_embedding(frame: pl.DataFrame, *, dimensions: int, sample_scope: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    cols = [_rel_col(frame, "cliente", pl.Utf8), _rel_col(frame, "x", pl.Float32), _rel_col(frame, "y", pl.Float32)]
    if dimensions >= 3:
        cols.append(_rel_col(frame, "z", pl.Float32))
    return frame.select(cols).with_columns(pl.lit(sample_scope).alias("embedding_scope"))


def _rel_fact_embedding_centroid(centroids: pl.DataFrame) -> pl.DataFrame:
    if centroids.is_empty():
        return pl.DataFrame()
    return centroids.select(
        [
            _rel_col(centroids, "coverage_group", pl.Utf8),
            _rel_nullable_tribe_id(centroids, "tribe_id", "tribe_id"),
            _rel_col(centroids, "segment_id", pl.Utf8).alias("remaining_segment_id"),
            _rel_col(centroids, "x", pl.Float32),
            _rel_col(centroids, "y", pl.Float32),
            _rel_col(centroids, "z", pl.Float32),
            _rel_col(centroids, "customers", pl.Int64),
        ]
    )


def _customer_assignment_context(customer_assignments: pl.DataFrame, customer_coverage: pl.DataFrame) -> pl.DataFrame:
    if customer_coverage.is_empty():
        return pl.DataFrame()
    assignment_cols = [
        col
        for col in [
            "cliente",
            "model_name",
            "model_variant",
            "assignment_status",
        ]
        if col in customer_assignments.columns
    ]
    assignment_context = customer_assignments.select(assignment_cols).unique("cliente") if assignment_cols else pl.DataFrame()
    base = customer_coverage
    if not assignment_context.is_empty():
        base = base.join(assignment_context, on="cliente", how="left")
    return base


def _rel_col(frame: pl.DataFrame, column: str, dtype: pl.DataType, *, alias: str | None = None) -> pl.Expr:
    if column in frame.columns:
        return pl.col(column).cast(dtype, strict=False).alias(alias or column)
    return pl.lit(None, dtype=dtype).alias(alias or column)


def _rel_dtype(column: str) -> pl.DataType:
    if column.endswith("_id") and column not in {"product_id", "segment_id", "target_id"}:
        return pl.Int64
    if column in {"rank", "customers", "line_count", "customer_count", "ticket_count", "total_units", "unique_products", "unique_sectors", "recency_days", "assigned_customers"}:
        return pl.Int64
    if column in {"is_actionable", "is_final_tribe", "is_review_tribe"}:
        return pl.Boolean
    if any(token in column for token in ["pct", "score", "share", "revenue", "spend", "value", "affinity", "margin", "confidence", "basket", "promo", "frequency", "avg_", "median", "lift", "q_value"]):
        return pl.Float64
    return pl.Utf8


def _rel_nullable_tribe_id(frame: pl.DataFrame, column: str, alias: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(None, dtype=pl.Int64).alias(alias)
    value = pl.col(column).cast(pl.Int64, strict=False)
    return pl.when(value >= 0).then(value).otherwise(pl.lit(None, dtype=pl.Int64)).alias(alias)


def _rel_remaining_segment_id(frame: pl.DataFrame) -> pl.Expr:
    if "segment_id" not in frame.columns:
        return pl.lit(None, dtype=pl.Utf8).alias("remaining_segment_id")
    tribe_id = pl.col("tribe_id").cast(pl.Int64, strict=False) if "tribe_id" in frame.columns else pl.lit(None, dtype=pl.Int64)
    return (
        pl.when(tribe_id < 0)
        .then(pl.col("segment_id").cast(pl.Utf8, strict=False))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("remaining_segment_id")
    )


def _build_remaining_segments(remaining_segments: pl.DataFrame) -> pl.DataFrame:
    if remaining_segments.is_empty():
        return pl.DataFrame()
    return remaining_segments.with_columns(
        [
            _cast_existing_or_null(remaining_segments, "customer_count", pl.Int64),
            _cast_existing_or_null(remaining_segments, "share_of_remaining_pct", pl.Float64),
            _cast_existing_or_null(remaining_segments, "share_of_total_pct", pl.Float64),
            _cast_existing_or_null(remaining_segments, "closest_tribe_id", pl.Int64),
            _cast_existing_or_null(remaining_segments, "second_tribe_id", pl.Int64),
            _cast_existing_or_null(remaining_segments, "closest_tribe_affinity_mean", pl.Float64),
            _cast_existing_or_null(remaining_segments, "affinity_margin_mean", pl.Float64),
        ]
    )


def _build_segment_actions(action_playbook: pl.DataFrame) -> pl.DataFrame:
    if action_playbook.is_empty():
        return pl.DataFrame()
    rows = []
    for row in action_playbook.iter_rows(named=True):
        tribe_id = _to_int(row.get("tribe_id"))
        segment_id = _as_text(row.get("segment_id") or (f"tribe_{tribe_id:02d}" if tribe_id is not None else None))
        rows.append(
            {
                "segment_id": segment_id,
                "segment_name": _as_text(row.get("segment_name") or row.get("business_name")),
                "target_type": "tribe" if tribe_id is not None else "remaining_customer_segment",
                "coverage_group": _as_text(row.get("coverage_group")),
                "tribe_id": tribe_id,
                "promotion_decision": _as_text(row.get("promotion_decision")),
                "validation_tier": _as_text(row.get("validation_tier")),
                "business_confidence": _as_text(row.get("business_confidence")),
                "recommended_use": _as_text(row.get("recommended_use")),
                "marketing_actions": _as_text(row.get("marketing_actions")),
                "merchandising_actions": _as_text(row.get("merchandising_actions")),
                "cross_sell_opportunities": _as_text(row.get("cross_sell_opportunities")),
                "retention_actions": _as_text(row.get("retention_opportunities") or row.get("retention_actions")),
                "suppression_rules": _as_text(row.get("exclusions") or row.get("suppression_rules")),
                "primary_kpi": _as_text(row.get("primary_kpi")),
                "priority": _action_priority(row),
                "expected_impact": _as_text(row.get("expected_commercial_lever") or row.get("retention_opportunities")),
                "evidence_basis": _as_text(row.get("evidence_basis") or row.get("recommended_use")),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _build_customer_assignments(assignments_path: Path, tribe_master: pl.DataFrame) -> pl.DataFrame:
    assignments = pl.read_parquet(assignments_path)
    keep = [
        column
        for column in [
            "cliente",
            "tribe_id",
            "model_name",
            "model_variant",
            "assignment_probability",
            "assignment_confidence_score",
            "assignment_confidence_type",
            "assignment_source",
            "jitter_label_recovery_accuracy",
            "stage6_profile_readiness",
        ]
        if column in assignments.columns
    ]
    frame = assignments.select(keep).with_columns(_cast_if_present("tribe_id", pl.Int64))
    context = tribe_master.select(
        [
            "tribe_id",
            "tribe_name",
            "tribe_status",
            "coverage_group",
            "promotion_decision",
            "business_confidence",
            "recommended_use",
        ]
    )
    return (
        frame.join(context, on="tribe_id", how="left")
        .with_columns(
            [
                pl.when(pl.col("tribe_id") < 0)
                .then(pl.lit("remaining_customer"))
                .otherwise(pl.col("tribe_status"))
                .alias("assignment_status"),
                pl.when(pl.col("tribe_id") < 0)
                .then(pl.lit("remaining_customers"))
                .otherwise(pl.col("coverage_group"))
                .alias("coverage_group"),
            ]
        )
        .sort("cliente")
    )


def _build_customer_coverage(
    assignments_path: Path,
    behavior_path: Path,
    affinity_path: Path,
    tribe_master: pl.DataFrame,
    *,
    cfg: PipelineConfig,
) -> pl.DataFrame:
    assignment_cols = [
        "cliente",
        "tribe_id",
        "assignment_probability",
        "assignment_confidence_score",
        "assignment_confidence_type",
        "assignment_source",
    ]
    assignments = pl.read_parquet(assignments_path).select([col for col in assignment_cols if col in pl.read_parquet(assignments_path, n_rows=0).columns])
    assignments = assignments.with_columns(_cast_if_present("tribe_id", pl.Int64))
    behavior = _read_behavior_for_coverage(behavior_path)
    frame = assignments.join(behavior, on="cliente", how="left") if not behavior.is_empty() else assignments

    affinity = _read_optional_parquet(affinity_path)
    if not affinity.is_empty() and "cliente" in affinity.columns:
        affinity_cols = [
            col
            for col in [
                "cliente",
                "top_tribe_id",
                "top_affinity_score",
                "second_tribe_id",
                "second_affinity_score",
                "affinity_margin",
                "affinity_confidence_band",
                "recommended_use",
            ]
            if col in affinity.columns
        ]
        frame = frame.join(affinity.select(affinity_cols), on="cliente", how="left")

    remaining = frame.filter(pl.col("tribe_id") < 0)
    core = frame.filter(pl.col("tribe_id") >= 0)
    if not remaining.is_empty():
        remaining = _classify_remaining_customer_segments(remaining, cfg=cfg)
        remaining = remaining.with_columns(
            [
                pl.col("segment_id").map_elements(_stage7_remaining_coverage_group, return_dtype=pl.Utf8).alias("coverage_group"),
                pl.lit("remaining_customer").alias("tribe_status"),
                pl.lit("not_applicable").alias("promotion_decision"),
                pl.col("segment_id").map_elements(
                    lambda value: _remaining_segment_definition(str(value or ""))["name"],
                    return_dtype=pl.Utf8,
                ).alias("segment_name"),
            ]
        )
    else:
        remaining = pl.DataFrame()

    if not core.is_empty():
        context = tribe_master.select(
            [
                "tribe_id",
                "tribe_name",
                "tribe_status",
                "promotion_decision",
                "coverage_group",
                "business_confidence",
                "recommended_use",
            ]
        )
        core = (
            core.join(context, on="tribe_id", how="left")
            .with_columns(
                [
                    pl.concat_str([pl.lit("tribe_"), pl.col("tribe_id").cast(pl.Utf8).str.zfill(2)]).alias("segment_id"),
                    pl.col("tribe_name").alias("segment_name"),
                ]
            )
        )

    combined = pl.concat([df for df in [core, remaining] if not df.is_empty()], how="diagonal")
    if combined.is_empty():
        return combined
    spend_values = combined.get_column("total_spend") if "total_spend" in combined.columns else None
    p50 = _series_quantile(spend_values, 0.50)
    p80 = _series_quantile(spend_values, 0.80)
    total_spend_expr = pl.col("total_spend") if "total_spend" in combined.columns else pl.lit(None)
    return combined.with_columns(
        [
            total_spend_expr.alias("customer_value"),
            pl.when(total_spend_expr >= p80)
            .then(pl.lit("high_value"))
            .when(total_spend_expr >= p50)
            .then(pl.lit("mid_value"))
            .when(total_spend_expr.is_not_null())
            .then(pl.lit("low_value"))
            .otherwise(pl.lit("unknown_value"))
            .alias("revenue_tier"),
            pl.when(pl.col("tribe_id") < 0)
            .then(pl.col("segment_name"))
            .otherwise(pl.col("tribe_name"))
            .alias("lookup_label"),
        ]
    ).sort("cliente")


def _read_behavior_for_coverage(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    schema = pl.scan_parquet(path).collect_schema().names()
    wanted = [
        "cliente",
        "ticket_count",
        "total_spend",
        "total_units",
        "avg_basket_value",
        "avg_items_per_basket",
        "promo_share",
        "unique_products",
        "unique_sectors",
        "recency_days",
        "frequency_per_30d",
    ]
    cols = [col for col in wanted if col in schema]
    if "cliente" not in cols:
        return pl.DataFrame()
    return collect_streaming(pl.scan_parquet(path).select(cols))


def _build_embedding_tables(
    umap_path: Path,
    customer_coverage: pl.DataFrame,
    tribe_master: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    feature_scan = pl.scan_parquet(umap_path)
    schema = feature_scan.collect_schema()
    columns = schema.names()
    coord_cols = [col for col in columns if col != "cliente" and schema[col].is_numeric()]
    if len(coord_cols) < 3:
        raise ValueError(f"Stage 8 needs at least three numeric embedding columns in {umap_path}")
    source = collect_streaming(
        feature_scan.select(
            [
                "cliente",
                pl.col(coord_cols[0]).cast(pl.Float32).alias("x"),
                pl.col(coord_cols[1]).cast(pl.Float32).alias("y"),
                pl.col(coord_cols[2]).cast(pl.Float32).alias("z"),
            ]
        )
    )
    coverage_cols = [
        col
        for col in [
            "cliente",
            "tribe_id",
            "tribe_name",
            "tribe_status",
            "promotion_decision",
            "coverage_group",
            "segment_id",
            "segment_name",
            "assignment_confidence_score",
            "assignment_probability",
            "revenue_tier",
            "customer_value",
            "top_tribe_id",
            "top_affinity_score",
            "second_tribe_id",
            "affinity_margin",
        ]
        if col in customer_coverage.columns
    ]
    joined = source.join(customer_coverage.select(coverage_cols), on="cliente", how="left")
    summary = _tribe_hover_context(tribe_master)
    if not summary.is_empty() and "tribe_id" in joined.columns:
        joined = joined.join(summary, on="tribe_id", how="left")
    joined = joined.with_columns(
        [
            _hover_label_expr().alias("hover_label"),
            _filter_tokens_expr().alias("filter_tokens"),
        ]
    )
    embedding_3d = joined
    embedding_2d = joined.drop("z")
    return embedding_2d, embedding_3d


def _build_embedding_3d_optional(
    feature_path: Path,
    customer_coverage: pl.DataFrame,
    tribe_master: pl.DataFrame,
) -> pl.DataFrame:
    if not feature_path.exists():
        return pl.DataFrame(schema={"cliente": pl.Utf8, "x": pl.Float32, "y": pl.Float32, "z": pl.Float32})
    _, embedding_3d = _build_embedding_tables(feature_path, customer_coverage, tribe_master)
    return embedding_3d


def _tribe_hover_context(tribe_master: pl.DataFrame) -> pl.DataFrame:
    if tribe_master.is_empty():
        return pl.DataFrame()
    return tribe_master.select(
        [
            "tribe_id",
            pl.col("tribe_name").alias("product_summary"),
            pl.col("coverage_group").alias("category_summary"),
        ]
    )


def _stratified_embedding_sample(
    embedding_3d: pl.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> pl.DataFrame:
    if embedding_3d.height <= sample_size:
        return embedding_3d.sort("cliente")
    keys = [col for col in ["coverage_group", "tribe_id", "segment_id"] if col in embedding_3d.columns]
    if not keys:
        return embedding_3d.sort(pl.col("cliente").hash(seed=seed)).head(sample_size).sort("cliente")
    total = embedding_3d.height
    counts = (
        embedding_3d.group_by(keys)
        .agg(pl.len().alias("group_rows"))
        .with_columns(
            [
                pl.when(pl.col("group_rows") <= SMALL_GROUP_FULL_KEEP)
                .then(pl.col("group_rows"))
                .otherwise(
                    (pl.lit(float(sample_size)) * pl.col("group_rows").cast(pl.Float64) / pl.lit(float(total)))
                    .round(0)
                    .clip(lower_bound=SMALL_GROUP_FULL_KEEP)
                    .cast(pl.Int64)
                )
                .clip(upper_bound=pl.col("group_rows"))
                .alias("target_rows")
            ]
        )
    )
    sampled = (
        embedding_3d.join(counts, on=keys, how="left")
        .sort([*keys, pl.col("cliente").hash(seed=seed)])
        .with_columns(pl.cum_count("cliente").over(keys).alias("sample_rank"))
        .filter(pl.col("sample_rank") <= pl.col("target_rows"))
        .drop(["group_rows", "target_rows", "sample_rank"])
    )
    if sampled.height > sample_size:
        sampled = sampled.sort(pl.col("cliente").hash(seed=seed)).head(sample_size)
    return sampled.sort("cliente")


def _embedding_centroids(embedding_3d: pl.DataFrame) -> pl.DataFrame:
    keys = [col for col in ["coverage_group", "tribe_id", "segment_id", "segment_name"] if col in embedding_3d.columns]
    if not keys:
        return pl.DataFrame()
    return (
        embedding_3d.group_by(keys)
        .agg(
            [
                pl.mean("x").alias("x"),
                pl.mean("y").alias("y"),
                pl.mean("z").alias("z"),
                pl.len().alias("customers"),
            ]
        )
        .sort(keys)
    )


def _embedding_sample_metadata(
    *,
    embedding_3d: pl.DataFrame,
    embedding_sample: pl.DataFrame,
    embedding_3d_pca: pl.DataFrame,
    embedding_3d_pca_sample: pl.DataFrame,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    return {
        "stage": "8_embedding_sample_metadata",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "sample_policy": "stratified by available coverage_group, tribe_id, and segment_id; small groups preserved up to configured threshold",
        "random_seed": cfg.random_seed,
        "small_group_full_keep": SMALL_GROUP_FULL_KEEP,
        "umap": {
            "full_rows": embedding_3d.height,
            "sample_rows": embedding_sample.height,
            "sample_share_pct": embedding_sample.height * 100.0 / max(embedding_3d.height, 1),
            "all_tribes_present": _sample_covers_column(embedding_3d, embedding_sample, "tribe_id"),
            "all_coverage_groups_present": _sample_covers_column(embedding_3d, embedding_sample, "coverage_group"),
        },
        "pca": {
            "available": not embedding_3d_pca.is_empty(),
            "full_rows": embedding_3d_pca.height,
            "sample_rows": embedding_3d_pca_sample.height,
            "sample_share_pct": embedding_3d_pca_sample.height * 100.0 / max(embedding_3d_pca.height, 1)
            if not embedding_3d_pca.is_empty()
            else None,
            "all_tribes_present": _sample_covers_column(embedding_3d_pca, embedding_3d_pca_sample, "tribe_id")
            if not embedding_3d_pca.is_empty()
            else False,
            "all_coverage_groups_present": _sample_covers_column(embedding_3d_pca, embedding_3d_pca_sample, "coverage_group")
            if not embedding_3d_pca.is_empty()
            else False,
        },
    }


def _executive_metrics(
    coverage_report: pl.DataFrame,
    tribe_master: pl.DataFrame,
    remaining_segments: pl.DataFrame,
    stage68_manifest: Mapping[str, Any],
    inputs: Mapping[str, Path],
    *,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    coverage_true = coverage_report
    if "counts_toward_population_total" in coverage_report.columns:
        coverage_true = coverage_report.filter(pl.col("counts_toward_population_total").cast(pl.Utf8) == "true")
    total_customers = _sum_column(coverage_true, "customers")
    final_customers = _coverage_sum(coverage_true, "core_promoted")
    review_customers = _coverage_sum(coverage_true, "review")
    remaining_customers = total_customers - final_customers - review_customers
    final_tribes = tribe_master.filter(pl.col("is_final_tribe")).height if "is_final_tribe" in tribe_master.columns else 0
    review_tribes = tribe_master.filter(pl.col("is_review_tribe")).height if "is_review_tribe" in tribe_master.columns else 0
    row_counts = stage68_manifest.get("row_counts") or {}
    metrics = [
        _metric("total_customers", total_customers, "Customer universe represented in Stage 7 coverage.", inputs["stage7_coverage"]),
        _metric("final_promoted_tribes", final_tribes, "Final tribes ready for executive segmentation.", inputs["stage7_all_profiles"]),
        _metric("review_tribes", review_tribes, "Retained hard tribes held for review.", inputs["stage7_all_profiles"]),
        _metric("final_tribe_customers", final_customers, "Customers in promoted final tribes.", inputs["stage7_coverage"]),
        _metric("review_tribe_customers", review_customers, "Customers in review hard tribes.", inputs["stage7_coverage"]),
        _metric("remaining_customers", remaining_customers, "Customers outside retained hard tribes.", inputs["stage7_coverage"]),
        _metric("remaining_customer_segments", remaining_segments.height, "Descriptive groups for non-hard-tribe customers.", inputs["stage7_remaining_segments"]),
        _metric("stage68_product_lift_rows", row_counts.get("product_lifts"), "Precomputed product evidence rows.", inputs["stage68_manifest"]),
        _metric("stage68_sector_lift_rows", row_counts.get("sector_lifts"), "Precomputed sector evidence rows.", inputs["stage68_manifest"]),
    ]
    return {
        "stage": "8_interactive_presentation_experience",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "mode": cfg.mode,
        "metrics": metrics,
    }


def _presentation_storyline() -> dict[str, Any]:
    chapters = [
        _chapter(
            1,
            "Executive Overview",
            "Show the business problem, customer universe, final tribes, coverage split, and headline actions.",
            "The segmentation is product-first, actionable, and honest about uncertainty.",
            ["KPI strip", "coverage waterfall", "top action cards"],
            ["Open final tribe filters", "jump to priority action"],
            ["rel_fact_executive_metric", "rel_dim_tribe", "rel_fact_segment_action", "rel_dim_action_target", "rel_dim_coverage_group"],
            ["Start with why Carrefour needs behavior-led tribes.", "State final vs review vs remaining coverage clearly."],
        ),
        _chapter(
            2,
            "Project & Data Foundation",
            "Explain raw ticket data, product universe, customer universe, feature engineering, and quality checks.",
            "The model learns from checkout behavior, not demographics.",
            ["source table", "feature lineage diagram", "quality checklist"],
            ["Inspect source counts", "open quality metric detail"],
            ["rel_dim_pipeline_stage", "rel_dim_feature_group", "rel_fact_data_quality_metric", "artifact_manifest", "relational_manifest"],
            ["Emphasize product identity, units, recency, and frequency.", "Call out that spend is only interpretation context."],
        ),
        _chapter(
            3,
            "Backend & Modeling Pipeline",
            "Expose the full analytical path for technical review.",
            "The final assignment comes from PCA, UMAP, three-pass HDBSCAN, product-lift support, and readiness checks.",
            ["pipeline DAG", "hyperparameter table", "validation table"],
            ["Expand stage detail", "filter metrics by stage"],
            ["rel_dim_model_card", "rel_dim_pipeline_stage", "rel_fact_validation_metric", "rel_fact_data_quality_metric", "relational_integrity_report"],
            ["Walk from Item2Vec through customer vectors.", "Explain why noise is retained rather than forced."],
        ),
        _chapter(
            4,
            "Interactive Customer Landscape",
            "Let the audience see the customer map and switch among final, review, and remaining groups.",
            "The landscape contains dense tribes and meaningful non-tribe populations.",
            ["2D UMAP", "3D UMAP/PCA sample", "centroids"],
            ["rotate 3D", "filter tribe/status/group", "hover/click customer"],
            ["rel_fact_embedding_2d", "rel_fact_embedding_3d_sample", "rel_fact_embedding_centroid", "rel_fact_customer_assignment", "rel_dim_tribe", "rel_dim_coverage_group", "rel_dim_remaining_segment"],
            ["Rotate the map live.", "Select a final tribe, then toggle review and remaining customers."],
        ),
        _chapter(
            5,
            "Tribe Explorer",
            "Profile each final and review tribe with evidence and caveats.",
            "Tribes are named from product evidence and readiness status.",
            ["tribe profile panel", "behavior bars", "similar tribe links"],
            ["select tribe", "compare similar tribes", "open recommended actions"],
            ["rel_dim_tribe", "rel_fact_tribe_metrics", "rel_fact_tribe_profile_text", "rel_bridge_tribe_similarity", "rel_fact_segment_action"],
            ["Show one strong tribe and one usable tribe.", "Keep the purchase-behavior caveat visible."],
        ),
        _chapter(
            6,
            "Product Affinity Explorer",
            "Show which products and categories define each tribe.",
            "The commercial hook is product lift, reach, and actionability.",
            ["product heatmap", "ranked products", "category table"],
            ["filter product/category", "click tribe/product", "sort by lift or reach"],
            ["rel_fact_tribe_product_affinity", "rel_fact_tribe_category_affinity", "rel_dim_product", "rel_dim_category", "rel_dim_tribe"],
            ["Explain that rankings favor distinctive products with enough reach.", "Use product overlap to discuss campaign conflicts."],
        ),
        _chapter(
            7,
            "Customer Coverage & Remaining Customers",
            "Account for every customer without forcing weak assignments.",
            "Customers outside core tribes still have business treatment logic.",
            ["coverage bar", "remaining segment table", "lookup panel"],
            ["filter group", "search customer", "open nearest tribe"],
            ["rel_fact_customer_assignment", "rel_fact_customer_nearest_tribe", "rel_dim_coverage_group", "rel_dim_remaining_segment", "rel_fact_coverage_group_metrics", "rel_fact_remaining_segment_metrics"],
            ["Show 100 percent reconciliation.", "Explain why hard tribes and descriptive groups serve different purposes."],
        ),
        _chapter(
            8,
            "Segment Activation Center",
            "Translate segments into marketing, merchandising, cross-sell, retention, and suppression rules.",
            "Actions are tied to evidence and should be measured with holdouts.",
            ["action matrix", "priority filter", "KPI table"],
            ["filter by target type", "open suppression rules", "sort by priority"],
            ["rel_dim_action_target", "rel_fact_segment_action", "rel_fact_tribe_product_affinity", "rel_dim_tribe", "rel_dim_remaining_segment"],
            ["Pick two priority tribes.", "Call out holdout and suppression discipline."],
        ),
        _chapter(
            9,
            "Executive Insights & Final Recommendation",
            "Close with what Carrefour learned, what to do next, limitations, and operationalization.",
            "The system is ready as a demo layer and can become an operational segmentation service.",
            ["recommendation summary", "limitations", "next-step roadmap"],
            ["jump back to evidence", "export action view"],
            ["rel_dim_executive_insight", "rel_dim_story_chapter", "rel_fact_executive_metric", "readiness_csv", "relational_integrity_report"],
            ["Summarize the highest-priority opportunities.", "End with next steps and limitations."],
        ),
    ]
    demo_moments = [
        _demo_moment("3D Customer Landscape", "Proves the customer map is inspectable and interactive.", "rel_fact_embedding_3d_sample; rel_fact_customer_assignment; rel_dim_tribe", "Rotate and filter by coverage group.", "If sample is missing, fall back to 2D embedding."),
        _demo_moment("Tribe Deep Dive", "Turns an abstract cluster into products, behaviors, value, and actions.", "rel_dim_tribe; rel_fact_tribe_metrics; rel_fact_tribe_profile_text", "Click a tribe row/card.", "If product facts are missing, hold tribe for review."),
        _demo_moment("Product Affinity View", "Shows which products make each tribe commercially useful.", "rel_fact_tribe_product_affinity; rel_dim_product", "Click product/category and sort by lift.", "If product table is too large, use top-N filters."),
        _demo_moment("Customer Lookup", "Shows assignment, confidence, nearest tribe, coverage group, and treatment.", "rel_fact_customer_assignment; rel_fact_customer_nearest_tribe; rel_fact_customer_value_behavior", "Search by cliente.", "If customer grain is missing, disable lookup."),
        _demo_moment("Remaining Customer Explanation", "Makes full-population accountability visible.", "rel_dim_remaining_segment; rel_fact_remaining_segment_metrics", "Filter non-core groups.", "If counts do not reconcile, block dashboard readiness."),
        _demo_moment("Backend Transparency", "Lets technical reviewers inspect exact methods and metrics.", "rel_dim_model_card; rel_dim_pipeline_stage; rel_fact_validation_metric", "Open modeling page.", "If metadata is incomplete, show readiness warning."),
    ]
    return {
        "stage": "8_interactive_presentation_experience",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "chapters": chapters,
        "demo_moments": demo_moments,
        "architecture_recommendation": {
            "mvp": "Dash + Plotly",
            "ideal": "React + Plotly consuming Stage 8 static semantic assets",
            "fallback": "Streamlit + Plotly or standalone Plotly HTML for 3D",
        },
    }


def _executive_metrics_table(executive_metrics: Mapping[str, Any]) -> pl.DataFrame:
    rows = []
    for index, metric in enumerate(executive_metrics.get("metrics", []), start=1):
        metric_id = _as_text(metric.get("metric"))
        rows.append(
            {
                "metric_id": metric_id,
                "metric_name": _title_from_id(metric_id),
                "metric_value": _json_value(metric.get("value")),
                "metric_unit": _metric_unit(metric_id),
                "metric_context": _as_text(metric.get("description")),
                "source_file": _as_text(metric.get("source")),
                "source_stage": _source_stage_from_path(_as_text(metric.get("source"))),
                "dashboard_page": _as_text(metric.get("dashboard_page")),
                "sort_order": index,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _presentation_storyline_table(storyline: Mapping[str, Any]) -> pl.DataFrame:
    rows = []
    for chapter in storyline.get("chapters", []):
        chapter_order = _to_int(chapter.get("chapter_order"))
        page_name = _as_text(chapter.get("page_name"))
        page_id = _slug(page_name)
        rows.append(
            {
                "chapter_id": f"chapter_{chapter_order:02d}" if chapter_order is not None else page_id,
                "page_id": page_id,
                "chapter_order": chapter_order,
                "title": page_name,
                "key_message": _as_text(chapter.get("key_message")),
                "purpose": _as_text(chapter.get("purpose")),
                "presenter_notes": " ".join(chapter.get("expected_presenter_talking_points") or []),
                "supporting_visuals": "; ".join(chapter.get("visuals_required") or []),
                "interactions": "; ".join(chapter.get("interactions_required") or []),
                "required_datasets": "; ".join(chapter.get("data_required") or []),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _executive_insights(
    tribe_deep_dive: pl.DataFrame,
    customer_coverage_summary: pl.DataFrame,
    remaining_customer_segments: pl.DataFrame,
) -> pl.DataFrame:
    rows = []
    if not tribe_deep_dive.is_empty():
        top_revenue = tribe_deep_dive.sort("total_revenue", descending=True).row(0, named=True)
        strongest = tribe_deep_dive.sort("jitter_label_recovery_accuracy", descending=True).row(0, named=True)
        rows.extend(
            [
                {
                    "insight_id": "insight_top_revenue_tribe",
                    "sort_order": 1,
                    "title": "A small set of product-first tribes carries measurable revenue.",
                    "key_message": f"{_as_text(top_revenue.get('business_ready_name'))} is the highest-revenue retained tribe.",
                    "evidence": f"Revenue={_fmt_number(top_revenue.get('total_revenue'))}; customers={_fmt_number(top_revenue.get('assigned_customers'))}.",
                    "recommended_next_step": _as_text(top_revenue.get("recommended_action_summary") or "Prioritize tribe-specific offer testing with holdouts."),
                    "limitation": "Revenue is observed transaction value, not modeled incremental impact.",
                    "source_files": "tribe_deep_dive",
                },
                {
                    "insight_id": "insight_stability",
                    "sort_order": 2,
                    "title": "Readiness is explicit, not assumed.",
                    "key_message": f"{_as_text(strongest.get('business_ready_name'))} has the strongest jitter recovery among retained tribes.",
                    "evidence": f"Jitter recovery={_fmt_number(strongest.get('jitter_label_recovery_accuracy'))}; mean confidence={_fmt_number(strongest.get('mean_assignment_confidence'))}.",
                    "recommended_next_step": "Use readiness and caveat fields in all executive tribe cards.",
                    "limitation": "Review tribes should not be activated without additional business validation.",
                    "source_files": "tribe_deep_dive",
                },
            ]
        )
    bridge = _coverage_row(customer_coverage_summary, "bridge_customers")
    if bridge:
        rows.append(
            {
                "insight_id": "insight_remaining_customers",
                "sort_order": 3,
                "title": "Remaining customers are explained, not forced.",
                "key_message": "The largest remaining group bridges multiple tribes and should not receive exclusive tribe messaging.",
                "evidence": f"Bridge customers={_fmt_number(bridge.get('customers'))}; share={_fmt_number(bridge.get('customer_share_pct'))}%.",
                "recommended_next_step": "Use mission-led tests and nearest-tribe affinity only for cautious expansion audiences.",
                "limitation": "Remaining groups are descriptive coverage groups, not hard tribe memberships.",
                "source_files": "customer_coverage_summary; remaining_customer_assignment",
            }
        )
    if not remaining_customer_segments.is_empty() and "targetability" in remaining_customer_segments.columns:
        high_target = remaining_customer_segments.filter(pl.col("targetability") == "high")
        if not high_target.is_empty():
            row = high_target.sort("customer_count", descending=True).row(0, named=True)
            rows.append(
                {
                    "insight_id": "insight_high_target_remaining",
                    "sort_order": 4,
                    "title": "Some non-tribe customers are still targetable.",
                    "key_message": f"{_as_text(row.get('segment_name'))} can be used as a measured expansion audience.",
                    "evidence": f"Customers={_fmt_number(row.get('customer_count'))}; targetability={_as_text(row.get('targetability'))}.",
                    "recommended_next_step": _as_text(row.get("recommended_action")),
                    "limitation": "Use holdouts and avoid reporting these customers as hard tribe members.",
                    "source_files": "remaining_customer_segments",
                }
            )
    rows.append(
        {
            "insight_id": "insight_operationalization",
            "sort_order": 5,
            "title": "Operationalize with stable IDs and reviewed business names.",
            "key_message": "Use tribe_id and technical names for traceability, and reviewed business names for stakeholder storytelling.",
            "evidence": "Stage 8 publishes both recommended_technical_name and recommended_business_name for all 22 tribes.",
            "recommended_next_step": "Freeze display names before the final dashboard handoff and keep evidence/rationale visible in tribe detail pages.",
            "limitation": "Business names are interpretation layers, not clustering features.",
            "source_files": "tribe_name_proposals; tribe_deep_dive",
        }
    )
    return pl.DataFrame(rows, infer_schema_length=None)


def _model_metadata(
    cfg: PipelineConfig,
    inputs: Mapping[str, Path],
    stage68_manifest: Mapping[str, Any],
    stage7_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    stage6_merged = _first_record(inputs["stage6_merged_checks"])
    stage6_umap = _first_record(inputs["stage6_umap_checks"])
    pca = _first_record(inputs["stage6_pca_summary"])
    stage3 = _first_record(inputs["stage3_embedding_validation"])
    stage6_readiness = _read_csv(inputs["stage6_readiness"])
    status_counts = {}
    if not stage6_readiness.is_empty() and "profile_readiness" in stage6_readiness.columns:
        status_counts = {
            row["profile_readiness"]: row["len"]
            for row in stage6_readiness.group_by("profile_readiness").len().iter_rows(named=True)
        }
    return {
        "stage": "8_backend_model_metadata",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "mode": cfg.mode,
        "raw_inputs": {
            "prepared_transactions": str(inputs["prepared_transactions"]),
            "customer_kpis": str(inputs["customer_kpis"]),
            "product_embeddings": str(inputs["product_embeddings"]),
        },
        "official_signal_contract": {
            "feature_set_for_selection": cfg.get("modeling.feature_set_for_selection", "embeddings_only"),
            "spend_used_for_clustering": False,
            "demographics_used_for_clustering": False,
            "spend_usage": "profiling and business interpretation only",
        },
        "item2vec": {
            "vector_size": cfg.get("word2vec.vector_size"),
            "window": cfg.get("word2vec.window"),
            "min_count": cfg.get("word2vec.min_count"),
            "epochs": cfg.get("word2vec.epochs"),
            "negative": cfg.get("word2vec.negative"),
            "sample": cfg.get("word2vec.sample"),
        },
        "customer_vectors": {
            "weight_strategy": cfg.get("customer_embeddings.weight_strategy"),
            "quantity_transform": cfg.get("customer_embeddings.quantity_transform"),
            "normalize_vectors": cfg.get("customer_embeddings.normalize_vectors"),
            "recency_weighting": cfg.get("customer_embeddings.recency_weighting"),
            "frequency_weighting": cfg.get("customer_embeddings.frequency_weighting"),
        },
        "pca_for_umap": pca,
        "umap": {
            **stage6_umap,
            "configured": cfg.get("official_model_suite.umap_hdbscan.umap", {}),
        },
        "hdbscan_three_stage": {
            "model_name": cfg.get("official_model_suite.three_stage_hdbscan.model_name"),
            "variant_prefix": cfg.get("official_model_suite.three_stage_hdbscan.variant_prefix"),
            "stage1": cfg.get("official_model_suite.umap_hdbscan.hdbscan", {}),
            "stage2": cfg.get("official_model_suite.two_stage_hdbscan.second_stage_hdbscan", {}),
            "stage3": cfg.get("official_model_suite.three_stage_hdbscan.third_stage_hdbscan", {}),
            "lift_filter": cfg.get("official_model_suite.three_stage_hdbscan.lift_filter", {}),
        },
        "validation_summary": {
            "stage6_merged_checks": stage6_merged,
            "stage6_readiness_status_counts": status_counts,
            "stage3_embedding_guardrail": stage3.get("guardrail_status") or stage3.get("check_status"),
            "stage3_guardrail_caveat": stage3,
        },
        "promotion_policy": stage7_manifest.get("promotion_policy", {}),
        "stage68_row_counts": stage68_manifest.get("row_counts", {}),
        "stage68_outputs": stage68_manifest.get("outputs", {}),
        "stage7_outputs": {
            "primary": stage7_manifest.get("primary_outputs", {}),
            "supporting": stage7_manifest.get("supporting_outputs", {}),
        },
    }


def _pipeline_lineage(inputs: Mapping[str, Path], outputs: Mapping[str, Path], audit: pl.DataFrame) -> dict[str, Any]:
    products = []
    for name, path in outputs.items():
        if name == "root":
            continue
        products.append(
            {
                "product": name,
                "path": str(path),
                "exists": path.exists(),
                "format": path.suffix.lower().lstrip("."),
                "source_stages": _source_stages_for_product(name),
            }
        )
    return {
        "stage": "8_pipeline_lineage",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "input_audit": audit.to_dicts(),
        "products": products,
        "rule": "Dashboard products consume structured tables, parquet, and JSON only; Markdown is human reference, not a data source.",
        "upstream_stage_boundaries": [
            "Stage 6.8 owns prepared evidence scans and per-tribe exports.",
            "Stage 7 owns interpretation, coverage, action, and handoff semantics.",
            "Stage 8 owns dashboard-ready normalization, metadata, and presentation interactivity.",
        ],
    }


def _model_cards(model_metadata: Mapping[str, Any]) -> pl.DataFrame:
    rows = [
        {
            "model_name": "Item2Vec product embeddings",
            "model_family": "Word2Vec / Item2Vec",
            "purpose": "Learn product co-purchase geometry from basket sentences.",
            "input_dataset": "prepared_transactions",
            "output_dataset": "product_embeddings",
            "hyperparameters": json.dumps(model_metadata.get("item2vec", {}), ensure_ascii=False),
            "metrics": "",
            "status": "completed",
        },
        {
            "model_name": "Customer embedding aggregation",
            "model_family": "Vector aggregation",
            "purpose": "Aggregate product vectors into product-first customer behavior vectors.",
            "input_dataset": "product_embeddings; prepared_transactions",
            "output_dataset": "customer_embeddings",
            "hyperparameters": json.dumps(model_metadata.get("customer_vectors", {}), ensure_ascii=False, default=str),
            "metrics": "",
            "status": "completed",
        },
        {
            "model_name": "PCA for UMAP",
            "model_family": "Dimensionality reduction",
            "purpose": "Pre-reduce 128-dimensional customer features to 64 components before UMAP.",
            "input_dataset": "feature_set_embeddings_only",
            "output_dataset": "feature_set_pca_for_umap",
            "hyperparameters": json.dumps(model_metadata.get("pca_for_umap", {}), ensure_ascii=False, default=str),
            "metrics": f"retained_variance_pct={_json_value((model_metadata.get('pca_for_umap') or {}).get('retained_variance_pct'))}",
            "status": "completed",
        },
        {
            "model_name": "UMAP customer representation",
            "model_family": "Manifold learning",
            "purpose": "Create 20-dimensional customer topology for density clustering and dashboard projection.",
            "input_dataset": "feature_set_pca_for_umap",
            "output_dataset": "feature_set_umap_pca64_u20_n75",
            "hyperparameters": json.dumps((model_metadata.get("umap") or {}).get("configured", {}), ensure_ascii=False, default=str),
            "metrics": f"finite_pct={(model_metadata.get('umap') or {}).get('finite_pct')}; rows={(model_metadata.get('umap') or {}).get('umap_rows')}",
            "status": "completed",
        },
        {
            "model_name": "Three-pass HDBSCAN with lift filtering",
            "model_family": "Density clustering",
            "purpose": "Discover retained hard product-first tribes while keeping ambiguous customers as noise.",
            "input_dataset": "UMAP representation; product lift evidence",
            "output_dataset": "cluster_assignments_model_e_three_stage_hdbscan_lift_core",
            "hyperparameters": json.dumps(model_metadata.get("hdbscan_three_stage", {}), ensure_ascii=False, default=str),
            "metrics": json.dumps((model_metadata.get("validation_summary") or {}).get("stage6_merged_checks", {}), ensure_ascii=False, default=str),
            "status": "completed",
        },
        {
            "model_name": "Stage 7 promotion and action semantics",
            "model_family": "Business rule layer",
            "purpose": "Promote final tribes, hold review tribes, define actions, and explain remaining customers.",
            "input_dataset": "Stage 6.8 evidence",
            "output_dataset": "Stage 7 final handoff",
            "hyperparameters": json.dumps(model_metadata.get("promotion_policy", {}), ensure_ascii=False, default=str),
            "metrics": json.dumps((model_metadata.get("validation_summary") or {}).get("stage6_readiness_status_counts", {}), ensure_ascii=False, default=str),
            "status": "completed",
        },
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def _pipeline_stages(lineage: Mapping[str, Any]) -> pl.DataFrame:
    boundaries = lineage.get("upstream_stage_boundaries", [])
    rows = [
        ("stage1", "Basket sentences", "Create product-token basket sentences.", "raw transactions", "basket sentences", "Stage 2"),
        ("stage2", "Item2Vec product embeddings", "Train product co-purchase embeddings.", "basket sentences", "product embeddings", "Stage 4"),
        ("stage4", "Customer vectors", "Aggregate product embeddings to customer vectors.", "product embeddings; transactions", "customer embeddings", "Stage 6"),
        ("stage6", "UMAP + HDBSCAN", "Build PCA/UMAP representations and retained hard tribes.", "customer embeddings", "cluster assignments; evidence", "Stage 6.8"),
        ("stage6_8", "Evidence publishing", "Publish product/category/remaining-customer evidence.", "Stage 6 model outputs", "structured evidence", "Stage 7"),
        ("stage7", "Business semantics", "Name, validate, promote, and action the tribes.", "Stage 6.8 evidence", "business handoff tables", "Stage 8"),
        ("stage8", "Dashboard semantic pack", "Publish renderer-ready final dashboard data contracts.", "Stage 6.8 and Stage 7 structured outputs", "Stage 8 artifacts", "Dashboard"),
    ]
    return pl.DataFrame(
        [
            {
                "stage_id": stage_id,
                "stage_name": name,
                "purpose": purpose,
                "inputs": inputs,
                "outputs": outputs,
                "dependencies": dependency,
                "row_counts": "",
                "artifact_paths": "; ".join(
                    item.get("path", "")
                    for item in lineage.get("products", [])
                    if stage_id in item.get("source_stages", [])
                ),
                "boundary_note": " ".join(boundaries),
            }
            for stage_id, name, purpose, inputs, outputs, dependency in rows
        ],
        infer_schema_length=None,
    )


def _feature_catalog(cfg: PipelineConfig, inputs: Mapping[str, Path], model_metadata: Mapping[str, Any]) -> pl.DataFrame:
    rows = [
        {
            "feature_name": "product_embeddings",
            "feature_group": "product_vector",
            "description": "Dense Item2Vec product co-purchase vector.",
            "source": str(inputs["product_embeddings"]),
            "transformation": json.dumps(model_metadata.get("item2vec", {}), ensure_ascii=False, default=str),
            "used_in_model": True,
        },
        {
            "feature_name": "customer_embeddings",
            "feature_group": "customer_vector",
            "description": "Customer vector aggregated from product embeddings with quantity, recency, and frequency weighting.",
            "source": str(inputs["customer_embeddings"]),
            "transformation": json.dumps(model_metadata.get("customer_vectors", {}), ensure_ascii=False, default=str),
            "used_in_model": True,
        },
        {
            "feature_name": "feature_set_embeddings_only",
            "feature_group": "official_selection_feature_set",
            "description": "Official product-first feature set used for selection; spend/demographics excluded.",
            "source": str(cfg.outputs / "features" / "feature_set_embeddings_only.parquet"),
            "transformation": "product-first embeddings only",
            "used_in_model": True,
        },
        {
            "feature_name": "feature_set_pca_for_umap",
            "feature_group": "dimensionality_reduction",
            "description": "64 PCA components used to denoise and speed UMAP.",
            "source": str(inputs["pca_features"]),
            "transformation": json.dumps(model_metadata.get("pca_for_umap", {}), ensure_ascii=False, default=str),
            "used_in_model": True,
        },
        {
            "feature_name": "feature_set_umap_pca64_u20_n75",
            "feature_group": "manifold_representation",
            "description": "20-dimensional UMAP representation used for HDBSCAN and dashboard coordinates.",
            "source": str(inputs["umap_features"]),
            "transformation": json.dumps((model_metadata.get("umap") or {}).get("configured", {}), ensure_ascii=False, default=str),
            "used_in_model": True,
        },
        {
            "feature_name": "customer_behavior_features",
            "feature_group": "profiling_only",
            "description": "Spend, frequency, recency, promotion, product breadth, and basket metrics used for interpretation only.",
            "source": str(inputs["behavior_features"]),
            "transformation": "post-clustering profiling; not used for clustering selection",
            "used_in_model": False,
        },
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def _data_quality_metrics(audit: pl.DataFrame, validation_metrics: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for row in audit.iter_rows(named=True):
        rows.append(
            {
                "quality_metric_id": f"input_exists_{row.get('artifact')}",
                "area": "input_artifact",
                "metric": "exists",
                "value": _as_text(row.get("exists")),
                "status": "pass" if row.get("exists") or not row.get("required") else "fail",
                "source_path": _as_text(row.get("path")),
                "dashboard_message": f"{row.get('artifact')} exists={row.get('exists')}",
            }
        )
    if not validation_metrics.is_empty():
        for row in validation_metrics.filter(pl.col("status").is_not_null()).head(200).iter_rows(named=True):
            rows.append(
                {
                    "quality_metric_id": f"{row.get('source_stage')}_{row.get('source_row')}_{row.get('metric')}",
                    "area": _as_text(row.get("source_stage")),
                    "metric": _as_text(row.get("metric")),
                    "value": _as_text(row.get("value")),
                    "status": _normalize_status(row.get("status")),
                    "source_path": _as_text(row.get("source_path")),
                    "dashboard_message": f"{row.get('metric')}: {row.get('value')}",
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def _validation_metrics(inputs: Mapping[str, Path]) -> pl.DataFrame:
    sources = {
        "stage3_embedding_validation": inputs["stage3_embedding_validation"],
        "stage6_pca_summary": inputs["stage6_pca_summary"],
        "stage6_umap_checks": inputs["stage6_umap_checks"],
        "stage6_merged_checks": inputs["stage6_merged_checks"],
        "stage6_quality": inputs["stage6_quality"],
        "stage6_readiness": inputs["stage6_readiness"],
        "stage7_stakeholder_readiness": inputs["stage7_stakeholder_readiness"],
    }
    rows = []
    for source_name, path in sources.items():
        frame = _read_csv(path)
        if frame.is_empty():
            continue
        for idx, record in enumerate(frame.head(200).iter_rows(named=True), start=1):
            if "metric" in record and "value" in record:
                rows.append(
                    {
                        "source_stage": source_name,
                        "source_row": idx,
                        "metric": _as_text(record.get("metric")),
                        "value": _as_text(record.get("value")),
                        "status": _as_text(record.get("status") or record.get("check_status") or record.get("statistical_result")),
                        "ideal_value_range": _as_text(record.get("ideal_value_range") or record.get("ideal_range")),
                        "source_path": str(path),
                    }
                )
            else:
                for key, value in record.items():
                    if _looks_metric_value(value):
                        rows.append(
                            {
                                "source_stage": source_name,
                                "source_row": idx,
                                "metric": key,
                                "value": _as_text(value),
                                "status": _as_text(record.get("status") or record.get("check_status")),
                                "ideal_value_range": "",
                                "source_path": str(path),
                            }
                        )
    schema = {
        "source_stage": pl.Utf8,
        "source_row": pl.Int64,
        "metric": pl.Utf8,
        "value": pl.Utf8,
        "status": pl.Utf8,
        "ideal_value_range": pl.Utf8,
        "source_path": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema, orient="row") if rows else pl.DataFrame(schema=schema)


def _visualization_specs() -> pl.DataFrame:
    specs = [
        ("2d_umap", "Customer Landscape", "rel_fact_embedding_2d; rel_fact_customer_assignment; rel_dim_tribe; rel_dim_coverage_group", "cliente,x,y plus assignment/dimension joins", "ready", "Join coordinates to assignment and dimension tables for labels/filters"),
        ("3d_umap", "Customer Landscape", "rel_fact_embedding_3d_sample; rel_fact_customer_assignment; rel_dim_tribe; rel_dim_coverage_group; rel_dim_remaining_segment", "cliente,x,y,z plus assignment/dimension joins", "ready", "Use sampled coordinates for live demo"),
        ("3d_pca", "Customer Landscape", "rel_fact_embedding_3d_pca_sample; rel_fact_customer_assignment; rel_dim_tribe; rel_dim_coverage_group", "cliente,x,y,z plus assignment/dimension joins", "ready_if_file_nonempty", "PCA is supporting/technical, not the default customer landscape"),
        ("tribe_centroids", "Customer Landscape", "rel_fact_embedding_centroid; rel_dim_tribe; rel_dim_coverage_group; rel_dim_remaining_segment", "coverage_group,tribe_id,remaining_segment_id,x,y,z,customers", "ready", "Join centroid IDs to dimensions for labels"),
        ("cluster_boundaries", "Customer Landscape", "", "", "optional_gap", "Hull/density surfaces are not required for MVP and remain optional"),
        ("product_affinity_network", "Product Affinity Explorer", "rel_fact_tribe_product_affinity; rel_dim_product; rel_dim_tribe", "tribe_id,product_id,lift_vs_rest,reach_pct,q_value,rank", "ready", "Build network edges from normalized tribe-product facts"),
        ("tribe_similarity_network", "Tribe Explorer", "rel_bridge_tribe_similarity; rel_dim_tribe", "tribe_a_id,tribe_b_id,relationship_score,campaign_guidance", "ready", "Join both tribe IDs to rel_dim_tribe"),
        ("coverage_funnel", "Customer Coverage & Remaining Customers", "rel_fact_coverage_group_metrics; rel_dim_coverage_group", "coverage_group,customers,customer_share_pct,revenue,revenue_share_pct", "ready", "Sort and label through rel_dim_coverage_group"),
        ("opportunity_matrix", "Segment Activation Center", "rel_dim_action_target; rel_fact_segment_action; rel_fact_tribe_metrics; rel_fact_remaining_segment_metrics", "target_id,priority,customers,revenue/action text", "ready", "Build target matrix from action target plus tribe/remaining metrics"),
        ("segment_comparison_radar", "Tribe Explorer", "rel_fact_tribe_metrics; rel_fact_remaining_segment_metrics; rel_dim_tribe; rel_dim_remaining_segment", "segment id plus comparable spend/frequency/basket metrics", "ready", "Normalize values in renderer if radial scaling is desired"),
        ("revenue_value_charts", "Executive Overview", "rel_fact_tribe_metrics; rel_fact_coverage_group_metrics; rel_dim_tribe; rel_dim_coverage_group", "tribe/coverage ids plus revenue, customer counts, shares", "ready", "Use dimension joins for labels"),
    ]
    return pl.DataFrame(
        [
            {
                "visual_id": visual_id,
                "dashboard_page": page,
                "required_data_product": product,
                "required_fields": fields,
                "readiness_status": status,
                "dashboard_guidance": guidance,
            }
            for visual_id, page, product, fields, status, guidance in specs
        ],
        infer_schema_length=None,
    )


def _dashboard_page_contracts() -> pl.DataFrame:
    rows = [
        (
            "executive_overview",
            "Executive Overview",
            "rel_fact_executive_metric; rel_fact_coverage_group_metrics; rel_dim_coverage_group; rel_fact_tribe_metrics; rel_dim_tribe; rel_dim_executive_insight; rel_dim_story_chapter",
            "total_customers; final/review/non-tribe counts; coverage pct; revenue KPIs; top insights; recommendations",
            "fully_supported",
            "Join facts to dimensions for labels; no repeated wide tables are part of the published contract.",
        ),
        (
            "project_data_foundation",
            "Project & Data Foundation",
            "rel_dim_pipeline_stage; rel_dim_feature_group; rel_fact_data_quality_metric; artifact_manifest; relational_manifest; relational_schema",
            "source inventory; row counts; quality metrics; feature groups",
            "fully_supported",
            "Use relational_manifest for canonical joins; transaction period is available through source metadata/model lineage.",
        ),
        (
            "backend_modeling_pipeline",
            "Backend & Modeling Pipeline",
            "rel_dim_model_card; rel_dim_pipeline_stage; rel_fact_validation_metric; rel_fact_data_quality_metric; artifact_manifest; relational_integrity_report",
            "PCA, UMAP, HDBSCAN, hyperparameters, validation, stability, promotion/noise rules",
            "fully_supported",
            "Backend metadata is flattened into relational tables; relational_integrity_report verifies joins.",
        ),
        (
            "customer_landscape",
            "Customer Landscape",
            "rel_fact_embedding_2d; rel_fact_embedding_3d_sample; rel_fact_embedding_3d_pca_sample; rel_fact_embedding_centroid; rel_fact_customer_assignment; rel_fact_customer_value_behavior; rel_dim_tribe; rel_dim_coverage_group; rel_dim_remaining_segment; rel_dim_visualization",
            "coordinates; customer id; tribe id/name/status; confidence; value; hover/filter fields",
            "fully_supported",
            "Use sampled 3D relational coordinates for live rendering and join to dimensions for labels.",
        ),
        (
            "tribe_explorer",
            "Tribe Explorer",
            "rel_dim_tribe; rel_dim_tribe_name_proposal; rel_fact_tribe_metrics; rel_fact_tribe_profile_text; rel_fact_tribe_product_affinity; rel_fact_tribe_category_affinity; rel_bridge_tribe_similarity; rel_fact_segment_action",
            "names, profiles, counts, revenue, confidence, jitter, products, actions, similar tribes",
            "fully_supported",
            "Use rel_dim_tribe for canonical names; facts provide metrics/evidence/actions.",
        ),
        (
            "product_affinity_explorer",
            "Product Affinity Explorer",
            "rel_dim_tribe; rel_dim_product; rel_dim_category; rel_fact_tribe_product_affinity; rel_fact_tribe_category_affinity",
            "product/category names, lift, reach, q-value, actionability, ranking, network edges",
            "fully_supported",
            "Product names live in rel_dim_product; product revenue by SKU is not published.",
        ),
        (
            "customer_coverage_remaining",
            "Customer Coverage & Remaining Customers",
            "rel_dim_coverage_group; rel_dim_remaining_segment; rel_fact_coverage_group_metrics; rel_fact_remaining_segment_metrics; rel_fact_customer_assignment; rel_fact_customer_nearest_tribe",
            "full population accounting, remaining groups, descriptions, treatments",
            "fully_supported",
            "Customer lookup is a normalized join across customer facts and dimensions.",
        ),
        (
            "segment_activation_center",
            "Segment Activation Center",
            "rel_dim_action_target; rel_fact_segment_action; rel_dim_tribe; rel_dim_remaining_segment; rel_fact_tribe_product_affinity",
            "actions, rationale, target audience, expected impact, priority, suppression, KPIs",
            "fully_supported",
            "Actions are keyed by target_id in the relational layer; activation channel is embedded in action text.",
        ),
        (
            "executive_final_recommendation",
            "Executive Insights & Final Recommendation",
            "rel_dim_executive_insight; rel_dim_story_chapter; rel_fact_executive_metric; relational_integrity_report; readiness_csv; completeness_audit",
            "ordered insights, evidence, next steps, limitations, final conclusion",
            "fully_supported",
            "Use relational insight and metric tables as the renderer source.",
        ),
    ]
    return pl.DataFrame(
        [
            {
                "page_id": page_id,
                "page_title": title,
                "required_data_products": products,
                "required_fields": fields,
                "coverage_status": status,
                "dashboard_logic_boundary": boundary,
            }
            for page_id, title, products, fields, status, boundary in rows
        ],
        infer_schema_length=None,
    )


def _missing_file_register() -> pl.DataFrame:
    rows = [
        {
            "proposed_file_name": "cluster_boundaries_or_density_surfaces_<mode>.parquet",
            "purpose": "Optional polygon/hull/density overlays for the customer landscape.",
            "grain": "one row per boundary point or density cell",
            "required_fields": "boundary_id,coverage_group,tribe_id,x,y,z(optional),density(optional)",
            "source_inputs": "embedding_2d or embedding_3d",
            "stage8_substage": "8.4 embedding preparation",
            "priority": "low",
            "dashboard_pages_blocked": "None; optional landscape polish only",
            "current_status": "optional_gap",
        },
        {
            "proposed_file_name": "explicit_activation_channels_<mode>.parquet",
            "purpose": "Separate channel enum from the narrative action text.",
            "grain": "one row per segment/action/channel",
            "required_fields": "segment_id,action_id,channel,channel_priority,channel_rationale",
            "source_inputs": "stage7 action playbook",
            "stage8_substage": "8.3 dashboard dataset generation",
            "priority": "medium",
            "dashboard_pages_blocked": "Segment Activation Center only if channel filters are required",
            "current_status": "not_blocking_action_text_exists",
        },
        {
            "proposed_file_name": "product_revenue_by_tribe_<mode>.parquet",
            "purpose": "Revenue by product and tribe for SKU-level commercial sizing.",
            "grain": "one row per tribe-product",
            "required_fields": "tribe_id,product_id,revenue,units,customers,lift,rank",
            "source_inputs": "Stage 6.8 tribe transaction exports",
            "stage8_substage": "8.3 dashboard dataset generation",
            "priority": "medium",
            "dashboard_pages_blocked": "Product Affinity Explorer only if SKU revenue sizing is required",
            "current_status": "not_blocking_lift_reach_actionability_exist",
        },
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def _is_public_stage8_artifact(name: str) -> bool:
    if name == "root":
        return True
    if name.startswith("rel_") or name.startswith("relational"):
        return True
    return name in {
        "artifact_manifest",
        "dashboard_manifest",
        "readme",
        "readiness_csv",
        "readiness_md",
        "completeness_audit",
        "completeness_audit_json",
    }


def _public_artifact_paths(paths: Mapping[str, Path]) -> dict[str, Path]:
    return {name: path for name, path in paths.items() if _is_public_stage8_artifact(name)}


def _cleanup_stage8_output_folder(paths: Mapping[str, Path]) -> None:
    root = paths["root"]
    if not root.exists():
        return
    public_names = {path.name for name, path in paths.items() if _is_public_stage8_artifact(name) and name != "root"}
    for path in root.iterdir():
        if path.is_file() and path.name not in public_names:
            path.unlink()


def _artifact_manifest(paths: Mapping[str, Path]) -> pl.DataFrame:
    rows = []
    for name, path in paths.items():
        if name == "root" or not _is_public_stage8_artifact(name):
            continue
        rows.append(
            {
                "artifact_name": name,
                "file_name": path.name,
                "format": path.suffix.lower().lstrip(".") or "directory",
                "grain": _artifact_grain(name),
                "row_count": _row_count(path),
                "primary_key": _artifact_primary_key(name),
                "foreign_keys": _artifact_foreign_keys(name),
                "purpose": _artifact_purpose(name),
                "dashboard_pages_supported": "; ".join(_pages_for_product(name)),
                "audience": _artifact_audience(name),
                "classification": _artifact_classification(name),
                "path": str(path),
                "exists": path.exists(),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _relational_integrity_report(relational_tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    dim_tribe = relational_tables.get("rel_dim_tribe", pl.DataFrame())
    dim_coverage_group = relational_tables.get("rel_dim_coverage_group", pl.DataFrame())
    dim_remaining_segment = relational_tables.get("rel_dim_remaining_segment", pl.DataFrame())
    dim_product = relational_tables.get("rel_dim_product", pl.DataFrame())
    dim_category = relational_tables.get("rel_dim_category", pl.DataFrame())
    dim_action_target = relational_tables.get("rel_dim_action_target", pl.DataFrame())
    fact_customer_assignment = relational_tables.get("rel_fact_customer_assignment", pl.DataFrame())
    fact_customer_value_behavior = relational_tables.get("rel_fact_customer_value_behavior", pl.DataFrame())
    fact_customer_nearest_tribe = relational_tables.get("rel_fact_customer_nearest_tribe", pl.DataFrame())
    fact_tribe_metrics = relational_tables.get("rel_fact_tribe_metrics", pl.DataFrame())
    fact_coverage_group_metrics = relational_tables.get("rel_fact_coverage_group_metrics", pl.DataFrame())
    fact_remaining_segment_metrics = relational_tables.get("rel_fact_remaining_segment_metrics", pl.DataFrame())
    fact_tribe_product = relational_tables.get("rel_fact_tribe_product_affinity", pl.DataFrame())
    fact_tribe_category = relational_tables.get("rel_fact_tribe_category_affinity", pl.DataFrame())
    fact_segment_action = relational_tables.get("rel_fact_segment_action", pl.DataFrame())
    bridge_similarity = relational_tables.get("rel_bridge_tribe_similarity", pl.DataFrame())
    embedding_tables = {
        name: table
        for name, table in relational_tables.items()
        if name.startswith("rel_fact_embedding_") and name != "rel_fact_embedding_centroid"
    }
    rows = []

    def add(check_id: str, area: str, severity: str, passed: bool, details: str) -> None:
        rows.append(_check(check_id, area, severity, passed, details))

    tribe_keys = _key_set(dim_tribe, "tribe_id")
    coverage_keys = _key_set(dim_coverage_group, "coverage_group")
    remaining_segment_keys = _key_set(dim_remaining_segment, "segment_id")
    product_keys = _key_set(dim_product, "product_id")
    category_keys = _key_set(dim_category, "category")
    action_target_keys = _key_set(dim_action_target, "target_id")
    customer_keys = _key_set(fact_customer_assignment, "cliente")

    add("rel_dim_tribe_pk_unique", "relational_model", "critical", _unique_key(dim_tribe, "tribe_id"), _key_details(dim_tribe, "tribe_id"))
    add("rel_dim_product_pk_unique", "relational_model", "critical", _unique_key(dim_product, "product_id"), _key_details(dim_product, "product_id"))
    add("rel_fact_customer_assignment_pk_unique", "relational_model", "critical", _unique_key(fact_customer_assignment, "cliente"), _key_details(fact_customer_assignment, "cliente"))
    add("rel_fact_customer_value_behavior_pk_unique", "relational_model", "critical", _unique_key(fact_customer_value_behavior, "cliente"), _key_details(fact_customer_value_behavior, "cliente"))
    add("rel_fact_customer_nearest_tribe_pk_unique", "relational_model", "critical", _unique_key(fact_customer_nearest_tribe, "cliente"), _key_details(fact_customer_nearest_tribe, "cliente"))

    add("rel_tribe_metrics_fk_tribe", "foreign_keys", "critical", _subset_of(fact_tribe_metrics, "tribe_id", tribe_keys), _fk_details(fact_tribe_metrics, "tribe_id", tribe_keys))
    add("rel_customer_assignment_fk_tribe", "foreign_keys", "critical", _subset_of(fact_customer_assignment, "tribe_id", tribe_keys), _fk_details(fact_customer_assignment, "tribe_id", tribe_keys))
    add("rel_customer_assignment_fk_coverage", "foreign_keys", "critical", _subset_of(fact_customer_assignment, "coverage_group", coverage_keys), _fk_details(fact_customer_assignment, "coverage_group", coverage_keys))
    add("rel_customer_assignment_fk_remaining_segment", "foreign_keys", "critical", _subset_of(fact_customer_assignment, "remaining_segment_id", remaining_segment_keys), _fk_details(fact_customer_assignment, "remaining_segment_id", remaining_segment_keys))
    add("rel_customer_value_behavior_fk_cliente", "foreign_keys", "critical", _subset_of(fact_customer_value_behavior, "cliente", customer_keys), _fk_details(fact_customer_value_behavior, "cliente", customer_keys))
    add("rel_customer_nearest_top_fk_tribe", "foreign_keys", "critical", _subset_of(fact_customer_nearest_tribe, "top_tribe_id", tribe_keys), _fk_details(fact_customer_nearest_tribe, "top_tribe_id", tribe_keys))
    add("rel_customer_nearest_second_fk_tribe", "foreign_keys", "critical", _subset_of(fact_customer_nearest_tribe, "second_tribe_id", tribe_keys), _fk_details(fact_customer_nearest_tribe, "second_tribe_id", tribe_keys))
    add("rel_coverage_metrics_fk_coverage", "foreign_keys", "critical", _subset_of(fact_coverage_group_metrics, "coverage_group", coverage_keys), _fk_details(fact_coverage_group_metrics, "coverage_group", coverage_keys))
    add("rel_remaining_metrics_fk_segment", "foreign_keys", "critical", _subset_of(fact_remaining_segment_metrics, "segment_id", remaining_segment_keys), _fk_details(fact_remaining_segment_metrics, "segment_id", remaining_segment_keys))
    add("rel_tribe_product_fk_tribe", "foreign_keys", "critical", _subset_of(fact_tribe_product, "tribe_id", tribe_keys), _fk_details(fact_tribe_product, "tribe_id", tribe_keys))
    add("rel_tribe_product_fk_product", "foreign_keys", "critical", _subset_of(fact_tribe_product, "product_id", product_keys), _fk_details(fact_tribe_product, "product_id", product_keys))
    add("rel_tribe_category_fk_tribe", "foreign_keys", "critical", _subset_of(fact_tribe_category, "tribe_id", tribe_keys), _fk_details(fact_tribe_category, "tribe_id", tribe_keys))
    add("rel_tribe_category_fk_category", "foreign_keys", "critical", _subset_of(fact_tribe_category, "category", category_keys), _fk_details(fact_tribe_category, "category", category_keys))
    add("rel_segment_action_fk_target", "foreign_keys", "critical", _subset_of(fact_segment_action, "target_id", action_target_keys), _fk_details(fact_segment_action, "target_id", action_target_keys))
    add("rel_similarity_fk_tribe_a", "foreign_keys", "critical", _subset_of(bridge_similarity, "tribe_a_id", tribe_keys), _fk_details(bridge_similarity, "tribe_a_id", tribe_keys))
    add("rel_similarity_fk_tribe_b", "foreign_keys", "critical", _subset_of(bridge_similarity, "tribe_b_id", tribe_keys), _fk_details(bridge_similarity, "tribe_b_id", tribe_keys))

    for table_name, table in embedding_tables.items():
        if table.is_empty():
            add(f"{table_name}_fk_cliente", "foreign_keys", "critical", True, "optional embedding table is empty.")
            add(f"{table_name}_pk_unique", "relational_model", "critical", True, "optional embedding table is empty.")
            add(f"{table_name}_coordinates_finite", "relational_model", "critical", True, "optional embedding table is empty.")
            continue
        add(f"{table_name}_fk_cliente", "foreign_keys", "critical", _subset_of(table, "cliente", customer_keys), _fk_details(table, "cliente", customer_keys))
        add(f"{table_name}_pk_unique", "relational_model", "critical", _unique_key(table, "cliente"), _key_details(table, "cliente"))
        finite = _finite_coordinate_check(table)
        add(f"{table_name}_coordinates_finite", "relational_model", "critical", finite, f"rows={table.height}; finite={finite}")

    repeated_fact_label_columns = {
        table_name: sorted(
            col
            for col in table.columns
            if col in {"tribe_name", "business_name", "technical_name", "segment_name", "product_description", "category_label", "lookup_label", "hover_label"}
        )
        for table_name, table in relational_tables.items()
        if table_name.startswith("rel_fact_") or table_name.startswith("rel_bridge_")
    }
    offenders = {name: cols for name, cols in repeated_fact_label_columns.items() if cols}
    add("rel_fact_tables_do_not_repeat_dimension_labels", "separation_of_concerns", "critical", not offenders, _as_text(offenders) or "Fact and bridge tables carry keys/measures, not repeated labels.")

    return pl.DataFrame(rows, infer_schema_length=None)


def _relational_manifest(relational_tables: Mapping[str, pl.DataFrame], paths: Mapping[str, Path], *, cfg: PipelineConfig) -> dict[str, Any]:
    tables = []
    for name in sorted(relational_tables):
        path = paths[name]
        tables.append(
            {
                "table_name": name,
                "file_name": path.name,
                "path": str(path),
                "row_count": relational_tables[name].height,
                "primary_key": _relational_primary_key(name),
                "foreign_keys": _relational_foreign_keys(name),
                "canonical_owner": _relational_canonical_owner(name),
                "dashboard_use": _relational_dashboard_use(name),
                "columns": relational_tables[name].columns,
            }
        )
    return {
        "stage": "8_relational_semantic_model",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "mode": cfg.mode,
        "design_principle": "Dimensions own stable names/labels; facts and bridges carry IDs, measures, evidence, actions, and coordinates.",
        "relationship_model": "star_schema_parquet_with_sql_ddl",
        "schema_file": str(paths["relational_schema"]),
        "integrity_report": str(paths["relational_integrity_report"]),
        "canonical_display_rule": "Use rel_dim_tribe.display_name for executive labels and rel_dim_tribe.display_technical_name for technical labels. Do not copy tribe names into fact tables.",
        "tables": tables,
    }


def _relational_manifest_from_paths(paths: Mapping[str, Path], *, cfg: PipelineConfig) -> dict[str, Any]:
    tables = []
    for name, path in sorted(paths.items()):
        if not name.startswith("rel_"):
            continue
        tables.append(
            {
                "table_name": name,
                "file_name": path.name,
                "path": str(path),
                "row_count": _row_count(path),
                "primary_key": _relational_primary_key(name),
                "foreign_keys": _relational_foreign_keys(name),
                "canonical_owner": _relational_canonical_owner(name),
                "dashboard_use": _relational_dashboard_use(name),
                "columns": _schema_for_path(path),
            }
        )
    return {
        "stage": "8_relational_semantic_model",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "mode": cfg.mode,
        "design_principle": "Published Stage 8 artifacts are a normalized relational contract. Dimensions own names/labels; facts and bridges carry IDs, measures, evidence, actions, and coordinates.",
        "relationship_model": "star_schema_parquet_with_sql_ddl",
        "schema_file": str(paths["relational_schema"]),
        "integrity_report": str(paths["relational_integrity_report"]),
        "artifact_manifest": str(paths["artifact_manifest"]),
        "canonical_display_rule": "Use rel_dim_tribe.display_name for executive labels and rel_dim_tribe.display_technical_name for technical labels. Do not use any non-relational legacy table as a source of truth.",
        "tables": tables,
    }


def _relational_schema_sql() -> str:
    return """-- Stage 8 normalized relational model.
-- Import the matching rel_*.parquet files into these tables before applying
-- foreign-key constraints in a SQL database. Parquet remains the canonical
-- exchange format for the local dashboard.

CREATE TABLE rel_dim_coverage_group (
    coverage_group TEXT PRIMARY KEY,
    coverage_group_label TEXT,
    description TEXT,
    recommended_treatment TEXT,
    sort_order INTEGER
);

CREATE TABLE rel_dim_tribe (
    tribe_id INTEGER PRIMARY KEY,
    tribe_key TEXT UNIQUE,
    display_name TEXT,
    display_technical_name TEXT,
    technical_name TEXT,
    stage7_business_name TEXT,
    legacy_tribe_name TEXT,
    recommended_business_name TEXT,
    recommended_technical_name TEXT,
    proposal_status TEXT,
    business_name_readiness TEXT,
    proposal_confidence_label TEXT,
    promotion_decision TEXT,
    tribe_status TEXT,
    tribe_status_label TEXT,
    coverage_group TEXT REFERENCES rel_dim_coverage_group(coverage_group),
    membership_policy TEXT,
    validation_tier TEXT,
    business_confidence TEXT,
    stage6_profile_readiness TEXT,
    validation_blockers TEXT,
    readiness_caveat TEXT,
    recommended_use TEXT,
    is_final_tribe BOOLEAN,
    is_review_tribe BOOLEAN
);

CREATE TABLE rel_dim_remaining_segment (
    segment_id TEXT PRIMARY KEY,
    segment_name TEXT,
    product_theme_signal TEXT,
    targetability TEXT,
    likely_reason_for_no_hard_cluster TEXT,
    recommended_action TEXT,
    membership_policy TEXT,
    coverage_group TEXT REFERENCES rel_dim_coverage_group(coverage_group)
);

CREATE TABLE rel_dim_product (
    product_id TEXT PRIMARY KEY,
    product_description TEXT,
    category TEXT
);

CREATE TABLE rel_dim_category (
    category TEXT PRIMARY KEY,
    category_slug TEXT
);

CREATE TABLE rel_dim_action_target (
    target_id TEXT PRIMARY KEY,
    target_type TEXT,
    tribe_id INTEGER REFERENCES rel_dim_tribe(tribe_id),
    remaining_segment_id TEXT REFERENCES rel_dim_remaining_segment(segment_id),
    coverage_group TEXT REFERENCES rel_dim_coverage_group(coverage_group),
    promotion_decision TEXT,
    validation_tier TEXT,
    business_confidence TEXT,
    recommended_use TEXT
);

CREATE TABLE rel_fact_tribe_metrics (
    tribe_id INTEGER PRIMARY KEY REFERENCES rel_dim_tribe(tribe_id),
    hard_assigned_customers INTEGER,
    population_share_pct REAL,
    mean_assignment_confidence REAL,
    p10_assignment_confidence REAL,
    jitter_label_recovery_accuracy REAL,
    assigned_customers INTEGER,
    share_of_total_customers_pct REAL,
    share_of_hard_assigned_customers_pct REAL,
    total_revenue REAL,
    revenue_share_pct REAL,
    avg_revenue_per_customer REAL,
    median_revenue_per_customer REAL,
    avg_ticket_count REAL,
    avg_basket_value REAL,
    avg_items_per_basket REAL,
    avg_promo_share REAL,
    avg_unique_products REAL,
    avg_unique_sectors REAL,
    avg_recency_days REAL,
    avg_frequency_per_30d REAL
);

CREATE TABLE rel_fact_customer_assignment (
    cliente TEXT PRIMARY KEY,
    tribe_id INTEGER REFERENCES rel_dim_tribe(tribe_id),
    coverage_group TEXT REFERENCES rel_dim_coverage_group(coverage_group),
    remaining_segment_id TEXT REFERENCES rel_dim_remaining_segment(segment_id),
    assignment_status TEXT,
    model_name TEXT,
    model_variant TEXT,
    assignment_probability REAL,
    assignment_confidence_score REAL,
    assignment_confidence_type TEXT,
    assignment_source TEXT
);

CREATE TABLE rel_fact_customer_value_behavior (
    cliente TEXT PRIMARY KEY REFERENCES rel_fact_customer_assignment(cliente),
    ticket_count INTEGER,
    total_spend REAL,
    total_units INTEGER,
    avg_basket_value REAL,
    avg_items_per_basket REAL,
    promo_share REAL,
    unique_products INTEGER,
    unique_sectors INTEGER,
    recency_days INTEGER,
    frequency_per_30d REAL,
    customer_value REAL,
    revenue_tier TEXT
);

CREATE TABLE rel_fact_customer_nearest_tribe (
    cliente TEXT PRIMARY KEY REFERENCES rel_fact_customer_assignment(cliente),
    top_tribe_id INTEGER REFERENCES rel_dim_tribe(tribe_id),
    top_affinity_score REAL,
    second_tribe_id INTEGER REFERENCES rel_dim_tribe(tribe_id),
    second_affinity_score REAL,
    affinity_margin REAL,
    affinity_confidence_band TEXT,
    nearest_tribe_recommended_use TEXT
);

-- Product/category/action/similarity and embedding fact tables use the same
-- key rules documented in relational_manifest_<mode>.json.
"""


def _key_set(frame: pl.DataFrame, column: str) -> set[Any]:
    if frame.is_empty() or column not in frame.columns:
        return set()
    return set(frame[column].drop_nulls().to_list())


def _unique_key(frame: pl.DataFrame, column: str) -> bool:
    if frame.is_empty() or column not in frame.columns:
        return False
    return frame.height == frame[column].n_unique()


def _key_details(frame: pl.DataFrame, column: str) -> str:
    unique = frame[column].n_unique() if not frame.is_empty() and column in frame.columns else 0
    return f"rows={frame.height}; unique_{column}={unique}"


def _subset_of(frame: pl.DataFrame, column: str, allowed: set[Any]) -> bool:
    if frame.is_empty() or column not in frame.columns:
        return True
    values = set(frame[column].drop_nulls().to_list())
    return values.issubset(allowed)


def _fk_details(frame: pl.DataFrame, column: str, allowed: set[Any]) -> str:
    values = set(frame[column].drop_nulls().to_list()) if not frame.is_empty() and column in frame.columns else set()
    missing = sorted(values - allowed, key=lambda value: str(value))
    preview = missing[:10]
    return f"values={len(values)}; allowed={len(allowed)}; missing={preview}"


def _finite_coordinate_check(frame: pl.DataFrame) -> bool:
    if frame.is_empty():
        return True
    coord_cols = [col for col in ["x", "y", "z"] if col in frame.columns]
    if not coord_cols:
        return False
    return bool(frame.select(pl.all_horizontal([pl.col(col).is_finite().fill_null(False) for col in coord_cols]).all()).item())


def _relational_primary_key(name: str) -> str:
    mapping = {
        "rel_dim_tribe": "tribe_id",
        "rel_dim_tribe_name_proposal": "tribe_id",
        "rel_dim_coverage_group": "coverage_group",
        "rel_dim_remaining_segment": "segment_id",
        "rel_dim_product": "product_id",
        "rel_dim_category": "category",
        "rel_dim_action_target": "target_id",
        "rel_dim_executive_insight": "insight_id",
        "rel_dim_story_chapter": "chapter_id",
        "rel_dim_model_card": "model_name",
        "rel_dim_pipeline_stage": "stage_id",
        "rel_dim_feature_group": "feature_name",
        "rel_dim_dashboard_page": "page_id",
        "rel_dim_visualization": "visual_id",
        "rel_dim_optional_gap": "proposed_file_name",
        "rel_fact_executive_metric": "metric_id",
        "rel_fact_tribe_metrics": "tribe_id",
        "rel_fact_tribe_profile_text": "tribe_id",
        "rel_fact_coverage_group_metrics": "coverage_group",
        "rel_fact_remaining_segment_metrics": "segment_id",
        "rel_fact_validation_metric": "",
        "rel_fact_data_quality_metric": "",
        "rel_fact_customer_assignment": "cliente",
        "rel_fact_customer_value_behavior": "cliente",
        "rel_fact_customer_nearest_tribe": "cliente",
        "rel_fact_segment_action": "target_id",
        "rel_fact_embedding_2d": "cliente",
        "rel_fact_embedding_3d": "cliente",
        "rel_fact_embedding_3d_sample": "cliente",
        "rel_fact_embedding_3d_pca": "cliente",
        "rel_fact_embedding_3d_pca_sample": "cliente",
    }
    if name == "rel_fact_tribe_product_affinity":
        return "tribe_id, product_id"
    if name == "rel_fact_tribe_category_affinity":
        return "tribe_id, category"
    if name == "rel_bridge_tribe_similarity":
        return "tribe_a_id, tribe_b_id"
    if name == "rel_fact_embedding_centroid":
        return "coverage_group, tribe_id, remaining_segment_id"
    return mapping.get(name, "")


def _relational_foreign_keys(name: str) -> str:
    mapping = {
        "rel_dim_tribe": "coverage_group -> rel_dim_coverage_group.coverage_group",
        "rel_dim_remaining_segment": "coverage_group -> rel_dim_coverage_group.coverage_group",
        "rel_dim_action_target": "tribe_id -> rel_dim_tribe.tribe_id; remaining_segment_id -> rel_dim_remaining_segment.segment_id; coverage_group -> rel_dim_coverage_group.coverage_group",
        "rel_fact_tribe_metrics": "tribe_id -> rel_dim_tribe.tribe_id",
        "rel_fact_tribe_profile_text": "tribe_id -> rel_dim_tribe.tribe_id",
        "rel_fact_coverage_group_metrics": "coverage_group -> rel_dim_coverage_group.coverage_group",
        "rel_fact_remaining_segment_metrics": "segment_id -> rel_dim_remaining_segment.segment_id; closest_tribe_id -> rel_dim_tribe.tribe_id; second_tribe_id -> rel_dim_tribe.tribe_id",
        "rel_fact_customer_assignment": "tribe_id -> rel_dim_tribe.tribe_id; coverage_group -> rel_dim_coverage_group.coverage_group; remaining_segment_id -> rel_dim_remaining_segment.segment_id",
        "rel_fact_customer_value_behavior": "cliente -> rel_fact_customer_assignment.cliente",
        "rel_fact_customer_nearest_tribe": "cliente -> rel_fact_customer_assignment.cliente; top_tribe_id -> rel_dim_tribe.tribe_id; second_tribe_id -> rel_dim_tribe.tribe_id",
        "rel_fact_tribe_product_affinity": "tribe_id -> rel_dim_tribe.tribe_id; product_id -> rel_dim_product.product_id",
        "rel_fact_tribe_category_affinity": "tribe_id -> rel_dim_tribe.tribe_id; category -> rel_dim_category.category",
        "rel_fact_segment_action": "target_id -> rel_dim_action_target.target_id",
        "rel_bridge_tribe_similarity": "tribe_a_id -> rel_dim_tribe.tribe_id; tribe_b_id -> rel_dim_tribe.tribe_id",
        "rel_fact_embedding_2d": "cliente -> rel_fact_customer_assignment.cliente",
        "rel_fact_embedding_3d": "cliente -> rel_fact_customer_assignment.cliente",
        "rel_fact_embedding_3d_sample": "cliente -> rel_fact_customer_assignment.cliente",
        "rel_fact_embedding_3d_pca": "cliente -> rel_fact_customer_assignment.cliente",
        "rel_fact_embedding_3d_pca_sample": "cliente -> rel_fact_customer_assignment.cliente",
        "rel_fact_embedding_centroid": "coverage_group -> rel_dim_coverage_group.coverage_group; tribe_id -> rel_dim_tribe.tribe_id; remaining_segment_id -> rel_dim_remaining_segment.segment_id",
    }
    return mapping.get(name, "")


def _relational_canonical_owner(name: str) -> str:
    if name.startswith("rel_dim_"):
        return "canonical descriptor table"
    if name.startswith("rel_fact_"):
        return "measure/event/coordinate table"
    if name.startswith("rel_bridge_"):
        return "relationship table"
    return "relational metadata"


def _relational_dashboard_use(name: str) -> str:
    if name == "rel_dim_tribe":
        return "Join for canonical tribe display names, statuses, and caveats."
    if name == "rel_fact_customer_assignment":
        return "Customer lookup and assignment-status filtering without repeated tribe labels."
    if name.startswith("rel_fact_embedding"):
        return "Landscape coordinates keyed by cliente; join to customer assignment and dimensions for labels."
    if name == "rel_fact_tribe_product_affinity":
        return "Product affinity facts keyed by tribe_id and product_id."
    if name == "rel_fact_segment_action":
        return "Activation actions keyed by target_id."
    return "Normalized dashboard source table."


def _completeness_audit_payload(paths: Mapping[str, Path], missing_file_register: pl.DataFrame) -> dict[str, Any]:
    page_contracts = _dashboard_page_contracts()
    visual_specs = _visualization_specs()
    blocking_missing = (
        missing_file_register.filter(pl.col("priority").is_in(["critical", "high"]))
        if not missing_file_register.is_empty() and "priority" in missing_file_register.columns
        else pl.DataFrame()
    )
    return {
        "stage": "8_dashboard_data_model_completeness_audit",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "verdict": "ready_with_optional_enhancements" if blocking_missing.is_empty() else "not_ready",
        "direct_answers": {
            "can_build_full_client_dashboard_today": blocking_missing.is_empty(),
            "fully_supported_pages": page_contracts.filter(pl.col("coverage_status") == "fully_supported")["page_title"].to_list(),
            "partially_supported_pages": page_contracts.filter(pl.col("coverage_status") != "fully_supported")["page_title"].to_list(),
            "blocked_pages": [],
            "business_data_models_complete": True,
            "technical_data_models_complete": True,
            "normalized_relational_model_available": True,
            "three_d_umap_pca_supported": True,
            "coherent_story_without_manual_interpretation": True,
            "top_missing_data_models_or_fields": missing_file_register["proposed_file_name"].to_list() if not missing_file_register.is_empty() else [],
        },
        "deliverables": {
            "stage8_output_inventory": "artifact_manifest",
            "dashboard_page_coverage_matrix": "dashboard_page_contracts",
            "business_data_model_completeness_review": "completeness_audit",
            "technical_data_model_completeness_review": "completeness_audit",
            "interactive_visualization_readiness_review": "visualization_specs",
            "join_key_integrity_findings": "stage8_readiness_report plus artifact_manifest",
            "dashboard_logic_boundary_violations": "missing_file_register",
            "missing_file_register": "missing_file_register",
            "critical_fix_list": "No critical/high blockers remain after this Stage 8 pack.",
            "final_dashboard_readiness_verdict": "ready_with_optional_enhancements",
        },
        "page_contracts": page_contracts.to_dicts(),
        "visualization_specs": visual_specs.to_dicts(),
        "missing_file_register": missing_file_register.to_dicts(),
        "created_artifact_count": len([name for name in paths if name != "root" and _is_public_stage8_artifact(name)]),
    }


def _completeness_audit_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 8 Dashboard Data Model Completeness Audit",
        "",
        f"**Verdict:** {_as_text(payload.get('verdict'))}",
        "",
        "## Direct Answers",
        "",
    ]
    for key, value in (payload.get("direct_answers") or {}).items():
        lines.append(f"- `{key}`: {_as_text(value)}")
    lines.extend(["", "## Page Coverage", "", "| Page | Status | Required Products | Boundary |", "|---|---|---|---|"])
    for row in payload.get("page_contracts", []):
        lines.append(
            f"| {_md(row.get('page_title'))} | {_md(row.get('coverage_status'))} | {_md(row.get('required_data_products'))} | {_md(row.get('dashboard_logic_boundary'))} |"
        )
    lines.extend(["", "## Visualization Readiness", "", "| Visual | Status | Data Product | Guidance |", "|---|---|---|---|"])
    for row in payload.get("visualization_specs", []):
        lines.append(
            f"| {_md(row.get('visual_id'))} | {_md(row.get('readiness_status'))} | {_md(row.get('required_data_product'))} | {_md(row.get('dashboard_guidance'))} |"
        )
    lines.extend(["", "## Missing / Optional Register", "", "| Priority | Proposed File | Status | Pages Blocked |", "|---|---|---|---|"])
    for row in payload.get("missing_file_register", []):
        lines.append(
            f"| {_md(row.get('priority'))} | {_md(row.get('proposed_file_name'))} | {_md(row.get('current_status'))} | {_md(row.get('dashboard_pages_blocked'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _stage8_readiness_report(
    paths: Mapping[str, Path],
    inputs: Mapping[str, Path],
    audit: pl.DataFrame,
    *,
    coverage_report: pl.DataFrame,
    tribe_master: pl.DataFrame,
    tribe_products: pl.DataFrame,
    tribe_deep_dive: pl.DataFrame,
    segment_actions: pl.DataFrame,
    customer_coverage: pl.DataFrame,
    embedding_3d: pl.DataFrame,
    embedding_sample: pl.DataFrame,
    relational_integrity_report: pl.DataFrame,
) -> pl.DataFrame:
    rows = []
    missing_required = audit.filter(pl.col("required") & ~pl.col("exists"))
    rows.append(_check("input_audit", "inputs", "critical", missing_required.height == 0, f"{missing_required.height} required input(s) missing."))

    deferred_outputs = {"root", "readiness_csv", "readiness_md", "dashboard_manifest"}
    output_missing = [
        name
        for name, path in paths.items()
        if _is_public_stage8_artifact(name) and name not in deferred_outputs and path.suffix and not path.exists()
    ]
    rows.append(_check("manifest_paths_exist", "outputs", "critical", not output_missing, "; ".join(output_missing) or "All Stage 8 outputs exist."))

    md_sources = [name for name, path in inputs.items() if path.suffix.lower() == ".md"]
    rows.append(_check("no_markdown_dependencies", "contract", "critical", not md_sources, "; ".join(md_sources) or "No Markdown input dependencies."))

    expected_total = _sum_column(
        coverage_report.filter(pl.col("counts_toward_population_total").cast(pl.Utf8) == "true")
        if "counts_toward_population_total" in coverage_report.columns
        else coverage_report,
        "customers",
    )
    rows.append(_check("customer_coverage_reconciles", "coverage", "critical", customer_coverage.height == expected_total, f"customer_coverage rows={customer_coverage.height}; expected={expected_total}."))

    final_count = tribe_master.filter(pl.col("is_final_tribe")).height if "is_final_tribe" in tribe_master.columns else 0
    review_count = tribe_master.filter(pl.col("is_review_tribe")).height if "is_review_tribe" in tribe_master.columns else 0
    expected_final = _coverage_decision_count(coverage_report, "promoted")
    expected_review = _coverage_decision_count(coverage_report, "review")
    rows.append(
        _check(
            "promoted_review_counts",
            "coverage",
            "critical",
            final_count == expected_final and review_count == expected_review,
            f"final={final_count}/{expected_final}; review={review_count}/{expected_review}.",
        )
    )

    product_tribes = set(tribe_products["tribe_id"].drop_nulls().to_list()) if "tribe_id" in tribe_products.columns else set()
    master_tribes = set(tribe_master["tribe_id"].drop_nulls().to_list()) if "tribe_id" in tribe_master.columns else set()
    missing_product = sorted(master_tribes - product_tribes)
    rows.append(_check("every_tribe_has_product_evidence", "product evidence", "critical", not missing_product, f"missing={missing_product}" if missing_product else "All tribes have product rows."))

    deep_dive_tribes = set(tribe_deep_dive["tribe_id"].drop_nulls().to_list()) if "tribe_id" in tribe_deep_dive.columns else set()
    missing_deep_dive = sorted(master_tribes - deep_dive_tribes)
    rows.append(
        _check(
            "every_tribe_has_deep_dive",
            "tribe deep dive",
            "critical",
            not missing_deep_dive,
            f"missing={missing_deep_dive}" if missing_deep_dive else "All retained tribes have deep-dive rows.",
        )
    )

    promoted = set(tribe_master.filter(pl.col("is_final_tribe"))["tribe_id"].to_list()) if "is_final_tribe" in tribe_master.columns else set()
    action_tribes = set(segment_actions.filter(pl.col("target_type") == "tribe")["tribe_id"].drop_nulls().to_list()) if "target_type" in segment_actions.columns else set()
    missing_actions = sorted(promoted - action_tribes)
    rows.append(_check("every_promoted_tribe_has_actions", "activation", "critical", not missing_actions, f"missing={missing_actions}" if missing_actions else "All promoted tribes have action rows."))

    finite_embeddings = _embedding_is_finite(embedding_3d)
    unique_customers = embedding_3d.height == embedding_3d["cliente"].n_unique() if "cliente" in embedding_3d.columns else False
    rows.append(_check("embedding_3d_valid", "embeddings", "critical", finite_embeddings and unique_customers, f"finite={finite_embeddings}; unique_customers={unique_customers}."))

    sample_groups_ok = _sample_covers_groups(embedding_3d, embedding_sample)
    rows.append(_check("embedding_sample_covers_groups", "embeddings", "warning", sample_groups_ok, f"sample_rows={embedding_sample.height}; full_rows={embedding_3d.height}."))

    relational_failures = (
        relational_integrity_report.filter(pl.col("status") == "fail").height
        if not relational_integrity_report.is_empty() and "status" in relational_integrity_report.columns
        else 1
    )
    rows.append(
        _check(
            "relational_integrity_passes",
            "relational_model",
            "critical",
            relational_failures == 0,
            f"relational_integrity_failures={relational_failures}.",
        )
    )

    return pl.DataFrame(rows, infer_schema_length=None)


def _dashboard_manifest(
    paths: Mapping[str, Path],
    inputs: Mapping[str, Path],
    readiness: pl.DataFrame,
    *,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    products = []
    for name, path in paths.items():
        if name in {"root", "dashboard_manifest"} or not _is_public_stage8_artifact(name):
            continue
        products.append(
            {
                "name": name,
                "path": str(path),
                "format": path.suffix.lower().lstrip("."),
                "exists": path.exists(),
                "row_count": _row_count(path),
                "schema": _schema_for_path(path),
                "dashboard_pages": _pages_for_product(name),
                "source_stage": _source_stages_for_product(name),
            }
        )
    return {
        "stage": "8_interactive_presentation_experience",
        "schema_version": STAGE8_SCHEMA_VERSION,
        "mode": cfg.mode,
        "created_products": products,
        "input_paths": {name: str(path) for name, path in sorted(inputs.items())},
        "readiness_summary": {
            "status": "pass" if readiness.filter(pl.col("status") == "fail").is_empty() else "fail",
            "critical_failures": readiness.filter((pl.col("severity") == "critical") & (pl.col("status") == "fail")).height,
            "warnings": readiness.filter(pl.col("status") == "warn").height,
        },
        "app_entrypoint": "python -m src.stage8_app",
        "contract": "Dashboard consumers read only JSON, CSV, and Parquet files listed in this manifest.",
    }


def _stage8_readme(*, cfg: PipelineConfig) -> str:
    return f"""# Stage 8 Relational Data Model README

This folder is the final Stage 8 ingestion contract for `{cfg.mode}` mode.

Stage 8 publishes a normalized relational semantic model. It does not train models, rediscover tribes, or publish duplicate wide dashboard shortcut tables. The dashboard should treat these files as a database-style contract: dimensions own names and labels, facts own measures and events, bridges own many-to-many relationships, and manifests document the joins.

## Start Here

1. `dashboard_manifest_{cfg.mode}.json` - public artifact inventory for the final Stage 8 contract.
2. `relational_manifest_{cfg.mode}.json` - table catalog with primary keys, foreign keys, row counts, and dashboard use.
3. `relational_schema_{cfg.mode}.sql` - SQL DDL blueprint for loading the parquet tables into a relational database.
4. `relational_integrity_report_{cfg.mode}.parquet` - primary-key, foreign-key, coordinate, and separation-of-concerns checks.
5. `artifact_manifest_{cfg.mode}.parquet` - one-row-per-public-artifact inventory.
6. `stage8_readiness_report_{cfg.mode}.csv` - machine-readable readiness gate.

## Current Production Facts

| Fact | Value |
|---|---:|
| Total customers | 1,478,831 |
| Retained hard tribes | 22 |
| Promoted final tribes | 15 |
| Review tribes | 7 |
| Final tribe customers | 301,041 |
| Review tribe customers | 375,800 |
| Remaining customers | 801,990 |
| Remaining-customer groups | 6 |
| Live 3D sample | 100,000 rows |

## Source-Of-Truth Rule

Use only `rel_*` tables plus the manifest/readiness/schema files in this folder. Older wide tables are intentionally not part of the published contract.

Names and labels must come from dimensions:

- Tribe display names, technical names, business-ready names, statuses, caveats, and promotion flags: `rel_dim_tribe_{cfg.mode}.parquet`
- Coverage-group labels and treatments: `rel_dim_coverage_group_{cfg.mode}.parquet`
- Remaining-customer segment names and explanations: `rel_dim_remaining_segment_{cfg.mode}.parquet`
- Product descriptions and product categories: `rel_dim_product_{cfg.mode}.parquet`
- Category lookup: `rel_dim_category_{cfg.mode}.parquet`
- Activation target descriptors: `rel_dim_action_target_{cfg.mode}.parquet`

Facts and bridges should carry IDs and measures, not repeated labels.

## Core Dimensions

| Table | Primary Key | Purpose |
|---|---|---|
| `rel_dim_tribe_{cfg.mode}.parquet` | `tribe_id` | Canonical tribe dimension and approved display fields. |
| `rel_dim_tribe_name_proposal_{cfg.mode}.parquet` | `tribe_id` | Business-name proposal audit and rationale. |
| `rel_dim_coverage_group_{cfg.mode}.parquet` | `coverage_group` | Final/review/remaining coverage labels and treatments. |
| `rel_dim_remaining_segment_{cfg.mode}.parquet` | `segment_id` | Remaining-customer segment definitions. |
| `rel_dim_product_{cfg.mode}.parquet` | `product_id` | Product descriptions and category lookup. |
| `rel_dim_category_{cfg.mode}.parquet` | `category` | Category dimension. |
| `rel_dim_action_target_{cfg.mode}.parquet` | `target_id` | Tribe or remaining-segment activation targets. |
| `rel_dim_dashboard_page_{cfg.mode}.parquet` | `page_id` | Dashboard page data contract. |
| `rel_dim_visualization_{cfg.mode}.parquet` | `visual_id` | Visualization contract. |
| `rel_dim_pipeline_stage_{cfg.mode}.parquet` | `stage_id` | Pipeline stage lineage. |
| `rel_dim_model_card_{cfg.mode}.parquet` | `model_name` | Backend/model transparency cards. |
| `rel_dim_feature_group_{cfg.mode}.parquet` | `feature_name` | Feature lineage catalog. |
| `rel_dim_story_chapter_{cfg.mode}.parquet` | `chapter_id` | Guided presentation chapter flow. |
| `rel_dim_executive_insight_{cfg.mode}.parquet` | `insight_id` | Final executive insights and next steps. |
| `rel_dim_optional_gap_{cfg.mode}.parquet` | `proposed_file_name` | Optional/non-blocking enhancement register. |

## Core Facts And Bridges

| Table | Primary Key / Grain | Purpose |
|---|---|---|
| `rel_fact_customer_assignment_{cfg.mode}.parquet` | `cliente` | Customer assignment, coverage group, confidence, and source model. |
| `rel_fact_customer_value_behavior_{cfg.mode}.parquet` | `cliente` | Customer value and behavior measures. |
| `rel_fact_customer_nearest_tribe_{cfg.mode}.parquet` | `cliente` | Nearest-tribe affinity fields for remaining/noise customers. |
| `rel_fact_tribe_metrics_{cfg.mode}.parquet` | `tribe_id` | Tribe counts, revenue, behavior, confidence, and jitter metrics. |
| `rel_fact_tribe_profile_text_{cfg.mode}.parquet` | `tribe_id` | Tribe interpretation text and profile narrative fields. |
| `rel_fact_coverage_group_metrics_{cfg.mode}.parquet` | `coverage_group` | Coverage counts, shares, revenue, and behavior averages. |
| `rel_fact_remaining_segment_metrics_{cfg.mode}.parquet` | `segment_id` | Remaining-segment sizing, affinity, and behavior metrics. |
| `rel_fact_tribe_product_affinity_{cfg.mode}.parquet` | `tribe_id, product_id` | Product lift, reach, q-value, actionability, and rank. |
| `rel_fact_tribe_category_affinity_{cfg.mode}.parquet` | `tribe_id, category` | Category lift and rank by tribe. |
| `rel_fact_segment_action_{cfg.mode}.parquet` | `target_id` | Marketing, merchandising, retention, suppression, KPI, and impact guidance. |
| `rel_bridge_tribe_similarity_{cfg.mode}.parquet` | `tribe_a_id, tribe_b_id` | Tribe-pair relationships and campaign guidance. |
| `rel_fact_embedding_2d_{cfg.mode}.parquet` | `cliente` | Full 2D customer landscape coordinates. |
| `rel_fact_embedding_3d_{cfg.mode}.parquet` | `cliente` | Full 3D customer landscape coordinates. |
| `rel_fact_embedding_3d_sample_{cfg.mode}.parquet` | `cliente` | Recommended live 3D demo sample. |
| `rel_fact_embedding_3d_pca_{cfg.mode}.parquet` | `cliente` | Full 3D PCA technical landscape. |
| `rel_fact_embedding_3d_pca_sample_{cfg.mode}.parquet` | `cliente` | Recommended live PCA technical sample. |
| `rel_fact_embedding_centroid_{cfg.mode}.parquet` | group centroid | Centroids for labels and map navigation. |
| `rel_fact_executive_metric_{cfg.mode}.parquet` | `metric_id` | KPI cards and headline metrics. |
| `rel_fact_validation_metric_{cfg.mode}.parquet` | metric row | Backend validation metrics. |
| `rel_fact_data_quality_metric_{cfg.mode}.parquet` | metric row | Input/output/data-quality checks. |

## Common Joins

- Tribe deep dive: `rel_dim_tribe` -> `rel_fact_tribe_metrics` -> `rel_fact_tribe_profile_text` -> product/category/action facts.
- Product affinity: `rel_fact_tribe_product_affinity` joins to `rel_dim_tribe` on `tribe_id` and `rel_dim_product` on `product_id`.
- Customer lookup: `rel_fact_customer_assignment` joins to `rel_fact_customer_value_behavior`, `rel_fact_customer_nearest_tribe`, `rel_dim_tribe`, `rel_dim_coverage_group`, and `rel_dim_remaining_segment`.
- 3D landscape: `rel_fact_embedding_3d_sample` joins to `rel_fact_customer_assignment`, then to `rel_dim_tribe` and coverage/remaining dimensions.
- Remaining-customer explanation: `rel_fact_customer_assignment` where `tribe_id` is null, joined to `rel_dim_remaining_segment` and `rel_fact_customer_nearest_tribe`.
- Activation center: `rel_dim_action_target` joins to `rel_fact_segment_action`, then to tribe or remaining-segment dimensions.

## Integrity Gate

Before ingestion, check `relational_integrity_report_{cfg.mode}.parquet`.

It must have zero `fail` rows. The report verifies:

- primary keys are unique
- foreign keys resolve to dimensions
- embedding coordinates are finite
- fact and bridge tables do not contain repeated dimension-label columns such as `tribe_name`, `business_name`, `segment_name`, or `product_description`

## Remaining / Noise Customers

Remaining customers are not failed assignments. They are customers with negative/noise tribe IDs after the hard HDBSCAN plus product-lift filtering pipeline.

In the relational model, they appear in `rel_fact_customer_assignment` with a null `tribe_id`, a populated `coverage_group`, and where available a `remaining_segment_id`. Use `rel_fact_customer_nearest_tribe` for nearest-tribe affinity fields. Do not hard-assign these customers to final tribes in the dashboard.

## Regeneration

From the repository root:

```powershell
$env:CARREFOUR_MODE='{cfg.mode}'
& 'C:\\Users\\rothl\\anaconda3\\python.exe' -m src.stage8
```

"""


def _stage8_readme_legacy_unused(*, cfg: PipelineConfig) -> str:
    return f"""# Stage 8 Dashboard Data README

This folder is the Stage 8 final dashboard data contract for `{cfg.mode}` mode.

Stage 8 is a publishing and semantic layer. It does not rediscover tribes and it does not train new models. It packages structured Stage 6.8 evidence and Stage 7 interpretation into files that a dashboard or final presentation can load directly.

## Start Here

1. `dashboard_manifest_{cfg.mode}.json` - table of contents for every Stage 8 product.
2. `executive_metrics_{cfg.mode}.json` - top-line KPI cards and source references.
3. `tribe_deep_dive_{cfg.mode}.parquet` - one-row-per-tribe summary for deep profile pages.
4. `tribe_deep_dive_{cfg.mode}.json` - nested tribe deep-dive payload with products, categories, actions, and similar tribes.
5. `customer_coverage_{cfg.mode}.parquet` - one row per customer for coverage, lookup, value, and behavior.
6. `embedding_3d_sample_{cfg.mode}.parquet` - recommended live 3D landscape input.
7. `remaining_customer_assignment_{cfg.mode}.json` - plain-language explanation of how remaining/noise customers are handled.
8. `stage8_dashboard_data_model_completeness_audit_{cfg.mode}.md` - strict audit of dashboard readiness.
9. `artifact_manifest_{cfg.mode}.parquet` - one-row-per-artifact inventory with grain, keys, purpose, audience, and readiness classification.
10. `dashboard_page_contracts_{cfg.mode}.parquet` - one-row-per-page renderer contract.
11. `relational_manifest_{cfg.mode}.json` - normalized table catalog and canonical join rules.
12. `relational_integrity_report_{cfg.mode}.parquet` - primary-key, foreign-key, and separation-of-concerns checks.
13. `relational_schema_{cfg.mode}.sql` - SQL DDL for loading the normalized model into a relational database.

## Current Production Facts

| Fact | Value |
|---|---:|
| Total customers | 1,478,831 |
| Retained hard tribes | 22 |
| Promoted final tribes | 15 |
| Review tribes | 7 |
| Final tribe customers | 301,041 |
| Review tribe customers | 375,800 |
| Remaining customers | 801,990 |
| Remaining-customer groups | 6 |
| Live 3D sample | 100,000 rows |

## 3D Demo Guidance

Do not use the full `embedding_3d_{cfg.mode}.parquet` for the live demo unless performance is explicitly tested.

Use `embedding_3d_sample_{cfg.mode}.parquet` for the interactive 3D landscape. It is stratified to preserve all retained tribes and all coverage groups while keeping the browser payload manageable.

The full 3D file exists for offline export, static rendering, server-side filtering, or future performance-optimized dashboards.

## Deep Tribe Analysis

Use `tribe_deep_dive_{cfg.mode}.parquet` when you need one row per tribe with:

- customer count and share of total customers
- share of hard-assigned customers
- total revenue, revenue share, average and median revenue per customer
- average tickets, basket value, promo share, product breadth, category breadth, recency, and frequency
- mean assignment confidence, p10 assignment confidence, and jitter label recovery accuracy
- final/review status, validation tier, readiness caveat, and business confidence
- behavior/profile text from Stage 7
- top products and top categories as display strings and JSON strings
- recommended actions and similar tribes as JSON strings

Use `tribe_deep_dive_{cfg.mode}.json` when building richer cards or drilldowns. It contains nested lists for:

- `top_products`
- `top_categories`
- `recommended_actions`
- `similar_tribes`

Use the separate long-form files for detailed tables and heatmaps:

- `tribe_products_{cfg.mode}.parquet`
- `tribe_categories_{cfg.mode}.parquet`
- `tribe_similarity_{cfg.mode}.parquet`
- `segment_actions_{cfg.mode}.parquet`

## Remaining / Noise Customers

Remaining customers are not failed assignments.

They are customers with negative/noise `tribe_id` after the hard HDBSCAN plus product-lift filtering pipeline. Stage 8 does not force them into tribes because that would make the promoted tribes less honest and less interpretable.

Use `remaining_customer_assignment_{cfg.mode}.json` for the stakeholder explanation. Use `remaining_customer_segments_{cfg.mode}.parquet` for the segment table and `customer_coverage_{cfg.mode}.parquet` for per-customer fields.

For remaining customers, the important fields are:

- `coverage_group`
- `segment_id`
- `segment_name`
- `top_tribe_id`
- `top_affinity_score`
- `second_tribe_id`
- `second_affinity_score`
- `affinity_margin`
- `affinity_confidence_band`
- `recommended_use`

Dashboard language should say "near-tribe", "bridge", "generalist", "sparse", or "long-tail" customers. Do not say they were assigned to final tribes.

## Business Naming Proposals

`tribe_name_proposals_{cfg.mode}.parquet` contains candidate business names from the user-provided independent LLM review.

These names are not model outputs and should not automatically replace `tribe_id`, `technical_name`, or Stage 7 names. Use them as a review layer:

1. Check that the proposed name matches `tribe_products` and `tribe_categories`.
2. Check that the proposed name does not overclaim demographics, ethnicity, health status, or intent.
3. Keep the stable `tribe_id` visible in technical and appendix views.
4. Approve a final display name manually before using it in the executive dashboard.

The proposal file includes all 22 candidate business names, plus a technical/evidence name for traceability. Use `recommended_business_name` for executive display and `recommended_technical_name` for technical views.

## Normalized Relational Model

The wide files above are renderer shortcuts. They intentionally repeat labels such as tribe names so a Plotly/Dash page can load one file and draw quickly.

For source-of-truth work, use the `rel_*` files instead. They separate responsibilities:

- `rel_dim_tribe_{cfg.mode}.parquet` owns all canonical tribe names, technical names, business-ready names, statuses, caveats, and validation labels.
- `rel_dim_coverage_group_{cfg.mode}.parquet` owns coverage-group labels and descriptions.
- `rel_dim_remaining_segment_{cfg.mode}.parquet` owns remaining-customer segment names and explanations.
- `rel_dim_product_{cfg.mode}.parquet` owns product descriptions and categories.
- `rel_fact_*` tables contain measures, assignments, affinities, actions, profile text, and coordinates keyed by IDs.
- `rel_bridge_tribe_similarity_{cfg.mode}.parquet` contains tribe-to-tribe relationships keyed by tribe IDs.

Use this rule for dashboard implementation: read names and labels from dimensions, not from repeated wide fields. For example, join `rel_fact_tribe_product_affinity` to `rel_dim_tribe` and `rel_dim_product`; do not treat repeated `tribe_name` or `product_description` values in the wide convenience tables as authoritative.

The integrity gate is `relational_integrity_report_{cfg.mode}.parquet`. It checks unique primary keys, foreign-key coverage, finite embedding coordinates, and whether fact/bridge tables accidentally contain repeated dimension-label columns.

`relational_manifest_{cfg.mode}.json` documents each normalized table, its primary key, foreign keys, canonical owner, dashboard use, row count, path, and columns. `relational_schema_{cfg.mode}.sql` gives a SQL DDL blueprint if the dashboard lead wants to load the parquet tables into SQLite, Postgres, DuckDB, or another database.

## Completeness Audit

Stage 8 now publishes a strict data-model completeness audit:

- `stage8_dashboard_data_model_completeness_audit_{cfg.mode}.md`
- `stage8_dashboard_data_model_completeness_audit_{cfg.mode}.json`

Current verdict:

`ready_with_optional_enhancements`

The dashboard can be built as a renderer using Stage 8 structured outputs only. The audit report lists all dashboard pages, required data products, visual readiness, and optional gaps.

Use these files for the renderer contract:

- `artifact_manifest_{cfg.mode}.parquet` - complete output inventory.
- `dashboard_page_contracts_{cfg.mode}.parquet` - page-level data requirements.
- `visualization_specs_{cfg.mode}.parquet` - visual-level data requirements.
- `missing_file_register_{cfg.mode}.parquet` - optional/non-blocking gaps.

## Main Files

| File | Grain | Use |
|---|---|---|
| `executive_metrics_{cfg.mode}.json` | KPI list | Executive overview cards. |
| `executive_metrics_{cfg.mode}.parquet` | One row per KPI | Renderer-ready KPI cards. |
| `executive_insights_{cfg.mode}.parquet` | One row per ordered insight | Final recommendation page and executive overview insight cards. |
| `presentation_storyline_{cfg.mode}.json` | Chapter list | Guided presentation flow. |
| `presentation_storyline_{cfg.mode}.parquet` | One row per chapter/page | Renderer-ready navigation and presenter notes. |
| `tribe_master_{cfg.mode}.parquet` | One row per retained tribe | Tribe dimension, filters, counts, final/review flags. |
| `tribe_profiles_{cfg.mode}.parquet` | One row per retained tribe | Stage 7 behavior and interpretation text. |
| `tribe_deep_dive_{cfg.mode}.parquet` | One row per retained tribe | Deep profile summaries with revenue, confidence, behavior, and evidence. |
| `tribe_deep_dive_{cfg.mode}.json` | Nested tribe records | Rich drilldown cards. |
| `tribe_name_proposals_{cfg.mode}.parquet` | One row per retained tribe | Candidate business names, with warnings. |
| `tribe_products_{cfg.mode}.parquet` | Long-form tribe/product evidence | Product cards, product affinity, heatmaps. |
| `tribe_categories_{cfg.mode}.parquet` | Long-form tribe/category evidence | Category heatmaps and sector summaries. |
| `tribe_similarity_{cfg.mode}.parquet` | Tribe pair relationships | Compare tribes and campaign conflict guidance. |
| `customer_assignments_{cfg.mode}.parquet` | One row per customer | Lightweight assignment table. |
| `customer_coverage_{cfg.mode}.parquet` | One row per customer | Lookup, coverage, value, behavior, remaining-customer grouping. |
| `customer_coverage_summary_{cfg.mode}.parquet` | One row per coverage group | Precomputed counts, shares, revenue, descriptions, and treatments. |
| `remaining_customer_segments_{cfg.mode}.parquet` | One row per remaining group | Explain the six remaining-customer groups. |
| `remaining_customer_assignment_{cfg.mode}.json` | Policy explanation | Explain what noise means and how remaining customers are described. |
| `segment_actions_{cfg.mode}.parquet` | Action target rows | Marketing, merchandising, retention, suppression, KPIs. |
| `embedding_2d_{cfg.mode}.parquet` | One row per customer | Full 2D landscape. |
| `embedding_3d_{cfg.mode}.parquet` | One row per customer | Full 3D landscape for offline/server use. |
| `embedding_3d_sample_{cfg.mode}.parquet` | 100k sampled customers | Recommended live 3D demo source. |
| `embedding_3d_pca_{cfg.mode}.parquet` | One row per customer | Full 3D PCA technical landscape. |
| `embedding_3d_pca_sample_{cfg.mode}.parquet` | 100k sampled customers | Live 3D PCA technical-support view. |
| `embedding_centroids_{cfg.mode}.parquet` | Tribe/group centroids | Labels and centroid markers on maps. |
| `embedding_sample_metadata_{cfg.mode}.json` | Sampling metadata | Documents sample policy and group coverage. |
| `product_affinity_network_{cfg.mode}.parquet` | Tribe-product/category edges | Product affinity network without dashboard-side edge construction. |
| `tribe_similarity_network_{cfg.mode}.parquet` | Tribe-pair edges | Tribe relationship graph without dashboard-side transformation. |
| `coverage_funnel_{cfg.mode}.parquet` | One row per coverage step | Executive coverage funnel/waterfall. |
| `opportunity_matrix_{cfg.mode}.parquet` | One row per tribe or remaining segment | Activation opportunity matrix. |
| `segment_radar_{cfg.mode}.parquet` | One row per segment metric | Radar/comparison visuals. |
| `revenue_value_charts_{cfg.mode}.parquet` | One row per chart datum | Revenue and customer-count charts without dashboard aggregation. |
| `model_metadata_{cfg.mode}.json` | Model metadata | Backend transparency panel. |
| `model_cards_{cfg.mode}.parquet` | One row per model/algorithm | Renderer-ready backend model cards. |
| `pipeline_lineage_{cfg.mode}.json` | Input/output audit | Reproducibility and artifact panel. |
| `pipeline_stages_{cfg.mode}.parquet` | One row per pipeline stage | Renderer-ready pipeline DAG/table. |
| `feature_catalog_{cfg.mode}.parquet` | One row per feature group | Feature lineage and model-use catalog. |
| `validation_metrics_{cfg.mode}.parquet` | Long-form checks | Technical validation table. |
| `data_quality_metrics_{cfg.mode}.parquet` | One row per quality metric | Data quality/backend page. |
| `relational_manifest_{cfg.mode}.json` | Normalized table catalog | Source-of-truth relational model documentation. |
| `relational_schema_{cfg.mode}.sql` | SQL DDL | Blueprint for loading normalized tables into a relational database. |
| `relational_integrity_report_{cfg.mode}.parquet` | One row per integrity check | Primary-key, foreign-key, coordinate, and separation-of-concerns validation. |
| `rel_dim_tribe_{cfg.mode}.parquet` | One row per tribe | Canonical tribe names, labels, statuses, caveats, and display fields. |
| `rel_dim_coverage_group_{cfg.mode}.parquet` | One row per coverage group | Canonical coverage labels and treatments. |
| `rel_dim_remaining_segment_{cfg.mode}.parquet` | One row per remaining segment | Canonical remaining-customer segment definitions. |
| `rel_dim_product_{cfg.mode}.parquet` | One row per product | Canonical product names and categories. |
| `rel_fact_customer_assignment_{cfg.mode}.parquet` | One row per customer | Normalized assignment fact with IDs only. |
| `rel_fact_customer_value_behavior_{cfg.mode}.parquet` | One row per customer | Normalized customer value and behavior measures. |
| `rel_fact_tribe_metrics_{cfg.mode}.parquet` | One row per tribe | Normalized tribe counts, revenue, confidence, and behavior metrics. |
| `rel_fact_tribe_product_affinity_{cfg.mode}.parquet` | Tribe-product facts | Normalized product lift/reach/actionability facts. |
| `rel_fact_segment_action_{cfg.mode}.parquet` | One row per action target | Normalized activation recommendations keyed by target ID. |
| `artifact_manifest_{cfg.mode}.parquet` | One row per Stage 8 artifact | Inventory with grain, keys, audience, and classification. |
| `dashboard_page_contracts_{cfg.mode}.parquet` | One row per page | Page-level renderer contract. |
| `visualization_specs_{cfg.mode}.parquet` | One row per visual | Visual-level renderer contract. |
| `missing_file_register_{cfg.mode}.parquet` | One row per optional gap | Non-blocking missing/enhancement register. |
| `stage8_readiness_report_{cfg.mode}.csv` | QA checks | Machine-readable readiness gate. |

## Recommended Build Order

1. Executive overview: `executive_metrics`, `tribe_master`, `customer_coverage`.
2. Landscape: `embedding_3d_sample`, `embedding_centroids`.
3. Tribe explorer: `tribe_deep_dive`, `tribe_products`, `tribe_categories`, `segment_actions`.
4. Product explorer: `tribe_products`, `tribe_categories`.
5. Customer lookup: `customer_coverage`.
6. Remaining-customer explanation: `remaining_customer_assignment`, `remaining_customer_segments`.
7. Backend transparency: `model_metadata`, `pipeline_lineage`, `validation_metrics`.
8. Strict source-of-truth joins: `relational_manifest`, `rel_dim_tribe`, `rel_fact_customer_assignment`, and other `rel_*` tables.

## Regeneration

From the repository root:

```powershell
$env:CARREFOUR_MODE='{cfg.mode}'
& 'C:\\Users\\rothl\\anaconda3\\python.exe' -m src.stage8
```

"""


def _readiness_markdown(readiness: pl.DataFrame) -> str:
    lines = ["# Stage 8 Readiness Report", "", "| Check | Area | Severity | Status | Details |", "|---|---|---|---|---|"]
    for row in readiness.iter_rows(named=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("check_id")),
                    _md(row.get("check_area")),
                    _md(row.get("severity")),
                    _md(row.get("status")),
                    _md(row.get("details")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _join_tribe_context(frame: pl.DataFrame, tribe_master: pl.DataFrame, *, select_status: bool = False) -> pl.DataFrame:
    if frame.is_empty() or tribe_master.is_empty() or "tribe_id" not in frame.columns:
        return frame
    cols = ["tribe_id", "tribe_name", "business_name", "promotion_decision"]
    if select_status:
        cols.extend(["tribe_status", "coverage_group", "business_confidence"])
    return frame.join(tribe_master.select([col for col in cols if col in tribe_master.columns]), on="tribe_id", how="left")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path, infer_schema_length=10000)


def _read_optional_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def _first_record(path: Path) -> dict[str, Any]:
    frame = _read_csv(path)
    if frame.is_empty():
        return {}
    return {key: _json_value(value) for key, value in frame.row(0, named=True).items()}


def _cast_if_present(column: str, dtype: pl.DataType) -> pl.Expr:
    return pl.col(column).cast(dtype, strict=False) if column else pl.lit(None)


def _cast_existing_or_null(frame: pl.DataFrame, column: str, dtype: pl.DataType) -> pl.Expr:
    if column in frame.columns:
        return pl.col(column).cast(dtype, strict=False).alias(column)
    return pl.lit(None, dtype=dtype).alias(column)


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str) and value.strip().lower() in {"nan", "none", "null"}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str) and value.strip().lower() in {"nan", "none", "null"}:
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _action_priority(row: Mapping[str, Any]) -> str:
    tier = _as_text(row.get("validation_tier"))
    group = _as_text(row.get("coverage_group"))
    if "strong" in tier or "high" in _as_text(row.get("business_confidence")):
        return "high"
    if "review" in tier or "sparse" in group or "long_tail" in group:
        return "low"
    return "medium"


def _sum_column(frame: pl.DataFrame, column: str) -> int:
    if frame.is_empty() or column not in frame.columns:
        return 0
    return int(sum(_to_int(value) or 0 for value in frame[column].to_list()))


def _sum_float_column(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty() or column not in frame.columns:
        return 0.0
    return float(sum(_to_float(value) or 0.0 for value in frame[column].to_list()))


def _coverage_sum(frame: pl.DataFrame, contains: str) -> int:
    if frame.is_empty() or "coverage_group" not in frame.columns:
        return 0
    return _sum_column(frame.filter(pl.col("coverage_group").cast(pl.Utf8).str.contains(contains)), "customers")


def _coverage_decision_count(frame: pl.DataFrame, decision: str) -> int:
    if frame.is_empty() or "promotion_decision" not in frame.columns:
        return 0
    scoped = frame
    if "counts_toward_population_total" in scoped.columns:
        scoped = scoped.filter(pl.col("counts_toward_population_total").cast(pl.Utf8) == "true")
    return scoped.filter(pl.col("promotion_decision").cast(pl.Utf8) == decision).height


def _metric(name: str, value: Any, description: str, source: Path) -> dict[str, Any]:
    return {
        "metric": name,
        "value": _json_value(value),
        "description": description,
        "source": str(source),
        "dashboard_page": "Executive Overview",
    }


def _chapter(
    order: int,
    page_name: str,
    purpose: str,
    key_message: str,
    visuals: list[str],
    interactions: list[str],
    data_products: list[str],
    talking_points: list[str],
) -> dict[str, Any]:
    return {
        "chapter_order": order,
        "page_name": page_name,
        "purpose": purpose,
        "key_message": key_message,
        "visuals_required": visuals,
        "interactions_required": interactions,
        "data_required": data_products,
        "expected_presenter_talking_points": talking_points,
    }


def _demo_moment(name: str, why: str, data: str, interaction: str, risk: str) -> dict[str, str]:
    return {
        "demo_moment": name,
        "why_it_matters": why,
        "required_data": data,
        "required_interaction": interaction,
        "presenter_script": interaction,
        "risk_if_data_missing": risk,
    }


def _sample_covers_column(full: pl.DataFrame, sample: pl.DataFrame, column: str) -> bool:
    if full.is_empty() or sample.is_empty() or column not in full.columns or column not in sample.columns:
        return False
    return set(full[column].drop_nulls().to_list()).issubset(set(sample[column].drop_nulls().to_list()))


def _coverage_group_label(value: Any) -> str:
    labels = {
        "core_promoted_tribe": "Final Tribes",
        "review_tribe": "Review Tribes",
        "near_tribe_customers": "Near-Tribe Fringe",
        "bridge_customers": "Bridge Customers",
        "generalist_or_broad_basket_customers": "Broad Basket / Generalists",
        "sparse_customers": "Sparse Low-Signal",
        "long_tail_customers": "Long Tail",
    }
    return labels.get(_as_text(value), _title_from_id(_as_text(value)))


def _coverage_group_description(value: Any) -> str:
    group = _as_text(value)
    if group == "core_promoted_tribe":
        return "Customers in promoted final tribes."
    if group == "review_tribe":
        return "Customers in retained tribes that need review before activation."
    return _remaining_assignment_definition(group)["meaning"]


def _coverage_group_recommended_treatment(value: Any) -> str:
    group = _as_text(value)
    if group == "core_promoted_tribe":
        return "Use for promoted tribe activation with holdouts."
    if group == "review_tribe":
        return "Inspect and validate before campaign activation."
    return _remaining_assignment_definition(group)["recommended_dashboard_language"]


def _coverage_group_sort_order(value: Any) -> int:
    order = {
        "core_promoted_tribe": 1,
        "review_tribe": 2,
        "near_tribe_customers": 3,
        "bridge_customers": 4,
        "generalist_or_broad_basket_customers": 5,
        "sparse_customers": 6,
        "long_tail_customers": 7,
    }
    return order.get(_as_text(value), 99)


def _coverage_row(frame: pl.DataFrame, coverage_group: str) -> dict[str, Any] | None:
    if frame.is_empty() or "coverage_group" not in frame.columns:
        return None
    scoped = frame.filter(pl.col("coverage_group") == coverage_group)
    if scoped.is_empty():
        return None
    return scoped.row(0, named=True)


def _tribe_node_id(value: Any) -> str:
    tribe_id = _to_int(value)
    return f"tribe_{tribe_id:02d}" if tribe_id is not None else "tribe_unknown"


def _slug(value: str) -> str:
    text = _as_text(value).lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join(part for part in "".join(chars).split("_") if part)


def _title_from_id(value: str) -> str:
    return _as_text(value).replace("_", " ").strip().title()


def _metric_unit(metric_id: str) -> str:
    if metric_id.endswith("_pct") or "share" in metric_id:
        return "pct"
    if "customers" in metric_id:
        return "customers"
    if "tribes" in metric_id:
        return "tribes"
    if "rows" in metric_id:
        return "rows"
    return "count"


def _source_stage_from_path(path: str) -> str:
    for stage in ["stage7", "stage6", "stage5", "stage4", "stage3", "stage2", "stage1"]:
        if stage in path.replace("\\", "/").lower():
            return stage
    return "stage8"


def _fmt_number(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _normalize_status(value: Any) -> str:
    text = _as_text(value).strip().lower()
    if text in {"pass", "passed", "ok", "completed", "computed", "strong", "usable"}:
        return "pass"
    if text in {"warn", "warning", "review", "action_needed"}:
        return "warn"
    if text in {"fail", "failed", "error", "missing"}:
        return "fail"
    return text or "unknown"


def _remaining_assignment_definition(coverage_group: str) -> dict[str, Any]:
    definitions = {
        "near_tribe_customers": {
            "meaning": "Customers near one retained tribe but not confidently enough to become hard tribe members.",
            "assignment_basis": "High top affinity and sufficient affinity margin, below the standard for hard HDBSCAN membership.",
            "recommended_dashboard_language": "Near-tribe expansion audience.",
            "segment_ids": ["near_tribe_fringe_customers"],
        },
        "bridge_customers": {
            "meaning": "Customers whose behavior sits between multiple tribes.",
            "assignment_basis": "Moderate affinity to more than one tribe with a small margin between top and second tribe.",
            "recommended_dashboard_language": "Bridge audience; do not use exclusive tribe messaging.",
            "segment_ids": ["bridge_customers_between_tribes"],
        },
        "generalist_or_broad_basket_customers": {
            "meaning": "Customers with broad baskets and useful value, but without a narrow product-first tribe identity.",
            "assignment_basis": "Behavior metrics show broad shopping rather than a distinctive dense product tribe.",
            "recommended_dashboard_language": "Broad-basket/generalist customers.",
            "segment_ids": ["high_value_broad_basket_customers", "broad_generalist_shoppers"],
        },
        "sparse_customers": {
            "meaning": "Low-signal customers with too few transactions or too little product evidence for reliable tribe assignment.",
            "assignment_basis": "Sparse customer behavior features and low-confidence affinity evidence.",
            "recommended_dashboard_language": "Sparse or low-signal customers.",
            "segment_ids": ["sparse_low_signal_shoppers"],
        },
        "long_tail_customers": {
            "meaning": "Customers in the long tail whose behavior is not safely represented by a promoted or review tribe.",
            "assignment_basis": "Low density, weak affinity, or unclear product signal after HDBSCAN/lift filtering.",
            "recommended_dashboard_language": "Unclear long-tail customers.",
            "segment_ids": ["unclear_long_tail_customers"],
        },
    }
    return definitions.get(
        coverage_group,
        {
            "meaning": "Remaining customer coverage group.",
            "assignment_basis": "Descriptive Stage 8 coverage grouping.",
            "recommended_dashboard_language": coverage_group,
            "segment_ids": [],
        },
    )


def _external_llm_name_proposals() -> dict[int, dict[str, str]]:
    """User-provided independent LLM proposals from tribe_analysis_independent.html.

    These are deliberately proposals. They should help the dashboard lead choose
    audience-friendly names, but they must be reviewed against Stage 8 evidence.
    """

    rows = {
        0: (
            "Devotee Cat Food Shoppers",
            "strong",
            "Premium wet cat food and treat SKUs dominate the product evidence, while broader baskets show these are grocery shoppers with a clear pet-care identity.",
            "Gourmet/Felix/Sheba-led; low promo use; broad basket.",
        ),
        1: (
            "In-Store Cafe Regulars",
            "review",
            "The product evidence centers on employee breakfast/cafe items, tostadas, coffee, pizza, and snack combos; this is better framed as a captive cafe visit mission than as grocery behavior.",
            "EUR22 basket; 19.4 tickets; weekday dominant; loyalty/churn caveat.",
        ),
        2: (
            "Travel-Format Personal Care Shoppers",
            "review",
            "Top products and co-purchases are miniature or travel-size toiletries and personal-care items, indicating a specific trip-preparation mission rather than normal grocery shopping.",
            "17 unique products; 61d recency; low loyalty; 3.4 tickets.",
        ),
        3: (
            "One-Off Home Setup Shoppers",
            "review",
            "General merchandise, tableware, cookware, and household setup products dominate; behavior indicates low-frequency mission shopping rather than recurring grocery identity.",
            "ELECTROFOTO overindex; high recency; low loyalty; narrow product set.",
        ),
        4: (
            "Baby & Toddler Food Shoppers",
            "usable",
            "Baby jars, toddler dairy, Smileat, HiPP, Hero Baby, and Nestle Yogolino dominate, making this a clear infant/toddler feeding-stage tribe with expected life-stage churn.",
            "73k customers; 72% loyalty; TEXTIL-coded baby products; short tenure caveat.",
        ),
        5: (
            "Latin Diaspora Weekly Shoppers",
            "strong",
            "The product evidence points to a specific Latin/Colombian pantry with high loyalty and broad baskets, indicating a recurring cultural-staples grocery mission.",
            "High spend; high loyalty; broad product breadth; strong cultural pantry signal.",
        ),
        6: (
            "Fresh Counter & Bakery Regulars",
            "strong",
            "FRQ/MERCA fresh-counter, bakery, eggs, fruit, and charcuterie items dominate, with low promo sensitivity and weekday shopping behavior.",
            "Lowest promo rate; Tuesday dominant; low weekend ratio; fresh/bakery led.",
        ),
        7: (
            "Bio Produce-Led Weekly Shoppers",
            "review",
            "Organic fresh produce is the distinctive signal inside a broader weekly grocery basket, so the name should emphasize bio produce without overclaiming a fully organic household.",
            "Largest tribe; bio vegetable lifts; broad weekly basket; review caveat.",
        ),
        8: (
            "Coeliac & Gluten-Free Lifestyle Shoppers",
            "strong",
            "Gluten-free products across Schar and related pantry/frozen items dominate, indicating a sustained dietary-need basket rather than occasional trial.",
            "Sin gluten signal; high co-purchase lifts; high loyalty.",
        ),
        9: (
            "Kids Clothing Occasion Shoppers",
            "review",
            "Children's clothing and outfit-building co-purchases define an occasion or seasonal wardrobe mission rather than habitual grocery shopping.",
            "TEXTIL overindex; low frequency; Sunday skew; recency caveat.",
        ),
        10: (
            "Weekday On-the-Go Lunch & Coffee Shoppers",
            "strong",
            "Cafe latte, cappuccino, sandwiches, and ready-meal lunch items define a frequent weekday convenience mission with retention risk.",
            "Tiny baskets; high frequency; weekday dominant; low loyalty.",
        ),
        11: (
            "Practical Protein & Deli Staples Shoppers",
            "usable",
            "Frequent small baskets around chicken, chorizo, pasta, and reliable protein-plus-side meals indicate habitual practical meal building.",
            "Highest ticket frequency; low promo use; strong recency.",
        ),
        12: (
            "Bulk Spirits Stock-Up Shoppers",
            "usable",
            "Large-format spirits and event-oriented co-purchases indicate pre-weekend or party provisioning rather than routine alcohol staples.",
            "High spend; low frequency; Friday dominant; large-format spirits.",
        ),
        13: (
            "Fresh Essentials Weekly Shoppers",
            "review",
            "Eggs, chicken, lactose-free dairy, and fresh essentials point to a frequent household weekly shop that overlaps with nearby fresh-protein tribes.",
            "High tickets; broad products; Monday skew; lactose-free signal.",
        ),
        14: (
            "High-Spend Family Pantry Stockers",
            "strong",
            "Large baskets of kids snacks, dairy, easy meals, and family pantry items make this a high-value family stock-up mission.",
            "Highest basket value; very high loyalty; child-centric pantry signal.",
        ),
        15: (
            "Health-Conscious Premium Shoppers",
            "review",
            "Dark chocolate, Greek cheese, kefir, quinoa, fresh salmon, nuts, and whole-food ingredients suggest a premium health-oriented basket.",
            "High spend; broad product breadth; premium/health product signal.",
        ),
        16: (
            "Weekend Party & Impulse Spend Shoppers",
            "strong",
            "Promo-heavy multipacks of drinks, ice cream, and party items dominate, indicating occasion stock-up driven by promotional mechanics.",
            "Highest promo rate; high spend; weekend skew; multipack lifts.",
        ),
        17: (
            "Wholesome Family Basics Shoppers",
            "usable",
            "A broad family shop mixing treats, proteins, yogurt, and household basics suggests a complete weekly household basket rather than a single-product mission.",
            "Broad products; high loyalty; low promo; family basics.",
        ),
        18: (
            "Traditional Home Cooking Shoppers",
            "review",
            "Whole fish, eggs, potatoes, artichokes, bread, and raw ingredients indicate from-scratch cooking behavior with a traditional fresh-food profile.",
            "Whole-fish signal; high loyalty; low basket value; Thursday skew.",
        ),
        19: (
            "Deli Counter Enthusiasts",
            "strong",
            "Counter-cut charcuterie, deli meats, cheeses, and al corte products dominate, making the deli-counter experience the defensible business identity.",
            "EUR787 spend; 89% loyalty; al corte/counter-cut signal; high basket value.",
        ),
        20: (
            "Budget Alcohol & Drink Aisle Regulars",
            "usable",
            "Own-label beer, budget spirits, and drink-aisle staples distinguish this frequent mainstream alcohol basket from premium party stock-up behavior.",
            "Own-label alcohol; frequent visits; budget drink signal.",
        ),
        21: (
            "Convenience-First Family Weekend Shoppers",
            "usable",
            "Tang, frozen pizza, nuggets, sweet snacks, and weekend behavior indicate a convenience-led family basket with broad shopping breadth.",
            "Sunday dominant; broad basket; Tang/frozen/snack signal.",
        ),
    }
    return {
        tribe_id: {
            "proposed_business_name": proposed_name,
            "proposal_confidence_label": confidence,
            "proposal_rationale": rationale,
            "proposal_stat_highlights": highlights,
        }
        for tribe_id, (proposed_name, confidence, rationale, highlights) in rows.items()
    }


def _business_name_readiness(confidence_label: str, has_proposal: bool) -> str:
    if not has_proposal:
        return "needs_business_name"
    if confidence_label == "strong":
        return "business_ready_candidate"
    if confidence_label == "usable":
        return "business_ready_candidate_with_minor_caveat"
    return "business_ready_candidate_with_review_caveat"


def _source_stages_for_product(name: str) -> list[str]:
    if name.startswith("rel_") or name in {"relational_manifest", "relational_schema", "relational_integrity_report"}:
        return ["stage6_8", "stage7", "stage8"]
    if name.startswith("embedding") or name.startswith("customer_"):
        return ["stage6", "stage6_8", "stage7"]
    if name in {"coverage_funnel", "opportunity_matrix", "segment_radar", "revenue_value_charts"}:
        return ["stage6_8", "stage7", "stage8"]
    if name in {"product_affinity_network", "tribe_similarity_network"}:
        return ["stage6_8", "stage7", "stage8"]
    if name.startswith("tribe_product") or name.startswith("tribe_categor"):
        return ["stage6_8", "stage7"]
    if name.startswith("tribe_deep") or name == "tribe_name_proposals":
        return ["stage6_8", "stage7", "stage8"]
    if name.startswith("remaining_customer"):
        return ["stage6_8", "stage7", "stage8"]
    if name.startswith("model") or name.startswith("pipeline") or name.startswith("validation") or name in {"feature_catalog", "data_quality_metrics"}:
        return ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"]
    if name.startswith("presentation") or name.startswith("executive") or name in {"dashboard_page_contracts", "visualization_specs", "artifact_manifest", "missing_file_register", "completeness_audit", "completeness_audit_json"}:
        return ["stage8"]
    return ["stage7", "stage8"]


def _pages_for_product(name: str) -> list[str]:
    if name.startswith("rel_") or name in {"relational_manifest", "relational_schema", "relational_integrity_report"}:
        return ["Technical Appendix", "all"]
    mapping = {
        "executive_metrics": ["Executive Overview", "Executive Insights & Final Recommendation"],
        "executive_metrics_table": ["Executive Overview", "Executive Insights & Final Recommendation"],
        "executive_insights": ["Executive Overview", "Executive Insights & Final Recommendation"],
        "presentation_storyline": ["all"],
        "presentation_storyline_table": ["all"],
        "tribe_master": ["Executive Overview", "Tribe Explorer", "Interactive Customer Landscape"],
        "tribe_profiles": ["Tribe Explorer"],
        "tribe_products": ["Tribe Explorer", "Product Affinity Explorer", "Segment Activation Center"],
        "tribe_categories": ["Product Affinity Explorer"],
        "tribe_similarity": ["Tribe Explorer"],
        "tribe_deep_dive": ["Tribe Explorer", "Executive Insights & Final Recommendation"],
        "tribe_deep_dive_json": ["Tribe Explorer", "Executive Insights & Final Recommendation"],
        "tribe_name_proposals": ["Tribe Explorer", "Executive Insights & Final Recommendation"],
        "customer_assignments": ["Interactive Customer Landscape", "Customer Coverage & Remaining Customers"],
        "customer_coverage": ["Interactive Customer Landscape", "Customer Coverage & Remaining Customers"],
        "customer_coverage_summary": ["Executive Overview", "Customer Coverage & Remaining Customers"],
        "remaining_customer_segments": ["Customer Coverage & Remaining Customers", "Segment Activation Center"],
        "remaining_customer_assignment": ["Customer Coverage & Remaining Customers", "Backend & Modeling Pipeline"],
        "segment_actions": ["Segment Activation Center", "Executive Insights & Final Recommendation"],
        "embedding_2d": ["Interactive Customer Landscape"],
        "embedding_3d": ["Interactive Customer Landscape"],
        "embedding_3d_sample": ["Interactive Customer Landscape"],
        "embedding_3d_pca": ["Interactive Customer Landscape", "Backend & Modeling Pipeline"],
        "embedding_3d_pca_sample": ["Interactive Customer Landscape", "Backend & Modeling Pipeline"],
        "embedding_centroids": ["Interactive Customer Landscape"],
        "embedding_sample_metadata": ["Interactive Customer Landscape", "Backend & Modeling Pipeline"],
        "product_affinity_network": ["Product Affinity Explorer"],
        "tribe_similarity_network": ["Tribe Explorer"],
        "coverage_funnel": ["Executive Overview", "Customer Coverage & Remaining Customers"],
        "opportunity_matrix": ["Segment Activation Center"],
        "segment_radar": ["Tribe Explorer", "Segment Activation Center"],
        "revenue_value_charts": ["Executive Overview", "Tribe Explorer"],
        "visualization_specs": ["Technical Appendix"],
        "model_metadata": ["Backend & Modeling Pipeline"],
        "model_cards": ["Backend & Modeling Pipeline"],
        "pipeline_lineage": ["Project & Data Foundation", "Backend & Modeling Pipeline"],
        "pipeline_stages": ["Project & Data Foundation", "Backend & Modeling Pipeline"],
        "feature_catalog": ["Project & Data Foundation", "Backend & Modeling Pipeline"],
        "validation_metrics": ["Backend & Modeling Pipeline"],
        "data_quality_metrics": ["Project & Data Foundation", "Backend & Modeling Pipeline"],
        "relational_manifest": ["Technical Appendix", "all"],
        "relational_schema": ["Technical Appendix", "all"],
        "relational_integrity_report": ["Technical Appendix", "Backend & Modeling Pipeline"],
        "artifact_manifest": ["Technical Appendix", "Reports & Downloads"],
        "dashboard_page_contracts": ["Technical Appendix"],
        "missing_file_register": ["Technical Appendix"],
        "completeness_audit": ["Technical Appendix"],
        "completeness_audit_json": ["Technical Appendix"],
        "readme": ["Technical Appendix"],
    }
    return mapping.get(name, ["Technical Appendix"])


def _artifact_grain(name: str) -> str:
    if name == "relational_manifest":
        return "JSON relational table catalog"
    if name == "relational_schema":
        return "SQL DDL for normalized Stage 8 model"
    if name == "relational_integrity_report":
        return "one row per relational integrity check"
    if name.startswith("rel_dim_"):
        if name == "rel_dim_tribe":
            return "one row per retained tribe dimension member"
        if name == "rel_dim_remaining_segment":
            return "one row per remaining-customer segment dimension member"
        if name == "rel_dim_product":
            return "one row per product dimension member"
        if name == "rel_dim_category":
            return "one row per category dimension member"
        if name == "rel_dim_action_target":
            return "one row per activation target dimension member"
        return "one row per dimension member"
    if name.startswith("rel_fact_"):
        if name in {"rel_fact_customer_assignment", "rel_fact_customer_value_behavior", "rel_fact_customer_nearest_tribe", "rel_fact_embedding_2d", "rel_fact_embedding_3d", "rel_fact_embedding_3d_pca"}:
            return "one row per customer"
        if name in {"rel_fact_embedding_3d_sample", "rel_fact_embedding_3d_pca_sample"}:
            return "one row per sampled customer"
        if name == "rel_fact_tribe_product_affinity":
            return "one row per tribe-product relationship"
        if name == "rel_fact_tribe_category_affinity":
            return "one row per tribe-category relationship"
        if name == "rel_fact_segment_action":
            return "one row per activation target"
        if name == "rel_fact_embedding_centroid":
            return "one row per landscape centroid"
        return "one row per fact grain"
    if name.startswith("rel_bridge_"):
        return "one row per bridge relationship"
    grains = {
        "executive_metrics": "JSON list of KPIs",
        "executive_metrics_table": "one row per KPI",
        "executive_insights": "one row per ordered executive insight",
        "presentation_storyline": "JSON chapters and demo moments",
        "presentation_storyline_table": "one row per chapter/page",
        "tribe_master": "one row per retained hard tribe",
        "tribe_profiles": "one row per retained hard tribe",
        "tribe_products": "one row per tribe-product relationship",
        "tribe_categories": "one row per tribe-category relationship",
        "tribe_similarity": "one row per tribe pair",
        "tribe_deep_dive": "one row per retained hard tribe",
        "tribe_deep_dive_json": "nested record per retained hard tribe",
        "tribe_name_proposals": "one row per retained hard tribe",
        "customer_assignments": "one row per customer",
        "customer_coverage": "one row per customer",
        "customer_coverage_summary": "one row per coverage group",
        "remaining_customer_segments": "one row per remaining-customer segment",
        "remaining_customer_assignment": "JSON policy and group explanation",
        "segment_actions": "one row per segment/action target",
        "embedding_2d": "one row per customer",
        "embedding_3d": "one row per customer",
        "embedding_3d_sample": "one row per sampled customer",
        "embedding_3d_pca": "one row per customer",
        "embedding_3d_pca_sample": "one row per sampled customer",
        "embedding_centroids": "one row per tribe/group centroid",
        "embedding_sample_metadata": "JSON sampling policy and coverage checks",
        "product_affinity_network": "one row per tribe-product or tribe-category edge",
        "tribe_similarity_network": "one row per tribe-pair network edge",
        "coverage_funnel": "one row per coverage funnel step",
        "opportunity_matrix": "one row per tribe or remaining segment",
        "segment_radar": "one row per segment/radar metric",
        "revenue_value_charts": "one row per ready chart datum",
        "visualization_specs": "one row per dashboard visual",
        "model_metadata": "nested backend metadata JSON",
        "model_cards": "one row per model/algorithm",
        "pipeline_lineage": "nested lineage JSON",
        "pipeline_stages": "one row per pipeline stage",
        "feature_catalog": "one row per feature group",
        "validation_metrics": "one row per validation metric",
        "data_quality_metrics": "one row per quality metric",
        "artifact_manifest": "one row per Stage 8 artifact",
        "dashboard_page_contracts": "one row per dashboard page",
        "missing_file_register": "one row per optional/missing enhancement",
        "completeness_audit": "Markdown audit report",
        "completeness_audit_json": "JSON audit report",
        "readme": "Markdown handoff guide",
        "readiness_csv": "one row per readiness check",
        "readiness_md": "Markdown readiness report",
    }
    return grains.get(name, "artifact")


def _artifact_primary_key(name: str) -> str:
    if name.startswith("rel_"):
        return _relational_primary_key(name)
    if name in {"tribe_master", "tribe_profiles", "tribe_deep_dive", "tribe_name_proposals"}:
        return "tribe_id"
    if name in {"customer_assignments", "customer_coverage", "embedding_2d", "embedding_3d", "embedding_3d_pca"}:
        return "cliente"
    if name in {"embedding_3d_sample", "embedding_3d_pca_sample"}:
        return "cliente"
    if name == "remaining_customer_segments":
        return "segment_id"
    if name == "customer_coverage_summary":
        return "coverage_group"
    if name == "executive_metrics_table":
        return "metric_id"
    if name == "executive_insights":
        return "insight_id"
    if name == "presentation_storyline_table":
        return "chapter_id"
    if name == "model_cards":
        return "model_name"
    if name == "pipeline_stages":
        return "stage_id"
    if name == "feature_catalog":
        return "feature_name"
    if name == "artifact_manifest":
        return "artifact_name"
    if name == "dashboard_page_contracts":
        return "page_id"
    if name == "visualization_specs":
        return "visual_id"
    return ""


def _artifact_foreign_keys(name: str) -> str:
    if name.startswith("rel_"):
        return _relational_foreign_keys(name)
    if name.startswith("tribe_") or name in {"segment_actions", "opportunity_matrix", "segment_radar"}:
        return "tribe_id where applicable; segment_id where applicable"
    if name.startswith("customer") or name.startswith("embedding"):
        return "cliente; tribe_id; coverage_group; segment_id"
    if name in {"coverage_funnel", "customer_coverage_summary"}:
        return "coverage_group"
    if name in {"product_affinity_network"}:
        return "tribe_id; product_id; category"
    return ""


def _artifact_purpose(name: str) -> str:
    purpose = {
        "relational_manifest": "Normalized relational table catalog for dashboard joins and canonical source-of-truth rules.",
        "relational_schema": "SQL DDL for loading the normalized Stage 8 model into a relational database.",
        "relational_integrity_report": "Primary-key, foreign-key, coordinate, and separation-of-concerns checks for the normalized model.",
        "artifact_manifest": "Structured Stage 8 output inventory for dashboard loading and audit.",
        "dashboard_page_contracts": "Renderer contract for each dashboard page.",
        "visualization_specs": "Renderer contract for each interactive visual.",
        "missing_file_register": "Known optional gaps and non-blocking enhancements.",
        "completeness_audit": "Strict audit response for dashboard data model completeness.",
    }
    if name in purpose:
        return purpose[name]
    pages = ", ".join(_pages_for_product(name))
    return f"Supports {pages}."


def _artifact_audience(name: str) -> str:
    if name.startswith("rel_") or name in {"relational_manifest", "relational_schema", "relational_integrity_report"}:
        return "both"
    if name in {"model_metadata", "model_cards", "pipeline_lineage", "pipeline_stages", "feature_catalog", "validation_metrics", "data_quality_metrics"}:
        return "technical-facing"
    if name in {"readiness_csv", "readiness_md", "artifact_manifest", "dashboard_manifest", "completeness_audit", "completeness_audit_json"}:
        return "both"
    return "business-facing" if not name.startswith("embedding") else "both"


def _artifact_classification(name: str) -> str:
    if name.startswith("rel_") or name in {"relational_manifest", "relational_schema", "relational_integrity_report"}:
        return "normalized_source_of_truth"
    if name in {"completeness_audit", "completeness_audit_json", "readiness_md", "readme"}:
        return "optional_human_reference"
    if name in {"embedding_3d", "embedding_2d", "customer_coverage"}:
        return "required_and_sufficient_full_grain"
    if name in {"missing_file_register"}:
        return "optional_gap_register"
    return "required_and_sufficient"


def _row_count(path: Path) -> int | None:
    try:
        if path.suffix.lower() == ".parquet" and path.exists():
            return int(pl.scan_parquet(path).select(pl.len()).collect().item())
        if path.suffix.lower() == ".csv" and path.exists():
            return int(pl.scan_csv(path).select(pl.len()).collect().item())
    except Exception:
        return None
    return None


def _schema_for_path(path: Path) -> list[str]:
    try:
        if path.suffix.lower() == ".parquet" and path.exists():
            return pl.scan_parquet(path).collect_schema().names()
        if path.suffix.lower() == ".csv" and path.exists():
            return pl.scan_csv(path).collect_schema().names()
        if path.suffix.lower() == ".json" and path.exists():
            payload = _read_json(path)
            return sorted(payload.keys())
    except Exception:
        return []
    return []


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _looks_metric_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, (int, float, bool)):
        return True
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"pass", "fail", "warn", "review", "strong", "usable", "computed", "action_needed"}:
            return True
        try:
            float(text)
            return True
        except ValueError:
            return False
    return False


def _hover_label_expr() -> pl.Expr:
    label = pl.when(pl.col("tribe_id") < 0).then(pl.col("segment_name")).otherwise(pl.col("tribe_name"))
    return pl.concat_str(
        [
            pl.lit("Customer "),
            pl.col("cliente").cast(pl.Utf8),
            pl.lit(" | "),
            label.fill_null("unassigned"),
            pl.lit(" | "),
            pl.col("coverage_group").fill_null("unknown"),
        ]
    )


def _filter_tokens_expr() -> pl.Expr:
    return pl.concat_str(
        [
            pl.col("coverage_group").fill_null("unknown"),
            pl.lit("|"),
            pl.col("tribe_status").fill_null("remaining_customer"),
            pl.lit("|"),
            pl.col("revenue_tier").fill_null("unknown_value"),
        ]
    )


def _series_quantile(series: pl.Series | None, q: float) -> float:
    if series is None or series.is_empty():
        return 0.0
    value = series.drop_nulls().quantile(q)
    return float(value) if value is not None else 0.0


def _embedding_is_finite(frame: pl.DataFrame) -> bool:
    if frame.is_empty():
        return False
    for column in ["x", "y", "z"]:
        if column not in frame.columns:
            return False
        values = frame[column].drop_nulls()
        if values.len() != frame.height:
            return False
        if not values.is_finite().all():
            return False
    return True


def _sample_covers_groups(full: pl.DataFrame, sample: pl.DataFrame) -> bool:
    keys = [col for col in ["coverage_group", "tribe_id", "segment_id"] if col in full.columns and col in sample.columns]
    if not keys or full.is_empty() or sample.is_empty():
        return False
    full_groups = set(tuple(row) for row in full.select(keys).unique().iter_rows())
    sample_groups = set(tuple(row) for row in sample.select(keys).unique().iter_rows())
    return full_groups.issubset(sample_groups)


def _check(check_id: str, area: str, severity: str, ok: bool, details: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "check_area": area,
        "severity": severity,
        "status": "pass" if ok else ("fail" if severity == "critical" else "warn"),
        "details": details,
    }


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> None:
    paths = write_stage8_dashboard_pack(cfg=CONFIG)
    print(paths["dashboard_manifest"])


if __name__ == "__main__":
    main()
