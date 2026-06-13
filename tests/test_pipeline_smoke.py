from copy import deepcopy
from dataclasses import replace
from datetime import date
import hashlib
from pathlib import Path
import shutil

import polars as pl
import pytest

from src.basket_builder import build_basket_sentences
from src.cluster_validation import build_cluster_validity_stability_report
from src.customer_embeddings import build_customer_embeddings
from src.config import configure_mode, load_config
from src.experiment_sandbox import (
    evaluate_product_embedding_quality,
    run_feature_set_experiments,
    run_customer_embedding_experiments,
)
from src.feature_engineering import build_behavioral_features, build_feature_set
from src.model_selection import build_candidate_model_diagnostics, run_candidate_model_suite, run_umap_hdbscan_experiments
from src.profiling import flatten_profiles_for_csv
from src.visualization import (
    plot_basket_summary,
    plot_behavioral_feature_summary,
    plot_customer_embedding_diagnostics,
    plot_embedding_validation_summary,
    plot_feature_set_summary,
    plot_prepared_data_overview,
    plot_product_embedding_diagnostics,
    plot_stage6_model_diagnostics,
)


@pytest.fixture
def workspace_tmp(request):
    base = Path(".pt")
    root = base / hashlib.blake2b(request.node.nodeid.encode("utf-8"), digest_size=6).hexdigest()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
        try:
            base.rmdir()
        except OSError:
            pass


def _tiny_transactions() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "cliente": ["c1", "c1", "c1", "c2"],
            "ticket": ["t1", "t1", "t2", "t3"],
            "fecha": [date(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 5), date(2026, 1, 3)],
            "idarticu": [101, 102, 101, 103],
            "importe": [2.0, 3.0, 4.0, 5.0],
            "unidades": [1, 2, 1, 1],
            "idpromoc": ["No promo", "Promo", "No promo", "Promo"],
            "idsector": [1, 1, 2, 3],
            "desc_larga_articulo": ["A", "B", "A", "C"],
            "desc_sector": ["S1", "S1", "S2", "S3"],
        }
    ).lazy()


def test_load_config_dev_paths():
    cfg = load_config("dev")
    assert cfg.mode == "dev"
    assert cfg.data_processed.name == "dev"
    assert cfg.outputs.name == "dev"
    assert cfg.models == cfg.outputs / "models"
    assert cfg.experiments == cfg.outputs / "experiments"
    assert cfg.model_selection == cfg.reports / "model_selection"
    assert cfg.model_selection_cache == cfg.models / "model_selection"


def test_configure_mode_updates_notebook_globals():
    import src.config as config_module

    initial_mode = config_module.CONFIG.mode
    try:
        cfg = configure_mode("prod")
        assert cfg.mode == "prod"
        assert config_module.CONFIG.mode == "prod"
        assert config_module.MODE == "prod"
        assert config_module.OUTPUTS.name == "prod"
    finally:
        configure_mode(initial_mode)


def test_prod_config_matches_dev_method_with_scale_overrides(workspace_tmp):
    dev_cfg = load_config("dev")
    prod_cfg = replace(load_config("prod"), root=workspace_tmp / "prod_root")

    assert dev_cfg.experiments_enabled is True
    assert prod_cfg.experiments_enabled is False
    prod_cfg.ensure_directories()
    assert not prod_cfg.experiments.exists()

    assert prod_cfg.get("word2vec.vector_size") == dev_cfg.get("word2vec.vector_size")
    assert prod_cfg.get("word2vec.workers") == dev_cfg.get("word2vec.workers")
    assert prod_cfg.get("umap.n_components") == dev_cfg.get("umap.n_components")
    assert prod_cfg.get("umap.n_neighbors") == dev_cfg.get("umap.n_neighbors")
    assert prod_cfg.get("hdbscan.min_samples") == dev_cfg.get("hdbscan.min_samples")
    assert prod_cfg.get("hdbscan.cluster_selection_method") == dev_cfg.get("hdbscan.cluster_selection_method")
    assert prod_cfg.get("pca.n_components") == dev_cfg.get("pca.n_components")

    assert prod_cfg.get("modeling.evaluation_sample_size") > dev_cfg.get("modeling.evaluation_sample_size")
    assert prod_cfg.get("modeling.fit_sample_size") == 300000
    assert prod_cfg.get("hdbscan.min_cluster_size") > dev_cfg.get("hdbscan.min_cluster_size")


def test_experiment_helpers_are_disabled_in_prod(workspace_tmp):
    cfg = replace(load_config("prod"), root=workspace_tmp / "prod_experiment_root")
    feature_path = workspace_tmp / "does_not_need_to_exist.parquet"

    with pytest.raises(RuntimeError, match="disabled"):
        run_feature_set_experiments({"toy": feature_path}, cfg=cfg)

    with pytest.raises(RuntimeError, match="disabled"):
        run_umap_hdbscan_experiments(feature_path, trials=[], cfg=cfg)


def test_generate_dev_subset_module_imports():
    import src.generate_dev_subset as generate_dev_subset

    assert generate_dev_subset.MIN_TICKETS_PER_CUSTOMER == 3


def test_official_stage6_suite_uses_non_targeted_search_ranges(workspace_tmp):
    base_cfg = load_config("dev")
    values = deepcopy(base_cfg.values)
    values["official_model_suite"] = {
        "include_gmm": True,
        "include_raw_hdbscan": False,
        "include_umap_hdbscan": False,
        "include_autoencoder": False,
        "include_pca_kmeans": True,
    }
    values["modeling"] = {**values.get("modeling", {}), "evaluation_sample_size": 12, "fit_sample_size": None}
    values["gmm"] = {**values.get("gmm", {}), "components_min": 3, "components_max": 4, "max_iter": 50, "n_init": 1}
    values["pca"] = {**values.get("pca", {}), "n_components": 2}
    values["kmeans"] = {**values.get("kmeans", {}), "k_min": 3, "k_max": 4, "batch_size": 16, "n_init": 1}
    cfg = replace(base_cfg, root=workspace_tmp / "official_suite_root", values=values)

    feature_path = cfg.outputs / "features" / "toy_features.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 18
    pl.DataFrame(
        {
            "cliente": [f"c{i:02d}" for i in range(rows)],
            "emb_000": [float((i // 6) * 5 + (i % 3) * 0.1) for i in range(rows)],
            "emb_001": [float((i % 6) * 0.2) for i in range(rows)],
        }
    ).write_parquet(feature_path)

    suite = run_candidate_model_suite(feature_path, force=True, cfg=cfg)
    model_names = [row["model_name"] for row in suite["candidate_results"]]
    gmm_rows = pl.read_parquet(suite["result_paths"][next(key for key in suite["result_paths"] if key.startswith("model_a_gmm"))])
    kmeans_rows = pl.read_parquet(
        suite["result_paths"][next(key for key in suite["result_paths"] if key.startswith("model_c_pca_kmeans"))]
    )

    assert model_names == ["model_a_gmm", "model_c_pca_kmeans"]
    assert gmm_rows.height == 2
    assert kmeans_rows.height == 2
    assert all(path.parent == cfg.model_selection_cache for path in suite["result_paths"].values())
    assert (cfg.models / "model_a_gmm_model.pkl").exists()
    assert (cfg.models / "model_c_pca_kmeans_model.pkl").exists()


def test_default_ml_artifacts_use_output_folders(workspace_tmp):
    cfg = replace(load_config("dev"), root=workspace_tmp / "path_contract_root")
    transactions = _tiny_transactions()

    basket_path = build_basket_sentences(transactions=transactions, cfg=cfg, force=True)
    assert basket_path.parent == cfg.outputs / "embeddings"
    assert (cfg.outputs / ".artifact_metadata.json").exists()
    assert not basket_path.with_name(f"{basket_path.name}.meta.json").exists()

    product_embeddings_path = cfg.outputs / "embeddings" / "product_embeddings.parquet"
    product_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "idarticu": [101, 102, 103],
            "emb_000": [1.0, 0.0, 0.5],
            "emb_001": [0.0, 1.0, 0.5],
        }
    ).write_parquet(product_embeddings_path)

    customer_embedding_path = build_customer_embeddings(
        product_embeddings_path,
        transactions=transactions,
        cfg=cfg,
        force=True,
    )
    assert customer_embedding_path.parent == cfg.outputs / "features"

    behavior_path = build_behavioral_features(transactions=transactions, cfg=cfg, force=True)
    assert behavior_path.parent == cfg.outputs / "features"

    feature_set_path = build_feature_set(customer_embedding_path, cfg=cfg, force=True)
    assert feature_set_path.parent == cfg.outputs / "features"


def test_customer_embeddings_use_quantity_not_spend(workspace_tmp):
    cfg = replace(load_config("dev"), root=workspace_tmp / "quantity_embedding_root")
    embedding_path = cfg.outputs / "embeddings" / "toy_product_embeddings.parquet"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "idarticu": [101, 102],
            "emb_000": [1.0, 0.0],
            "emb_001": [0.0, 1.0],
        }
    ).write_parquet(embedding_path)
    transactions = pl.DataFrame(
        {
            "cliente": ["c1", "c1"],
            "ticket": ["t1", "t1"],
            "fecha": [date(2026, 1, 1), date(2026, 1, 1)],
            "idarticu": [101, 102],
            "unidades": [3, 1],
        }
    ).lazy()

    output = build_customer_embeddings(embedding_path, transactions=transactions, cfg=cfg, force=True)
    row = pl.read_parquet(output).row(0, named=True)

    assert row["emb_000"] == pytest.approx(0.75)
    assert row["emb_001"] == pytest.approx(0.25)
    assert row["embedding_weight_sum"] == pytest.approx(4.0)


def test_basket_builder_uses_ticket_sentences(workspace_tmp):
    output = workspace_tmp / "baskets.parquet"
    build_basket_sentences(
        transactions=_tiny_transactions(),
        output_path=output,
        repeat_product_by_quantity=False,
        force=True,
    )
    baskets = pl.read_parquet(output).sort("ticket")
    assert baskets.shape[0] == 3
    products = baskets.filter(pl.col("ticket") == "t1").select("products").row(0)[0]
    assert set(products) == {"101", "102"}
    assert len(products) == 2


def test_behavioral_features_parse_no_promo(workspace_tmp):
    output = workspace_tmp / "behavior.parquet"
    build_behavioral_features(transactions=_tiny_transactions(), output_path=output, force=True)
    features = pl.read_parquet(output).sort("cliente")
    c1 = features.filter(pl.col("cliente") == "c1").row(0, named=True)
    c2 = features.filter(pl.col("cliente") == "c2").row(0, named=True)
    assert c1["ticket_count"] == 2
    assert c1["promo_line_share"] == 1 / 3
    assert c1["promo_basket_share"] == 1 / 2
    assert c1["promo_share"] == 1 / 3
    assert c2["promo_line_share"] == 1.0
    assert c2["promo_basket_share"] == 1.0
    assert c2["promo_share"] == 1.0


def test_flatten_profiles_for_csv_serializes_nested_columns(workspace_tmp):
    profile_path = workspace_tmp / "nested_profiles.parquet"
    csv_path = workspace_tmp / "nested_profiles.csv"
    pl.DataFrame(
        {
            "tribe_id": [0],
            "n_customers": [12],
            "top_product_ids": [["101", "102"]],
            "top_products": [["Milk", "Bread"]],
            "top_product_lifts": [[2.5, 1.8]],
            "top_product_customer_counts": [[10, 8]],
            "top_sectors": [["Fresh", "Grocery"]],
            "top_sector_lifts": [[1.4, 1.2]],
        }
    ).write_parquet(profile_path)

    flatten_profiles_for_csv(profile_path, csv_path)
    exported = pl.read_csv(csv_path)

    row = exported.row(0, named=True)
    assert row["top_products"] == "Milk; Bread"
    assert row["top_product_customer_counts"] == "10; 8"


def test_stage6_diagnostics_collects_and_plots_candidates(workspace_tmp):
    cfg = replace(load_config("dev"), root=workspace_tmp / "diagnostics_root")
    result_dir = cfg.model_selection
    result_dir.mkdir(parents=True, exist_ok=True)
    gmm_results = result_dir / "model_a_gmm_grid_results.parquet"
    hdbscan_results = result_dir / "model_b_umap_hdbscan_results.parquet"

    pl.DataFrame(
        {
            "model": ["Model A", "Model A"],
            "model_id": ["model_a_gmm", "model_a_gmm"],
            "model_name": ["model_a_gmm", "model_a_gmm"],
            "algorithm_name": ["GaussianMixture", "GaussianMixture"],
            "model_variant": ["gmm_k6", "gmm_k7"],
            "cluster_count": [6, 7],
            "coverage_adjusted_silhouette": [0.2, 0.3],
            "silhouette": [0.2, 0.3],
            "davies_bouldin": [1.4, 1.1],
            "cluster_size_cv": [0.5, 0.4],
            "noise_pct": [0.0, 0.0],
            "coverage_pct": [100.0, 100.0],
            "passes_quality_gate": [True, True],
            "selected_within_family": [False, True],
        }
    ).write_parquet(gmm_results)
    pl.DataFrame(
        {
            "model": ["Model B"],
            "model_id": ["model_b_umap_hdbscan"],
            "model_name": ["model_b_umap_hdbscan"],
            "algorithm_name": ["UMAP_HDBSCAN"],
            "model_variant": ["umap_hdbscan_mcs150_ms5_leaf"],
            "cluster_count": [0],
            "coverage_adjusted_silhouette": [None],
            "silhouette": [None],
            "davies_bouldin": [None],
            "cluster_size_cv": [None],
            "noise_pct": [100.0],
            "coverage_pct": [0.0],
            "passes_quality_gate": [False],
            "selected_within_family": [False],
        }
    ).write_parquet(hdbscan_results)

    diagnostics = build_candidate_model_diagnostics(
        {"result_paths": {"a": gmm_results, "b": hdbscan_results}},
        cfg=cfg,
    )
    figure = plot_stage6_model_diagnostics(diagnostics["parquet"], cfg=cfg)
    ranked = pl.read_parquet(diagnostics["parquet"]).sort("stage6_rank")

    assert "csv" not in diagnostics
    assert diagnostics["parquet"].parent == cfg.model_selection
    assert ranked["candidate_id"].to_list()[0] == "model_a_gmm::gmm_k7"
    assert diagnostics["summary_csv"].exists()
    assert diagnostics["summary_md"].exists()
    assert figure.exists()


def test_cluster_validity_stability_report_scores_assignments(workspace_tmp):
    cfg = replace(load_config("dev"), root=workspace_tmp / "cluster_validity_root")
    values = deepcopy(cfg.values)
    values["stability"] = {"sample_size": 12, "repeats": 2, "jitter_scale": 0.01}
    cfg = replace(cfg, values=values)
    feature_path = cfg.outputs / "features" / "toy_features.parquet"
    assignment_path = cfg.model_selection_cache / "cluster_assignments_model_a_gmm.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 12
    clientes = [f"c{i:02d}" for i in range(rows)]
    pl.DataFrame(
        {
            "cliente": clientes,
            "emb_000": [0.0] * 6 + [5.0] * 6,
            "emb_001": list(range(6)) + list(range(6)),
        }
    ).write_parquet(feature_path)
    pl.DataFrame(
        {
            "cliente": clientes,
            "tribe_id": [0] * 6 + [1] * 6,
            "model_name": ["model_a_gmm"] * rows,
            "model_variant": ["gmm_k2"] * rows,
        }
    ).write_parquet(assignment_path)

    report = build_cluster_validity_stability_report(
        {"assignment_paths": {"model_a_gmm::gmm_k2": assignment_path}},
        feature_path,
        cfg=cfg,
    )
    scored = pl.read_parquet(report["parquet"]).row(0, named=True)

    assert report["summary_csv"].exists()
    assert report["summary_md"].exists()
    assert scored["cluster_count"] == 2
    assert scored["jitter_ari_mean"] is not None


def test_embedding_quality_scores_product_neighborhoods(workspace_tmp):
    cfg = replace(load_config("dev"), root=workspace_tmp / "embedding_sandbox_root")
    embedding_path = cfg.outputs / "embeddings" / "toy_product_embeddings.parquet"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "idarticu": [101, 102, 103],
            "emb_000": [1.0, 0.9, 0.0],
            "emb_001": [0.0, 0.1, 1.0],
        }
    ).write_parquet(embedding_path)

    metrics = evaluate_product_embedding_quality(
        embedding_path,
        transactions=_tiny_transactions(),
        trial_name="toy",
        cfg=cfg,
    )

    assert metrics["vector_dims"] == 2
    assert metrics["product_vocab_coverage_pct"] == 100.0
    assert metrics["same_sector_at_1_pct"] > 0
    assert metrics["embedding_quality_score"] > 0


def test_customer_embedding_experiments_write_ranked_diagnostics(workspace_tmp):
    cfg = replace(load_config("dev"), root=workspace_tmp / "customer_embedding_sandbox_root")
    embedding_path = cfg.outputs / "embeddings" / "toy_product_embeddings.parquet"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "idarticu": [101, 102, 103],
            "emb_000": [1.0, 0.0, 0.5],
            "emb_001": [0.0, 1.0, 0.5],
        }
    ).write_parquet(embedding_path)

    result = run_customer_embedding_experiments(
        embedding_path,
        trials=[
            {"name": "quantity", "weight_strategy": "quantity", "normalize_vectors": False},
            {"name": "quantity_idf", "weight_strategy": "quantity_idf", "normalize_vectors": False},
            {"name": "equal_l2", "weight_strategy": "equal", "normalize_vectors": True},
        ],
        experiment_name="toy_customer_embeddings",
        sample_size=4,
        transactions=_tiny_transactions(),
        force=True,
        cfg=cfg,
    )
    ranked = pl.read_parquet(result["diagnostics"]["parquet"]).sort("customer_embedding_sandbox_rank")
    equal_l2 = pl.read_parquet(result["customer_embedding_paths"]["equal_l2"]).sort("cliente")
    l2_norm = (equal_l2["emb_000"][0] ** 2 + equal_l2["emb_001"][0] ** 2) ** 0.5

    assert "csv" not in result["diagnostics"]
    assert result["diagnostics"]["summary_csv"].exists()
    assert result["diagnostics"]["summary_md"].exists()
    assert ranked.height == 3
    assert ranked["customer_embedding_sandbox_rank"].to_list() == [1, 2, 3]
    assert result["customer_embedding_paths"]["quantity"].parent == cfg.experiments / "toy_customer_embeddings"
    assert result["customer_embedding_paths"]["quantity_idf"].parent == cfg.experiments / "toy_customer_embeddings"
    assert abs(l2_norm - 1.0) < 1e-6


def test_stage_visualizations_write_figure_files(workspace_tmp):
    cfg = replace(load_config("dev"), root=workspace_tmp / "stage_figures_root")
    transactions = _tiny_transactions()

    basket_path = build_basket_sentences(transactions=transactions, cfg=cfg, force=True)
    embedding_path = cfg.outputs / "embeddings" / "toy_product_embeddings.parquet"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "idarticu": [101, 102, 103],
            "emb_000": [1.0, 0.0, 0.5],
            "emb_001": [0.0, 1.0, 0.5],
        }
    ).write_parquet(embedding_path)
    customer_embedding_path = build_customer_embeddings(
        embedding_path,
        transactions=transactions,
        cfg=cfg,
        force=True,
    )
    behavior_path = build_behavioral_features(transactions=transactions, cfg=cfg, force=True)
    feature_set_path = build_feature_set(customer_embedding_path, cfg=cfg, force=True)
    validation_csv = cfg.reports / "toy_embedding_validation.csv"
    validation_csv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "product_id": [101, 101, 102, 102],
            "product_sector": ["S1", "S1", "S1", "S1"],
            "neighbor_rank": [1, 2, 1, 2],
            "neighbor_sector": ["S1", "S2", "S1", "S3"],
            "cosine_similarity": [0.9, 0.7, 0.85, 0.5],
        }
    ).write_csv(validation_csv)

    figures = [
        plot_prepared_data_overview(transactions, cfg=cfg),
        plot_basket_summary(basket_path, cfg=cfg),
        plot_product_embedding_diagnostics(embedding_path, cfg=cfg),
        plot_embedding_validation_summary(validation_csv, cfg=cfg),
        plot_customer_embedding_diagnostics(customer_embedding_path, cfg=cfg),
        plot_behavioral_feature_summary(behavior_path, cfg=cfg),
        plot_feature_set_summary({"embeddings_only": feature_set_path}, cfg=cfg),
    ]

    assert all(path.exists() for path in figures)
    assert all(path.parent == cfg.figures for path in figures)


def test_feature_set_experiments_write_ranked_diagnostics(workspace_tmp):
    cfg = replace(load_config("dev"), root=workspace_tmp / "feature_sandbox_root")
    feature_dir = cfg.outputs / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows = 12
    feature_a = feature_dir / "feature_set_a.parquet"
    feature_b = feature_dir / "feature_set_b.parquet"
    pl.DataFrame(
        {
            "cliente": [f"c{i:02d}" for i in range(rows)],
            "emb_000": [0.0] * 6 + [5.0] * 6,
            "emb_001": list(range(6)) + list(range(6)),
        }
    ).write_parquet(feature_a)
    pl.DataFrame(
        {
            "cliente": [f"c{i:02d}" for i in range(rows)],
            "emb_000": [float(i % 3) for i in range(rows)],
            "emb_001": [float(i % 4) for i in range(rows)],
        }
    ).write_parquet(feature_b)

    result = run_feature_set_experiments(
        {"a": feature_a, "b": feature_b},
        experiment_name="toy_feature_sets",
        k_values=[2, 3],
        cfg=cfg,
    )
    ranked = pl.read_parquet(result["diagnostics"]["parquet"]).sort("feature_set_sandbox_rank")

    assert "csv" not in result["diagnostics"]
    assert result["diagnostics"]["summary_csv"].exists()
    assert result["diagnostics"]["summary_md"].exists()
    assert ranked.height == 4
    assert ranked["feature_set_sandbox_rank"].to_list() == [1, 2, 3, 4]


def test_experiment_notebook_is_separate_from_official_pipeline():
    pipeline = Path("notebooks/03_ml_pipeline.ipynb").read_text(encoding="utf-8")
    sandbox = Path("notebooks/04_experiment_sandbox.ipynb").read_text(encoding="utf-8")

    assert "Stage 2B" not in pipeline
    assert "Stage 6B" not in pipeline
    assert "04_experiment_sandbox.ipynb" in pipeline
    assert "04 Experiment Sandbox" in sandbox
    assert "run_umap_hdbscan_experiments" in sandbox
    assert "RUN_UMAP_HDBSCAN_SANDBOX = False" in sandbox
    assert "quantity_idf" in sandbox
    assert "target_15" not in sandbox
    assert "target_tribes" not in sandbox
    assert "run_candidate_model_suite" in pipeline
    assert "build_cluster_validity_stability_report" in pipeline
