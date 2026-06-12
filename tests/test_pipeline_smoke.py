from datetime import date
from pathlib import Path
from dataclasses import replace

import polars as pl

from src.basket_builder import build_basket_sentences
from src.customer_embeddings import build_customer_embeddings
from src.config import load_config
from src.feature_engineering import build_behavioral_features, build_feature_set
from src.model_selection import build_candidate_model_diagnostics
from src.profiling import flatten_profiles_for_csv
from src.visualization import plot_stage6_model_diagnostics


ARTIFACT_DIR = Path("outputs/dev/test_smoke")


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
    assert cfg.models.name == "dev"


def test_default_ml_artifacts_use_output_folders():
    cfg = replace(load_config("dev"), root=ARTIFACT_DIR / "path_contract_root")
    transactions = _tiny_transactions()

    basket_path = build_basket_sentences(transactions=transactions, cfg=cfg, force=True)
    assert basket_path.parent == cfg.outputs / "embeddings"

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


def test_basket_builder_uses_ticket_sentences():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = ARTIFACT_DIR / "baskets.parquet"
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


def test_behavioral_features_parse_no_promo():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = ARTIFACT_DIR / "behavior.parquet"
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


def test_flatten_profiles_for_csv_serializes_nested_columns():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = ARTIFACT_DIR / "nested_profiles.parquet"
    csv_path = ARTIFACT_DIR / "nested_profiles.csv"
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


def test_stage6_diagnostics_collects_and_plots_candidates():
    cfg = replace(load_config("dev"), root=ARTIFACT_DIR / "diagnostics_root")
    result_dir = cfg.outputs / "model_selection"
    result_dir.mkdir(parents=True, exist_ok=True)
    gmm_results = result_dir / "model_a_gmm_grid_results.parquet"
    hdbscan_results = result_dir / "model_b_hdbscan_results.parquet"

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
            "model_id": ["model_b_hdbscan"],
            "model_name": ["model_b_hdbscan"],
            "algorithm_name": ["HDBSCAN"],
            "model_variant": ["hdbscan_mcs150_ms5_leaf"],
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

    assert ranked["candidate_id"].to_list()[0] == "model_a_gmm::gmm_k7"
    assert figure.exists()
