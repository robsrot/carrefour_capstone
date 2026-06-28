import polars as pl

from src.config import PipelineConfig
from src.feature_engineering import build_feature_set, build_product_exposure_features


def _test_config(tmp_path):
    return PipelineConfig(
        values={
            "run": {"random_seed": 42},
            "paths": {
                "dev": "data/dev",
                "processed": "data/processed",
                "raw_csv": "data/raw/csv",
                "raw_parquet": "data/raw/parquet",
                "outputs": "outputs",
            },
            "data": {"prepared_transactions": "df_combined.parquet"},
            "product_exposure_features": {
                "output": "customer_product_exposure_features.parquet",
                "catalog_output_dir": "stage5",
                "catalog_output_csv": "product_exposure_feature_catalog.csv",
                "ticket_column": "ticket",
                "metrics": ["basket_share", "distinct_product_share"],
                "dimensions": {"themes": True, "sectors": True, "product_families": True},
                "min_customers": {"theme": 1, "sector": 1, "family": 1},
                "max_features": {"theme": 20, "sector": 20, "family": 20},
                "family_ngram_sizes": [2, 1],
                "max_families_per_product": 4,
            },
            "feature_sets": {
                "outputs": {
                    "embeddings_only": "feature_set_embeddings_only.parquet",
                    "embeddings_frequency": "feature_set_embeddings_frequency.parquet",
                    "embeddings_product_exposure": "feature_set_embeddings_product_exposure.parquet",
                },
                "standardize_product_exposure": False,
                "product_exposure_weight": 0.35,
                "frequency_anchor_weight": 0.12,
            },
        },
        mode="dev",
        root=tmp_path,
    )


def test_product_exposure_features_are_multi_label_and_customer_level(tmp_path):
    cfg = _test_config(tmp_path)
    transactions = pl.DataFrame(
        {
            "cliente": ["c1", "c1", "c2", "c3"],
            "ticket": ["t1", "t2", "t3", "t4"],
            "idarticu": [101, 102, 102, 103],
            "desc_larga_articulo": [
                "VESTIDO TEX BABY NINA ALGODON ECOLOGICO",
                "YOGUR BIO SIN LACTOSA NATURAL",
                "YOGUR BIO SIN LACTOSA NATURAL",
                "PIENSO PERRO ADULTO POLLO",
            ],
            "idsector": [1, 2, 2, 3],
            "desc_sector": ["TEXTIL", "P.G.C.", "P.G.C.", "MASCOTAS"],
            "unidades": [1.0, 2.0, 1.0, 1.0],
        }
    ).lazy()

    exposure_path = build_product_exposure_features(transactions=transactions, cfg=cfg)
    exposure = pl.read_parquet(exposure_path).sort("cliente")
    c1 = exposure.filter(pl.col("cliente") == "c1").row(0, named=True)
    catalog = pl.read_csv(cfg.artifacts / "stage5" / "product_exposure_feature_catalog.csv")

    assert "pdx_basket_share_theme_baby_girls_clothing" in exposure.columns
    assert "pdx_basket_share_theme_organic_bio" in exposure.columns
    assert "pdx_basket_share_theme_lactose_free" in exposure.columns
    assert "pdx_distinct_product_share_theme_baby_girls_clothing" in exposure.columns
    assert c1["pdx_basket_share_theme_baby_girls_clothing"] == 0.5
    assert c1["pdx_basket_share_theme_organic_bio"] == 1.0
    assert c1["pdx_distinct_product_share_theme_baby_girls_clothing"] == 0.5
    assert {"theme", "sector", "family"}.issubset(set(catalog["feature_type"].to_list()))


def test_embeddings_product_exposure_feature_set_joins_product_features(tmp_path):
    cfg = _test_config(tmp_path)
    exposure_path = tmp_path / "exposure.parquet"
    embeddings_path = tmp_path / "embeddings.parquet"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2"],
            "pdx_basket_share_theme_baby_food": [0.5, 0.0],
            "pdx_basket_share_theme_organic_bio": [0.5, 1.0],
        }
    ).write_parquet(exposure_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2"],
            "emb_000": [0.1, 0.2],
            "emb_001": [0.3, 0.4],
        }
    ).write_parquet(embeddings_path)

    feature_path = build_feature_set(
        embeddings_path,
        product_exposure_path=exposure_path,
        variant="embeddings_product_exposure",
        cfg=cfg,
    )
    features = pl.read_parquet(feature_path).sort("cliente")

    assert "emb_000" in features.columns
    assert "pdx_basket_share_theme_baby_food" in features.columns
    assert features.filter(pl.col("cliente") == "c1")[0, "pdx_basket_share_theme_baby_food"] == 0.5


def test_embeddings_frequency_feature_set_adds_only_scaled_frequency_anchor(tmp_path):
    cfg = _test_config(tmp_path)
    embeddings_path = tmp_path / "embeddings.parquet"
    behavior_path = tmp_path / "behavior.parquet"

    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3"],
            "emb_000": [0.1, 0.2, 0.3],
            "emb_001": [0.4, 0.5, 0.6],
        }
    ).write_parquet(embeddings_path)
    pl.DataFrame(
        {
            "cliente": ["c1", "c2", "c3"],
            "frequency_per_30d": [0.0, 1.0, 9.0],
            "total_spend": [100.0, 200.0, 300.0],
        }
    ).write_parquet(behavior_path)

    feature_path = build_feature_set(
        embeddings_path,
        behavior_path=behavior_path,
        variant="embeddings_frequency",
        cfg=cfg,
    )
    features = pl.read_parquet(feature_path).sort("cliente")

    assert features.columns == ["cliente", "emb_000", "emb_001", "freq_anchor"]
    assert features["freq_anchor"].dtype == pl.Float32
    assert abs(float(features["freq_anchor"].mean())) < 1e-7
    assert round(float(features["freq_anchor"].std(ddof=0)), 2) == 0.12
