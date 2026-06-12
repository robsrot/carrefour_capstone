from datetime import date
from pathlib import Path

import polars as pl

from src.basket_builder import build_basket_sentences
from src.config import load_config
from src.feature_engineering import build_behavioral_features


ARTIFACT_DIR = Path("data/dev/test_smoke")


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
    assert products == ["101", "102"]


def test_behavioral_features_parse_no_promo():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = ARTIFACT_DIR / "behavior.parquet"
    build_behavioral_features(transactions=_tiny_transactions(), output_path=output, force=True)
    features = pl.read_parquet(output).sort("cliente")
    c1 = features.filter(pl.col("cliente") == "c1").row(0, named=True)
    c2 = features.filter(pl.col("cliente") == "c2").row(0, named=True)
    assert c1["ticket_count"] == 2
    assert c1["promo_share"] == 1 / 3
    assert c2["promo_share"] == 1.0
