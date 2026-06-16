import polars as pl

from src.basket_builder import (
    build_basket_sentences,
    build_basket_staple_diagnostics,
)
from src.config import PipelineConfig


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
            "data": {
                "prepared_transactions": "df_combined.parquet",
                "customer_kpis": "customer_kpis.parquet",
                "product_master": "maestra_articulos.parquet",
            },
            "cache": {"force": False, "use_cached": True},
            "baskets": {
                "construction_strategy": "common_downsampled",
                "repeat_product_by_quantity": False,
                "output": "basket_sentences.parquet",
                "diagnostics": {
                    "output_dir": "stage1",
                    "product_ubiquity_output": "product_ubiquity_diagnostics.parquet",
                    "basket_exposure_output": "basket_common_product_exposure.parquet",
                    "common_products_csv": "common_product_candidates.csv",
                    "summary_md": "basket_common_product_diagnostics.md",
                    "top_n_common_products": 10,
                    "common_customer_penetration_threshold": 0.50,
                    "common_basket_penetration_threshold": 0.50,
                    "common_line_share_threshold": 0.50,
                },
                "downsampling": {
                    "manual_exclude_product_ids": [],
                    "auto_exclude": {
                        "enabled": True,
                        "customer_penetration_threshold": 0.75,
                        "basket_penetration_threshold": 0.75,
                        "line_share_threshold": None,
                        "max_products": 10,
                    },
                    "target_customer_penetration": 0.50,
                    "keep_probability_exponent": 1.0,
                    "min_keep_probability": 0.0,
                },
            },
        },
        mode="dev",
        root=tmp_path,
    )


def test_build_basket_staple_diagnostics_flags_common_products(tmp_path):
    transactions = pl.DataFrame(
        {
            "cliente": ["c1", "c1", "c2", "c2", "c3", "c3"],
            "ticket": ["t1", "t1", "t2", "t2", "t3", "t3"],
            "idarticu": ["milk", "niche_a", "milk", "niche_b", "milk", "niche_c"],
            "unidades": [1, 1, 2, 1, 1, 1],
            "desc_larga_articulo": ["Milk", "Niche A", "Milk", "Niche B", "Milk", "Niche C"],
            "desc_sector": ["Dairy", "Special", "Dairy", "Special", "Dairy", "Special"],
        }
    ).lazy()
    cfg = _test_config(tmp_path)

    paths = build_basket_staple_diagnostics(transactions=transactions, force=True, cfg=cfg)

    product_diagnostics = pl.read_parquet(paths["product_diagnostics"])
    milk = product_diagnostics.filter(pl.col("idarticu") == "milk").row(0, named=True)
    niche = product_diagnostics.filter(pl.col("idarticu") == "niche_a").row(0, named=True)
    exposure = pl.read_parquet(paths["basket_exposure"]).sort("ticket")

    assert milk["common_product_candidate"] is True
    assert niche["common_product_candidate"] is False
    assert milk["customer_penetration"] == 1.0
    assert exposure["n_common_product_candidates"].to_list() == [1, 1, 1]
    assert exposure["common_product_candidate_share"].to_list() == [0.5, 0.5, 0.5]
    assert paths["summary_md"].exists()
    assert paths["common_products_csv"].exists()


def test_build_basket_sentences_uses_single_downsampled_output(tmp_path):
    transactions = pl.DataFrame(
        {
            "cliente": ["c1", "c1", "c2", "c2"],
            "ticket": ["t1", "t1", "t2", "t2"],
            "idarticu": ["milk", "niche_a", "milk", "niche_b"],
            "unidades": [1, 1, 1, 1],
            "desc_larga_articulo": ["Milk", "Niche A", "Milk", "Niche B"],
            "desc_sector": ["Dairy", "Special", "Dairy", "Special"],
        }
    ).lazy()
    cfg = _test_config(tmp_path)

    basket_path = build_basket_sentences(transactions=transactions, force=True, cfg=cfg)
    baskets = pl.read_parquet(basket_path).sort("ticket")

    assert basket_path.name == "basket_sentences.parquet"
    assert baskets["ticket"].to_list() == ["t1", "t2"]
    assert baskets["products"].to_list() == [["niche_a"], ["niche_b"]]
    assert not (basket_path.parent / "basket_sentences_common_downsampled.parquet").exists()

    plan_path = cfg.artifacts / "stage1" / "common_product_downsampling_plan.parquet"
    summary_path = cfg.artifacts / "stage1" / "basket_downsampling_summary.csv"
    assert plan_path.exists()
    assert summary_path.exists()

    plan = pl.read_parquet(plan_path)
    summary = pl.read_csv(summary_path).row(0, named=True)
    milk_plan = plan.filter(pl.col("idarticu") == "milk").row(0, named=True)

    assert {"downsampling_action", "_keep_probability", "keep_probability_pct"}.issubset(set(plan.columns))
    assert milk_plan["downsampling_action"] == "auto_exclude"
    assert milk_plan["_keep_probability"] == 0.0
    assert summary["raw_baskets"] == 2
    assert summary["output_baskets"] == 2
    assert summary["raw_unique_ticket_product_pairs"] == 4
    assert summary["output_unique_ticket_product_pairs"] == 2
    assert summary["auto_excluded_products"] == 1
    assert summary["unique_pair_retention_pct"] == 50.0
    assert summary["basket_retention_pct"] == 100.0


def test_legacy_second_artifact_builder_is_not_public():
    import src.basket_builder as basket_builder
    import src.embeddings as embeddings

    assert not hasattr(basket_builder, "build_common_downsampled_basket_sentences")
    assert "build_common_downsampled_basket_sentences" not in embeddings.__all__
