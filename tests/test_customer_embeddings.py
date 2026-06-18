import polars as pl
import pytest

from src.config import PipelineConfig
from src.customer_embeddings import (
    build_customer_embeddings,
    build_customer_embedding_variant_diagnostics,
    build_customer_embedding_weight_diagnostics,
)


def _test_config(
    tmp_path,
    *,
    quantity_transform="log1p",
    max_customer_product_weight=None,
    write_top_products_csv=False,
    gates=None,
    variant_diagnostics=None,
    recency_weighting=None,
    frequency_weighting=None,
):
    customer_embedding_cfg = {
        "output": "customer_embeddings.parquet",
        "weight_strategy": "quantity",
        "quantity_transform": quantity_transform,
        "max_customer_product_weight": max_customer_product_weight,
        "normalize_vectors": False,
        "partition_count": 1,
        "diagnostics": {
            "enabled": True,
            "output_dir": "stage4",
            "output_prefix": "customer_embedding",
            "write_top_products_csv": write_top_products_csv,
            "top_n_products": 10,
        },
    }
    if gates is not None:
        customer_embedding_cfg["gates"] = gates
    if variant_diagnostics is not None:
        customer_embedding_cfg["variant_diagnostics"] = variant_diagnostics
    if recency_weighting is not None:
        customer_embedding_cfg["recency_weighting"] = recency_weighting
    if frequency_weighting is not None:
        customer_embedding_cfg["frequency_weighting"] = frequency_weighting
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
                "diagnostics": {
                    "output_dir": "stage1",
                    "product_ubiquity_output": "product_ubiquity_diagnostics.parquet",
                },
            },
            "customer_embeddings": customer_embedding_cfg,
        },
        mode="dev",
        root=tmp_path,
    )


def _write_product_embeddings(path, product_ids=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    product_ids = product_ids or ["common", "niche", "other"]
    rows = {
        "common": {"idarticu": "common", "emb_000": 1.0, "emb_001": 0.0},
        "niche": {"idarticu": "niche", "emb_000": 0.0, "emb_001": 1.0},
        "other": {"idarticu": "other", "emb_000": 0.0, "emb_001": -1.0},
    }
    pl.DataFrame(
        [rows[product_id] for product_id in product_ids]
    ).write_parquet(path)


def _transactions():
    return pl.DataFrame(
        {
            "cliente": ["c1", "c1", "c2", "c2"],
            "ticket": ["t1", "t1", "t2", "t2"],
            "idarticu": ["common", "niche", "common", "other"],
            "unidades": [100, 1, 1, 1],
            "desc_larga_articulo": ["Common Staple", "Niche Item", "Common Staple", "Other Item"],
            "desc_sector": ["Grocery", "Special", "Grocery", "Other"],
        }
    ).lazy()


def _temporal_transactions():
    return pl.DataFrame(
        {
            "cliente": ["c1", "c1", "c1", "c2"],
            "ticket": ["old_bulk", "recent_1", "recent_2", "recent_other"],
            "idarticu": ["common", "niche", "niche", "other"],
            "unidades": [10, 1, 1, 1],
            "fecha": [
                "2024-01-01",
                "2024-01-31",
                "2024-01-31",
                "2024-01-31",
            ],
            "desc_larga_articulo": ["Common Staple", "Niche Item", "Niche Item", "Other Item"],
            "desc_sector": ["Grocery", "Special", "Special", "Other"],
        }
    ).with_columns(pl.col("fecha").str.strptime(pl.Date)).lazy()


def _write_common_product_diagnostics(cfg):
    path = cfg.artifacts / "stage1" / "product_ubiquity_diagnostics.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "idarticu": ["common", "niche", "other"],
            "common_product_candidate": [True, False, False],
        }
    ).write_parquet(path)


def test_customer_embeddings_apply_log_quantity_transform_and_write_diagnostics(tmp_path):
    cfg = _test_config(tmp_path, quantity_transform="log1p")
    cfg.ensure_directories()
    _write_common_product_diagnostics(cfg)
    embeddings_path = cfg.outputs / "embeddings" / "product_embeddings.parquet"
    _write_product_embeddings(embeddings_path)

    output = build_customer_embeddings(embeddings_path, transactions=_transactions(), force=True, cfg=cfg)

    rows = pl.read_parquet(output).sort("cliente")
    c1 = rows.filter(pl.col("cliente") == "c1").row(0, named=True)
    assert 0.85 < c1["emb_000"] < 0.90
    assert 0.10 < c1["emb_001"] < 0.15

    summary = pl.read_csv(cfg.artifacts / "stage4" / "customer_embedding_weight_diagnostics.csv").row(0, named=True)

    assert summary["quantity_transform"] == "log1p"
    assert summary["line_coverage_pct"] == 100.0
    assert summary["unit_coverage_pct"] == 100.0
    assert summary["customer_coverage_pct"] == 100.0
    assert summary["passes_stage4_gates"] is True
    assert summary["stage4_gate_status"] == "disabled"
    assert summary["common_product_weight_share_pct"] > 0
    assert not (cfg.artifacts / "stage4" / "customer_embedding_top_weighted_products.csv").exists()


def test_customer_embedding_weight_diagnostics_can_be_built_without_rebuilding_vectors(tmp_path):
    cfg = _test_config(
        tmp_path,
        quantity_transform="raw",
        max_customer_product_weight=2.0,
        write_top_products_csv=True,
    )
    cfg.ensure_directories()
    _write_common_product_diagnostics(cfg)

    paths = build_customer_embedding_weight_diagnostics(
        transactions=_transactions(),
        output_dir=cfg.reports / "custom_stage4",
        output_prefix="capped",
        cfg=cfg,
    )
    summary = pl.read_csv(paths["summary_csv"]).row(0, named=True)

    assert summary["max_customer_product_weight"] == 2.0
    assert summary["top_product_weight_share_pct"] < 70.0
    assert paths["top_products_csv"].exists()


def test_customer_embedding_coverage_gate_fails_when_products_are_unembedded(tmp_path):
    cfg = _test_config(
        tmp_path,
        gates={
            "enabled": True,
            "fail_on_violation": True,
            "min_line_coverage_pct": 100.0,
            "min_unit_coverage_pct": 100.0,
            "min_customer_coverage_pct": 100.0,
        },
    )
    cfg.ensure_directories()
    _write_common_product_diagnostics(cfg)
    embeddings_path = cfg.outputs / "embeddings" / "product_embeddings.parquet"
    _write_product_embeddings(embeddings_path, product_ids=["common", "niche"])

    with pytest.raises(ValueError, match="Stage 4 customer embedding gates failed"):
        build_customer_embeddings(embeddings_path, transactions=_transactions(), force=True, cfg=cfg)

    summary = pl.read_csv(cfg.artifacts / "stage4" / "customer_embedding_weight_diagnostics.csv").row(0, named=True)
    assert summary["passes_stage4_gates"] is False
    assert summary["line_coverage_pct"] < 100.0
    assert "line_coverage_pct" in summary["stage4_gate_issues"]


def test_customer_embedding_dominance_gate_fails_when_common_product_dominates(tmp_path):
    cfg = _test_config(
        tmp_path,
        quantity_transform="raw",
        gates={
            "enabled": True,
            "fail_on_violation": True,
            "max_common_product_weight_share_pct": 10.0,
            "max_p95_customer_top_product_weight_share_pct": 80.0,
        },
    )
    cfg.ensure_directories()
    _write_common_product_diagnostics(cfg)
    embeddings_path = cfg.outputs / "embeddings" / "product_embeddings.parquet"
    _write_product_embeddings(embeddings_path)

    with pytest.raises(ValueError, match="Stage 4 customer embedding gates failed"):
        build_customer_embeddings(embeddings_path, transactions=_transactions(), force=True, cfg=cfg)

    summary = pl.read_csv(cfg.artifacts / "stage4" / "customer_embedding_weight_diagnostics.csv").row(0, named=True)
    assert summary["passes_stage4_gates"] is False
    assert summary["common_product_weight_share_pct"] > 10.0
    assert "common_product_weight_share_pct" in summary["stage4_gate_issues"]


def test_customer_embeddings_apply_recency_decay_and_product_basket_frequency(tmp_path):
    cfg = _test_config(
        tmp_path,
        quantity_transform="raw",
        recency_weighting={
            "enabled": True,
            "date_column": "fecha",
            "reference_date": "2024-01-31",
            "half_life_days": 10,
            "min_multiplier": 0.25,
        },
        frequency_weighting={
            "enabled": True,
            "ticket_column": "ticket",
            "transform": "log1p",
            "exponent": 1.0,
            "max_multiplier": 3.0,
        },
    )
    cfg.ensure_directories()
    _write_common_product_diagnostics(cfg)
    embeddings_path = cfg.outputs / "embeddings" / "product_embeddings.parquet"
    _write_product_embeddings(embeddings_path)

    output = build_customer_embeddings(embeddings_path, transactions=_temporal_transactions(), force=True, cfg=cfg)
    rows = pl.read_parquet(output).sort("cliente")
    c1 = rows.filter(pl.col("cliente") == "c1").row(0, named=True)
    summary = pl.read_csv(cfg.artifacts / "stage4" / "customer_embedding_weight_diagnostics.csv").row(0, named=True)

    assert c1["emb_001"] > c1["emb_000"]
    assert summary["recency_weighting_enabled"] is True
    assert summary["recency_reference_date"] == "2024-01-31"
    assert summary["frequency_weighting_enabled"] is True
    assert summary["frequency_transform"] == "log1p"
    assert summary["max_customer_product_basket_count"] == 2
    assert summary["max_frequency_multiplier_observed"] > 1.0


def test_customer_embedding_variant_diagnostics_compare_capped_and_idf_trials(tmp_path):
    cfg = _test_config(
        tmp_path,
        variant_diagnostics={
            "enabled": True,
            "output_dir": "stage4",
            "output_csv": "customer_embedding_variant_diagnostics.csv",
            "trials": [
                {
                    "name": "official_quantity_log1p",
                    "weight_strategy": "quantity",
                    "quantity_transform": "log1p",
                    "max_customer_product_weight": None,
                    "normalize_vectors": False,
                },
                {
                    "name": "capped_quantity_log1p_2",
                    "weight_strategy": "quantity",
                    "quantity_transform": "log1p",
                    "max_customer_product_weight": 2.0,
                    "normalize_vectors": False,
                },
                {
                    "name": "quantity_idf_log1p",
                    "weight_strategy": "quantity_idf",
                    "quantity_transform": "log1p",
                    "max_customer_product_weight": None,
                    "normalize_vectors": False,
                },
            ],
        },
        gates={
            "enabled": True,
            "fail_on_violation": False,
            "max_common_product_weight_share_pct": 80.0,
        },
    )
    cfg.ensure_directories()
    _write_common_product_diagnostics(cfg)
    embeddings_path = cfg.outputs / "embeddings" / "product_embeddings.parquet"
    _write_product_embeddings(embeddings_path)

    output = build_customer_embedding_variant_diagnostics(
        embeddings_path=embeddings_path,
        transactions=_transactions(),
        force=True,
        cfg=cfg,
    )
    diagnostics = pl.read_csv(output).sort("trial_name")
    official = diagnostics.filter(pl.col("trial_name") == "official_quantity_log1p").row(0, named=True)
    capped = diagnostics.filter(pl.col("trial_name") == "capped_quantity_log1p_2").row(0, named=True)
    idf = diagnostics.filter(pl.col("trial_name") == "quantity_idf_log1p").row(0, named=True)

    assert output == cfg.artifacts / "stage4" / "customer_embedding_variant_diagnostics.csv"
    assert diagnostics["trial_name"].to_list() == [
        "capped_quantity_log1p_2",
        "official_quantity_log1p",
        "quantity_idf_log1p",
    ]
    assert official["is_official_stage4_recipe"] is True
    assert capped["is_official_stage4_recipe"] is False
    assert idf["weight_strategy"] == "quantity_idf"
    assert capped["common_product_weight_share_pct"] < official["common_product_weight_share_pct"]
    assert idf["common_product_weight_share_pct"] < official["common_product_weight_share_pct"]
    assert "common_product_gate_margin_pct" in diagnostics.columns
