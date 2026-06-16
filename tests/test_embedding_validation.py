import polars as pl

from src.config import PipelineConfig
from src.embedding_validation import product_embedding_guardrail_status, validate_product_embeddings


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
                "diagnostics": {
                    "common_customer_penetration_threshold": 0.60,
                    "common_basket_penetration_threshold": 0.60,
                    "common_line_share_threshold": 0.60,
                },
            },
            "word2vec": {
                "window": 3,
                "full_basket_context": False,
                "sample": 0.0001,
                "min_count": 2,
            },
            "embedding_validation": {
                "output_dir": "stage3",
                "sample_size": 2,
                "neighbors": 2,
                "staple_sample_size": 1,
                "niche_sample_size": 1,
                "common_sample_size": 1,
                "rare_sample_size": 1,
                "niche_category_sample_size": 1,
                "category_report_max_products": 1,
                "niche_basket_penetration_max": 0.25,
                "common_neighbor_basket_penetration_threshold": 0.60,
                "generic_neighbor_warning_share_threshold": 0.50,
                "write_extract_figures": False,
                "guardrails": {
                    "max_generic_warning_product_share": 0.0,
                    "min_rare_same_sector_share": 0.50,
                    "max_common_rare_same_sector_gap": 0.25,
                    "max_top_hub_cross_sector_share": 0.80,
                    "max_hubness_lift": 10.0,
                },
                "niche_categories": "auto",
                "auto_niche_categories": {
                    "min_matched_products": 1,
                    "max_matched_products": 10,
                    "max_median_basket_penetration": 1.0,
                    "max_categories": 5,
                },
                "hubness_neighbors": 1,
                "hubness_sample_size": None,
                "hubness_chunk_size": 2,
                "output_csv": "embedding_validation.csv",
                "hubness_output_csv": "embedding_hubness.csv",
                "output_md": "embedding_validation.md",
            },
        },
        mode="dev",
        root=tmp_path,
    )


def test_validate_product_embeddings_writes_hubness_and_frequency_flags(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.ensure_directories()
    embeddings_path = cfg.outputs / "embeddings" / "product_embeddings.parquet"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "idarticu": [1, 2, 3, 4, 5],
            "emb_000": [1.0, 0.98, 0.96, -1.0, -0.95],
            "emb_001": [0.0, 0.04, -0.05, 0.0, 0.08],
        }
    ).write_parquet(embeddings_path)
    transactions = pl.DataFrame(
        {
            "cliente": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"],
            "ticket": ["t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4"],
            "idarticu": [1, 2, 1, 3, 1, 4, 1, 5],
            "unidades": [1, 1, 1, 1, 1, 1, 1, 1],
            "desc_larga_articulo": [
                "Staple",
                "Sin gluten bread",
                "Staple",
                "Organic milk",
                "Staple",
                "Pienso perro",
                "Staple",
                "D",
            ],
            "desc_sector": ["Grocery", "Grocery", "Grocery", "Grocery", "Grocery", "Pet", "Grocery", "Home"],
        }
    ).lazy()

    csv_path, md_path = validate_product_embeddings(
        embeddings_path,
        transactions=transactions,
        force=True,
        cfg=cfg,
    )
    guardrail_status = product_embedding_guardrail_status(csv_path, cfg=cfg)

    report = pl.read_csv(csv_path)
    hubness = pl.read_csv(cfg.artifacts / "stage3" / "embedding_hubness.csv")
    markdown = md_path.read_text(encoding="utf-8")

    assert {
        "sample_group",
        "validation_categories",
        "neighbor_is_common_filler",
        "neighbor_is_generic_staple",
        "neighbor_hubness_lift",
        "query_generic_neighbor_share",
        "generic_neighbor_warning",
    }.issubset(report.columns)
    assert {"hub_neighbor_count", "hubness_lift", "cross_sector_neighbor_share"}.issubset(hubness.columns)
    assert report.filter(pl.col("sample_group").str.contains("staple")).height > 0
    assert report.filter(pl.col("sample_group").str.contains("niche")).height > 0
    assert report.filter(pl.col("sample_group").str.contains("common_frequency")).height > 0
    assert report.filter(pl.col("sample_group").str.contains("rare_frequency")).height > 0
    assert report.filter(pl.col("sample_group").str.contains("category_")).height > 0
    assert report.filter(pl.col("generic_neighbor_warning")).height > 0
    assert "Top Embedding Hubs" in markdown
    assert "Niche Product Neighbor Checks" in markdown
    assert "Common vs Rare Neighbor Quality" in markdown
    assert "Gluten Free Niche Theme Neighbor Checks" in markdown
    assert "Data-Selected Niche Theme Coverage and Quality" in markdown
    assert "Generic Neighbor Warnings" in markdown
    assert "Dynamic Guardrail Status" in markdown
    assert "Upstream actions" in markdown
    assert guardrail_status["status"] == "action_needed"
    assert guardrail_status["issues"]
    assert guardrail_status["upstream_actions"]
