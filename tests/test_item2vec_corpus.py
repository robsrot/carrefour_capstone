import polars as pl

from src.config import PipelineConfig
from src.item2vec import BasketSentenceCorpus, item2vec_training_summary, write_item2vec_training_diagnostics


def test_basket_sentence_corpus_skips_short_baskets_and_caps_large_contexts(tmp_path):
    basket_path = tmp_path / "basket_sentences.parquet"
    pl.DataFrame(
        {
            "ticket": ["t1", "t2", "t3"],
            "products": [["1"], ["2", "3", "4", "5"], ["6", "7"]],
            "n_product_tokens": [1, 4, 2],
        }
    ).write_parquet(basket_path)

    corpus = BasketSentenceCorpus(
        basket_path,
        min_tokens_per_basket=2,
        max_tokens_per_basket=3,
    )

    assert list(corpus) == [["2", "3", "4"], ["6", "7"]]


def test_item2vec_training_diagnostics_capture_effective_corpus(tmp_path):
    basket_path = tmp_path / "basket_sentences.parquet"
    pl.DataFrame(
        {
            "ticket": ["t1", "t2", "t3"],
            "products": [["1"], ["2", "3", "4", "5"], ["6", "7"]],
            "n_product_tokens": [1, 4, 2],
        }
    ).write_parquet(basket_path)
    cfg = PipelineConfig(
        values={
            "run": {"random_seed": 42},
            "paths": {
                "dev": "data/dev",
                "processed": "data/processed",
                "raw_csv": "data/raw/csv",
                "raw_parquet": "data/raw/parquet",
                "outputs": "outputs",
            },
            "word2vec": {
                "window": 2,
                "full_basket_context": False,
                "min_tokens_per_basket": 2,
                "max_tokens_per_basket": 3,
                "min_count": 1,
                "sample": 0.0001,
                "negative": 5,
                "epochs": 1,
                "sg": 1,
                "diagnostics": {
                    "output_dir": "stage2",
                    "training_summary_csv": "item2vec_training_corpus_diagnostics.csv",
                },
            },
        },
        mode="dev",
        root=tmp_path,
    )

    summary = item2vec_training_summary(basket_path, cfg=cfg)
    output = write_item2vec_training_diagnostics(basket_path, cfg=cfg)
    written = pl.read_csv(output).row(0, named=True)

    assert summary["baskets"] == 3
    assert summary["training_baskets"] == 2
    assert summary["skipped_short_baskets"] == 1
    assert summary["capped_baskets"] == 1
    assert summary["training_product_tokens"] == 5
    assert abs(written["training_token_retention_pct"] - (5 / 7 * 100)) < 1e-9
