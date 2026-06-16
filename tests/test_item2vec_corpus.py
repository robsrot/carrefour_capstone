import polars as pl

from src.item2vec import BasketSentenceCorpus


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
