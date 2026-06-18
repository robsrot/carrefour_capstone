"""Compatibility layer for basket building and Item2Vec stages."""

from src.basket_builder import (
    basket_summary,
    build_basket_sentences,
    build_basket_staple_diagnostics,
)
from src.item2vec import (
    BasketSentenceCorpus,
    item2vec_training_summary,
    load_product_embeddings,
    save_product_embeddings,
    train_item2vec,
    write_item2vec_training_diagnostics,
)


def train_word2vec(*args, **kwargs):
    """Backward-compatible alias for the Item2Vec training function."""

    return train_item2vec(*args, **kwargs)


__all__ = [
    "BasketSentenceCorpus",
    "basket_summary",
    "build_basket_sentences",
    "build_basket_staple_diagnostics",
    "item2vec_training_summary",
    "load_product_embeddings",
    "save_product_embeddings",
    "train_item2vec",
    "train_word2vec",
    "write_item2vec_training_diagnostics",
]
