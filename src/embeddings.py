"""Phase 1 — Product embeddings via Word2Vec (Item2Vec).

Each shopping basket (ticket) is treated as a "sentence"; each product ID is a
"word". Word2Vec learns which products are behaviourally similar from co-purchase
patterns: products bought together end up geometrically close in the 100-dim space.

Public API
----------
build_basket_sentences()  → data/processed/basket_sentences.parquet
train_word2vec()          → models/word2vec_product.model
save_embeddings()         → data/processed/product_embeddings.parquet
sanity_check()            → prints nearest-neighbour table to stdout
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from gensim.models import Word2Vec

from src.config import (
    DATA_PROCESSED,
    MODELS,
    RANDOM_SEED,
    W2V_EPOCHS,
    W2V_MIN_COUNT,
    W2V_SG,
    W2V_VECTOR_SIZE,
    W2V_WINDOW,
)

_log = logging.getLogger(__name__)

_BASKET_CACHE     = DATA_PROCESSED / "basket_sentences.parquet"
_EMBEDDINGS_CACHE = DATA_PROCESSED / "product_embeddings.parquet"
_MODEL_PATH       = MODELS / "word2vec_product.model"


# ─── 1. Build basket sentences ────────────────────────────────────────────────

def build_basket_sentences(
    df_combined_path: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Group df_combined by ticket → one row per basket with a list of product IDs.

    Uses Polars streaming so the 190 M-row source is never fully materialised.
    Result is cached; subsequent calls are instant.

    Returns
    -------
    Path to basket_sentences.parquet  (columns: ticket int64, products list[int64])
    """
    if df_combined_path is None:
        df_combined_path = DATA_PROCESSED / "df_combined.parquet"

    if _BASKET_CACHE.exists() and not force:
        n = pl.scan_parquet(_BASKET_CACHE).select(pl.len()).collect().item()
        _log.info("Cache hit — %s baskets in %s", f"{n:,}", _BASKET_CACHE.name)
        return _BASKET_CACHE

    _log.info("Building basket sentences from %s ...", df_combined_path.name)
    baskets = (
        pl.scan_parquet(df_combined_path)
        .select(["ticket", "idarticu"])          # drop unused columns before grouping
        .group_by("ticket")
        .agg(pl.col("idarticu").alias("products"))
        .collect(engine="streaming")
    )
    baskets.write_parquet(_BASKET_CACHE, compression="zstd")
    _log.info(
        "Saved %s baskets → %s  (%.0f MB on disk)",
        f"{len(baskets):,}",
        _BASKET_CACHE.name,
        _BASKET_CACHE.stat().st_size / 1024 ** 2,
    )
    return _BASKET_CACHE


# ─── 2. Re-iterable sentence stream for gensim ───────────────────────────────

class _BasketIterator:
    """Re-iterable corpus for gensim Word2Vec.

    gensim makes (1 + n_epochs) full passes over the corpus during training.
    Strategy: load basket_sentences.parquet once into a compact Polars DataFrame
    (~1.7 GB in Arrow columnar format — far less than the equivalent Python lists),
    then yield sentences in batches for each pass. Peak extra memory per batch
    is ~batch_size x avg_basket_size x 60 bytes (Python strings), discarded after yield.
    """

    def __init__(self, parquet_path: Path, batch_size: int = 100_000):
        self.parquet_path = parquet_path
        self.batch_size   = batch_size
        self._df: pl.DataFrame | None = None

    def _load(self) -> pl.DataFrame:
        if self._df is None:
            _log.info("Loading basket sentences into memory (one-time) ...")
            self._df = pl.read_parquet(self.parquet_path, columns=["products"])
            _log.info("  %s baskets loaded", f"{len(self._df):,}")
        return self._df

    def __len__(self) -> int:
        return len(self._load())

    def __iter__(self):
        df = self._load()
        n  = len(df)
        for offset in range(0, n, self.batch_size):
            chunk = df["products"].slice(offset, self.batch_size).to_list()
            for products in chunk:
                # product IDs must be strings for gensim; skip any nulls
                yield [str(p) for p in products if p is not None]


# ─── 3. Train Word2Vec ────────────────────────────────────────────────────────

def train_word2vec(
    basket_path: Path | None = None,
    model_path: Path | None = None,
    *,
    workers: int = 4,
    force: bool = False,
) -> Word2Vec:
    """Train Word2Vec on the basket corpus; load from disk if already trained.

    Parameters
    ----------
    basket_path : path to basket_sentences.parquet (default: DATA_PROCESSED)
    model_path  : where to save the .model file (default: MODELS)
    workers     : parallel training threads — reduce to 2 if RAM is tight
    force       : re-train even if a cached model exists
    """
    if basket_path is None:
        basket_path = _BASKET_CACHE
    if model_path is None:
        model_path = _MODEL_PATH

    if model_path.exists() and not force:
        _log.info("Loading cached model from %s ...", model_path.name)
        return Word2Vec.load(str(model_path))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = _BasketIterator(basket_path)

    _log.info(
        "Training Word2Vec | %s baskets | vector_size=%d  window=%d  "
        "min_count=%d  epochs=%d  sg=%d  workers=%d",
        f"{len(corpus):,}",
        W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCHS, W2V_SG, workers,
    )

    model = Word2Vec(
        sentences=corpus,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=W2V_SG,
        epochs=W2V_EPOCHS,
        workers=workers,
        seed=RANDOM_SEED,
    )

    model.save(str(model_path))
    _log.info(
        "Saved → %s  |  vocab: %s products have embeddings",
        model_path.name,
        f"{len(model.wv):,}",
    )
    return model


# ─── 4. Export embeddings to parquet ─────────────────────────────────────────

def save_embeddings(
    model: Word2Vec,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Write (idarticu, embedding) to product_embeddings.parquet.

    embedding is a list[float32] of length W2V_VECTOR_SIZE (100).
    This is the lookup table consumed by Phase 2 (customer vector aggregation).
    """
    if _EMBEDDINGS_CACHE.exists() and not force:
        _log.info("Embeddings cache hit: %s", _EMBEDDINGS_CACHE.name)
        return pl.read_parquet(_EMBEDDINGS_CACHE)

    vocab   = list(model.wv.key_to_index.keys())
    vectors = model.wv.vectors          # (vocab_size, 100) numpy float32 array

    df = pl.DataFrame({
        "idarticu":  pl.Series([int(v) for v in vocab], dtype=pl.Int64),
        "embedding": pl.Series(vectors.tolist(), dtype=pl.List(pl.Float32)),
    })
    df.write_parquet(_EMBEDDINGS_CACHE, compression="zstd")
    _log.info(
        "Saved %s embeddings → %s  (%.1f MB on disk)",
        f"{len(df):,}",
        _EMBEDDINGS_CACHE.name,
        _EMBEDDINGS_CACHE.stat().st_size / 1024 ** 2,
    )
    return df


# ─── 5. Sanity check ─────────────────────────────────────────────────────────

def sanity_check(
    model: Word2Vec,
    df_articles: pl.DataFrame,
    probe_ids: list[int],
    topn: int = 5,
) -> None:
    """Print the top-N most similar products for each probe product ID.

    Similar products should make commercial sense (beer → wine/chips, olive oil →
    pasta/tomatoes). A failure here — unrelated sectors clustering together — means
    the basket building or training has a bug, not just a tuning issue.
    """
    id_to_name   = dict(zip(
        df_articles["idarticu"].to_list(),
        df_articles["desc_larga_articulo"].to_list(),
    ))
    id_to_sector = dict(zip(
        df_articles["idarticu"].to_list(),
        df_articles["desc_sector"].to_list(),
    ))

    for pid in probe_ids:
        key = str(pid)
        if key not in model.wv:
            print(f"\nProduct {pid} not in vocabulary (min_count={W2V_MIN_COUNT}).")
            continue

        name   = id_to_name.get(pid, f"id={pid}")
        sector = id_to_sector.get(pid, "?")
        print(f"\n{'='*72}")
        print(f"  PROBE  [{sector}]")
        print(f"  {name[:68]}")
        print(f"  idarticu = {pid}")
        print(f"{'='*72}")
        print(f"  {'Score':>8}  {'Sector':<26}  Product")
        print(f"  {'-'*8}  {'-'*26}  {'-'*34}")
        for sim_key, score in model.wv.most_similar(key, topn=topn):
            sim_id   = int(sim_key)
            sim_name = id_to_name.get(sim_id, f"id={sim_id}")
            sim_sec  = id_to_sector.get(sim_id, "?")
            print(f"  {score:>8.4f}  {sim_sec:<26}  {sim_name[:34]}")
