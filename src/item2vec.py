"""Item2Vec model training and product embedding persistence."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.progress import log_event
from src.utils import collect_streaming, file_fingerprint, should_use_cache, write_artifact_metadata


class BasketSentenceCorpus:
    """Iterable over ticket-level product-token lists saved by basket_builder."""

    def __init__(
        self,
        basket_path: str | Path,
        min_tokens_per_basket: int = 2,
        max_tokens_per_basket: int | None = None,
    ):
        self.basket_path = Path(basket_path)
        self.min_tokens_per_basket = max(int(min_tokens_per_basket), 1)
        self.max_tokens_per_basket = (
            max(int(max_tokens_per_basket), self.min_tokens_per_basket)
            if max_tokens_per_basket is not None
            else None
        )

    def __iter__(self) -> Iterator[list[str]]:
        df = pl.read_parquet(self.basket_path, columns=["products"])
        for row in df.iter_rows(named=True):
            products = row["products"] or []
            if self.max_tokens_per_basket is not None and len(products) > self.max_tokens_per_basket:
                products = products[: self.max_tokens_per_basket]
            if len(products) >= self.min_tokens_per_basket:
                yield [str(product) for product in products]


class _Item2VecProgress:
    """Small Gensim callback that prints one concise line per epoch."""

    def __init__(self, total_epochs: int):
        from gensim.models.callbacks import CallbackAny2Vec

        class _Callback(CallbackAny2Vec):
            def __init__(self, epochs: int):
                self.epochs = epochs
                self.epoch = 0
                self.previous_loss = 0.0
                self.epoch_started_at = 0.0
                self.training_started_at = time.perf_counter()

            def on_epoch_begin(self, model):
                self.epoch_started_at = time.perf_counter()
                print(f"Item2Vec epoch {self.epoch + 1}/{self.epochs} started", flush=True)

            def on_epoch_end(self, model):
                self.epoch += 1
                elapsed = time.perf_counter() - self.epoch_started_at
                total_elapsed = time.perf_counter() - self.training_started_at
                latest_loss = float(model.get_latest_training_loss())
                epoch_loss = latest_loss - self.previous_loss if latest_loss else None
                self.previous_loss = latest_loss
                loss_text = f", epoch_loss={epoch_loss:,.0f}" if epoch_loss is not None else ""
                print(
                    f"Item2Vec epoch {self.epoch}/{self.epochs} finished "
                    f"in {elapsed:,.1f}s{loss_text}; elapsed={total_elapsed:,.1f}s",
                    flush=True,
                )

        self.callback = _Callback(total_epochs)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _corpus_limits(cfg: PipelineConfig) -> dict[str, int | None]:
    min_tokens = max(int(cfg.get("word2vec.min_tokens_per_basket", 2)), 1)
    max_tokens = _optional_int(cfg.get("word2vec.max_tokens_per_basket", None))
    if max_tokens is not None:
        max_tokens = max(max_tokens, min_tokens)
    return {"min_tokens_per_basket": min_tokens, "max_tokens_per_basket": max_tokens}


def _basket_training_summary(
    basket_path: Path,
    cfg: PipelineConfig | None = None,
) -> dict[str, float | int | None]:
    lf = pl.scan_parquet(basket_path)
    names = set(lf.collect_schema().names())
    if "n_product_tokens" in names:
        limits = _corpus_limits(cfg) if cfg is not None else {"min_tokens_per_basket": 1, "max_tokens_per_basket": None}
        min_tokens = int(limits["min_tokens_per_basket"] or 1)
        max_tokens = limits["max_tokens_per_basket"]
        effective_tokens = pl.when(pl.col("n_product_tokens") >= min_tokens).then(pl.col("n_product_tokens")).otherwise(0)
        if max_tokens is not None:
            effective_tokens = pl.when(effective_tokens > max_tokens).then(max_tokens).otherwise(effective_tokens)
        row = collect_streaming(
            lf.select(
                [
                    pl.len().alias("baskets"),
                    pl.col("n_product_tokens").sum().alias("product_tokens"),
                    pl.col("n_product_tokens").mean().alias("avg_tokens_per_basket"),
                    pl.col("n_product_tokens").median().alias("median_tokens_per_basket"),
                    pl.col("n_product_tokens").max().alias("max_tokens_per_basket"),
                    (pl.col("n_product_tokens") >= min_tokens).sum().alias("training_baskets"),
                    (pl.col("n_product_tokens") < min_tokens).sum().alias("skipped_short_baskets"),
                    (pl.col("n_product_tokens") > max_tokens).sum().alias("capped_baskets")
                    if max_tokens is not None
                    else pl.lit(0).alias("capped_baskets"),
                    effective_tokens.sum().alias("training_product_tokens"),
                ]
            )
        ).row(0, named=True)
    else:
        row = collect_streaming(lf.select(pl.len().alias("baskets"))).row(0, named=True)
        row.update(
            {
                "product_tokens": None,
                "avg_tokens_per_basket": None,
                "median_tokens_per_basket": None,
                "max_tokens_per_basket": None,
                "training_baskets": None,
                "skipped_short_baskets": None,
                "capped_baskets": None,
                "training_product_tokens": None,
            }
        )
    return row


def _effective_window(basket_path: Path, cfg: PipelineConfig) -> int:
    configured_window = int(cfg.get("word2vec.window"))
    if not bool(cfg.get("word2vec.full_basket_context", False)):
        return configured_window
    summary = _basket_training_summary(basket_path, cfg=cfg)
    max_tokens = summary.get("max_tokens_per_basket") or configured_window
    capped_tokens = _corpus_limits(cfg)["max_tokens_per_basket"]
    if capped_tokens is not None:
        max_tokens = min(int(max_tokens), int(capped_tokens))
    return max(configured_window, int(max_tokens))


def _print_training_config(
    basket_path: Path,
    output: Path,
    cfg: PipelineConfig,
    cached: bool,
    effective_window: int,
) -> None:
    summary = _basket_training_summary(basket_path, cfg=cfg)
    mode = "loading cached model" if cached else "training model"
    print(f"Item2Vec: {mode}", flush=True)
    print(f"  baskets: {summary.get('baskets'):,}", flush=True)
    if summary.get("product_tokens") is not None:
        print(
            f"  product tokens: {summary.get('product_tokens'):,}; "
            f"avg/basket: {summary.get('avg_tokens_per_basket'):.2f}; "
            f"median/basket: {summary.get('median_tokens_per_basket'):.2f}",
            flush=True,
        )
        print(
            f"  training baskets: {summary.get('training_baskets'):,}; "
            f"training tokens: {summary.get('training_product_tokens'):,}; "
            f"skipped short baskets: {summary.get('skipped_short_baskets'):,}; "
            f"capped baskets: {summary.get('capped_baskets'):,}",
            flush=True,
        )
    print(f"  model path: {output}", flush=True)
    limits = _corpus_limits(cfg)
    print(
        "  config: "
        f"vector_size={cfg.get('word2vec.vector_size')}, "
        f"window={effective_window} "
        f"(configured={cfg.get('word2vec.window')}, "
        f"full_basket_context={cfg.get('word2vec.full_basket_context')}), "
        f"min_tokens_per_basket={limits['min_tokens_per_basket']}, "
        f"max_tokens_per_basket={limits['max_tokens_per_basket']}, "
        f"min_count={cfg.get('word2vec.min_count')}, "
        f"negative={cfg.get('word2vec.negative')}, "
        f"sample={cfg.get('word2vec.sample')}, "
        f"epochs={cfg.get('word2vec.epochs')}, "
        f"sg={cfg.get('word2vec.sg')}, "
        f"workers={cfg.get('word2vec.workers')}, "
        f"seed={cfg.random_seed}",
        flush=True,
    )


def train_item2vec(
    basket_path: str | Path,
    model_path: str | Path | None = None,
    force: bool | None = None,
    verbose: bool | None = None,
    cfg: PipelineConfig = CONFIG,
):
    """Train or load a Word2Vec model where tickets are sentences and products are words."""

    from gensim.models import Word2Vec

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    verbose = bool(cfg.get("word2vec.report_progress", True)) if verbose is None else verbose
    basket_file = Path(basket_path)
    output = Path(model_path) if model_path else cfg.models / cfg.get("word2vec.model_name")
    effective_window = _effective_window(basket_file, cfg)
    corpus_limits = _corpus_limits(cfg)
    cache_metadata = {
        "stage": "item2vec_model",
        "mode": cfg.mode,
        "basket_sentences": file_fingerprint(basket_file),
        "word2vec": {
            "vector_size": int(cfg.get("word2vec.vector_size")),
            "window": effective_window,
            "configured_window": int(cfg.get("word2vec.window")),
            "full_basket_context": bool(cfg.get("word2vec.full_basket_context", False)),
            "min_tokens_per_basket": int(corpus_limits["min_tokens_per_basket"] or 1),
            "max_tokens_per_basket": corpus_limits["max_tokens_per_basket"],
            "basket_context_policy": "deterministic_local_context",
            "min_count": int(cfg.get("word2vec.min_count")),
            "negative": int(cfg.get("word2vec.negative")),
            "sample": float(cfg.get("word2vec.sample")),
            "epochs": int(cfg.get("word2vec.epochs")),
            "sg": int(cfg.get("word2vec.sg")),
            "workers": int(cfg.get("word2vec.workers")),
            "seed": cfg.random_seed,
        },
    }
    cached = should_use_cache(
        output,
        force=force,
        use_cached=cfg.get("cache.use_cached", True),
        metadata=cache_metadata,
    )
    if verbose:
        _print_training_config(basket_file, output, cfg, cached=cached, effective_window=effective_window)
    if cached:
        model = Word2Vec.load(str(output))
        if verbose:
            print(
                f"Item2Vec cache loaded: vocab={len(model.wv):,}, vector_size={model.vector_size}",
                flush=True,
            )
        return model

    epochs = int(cfg.get("word2vec.epochs"))
    progress = _Item2VecProgress(epochs).callback if verbose else None
    corpus = BasketSentenceCorpus(
        basket_file,
        min_tokens_per_basket=int(corpus_limits["min_tokens_per_basket"] or 1),
        max_tokens_per_basket=corpus_limits["max_tokens_per_basket"],
    )
    model = Word2Vec(
        sentences=corpus,
        vector_size=int(cfg.get("word2vec.vector_size")),
        window=effective_window,
        min_count=int(cfg.get("word2vec.min_count")),
        negative=int(cfg.get("word2vec.negative")),
        sample=float(cfg.get("word2vec.sample")),
        epochs=epochs,
        sg=int(cfg.get("word2vec.sg")),
        workers=int(cfg.get("word2vec.workers")),
        seed=cfg.random_seed,
        sorted_vocab=1,
        compute_loss=bool(cfg.get("word2vec.compute_loss", True)) if verbose else False,
        callbacks=[progress] if progress is not None else [],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    write_artifact_metadata(output, cache_metadata)
    if verbose:
        print(
            f"Item2Vec model saved: vocab={len(model.wv):,}, "
            f"vector_size={model.vector_size}, path={output}",
            flush=True,
        )
    return model


def save_product_embeddings(
    model,
    output_path: str | Path | None = None,
    force: bool | None = None,
    cfg: PipelineConfig = CONFIG,
) -> Path:
    """Persist product embeddings as one row per product with wide numeric columns."""

    cfg.ensure_directories()
    force = cfg.get("cache.force", False) if force is None else force
    output = Path(output_path) if output_path else cfg.artifact_path(
        "word2vec",
        "embeddings_output",
        directory=cfg.outputs / "embeddings",
    )
    cache_metadata = {
        "stage": "product_embeddings",
        "mode": cfg.mode,
        "model_vector_size": int(model.vector_size),
        "model_vocab": int(len(model.wv)),
        "model_keys_head": list(model.wv.index_to_key[:10]),
    }
    if should_use_cache(
        output,
        force=force,
        use_cached=cfg.get("cache.use_cached", True),
        metadata=cache_metadata,
    ):
        log_event("Stage 2 product embeddings", "cache hit", cfg=cfg, path=output)
        return output

    keys = list(model.wv.index_to_key)
    vectors = np.asarray([model.wv[key] for key in keys], dtype=np.float32)
    data: dict[str, list | np.ndarray] = {"idarticu": [int(key) for key in keys]}
    for idx in range(vectors.shape[1]):
        data[f"emb_{idx:03d}"] = vectors[:, idx]

    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).sort("idarticu").write_parquet(output)
    write_artifact_metadata(output, cache_metadata)
    log_event("Stage 2 product embeddings", "wrote artifact", cfg=cfg, products=len(keys), path=output)
    return output


def load_product_embeddings(path: str | Path | None = None, cfg: PipelineConfig = CONFIG) -> pl.DataFrame:
    embedding_path = Path(path) if path else cfg.artifact_path(
        "word2vec",
        "embeddings_output",
        directory=cfg.outputs / "embeddings",
    )
    if not embedding_path.exists():
        raise FileNotFoundError(f"Product embeddings not found: {embedding_path}")
    return pl.read_parquet(embedding_path)
