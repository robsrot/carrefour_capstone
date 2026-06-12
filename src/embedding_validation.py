"""Qualitative validation for product embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from src.config import CONFIG, PipelineConfig
from src.data_loader import load_prepared_transactions
from src.utils import collect_streaming, deterministic_sample_indices, numeric_feature_columns


def _product_metadata(transactions: pl.LazyFrame) -> pl.DataFrame:
    available = set(transactions.collect_schema().names())
    cols = ["idarticu"]
    for col in ["desc_larga_articulo", "idsector", "desc_sector"]:
        if col in available:
            cols.append(col)
    return collect_streaming(transactions.select(cols).unique(subset=["idarticu"]))


def validate_product_embeddings(
    embeddings_path: str | Path,
    transactions: pl.LazyFrame | None = None,
    sample_product_ids: Sequence[int] | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> tuple[Path, Path]:
    """Create a nearest-neighbor report for sampled products."""

    cfg.ensure_directories()
    emb = pl.read_parquet(embeddings_path).sort("idarticu")
    feature_cols = numeric_feature_columns(emb, exclude=("idarticu",))
    matrix = emb.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    ids = emb["idarticu"].to_numpy()

    if sample_product_ids:
        chosen = [int(pid) for pid in sample_product_ids if int(pid) in set(ids.tolist())]
        indices = [int(np.where(ids == pid)[0][0]) for pid in chosen]
    else:
        indices = deterministic_sample_indices(
            len(ids),
            int(cfg.get("embedding_validation.sample_size", 25)),
            cfg.random_seed,
        ).tolist()

    lf = transactions if transactions is not None else load_prepared_transactions(cfg=cfg)
    meta = _product_metadata(lf)
    meta_lookup = {row["idarticu"]: row for row in meta.iter_rows(named=True)}

    rows = []
    n_neighbors = int(cfg.get("embedding_validation.neighbors", 8))
    for idx in indices:
        pid = int(ids[idx])
        scores = normalized @ normalized[idx]
        order = np.argsort(-scores)
        neighbors = [j for j in order if j != idx][:n_neighbors]
        source = meta_lookup.get(pid, {})
        for rank, neighbor_idx in enumerate(neighbors, start=1):
            neighbor_id = int(ids[neighbor_idx])
            neighbor = meta_lookup.get(neighbor_id, {})
            rows.append(
                {
                    "product_id": pid,
                    "product_description": source.get("desc_larga_articulo"),
                    "product_sector": source.get("desc_sector"),
                    "neighbor_rank": rank,
                    "neighbor_id": neighbor_id,
                    "neighbor_description": neighbor.get("desc_larga_articulo"),
                    "neighbor_sector": neighbor.get("desc_sector"),
                    "cosine_similarity": float(scores[neighbor_idx]),
                }
            )

    csv_path = Path(output_csv) if output_csv else cfg.reports / cfg.get("embedding_validation.output_csv")
    md_path = Path(output_md) if output_md else cfg.reports / cfg.get("embedding_validation.output_md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report = pl.DataFrame(rows)
    report.write_csv(csv_path)

    lines = [
        "# Product Embedding Validation",
        "",
        "Nearest neighbors should share basket missions, sectors, substitutes, or complements.",
        "",
    ]
    for pid in dict.fromkeys(row["product_id"] for row in rows):
        block = [row for row in rows if row["product_id"] == pid]
        first = block[0]
        lines.append(f"## {pid} - {first.get('product_description') or 'Unknown product'}")
        lines.append(f"Sector: {first.get('product_sector') or 'Unknown'}")
        for row in block:
            lines.append(
                f"- {row['neighbor_rank']}. {row['neighbor_id']} - "
                f"{row.get('neighbor_description') or 'Unknown'} "
                f"({row.get('neighbor_sector') or 'Unknown'}), cosine={row['cosine_similarity']:.3f}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
