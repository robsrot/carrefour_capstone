"""Compatibility layer for customer embedding and feature-set stages."""

from src.customer_embeddings import build_customer_embeddings, load_customer_embeddings
from src.feature_engineering import build_behavioral_features, build_feature_set


def build_customer_vectors(*args, **kwargs):
    """Backward-compatible alias for weighted customer embeddings."""

    return build_customer_embeddings(*args, **kwargs)


__all__ = [
    "build_behavioral_features",
    "build_customer_embeddings",
    "build_customer_vectors",
    "build_feature_set",
    "load_customer_embeddings",
]
