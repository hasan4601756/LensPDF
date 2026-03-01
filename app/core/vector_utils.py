"""
core/vector_utils.py
Cosine similarity and top-k semantic search utilities.
Works with 768-dim embeddings from intfloat/multilingual-e5-base.
"""

import numpy as np


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    already_normalized: bool = True,
) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        a:                  1D numpy array (embedding vector).
        b:                  1D numpy array (embedding vector).
        already_normalized: If True (default), uses fast dot product.
                            Set False if embeddings were not L2-normalized.

    Returns:
        Float in range [-1, 1]. Higher = more similar.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if already_normalized:
        return float(np.dot(a, b))

    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


def top_k_similar(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int = 5,
    already_normalized: bool = True,
) -> list[dict]:
    """
    Find the top-k most similar embeddings to a query embedding.

    Args:
        query_embedding:    1D numpy array of shape (dim,).
        corpus_embeddings:  2D numpy array of shape (N, dim).
        k:                  Number of top results to return.
        already_normalized: If True, uses dot product (fast).

    Returns:
        List of dicts sorted by score descending:
        [{"index": int, "score": float}, ...]
    """
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a 1D array.")
    if corpus_embeddings.ndim != 2:
        raise ValueError("corpus_embeddings must be a 2D array.")
    if query_embedding.shape[0] != corpus_embeddings.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query={query_embedding.shape[0]}, "
            f"corpus={corpus_embeddings.shape[1]}"
        )

    k = min(k, len(corpus_embeddings))

    if already_normalized:
        scores = corpus_embeddings @ query_embedding
    else:
        norms        = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
        normed       = corpus_embeddings / norms
        query_normed = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        scores       = normed @ query_normed

    top_indices = np.argsort(scores)[::-1][:k]
    return [{"index": int(i), "score": float(scores[i])} for i in top_indices]


def batch_cosine_similarity(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    already_normalized: bool = True,
) -> np.ndarray:
    """
    Compute cosine similarity between every query and every corpus embedding.

    Args:
        query_embeddings:   2D array of shape (Q, dim).
        corpus_embeddings:  2D array of shape (N, dim).
        already_normalized: If True, uses matrix multiplication directly.

    Returns:
        2D array of shape (Q, N) with similarity scores.
    """
    if already_normalized:
        return query_embeddings @ corpus_embeddings.T

    q_norms = np.linalg.norm(query_embeddings,  axis=1, keepdims=True) + 1e-10
    c_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
    return (query_embeddings / q_norms) @ (corpus_embeddings / c_norms).T
