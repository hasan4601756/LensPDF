"""
utils/helpers.py
Utilities for saving, loading, and inspecting embedding arrays.
"""

import os
import numpy as np


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    """
    Save an embeddings array to a .npy file.

    Args:
        embeddings: 2D numpy array of shape (N, dim).
        path:       File path ending in .npy (e.g. 'data/embeddings.npy').
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, embeddings)
    print(f"Saved  → {path}  |  shape={embeddings.shape}  |  "
          f"size={embeddings.nbytes / 1024:.1f} KB")


def load_embeddings(path: str) -> np.ndarray:
    """
    Load an embeddings array from a .npy file.

    Args:
        path: File path ending in .npy.

    Returns:
        2D numpy array of shape (N, dim).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    embeddings = np.load(path)
    print(f"Loaded ← {path}  |  shape={embeddings.shape}  |  "
          f"size={embeddings.nbytes / 1024:.1f} KB")
    return embeddings


def save_embeddings_with_metadata(
    embeddings: np.ndarray,
    metadata: list[dict],
    embeddings_path: str,
    metadata_path: str,
) -> None:
    """
    Save embeddings array and their associated metadata separately.

    Args:
        embeddings:      2D numpy array of shape (N, dim).
        metadata:        List of N dicts (one per chunk).
        embeddings_path: Path for the embeddings .npy file.
        metadata_path:   Path for the metadata .npy file.
    """
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Length mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata entries."
        )

    save_embeddings(embeddings, embeddings_path)
    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    np.save(metadata_path, np.array(metadata, dtype=object))
    print(f"Saved  → {metadata_path}  |  {len(metadata)} metadata entries")


def load_embeddings_with_metadata(
    embeddings_path: str,
    metadata_path: str,
) -> tuple[np.ndarray, list[dict]]:
    """
    Load embeddings and their associated metadata.

    Returns:
        Tuple of (embeddings np.ndarray, metadata list[dict]).
    """
    embeddings = load_embeddings(embeddings_path)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = np.load(metadata_path, allow_pickle=True).tolist()
    print(f"Loaded ← {metadata_path}  |  {len(metadata)} metadata entries")
    return embeddings, metadata


def inspect_embeddings(embeddings: np.ndarray) -> dict:
    """
    Print and return basic statistics about an embeddings array.

    Args:
        embeddings: 2D numpy array of shape (N, dim).

    Returns:
        Dict with stats: num_vectors, dim, dtype, size_kb, mean, std, min, max.
    """
    stats = {
        "num_vectors": embeddings.shape[0],
        "dim":         embeddings.shape[1],
        "dtype":       str(embeddings.dtype),
        "size_kb":     round(embeddings.nbytes / 1024, 2),
        "mean":        round(float(np.mean(embeddings)), 6),
        "std":         round(float(np.std(embeddings)),  6),
        "min":         round(float(np.min(embeddings)),  6),
        "max":         round(float(np.max(embeddings)),  6),
    }

    print("=== Embeddings Inspection ===")
    for k, v in stats.items():
        print(f"  {k:<14}: {v}")

    return stats
