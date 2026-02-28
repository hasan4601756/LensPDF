"""
Embedding model loading and vector generation.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import time


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingPipeline:
    """
    Loads a local sentence-transformers model and generates embeddings.
    No external API calls are made.

    Args:
        model_name:    Hugging Face model identifier.
        batch_size:    Number of chunks per forward pass.
        device:        'cpu', 'cuda', or None (auto-detect).
        normalize:     L2-normalize embeddings (recommended for cosine similarity).
        show_progress: Show progress bar during batch embedding.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        device: str | None = None,
        normalize: bool = True,
        show_progress: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.show_progress = show_progress

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model '{model_name}' on {self.device.upper()}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        return self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

    def embed_batch(self, chunks: list[str]) -> np.ndarray:
        """
        Embed a list of text chunks in batches.
        Empty/whitespace chunks are skipped with a warning and replaced
        with zero vectors in the output to preserve index alignment.

        Args:
            chunks: List of text strings.

        Returns:
            2D numpy array of shape (len(chunks), embedding_dim).
        """
        if not chunks:
            raise ValueError("chunks list is empty.")

        valid_indices = [i for i, c in enumerate(chunks) if c and c.strip()]
        valid_chunks  = [chunks[i] for i in valid_indices]
        skipped       = len(chunks) - len(valid_chunks)

        if skipped > 0:
            print(f"Warning: Skipped {skipped} empty/whitespace chunk(s).")
        if not valid_chunks:
            raise ValueError("All chunks are empty.")

        start = time.time()
        embeddings = self.model.encode(
            valid_chunks,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress,
        )
        elapsed = time.time() - start
        print(f"Embedded {len(valid_chunks)} chunks in {elapsed:.2f}s "
              f"({len(valid_chunks) / elapsed:.1f} chunks/sec)")

        if skipped > 0:
            full = np.zeros((len(chunks), self.embedding_dim), dtype=np.float32)
            for out_idx, orig_idx in enumerate(valid_indices):
                full[orig_idx] = embeddings[out_idx]
            return full

        return embeddings

    def embed_documents(self, documents: list[dict]) -> list[dict]:
        """
        Embed a list of chunk dicts produced by a text chunker.
        Each dict must contain a 'text' key.
        An 'embedding' key (np.ndarray) is added to each dict in-place.

        Args:
            documents: e.g. [{"text": "...", "metadata": {...}}, ...]

        Returns:
            Same list with 'embedding' added to every dict.
        """
        if not documents:
            raise ValueError("documents list is empty.")

        texts      = [doc.get("text", "") for doc in documents]
        embeddings = self.embed_batch(texts)

        for doc, emb in zip(documents, embeddings):
            doc["embedding"] = emb

        return documents

    def __repr__(self):
        return (f"EmbeddingPipeline(model='{self.model_name}', "
                f"dim={self.embedding_dim}, device={self.device}, "
                f"batch_size={self.batch_size})")
