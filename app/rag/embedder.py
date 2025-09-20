"""
Embedder functionality for the RAG pipeline.

This module provides text embedding capabilities using sentence-transformers
with support for batching and model loading.
"""

import uuid

import numpy as np
from sentence_transformers import SentenceTransformer
from tortoise.transactions import in_transaction

from app.core.config import Settings

# Global model instance for caching
_model = None
_settings = None


def load_model() -> SentenceTransformer:
    """
    Load the embedding model based on environment configuration.

    This function uses a global cache to store the model. For production use,
    it's recommended to call this function once at application startup
    to pre-load the model and avoid a delay on the first request.

    Returns:
        SentenceTransformer model instance
    """
    global _model, _settings

    if _model is None:
        if _settings is None:
            _settings = Settings()
        _model = SentenceTransformer(_settings.EMBEDDING_MODEL)

    return _model


def get_embedding_dimension() -> int:
    """Get the embedding dimension of the loaded model."""
    model = load_model()
    return model.get_sentence_embedding_dimension()


def embed_texts(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """
    Embed a list of texts with optional batching.

    Args:
        texts: List of text strings to embed
        batch_size: Optional batch size for processing

    Returns:
        NumPy array of embeddings with shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.array([])

    model = load_model()

    # Process in batches if batch_size is specified
    if batch_size and len(texts) > batch_size:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        result = np.vstack(embeddings)
    else:
        result = model.encode(texts, convert_to_numpy=True)

    # Ensure we have the right shape and no NaNs
    if len(result.shape) == 1:
        result = result.reshape(1, -1)

    # Check for NaNs
    if np.isnan(result).any():
        raise ValueError("Embeddings contain NaN values")

    return result


async def persist_embeddings(chunk_ids: list[str | uuid.UUID], embeddings: np.ndarray) -> int:
    """
    Persist embeddings for given chunk IDs in a transactional manner.

    Args:
        chunk_ids: List of chunk UUIDs to associate with embeddings
        embeddings: NumPy array of embeddings with shape (len(chunk_ids), embedding_dim)

    Returns:
        Number of embeddings successfully persisted

    Test:
        After call, select count(*) from embeddings equals chunk count.
    """
    if len(chunk_ids) != len(embeddings):
        raise ValueError("Number of chunk_ids must match number of embeddings")

    if not chunk_ids:
        return 0

    # Import here to avoid circular imports
    from app.models.embeddings import Embedding

    embedding_dim = get_embedding_dimension()

    persisted_count = 0

    async with in_transaction() as connection:
        for i, chunk_id in enumerate(chunk_ids):
            try:
                # Convert chunk_id to UUID if it's a string
                if isinstance(chunk_id, str):
                    chunk_uuid = uuid.UUID(chunk_id)
                else:
                    chunk_uuid = chunk_id

                # Create embedding record
                await Embedding.create(
                    chunk_id=chunk_uuid,
                    vector=embeddings[i].tolist(),  # Convert numpy array to list for storage
                    dim=embedding_dim,
                    using_db=connection,
                )
                persisted_count += 1

            except Exception as e:
                # Log error but continue with other embeddings
                print(f"Failed to persist embedding for chunk {chunk_id}: {e}")
                # In a real implementation, you'd use proper logging
                continue

    return persisted_count
