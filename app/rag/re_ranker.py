"""Reranking module for improving retrieval results using cross-encoder models."""

import logging

import anyio
from sentence_transformers import CrossEncoder
from torch import nn

from .schemas import Hit

logger = logging.getLogger(__name__)

# Global cross-encoder model cache
_cross_encoder_model: CrossEncoder | None = None


def get_cross_encoder_model() -> CrossEncoder:
    """
    Get or initialize the cross-encoder model.

    This function uses a global cache to store the model. For production use,
    it's recommended to call this function once at application startup
    to pre-load the model and avoid a delay on the first request.

    Returns:
        Loaded CrossEncoder model instance
    """
    global _cross_encoder_model

    # The model is loaded only once and then cached in the global variable.
    if _cross_encoder_model is None:
        logger.info("Loading cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _cross_encoder_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,  # or 384 if you want faster/cheaper
            default_activation_function=nn.Sigmoid(),  # returns [0,1] directly
        )
        logger.info("Cross-encoder model loaded successfully")

    return _cross_encoder_model


def topn(query: str, candidates: list[Hit], n: int = 6) -> list[Hit]:
    """
    Select top-N candidates based on cross-encoder re-ranking scores.

    Uses batch prediction for better performance when scoring multiple candidates.

    Args:
        query: The search query string
        candidates: List of Hit objects to re-rank
        n: Number of top results to return

    Returns:
        List of Hit objects with `re_rank_score` populated, sorted by `re_rank_score` (highest first), limited to n results
    """
    if not candidates:
        return []

    if not query or not query.strip():
        # Return candidates with zero scores if query is empty
        return [hit._replace(re_rank_score=0.0) for hit in candidates[:n]]

    try:
        model = get_cross_encoder_model()

        # Prepare query-text pairs for batch scoring
        scorable_hits = [hit for hit in candidates if hit.text and hit.text.strip()]

        if not scorable_hits:
            return [hit._replace(re_rank_score=0.0) for hit in candidates[:n]]

        query_text_pairs = [(query.strip(), hit.text.strip()) for hit in scorable_hits]

        # Batch predict for efficiency.
        # The model is configured with a Sigmoid activation function, so scores are already in the [0, 1] range.
        scores = model.predict(query_text_pairs)

        scored_hits = [hit._replace(re_rank_score=score) for hit, score in zip(scorable_hits, scores, strict=False)]

        scored_hits.sort(key=lambda x: x.re_rank_score, reverse=True)

        logger.debug(
            f"re_ranked {len(candidates)} candidates using cross-encoder, returning top {min(n, len(scored_hits))}"
        )

        return scored_hits[:n]

    except Exception as e:
        logger.error(f"Error in batch re_ranking: {e}")
        # Fallback: return original order with zero scores
        return [hit._replace(re_rank_score=0.0) for hit in candidates[:n]]


async def re_rank_hits(query: str, hits: list[Hit], n: int = 6) -> list[Hit]:
    """
    Re-rank search hits using cross-encoder model.

    This function takes the initial retrieval hits and applies cross-encoder
    re-ranking to improve the ordering based on semantic query-text relevance.

    Args:
        query: The search query string
        hits: List of Hit objects from retrieval, which must include text
        n: Number of top results to return after re-ranking

    Returns:
        List of re_ranked Hit objects with `re_rank_score` populated, limited to n results
    """
    if not hits:
        return []

    # Get re_ranked results using cross-encoder
    re_ranked_hits = await anyio.to_thread.run_sync(topn, query, hits, n)

    logger.debug(f"Cross-encoder re_ranked {len(hits)} hits down to {len(re_ranked_hits)} results")

    return re_ranked_hits
