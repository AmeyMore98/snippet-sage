"""Hybrid retrieval combining vector search and full-text search."""

import asyncio
import logging

import numpy as np
from tortoise.connection import connections

from ..schemas.hit import Hit

logger = logging.getLogger(__name__)


async def vector_search(query_vector: np.ndarray, k: int) -> list[Hit]:
    """
    Perform vector similarity search using pgvector.

    Args:
        query_vector: The query embedding as numpy array
        k: Number of results to return

    Returns:
        List of hits sorted by vector similarity (highest first)
    """
    conn = connections.get("default")

    # Convert numpy array to list for SQL parameter
    query_vec_list = str(query_vector.tolist())

    # Use cosine similarity (1 - cosine distance) for scoring
    # The <=> operator computes cosine distance
    query = """
        SELECT
            c.id as chunk_id,
            1 - (e.vector <=> $1) as vscore
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        ORDER BY e.vector <=> $1
        LIMIT $2
    """

    results = await conn.execute_query(query, [query_vec_list, k])

    hits = []
    for row in results[1]:  # results[1] contains the data rows
        chunk_id, vscore = row
        hits.append(Hit(chunk_id=str(chunk_id), retrieval_score=float(vscore) if vscore is not None else 0.0))

    logger.debug(f"Vector search returned {len(hits)} hits")
    return hits


async def lexical_search(query: str, k: int) -> list[Hit]:
    """
    Perform full-text search using PostgreSQL's FTS.

    Args:
        query: The search query string
        k: Number of results to return

    Returns:
        List of hits sorted by text relevance (highest first)
    """
    conn = connections.get("default")

    # Use plainto_tsquery for simple query processing
    # ts_rank provides relevance scoring
    query_sql = """
        SELECT
            id as chunk_id,
            ts_rank(fts, plainto_tsquery('english', $1)) as lscore
        FROM chunks
        WHERE fts @@ plainto_tsquery('english', $1)
        ORDER BY lscore DESC
        LIMIT $2
    """

    results = await conn.execute_query(query_sql, [query, k])

    hits = []
    for row in results[1]:  # results[1] contains the data rows
        chunk_id, lscore = row
        hits.append(Hit(chunk_id=str(chunk_id), retrieval_score=float(lscore) if lscore is not None else 0.0))

    logger.debug(f"Lexical search returned {len(hits)} hits")
    return hits


def normalize_scores(hits: list[Hit]) -> list[Hit]:
    """
    Normalize hit scores to [0,1] range using min-max scaling.

    Args:
        hits: List of hits to normalize

    Returns:
        List of hits with normalized scores, preserving order
    """
    if not hits:
        return hits

    scores = [hit.retrieval_score for hit in hits]
    min_score = min(scores)
    max_score = max(scores)

    # Handle edge case where all scores are the same
    if max_score == min_score:
        return [Hit(hit.chunk_id, 1.0) for hit in hits]

    # Min-max normalization
    normalized_hits = []
    for hit in hits:
        normalized_score = (hit.retrieval_score - min_score) / (max_score - min_score)
        normalized_hits.append(Hit(hit.chunk_id, normalized_score))

    return normalized_hits


def fuse(vector_hits: list[Hit], lexical_hits: list[Hit], alpha: float = 0.6, k: int = 12) -> list[Hit]:
    """
    Fuse vector and lexical search results using weighted combination.

    Args:
        vector_hits: Results from vector search (should be normalized)
        lexical_hits: Results from lexical search (should be normalized)
        alpha: Weight for vector scores (1-alpha for lexical scores)
        k: Maximum number of results to return

    Returns:
        Fused results sorted by combined score (highest first)
    """
    # Create lookup maps for efficient merging
    vector_map = {hit.chunk_id: hit for hit in vector_hits}
    lexical_map = {hit.chunk_id: hit for hit in lexical_hits}

    # Get all unique chunk IDs from both result sets
    all_chunk_ids = set(vector_map.keys()) | set(lexical_map.keys())

    fused_hits = []
    for chunk_id in all_chunk_ids:
        vector_hit = vector_map.get(chunk_id)
        lexical_hit = lexical_map.get(chunk_id)

        # Get scores, defaulting to 0 if not present in one of the searches
        vector_score = vector_hit.retrieval_score if vector_hit else 0.0
        lexical_score = lexical_hit.retrieval_score if lexical_hit else 0.0

        # Weighted combination
        combined_score = alpha * vector_score + (1 - alpha) * lexical_score

        fused_hits.append(Hit(chunk_id, retrieval_score=combined_score))

    # Sort by combined score (descending) and limit to k
    fused_hits.sort(key=lambda x: x.retrieval_score, reverse=True)
    return fused_hits[:k]


async def hybrid_search(query: str, query_vector: np.ndarray, k: int = 12, alpha: float = 0.6) -> list[Hit]:
    """
    Perform hybrid search combining vector and lexical approaches.

    Args:
        query: The search query string
        query_vector: The query embedding as numpy array
        k: Number of results to return
        alpha: Weight for vector scores vs lexical scores

    Returns:
        List of fused hits sorted by relevance
    """
    # We retrieve 2x more candidates than needed from each search method
    #  because fusion works better with more options.
    # If we only got the top-k from each method, we might miss great results that rank #11 in vector
    # search but #2 in lexical search.
    # The expanded candidate pool allows optimal fusion ranking.
    # But with a cap to avoid excessive computation
    kvec = min(k * 2, 50)
    klex = min(k * 2, 50)

    # Perform both searches in parallel for better performance
    vector_hits, lexical_hits = await asyncio.gather(vector_search(query_vector, kvec), lexical_search(query, klex))

    # Normalize scores separately for each method
    vector_hits_norm = normalize_scores(vector_hits)
    lexical_hits_norm = normalize_scores(lexical_hits)

    # Fuse the results
    fused_hits = fuse(vector_hits_norm, lexical_hits_norm, alpha, k)

    return fused_hits
