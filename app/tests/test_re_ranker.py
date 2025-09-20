"""Tests for the re_ranker module."""

import numpy as np
import pytest

from app.rag.re_ranker import re_rank_hits
from app.rag.schemas import Hit


class TestRerankHits:
    """Test the Hit re_ranking functionality with cross-encoder."""

    @pytest.mark.asyncio
    async def test_re_rank_empty_hits(self):
        """Empty hits list should return empty result."""
        result = await re_rank_hits("query", [], n=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_re_rank_empty_query(self):
        """Empty query should return original hits with zero re-rank scores."""
        hits = [Hit("chunk1", 0.5, "some text"), Hit("chunk2", 0.4, "other text")]
        result = await re_rank_hits("", hits, n=2)

        assert len(result) == 2
        for res_hit in result:
            assert res_hit.re_rank_score == 0.0
            assert res_hit.retrieval_score in [0.5, 0.4]  # Original scores preserved

    @pytest.mark.asyncio
    async def test_re_rank_single_hit(self):
        """Single hit should be returned with cross-encoder score."""
        hits = [Hit(chunk_id="chunk1", retrieval_score=0.5, text="python programming tutorial")]

        result = await re_rank_hits("python", hits, n=5)

        assert len(result) == 1
        assert result[0].chunk_id == "chunk1"
        # Rerank score should be populated
        assert 0.0 <= result[0].re_rank_score <= 1.0
        assert result[0].retrieval_score == 0.5  # Original score preserved

    @pytest.mark.asyncio
    async def test_re_rank_multiple_hits_by_relevance(self):
        """Multiple hits should be reordered by cross-encoder relevance."""
        hits = [
            Hit(chunk_id="chunk1", retrieval_score=0.9, text="cooking recipes and kitchen tools"),
            Hit(chunk_id="chunk2", retrieval_score=0.1, text="python machine learning tutorial"),
            Hit(chunk_id="chunk3", retrieval_score=0.5, text="general programming concepts overview"),
        ]

        result = await re_rank_hits("python programming", hits, n=3)

        assert len(result) == 3
        # chunk2 should now be first
        assert result[0].chunk_id == "chunk2"

        # Scores should be in descending order after re_ranking
        re_rank_scores = [hit.re_rank_score for hit in result]
        assert re_rank_scores == sorted(re_rank_scores, reverse=True)

        # All scores should be valid
        for hit in result:
            assert 0.0 <= hit.re_rank_score <= 1.0

    @pytest.mark.asyncio
    async def test_re_rank_respects_n_limit(self):
        """Should return at most n results after re_ranking."""
        hits = [
            Hit(chunk_id="chunk1", retrieval_score=0.8, text="machine learning basics"),
            Hit(chunk_id="chunk2", retrieval_score=0.7, text="deep learning tutorial"),
            Hit(chunk_id="chunk3", retrieval_score=0.6, text="neural networks guide"),
            Hit(chunk_id="chunk4", retrieval_score=0.5, text="AI fundamentals"),
        ]

        result = await re_rank_hits("machine learning", hits, n=2)
        assert len(result) == 2

        # Should return the top 2 most relevant after cross-encoder scoring
        for hit in result:
            assert 0.0 <= hit.re_rank_score <= 1.0

    @pytest.mark.asyncio
    async def test_re_rank_n_larger_than_hits(self):
        """N larger than hit count should return all re-ranked hits."""
        hits = [Hit("chunk1", 0.8, "python tutorial"), Hit("chunk2", 0.7, "java guide")]

        result = await re_rank_hits("programming", hits, n=10)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_re_rank_preserves_chunk_ids(self):
        """Chunk IDs should be preserved during re_ranking."""
        original_chunk_ids = {"chunk1", "chunk2", "chunk3"}
        hits = [
            Hit(chunk_id="chunk1", retrieval_score=0.8, text="python tutorial"),
            Hit(chunk_id="chunk2", retrieval_score=0.7, text="machine learning"),
            Hit(chunk_id="chunk3", retrieval_score=0.6, text="python basics"),
        ]

        result = await re_rank_hits("python", hits, n=3)

        result_chunk_ids = {hit.chunk_id for hit in result}
        assert result_chunk_ids == original_chunk_ids

    @pytest.mark.asyncio
    async def test_re_rank_handles_empty_text(self):
        """Should handle hits with empty text."""
        hits = [
            Hit(chunk_id="chunk1", retrieval_score=0.8, text=""),  # Empty text
            Hit(chunk_id="chunk2", retrieval_score=0.7, text="python programming guide"),
        ]
        result = await re_rank_hits("python", hits, n=2)

        # Should not crash and return results
        assert len(result) <= 2

        for hit in result:
            assert 0.0 <= hit.re_rank_score <= 1.0
            assert hit.chunk_id in ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_re_rank_new_scores_meaningful(self):
        """New scores should be meaningful (between 0 and 1)."""
        hits = [
            Hit(chunk_id="chunk1", retrieval_score=0.5, text="python programming tutorial"),
            Hit(chunk_id="chunk2", retrieval_score=0.3, text="java development guide"),
        ]
        result = await re_rank_hits("python", hits, n=2)

        for hit in result:
            assert 0.0 <= hit.re_rank_score <= 1.0
            assert isinstance(hit.re_rank_score, float | np.floating)

    @pytest.mark.asyncio
    async def test_re_rank_batch_processing(self):
        """Should handle batch processing efficiently for many hits."""
        # Create a larger set of candidates to test batch processing
        hits = [Hit(f"chunk{i}", 0.5, f"text content {i} about programming") for i in range(20)]

        result = await re_rank_hits("programming", hits, n=10)

        assert len(result) == 10
        # All should have valid scores and be sorted
        re_rank_scores = [hit.re_rank_score for hit in result]
        assert re_rank_scores == sorted(re_rank_scores, reverse=True)
        for res_hit in result:
            assert 0.0 <= res_hit.re_rank_score <= 1.0
