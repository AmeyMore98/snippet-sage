"""Tests for the hybrid retriever module."""

import pytest

from app.rag.retriever import fuse, normalize_scores
from app.rag.schemas import Hit


class TestNormalizeScores:
    """Test score normalization functionality."""

    def test_normalize_empty_list(self):
        """Empty list should return empty list."""
        result = normalize_scores([])
        assert result == []

    def test_normalize_single_hit(self):
        """Single hit should have score normalized to 1.0."""
        hits = [Hit("chunk1", 0.5)]
        result = normalize_scores(hits)
        assert len(result) == 1
        assert result[0].retrieval_score == 1.0
        assert result[0].chunk_id == "chunk1"

    def test_normalize_identical_scores(self):
        """Identical scores should all normalize to 1.0."""
        hits = [Hit("chunk1", 0.7), Hit("chunk2", 0.7), Hit("chunk3", 0.7)]
        result = normalize_scores(hits)
        assert len(result) == 3
        assert all(hit.retrieval_score == 1.0 for hit in result)

    def test_normalize_different_scores(self):
        """Different scores should normalize to [0, 1] range."""
        hits = [
            Hit("chunk1", 0.1),  # min -> 0.0
            Hit("chunk2", 0.5),  # mid -> 0.5
            Hit("chunk3", 0.9),  # max -> 1.0
        ]
        result = normalize_scores(hits)

        assert len(result) == 3
        assert result[0].retrieval_score == 0.0  # (0.1 - 0.1) / (0.9 - 0.1)
        assert result[1].retrieval_score == 0.5  # (0.5 - 0.1) / (0.9 - 0.1)
        assert result[2].retrieval_score == 1.0  # (0.9 - 0.1) / (0.9 - 0.1)

        # Check order preservation
        assert result[0].chunk_id == "chunk1"
        assert result[1].chunk_id == "chunk2"
        assert result[2].chunk_id == "chunk3"

    def test_normalize_preserves_monotonic_order(self):
        """Normalization should preserve relative ordering."""
        hits = [Hit("chunk1", 10.5), Hit("chunk2", 15.2), Hit("chunk3", 12.8)]
        result = normalize_scores(hits)

        # Original order: chunk1 (lowest), chunk3 (middle), chunk2 (highest)
        # Normalized scores should maintain this relative ordering
        assert result[0].retrieval_score < result[2].retrieval_score < result[1].retrieval_score
        assert result[1].retrieval_score == 1.0  # max score
        assert result[0].retrieval_score == 0.0  # min score


class TestFuse:
    """Test result fusion functionality."""

    def test_fuse_empty_lists(self):
        """Fusing empty lists should return empty list."""
        result = fuse([], [], k=10)
        assert result == []

    def test_fuse_one_empty_list(self):
        """Fusing with one empty list should use scores from the other."""
        vector_hits = [Hit("chunk1", 0.8), Hit("chunk2", 0.6)]
        lexical_hits = []

        result = fuse(vector_hits, lexical_hits, alpha=0.6, k=10)

        assert len(result) == 2
        # With alpha=0.6, scores should be 0.6 * vector_score + 0.4 * 0
        assert result[0].chunk_id == "chunk1"
        assert result[0].retrieval_score == 0.6 * 0.8  # 0.48
        assert result[1].chunk_id == "chunk2"
        assert result[1].retrieval_score == 0.6 * 0.6  # 0.36

    def test_fuse_no_overlap(self):
        """Fusing non-overlapping results should combine all hits."""
        vector_hits = [Hit("chunk1", 0.8)]
        lexical_hits = [Hit("chunk2", 0.9)]

        result = fuse(vector_hits, lexical_hits, alpha=0.6, k=10)

        assert len(result) == 2
        # Results should be sorted by combined score (descending)
        chunk_scores = {hit.chunk_id: hit.retrieval_score for hit in result}
        assert chunk_scores["chunk1"] == pytest.approx(0.6 * 0.8)  # 0.48
        assert chunk_scores["chunk2"] == pytest.approx(0.4 * 0.9)  # 0.36

    def test_fuse_with_overlap(self):
        """Fusing overlapping results should combine scores properly."""
        vector_hits = [Hit("chunk1", 1.0), Hit("chunk2", 0.5)]
        lexical_hits = [Hit("chunk1", 0.8), Hit("chunk3", 0.6)]

        result = fuse(vector_hits, lexical_hits, alpha=0.6, k=10)

        assert len(result) == 3
        chunk_scores = {hit.chunk_id: hit.retrieval_score for hit in result}

        # chunk1: 0.6 * 1.0 + 0.4 * 0.8 = 0.92
        # chunk2: 0.6 * 0.5 + 0.4 * 0.0 = 0.30
        # chunk3: 0.6 * 0.0 + 0.4 * 0.6 = 0.24
        assert chunk_scores["chunk1"] == pytest.approx(0.92)
        assert chunk_scores["chunk2"] == pytest.approx(0.30)
        assert chunk_scores["chunk3"] == pytest.approx(0.24)

    def test_fuse_respects_k_limit(self):
        """Fusion should respect the k limit parameter."""
        vector_hits = [Hit("chunk1", 1.0), Hit("chunk2", 0.8), Hit("chunk3", 0.6)]
        lexical_hits = [Hit("chunk4", 0.9), Hit("chunk5", 0.7)]

        result = fuse(vector_hits, lexical_hits, alpha=0.6, k=2)

        assert len(result) == 2
        # Should be top 2 by combined score
        assert result[0].retrieval_score >= result[1].retrieval_score

    def test_fuse_sorts_by_combined_score(self):
        """Results should be sorted by combined score descending."""
        vector_hits = [Hit("chunk1", 0.5)]
        lexical_hits = [
            Hit("chunk2", 1.0),  # This should rank higher
            Hit("chunk3", 0.3),
        ]

        result = fuse(vector_hits, lexical_hits, alpha=0.3, k=10)

        # chunk2: 0.3 * 0.0 + 0.7 * 1.0 = 0.70 (highest)
        # chunk1: 0.3 * 0.5 + 0.7 * 0.0 = 0.15
        # chunk3: 0.3 * 0.0 + 0.7 * 0.3 = 0.21
        assert result[0].chunk_id == "chunk2"
        assert result[1].chunk_id == "chunk3"
        assert result[2].chunk_id == "chunk1"

    def test_fuse_alpha_weighting(self):
        """Different alpha values should change ranking."""
        vector_hits = [Hit("chunk1", 1.0)]  # Strong vector match
        lexical_hits = [Hit("chunk2", 1.0)]  # Strong lexical match

        # High alpha (favor vector)
        result_high_alpha = fuse(vector_hits, lexical_hits, alpha=0.9, k=10)
        # chunk1: 0.9 * 1.0 + 0.1 * 0.0 = 0.90
        # chunk2: 0.9 * 0.0 + 0.1 * 1.0 = 0.10
        assert result_high_alpha[0].chunk_id == "chunk1"

        # Low alpha (favor lexical)
        result_low_alpha = fuse(vector_hits, lexical_hits, alpha=0.1, k=10)
        # chunk1: 0.1 * 1.0 + 0.9 * 0.0 = 0.10
        # chunk2: 0.1 * 0.0 + 0.9 * 1.0 = 0.90
        assert result_low_alpha[0].chunk_id == "chunk2"


class TestHybridSearchLogic:
    """Test hybrid search logic components."""

    def test_hit_named_tuple_creation(self):
        """Hit namedtuple should work correctly."""
        hit = Hit("chunk_id_123", 0.85)

        assert hit.chunk_id == "chunk_id_123"
        assert hit.retrieval_score == 0.85

    def test_hit_with_optional_preview(self):
        """Hit should work with None preview."""
        hit = Hit("chunk_id_123", 0.85)

        assert hit.chunk_id == "chunk_id_123"
        assert hit.retrieval_score == 0.85


# Note: The database-dependent tests (vector_search, lexical_search)
# would need integration tests with actual PostgreSQL + pgvector setup.
# Those are better tested as part of the full integration test suite.
