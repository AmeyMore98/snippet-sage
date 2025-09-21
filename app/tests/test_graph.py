"""
Tests for the LangGraph orchestration module.
"""

from unittest.mock import AsyncMock, patch

import numpy as np

from app.rag.graph import (
    _enrich_hits_with_text,
    answer_node,
    build_graph,
    cite_node,
    parse_node,
    re_rank_node,
    retrieve_node,
)
from app.rag.schemas import Hit


class TestParseNode:
    """Test cases for the parse_node function."""

    def test_parse_node_normalizes_question(self):
        """Test that parse node normalizes the input question."""
        state = {"question": "  What is AI?  "}
        result = parse_node(state)

        assert "q_norm" in result
        assert result["q_norm"] == "What is AI?"
        assert result["question"] == "  What is AI?  "  # Original preserved

    def test_parse_node_handles_multiline_question(self):
        """Test parse node with multiline input."""
        state = {"question": "What is\n\nmachine\tlearning?"}
        result = parse_node(state)

        assert result["q_norm"] == "What is machine learning?"

    def test_parse_node_handles_empty_question(self):
        """Test parse node with empty/whitespace question."""
        state = {"question": "   "}
        result = parse_node(state)

        assert result["q_norm"] == ""

    def test_parse_node_fallback_on_normalization_failure(self):
        """Test parse node falls back to original if normalization fails."""
        state = {"question": "test"}
        result = parse_node(state)

        assert result["q_norm"] == "test"


class TestRetrieveNode:
    """Test cases for the retrieve_node function."""

    @patch("app.rag.graph.hybrid_search")
    @patch("app.rag.graph.embed_texts")
    @patch("app.rag.graph._enrich_hits_with_text")
    async def test_retrieve_node_success(self, mock_enrich, mock_embed, mock_hybrid):
        """Test successful retrieval with mock data."""
        # Setup mocks
        mock_embed.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_hits = [Hit(chunk_id="chunk-1", retrieval_score=0.9), Hit(chunk_id="chunk-2", retrieval_score=0.8)]
        mock_hybrid.return_value = mock_hits
        mock_enrich.return_value = mock_hits

        state = {"q_norm": "test question", "k": 10}
        result = await retrieve_node(state)

        assert "query_vector" in result
        assert "hits" in result
        assert len(result["hits"]) == 2
        assert result["hits"][0].chunk_id == "chunk-1"
        mock_embed.assert_called_once_with(["test question"])
        mock_hybrid.assert_called_once()
        mock_enrich.assert_called_once_with(mock_hits)

    @patch("app.rag.graph.hybrid_search")
    @patch("app.rag.graph.embed_texts")
    @patch("app.rag.graph._enrich_hits_with_text")
    async def test_retrieve_node_no_results(self, mock_enrich, mock_embed, mock_hybrid):
        """Test retrieve node when no results are found."""
        mock_embed.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_hybrid.return_value = []
        mock_enrich.return_value = []

        state = {"q_norm": "no results", "k": 5}
        result = await retrieve_node(state)

        assert result["hits"] == []
        # Enrich is only called when there are hits
        mock_enrich.assert_not_called()

    @patch("app.rag.graph.hybrid_search")
    @patch("app.rag.graph.embed_texts")
    async def test_retrieve_node_uses_default_k(self, mock_embed, mock_hybrid):
        """Test retrieve node uses default k value."""
        mock_embed.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_hybrid.return_value = []

        state = {"q_norm": "test"}  # No k specified
        await retrieve_node(state)

        # Should call hybrid_search with default k=12
        mock_hybrid.assert_called_once_with("test", mock_embed.return_value[0], k=12)


class TestRerankNode:
    """Test cases for the re_rank_node function."""

    @patch("app.rag.graph.re_rank_hits")
    async def test_re_rank_node_success(self, mock_re_rank):
        """Test successful re_ranking."""
        input_hits = [
            Hit(chunk_id="chunk-1", retrieval_score=0.9, text="text 1"),
            Hit(chunk_id="chunk-2", retrieval_score=0.8, text="text 2"),
        ]
        re_ranked_hits = [
            Hit(chunk_id="chunk-2", retrieval_score=0.8, text="text 2", re_rank_score=0.95),
            Hit(chunk_id="chunk-1", retrieval_score=0.9, text="text 1", re_rank_score=0.85),
        ]
        mock_re_rank.return_value = re_ranked_hits

        state = {"q_norm": "test query", "hits": input_hits, "k": 10}
        result = await re_rank_node(state)

        assert "top_passages" in result
        assert len(result["top_passages"]) == 2
        assert result["top_passages"][0].chunk_id == "chunk-2"  # Reranked order
        mock_re_rank.assert_called_once_with("test query", input_hits, n=6)

    @patch("app.rag.graph.re_rank_hits")
    async def test_re_rank_node_empty_hits(self, mock_re_rank):
        """Test re_rank node with no hits."""
        state = {"q_norm": "test", "hits": []}
        result = await re_rank_node(state)

        assert result["top_passages"] == []
        mock_re_rank.assert_not_called()

    @patch("app.rag.graph.re_rank_hits")
    async def test_re_rank_node_respects_k_limit(self, mock_re_rank):
        """Test re_rank node respects k limit for top_n."""
        mock_re_rank.return_value = []

        # Test with k=3, should limit to min(6, 3) = 3
        state = {"q_norm": "test", "hits": [Hit("test", 0.5)], "k": 3}
        await re_rank_node(state)

        mock_re_rank.assert_called_once_with("test", [Hit("test", 0.5)], n=3)


class TestAnswerNode:
    """Test cases for the answer_node function."""

    @patch("app.rag.graph.run_answerer")
    def test_answer_node_with_passages(self, mock_answerer):
        """Test answer node with valid passages."""
        mock_answerer.return_value = {
            "answer": "This is the answer",
            "citations": ["chunk-1", "chunk-2"],
            "confidence": 0.85,
        }

        passages = [
            Hit(chunk_id="chunk-1", text="text 1", re_rank_score=0.9),
            Hit(chunk_id="chunk-2", text="text 2", re_rank_score=0.8),
        ]

        state = {"q_norm": "test question", "top_passages": passages}
        result = answer_node(state)

        assert "answer_bundle" in result
        bundle = result["answer_bundle"]
        assert bundle["answer"] == "This is the answer"
        assert bundle["citations"] == ["chunk-1", "chunk-2"]
        assert bundle["confidence"] == 0.85
        mock_answerer.assert_called_once_with("test question", passages)

    def test_answer_node_no_passages(self):
        """Test answer node with no passages."""
        state = {"q_norm": "test question", "top_passages": []}
        result = answer_node(state)

        bundle = result["answer_bundle"]
        assert bundle["answer"] == ""
        assert bundle["citations"] == []
        assert bundle["confidence"] == 0.0

    def test_answer_node_missing_top_passages(self):
        """Test answer node with missing top_passages key."""
        state = {"q_norm": "test question"}
        result = answer_node(state)

        bundle = result["answer_bundle"]
        assert bundle["answer"] == ""
        assert bundle["citations"] == []
        assert bundle["confidence"] == 0.0


class TestCiteNode:
    """Test cases for the cite_node function."""

    async def test_cite_node_formats_citations_correctly(self):
        """Test cite node formats citations with proper previews."""
        passages = [
            Hit(
                chunk_id="chunk-1",
                text="This is the first passage with some content",
                text_preview="This is the first passage with some content",
                re_rank_score=0.9,
            ),
            Hit(
                chunk_id="chunk-2",
                text="This is the second passage",
                text_preview="This is the second passage",
                re_rank_score=0.8,
            ),
        ]

        answer_bundle = {"answer": "The answer", "citations": ["chunk-1", "chunk-2"], "confidence": 0.85}

        state = {"answer_bundle": answer_bundle, "top_passages": passages}

        result = await cite_node(state)

        response = result["response"]
        assert response["answer"] == "The answer"
        assert response["confidence"] == 0.85
        assert len(response["citations"]) == 2

        # Check first citation
        citation1 = response["citations"][0]
        assert citation1["chunk_id"] == "chunk-1"
        assert citation1["preview"] == "This is the first passage with some content"
        assert citation1["score"] == 0.9

    async def test_cite_node_handles_missing_cited_chunks(self):
        """Test cite node handles citations for chunks not in top passages."""
        passages = [Hit(chunk_id="chunk-1", text="Available chunk", re_rank_score=0.9)]

        answer_bundle = {
            "answer": "The answer",
            "citations": ["chunk-1", "missing-chunk"],  # missing-chunk not in passages
            "confidence": 0.85,
        }

        state = {"answer_bundle": answer_bundle, "top_passages": passages}

        result = await cite_node(state)
        citations = result["response"]["citations"]

        # Should only include citations for chunks that exist in passages
        assert len(citations) == 1
        assert citations[0]["chunk_id"] == "chunk-1"

    async def test_cite_node_empty_state(self):
        """Test cite node with empty state."""
        state = {}
        result = await cite_node(state)

        response = result["response"]
        assert response["answer"] == ""
        assert response["citations"] == []
        assert response["confidence"] == 0.0


class TestEnrichHitsWithText:
    """Test cases for the _enrich_hits_with_text helper function."""

    @patch("app.rag.graph.Chunk")
    async def test_enrich_hits_success(self, mock_chunk):
        """Test successful hit enrichment with database data."""
        # Mock Chunk.filter().values() to return the expected data
        mock_chunk.filter.return_value.values = AsyncMock(
            return_value=[
                {"id": "chunk-1", "text": "Full text content 1"},
                {"id": "chunk-2", "text": "Full text content 2"},
            ]
        )

        hits = [Hit(chunk_id="chunk-1", retrieval_score=0.9), Hit(chunk_id="chunk-2", retrieval_score=0.8)]

        enriched = await _enrich_hits_with_text(hits)

        assert len(enriched) == 2
        assert enriched[0].chunk_id == "chunk-1"
        assert enriched[0].text == "Full text content 1"
        assert enriched[0].text_preview == "Full text content 1"
        assert enriched[0].retrieval_score == 0.9

    @patch("app.rag.graph.Chunk")
    async def test_enrich_hits_database_error(self, mock_chunk):
        """Test hit enrichment handles database errors gracefully."""
        mock_chunk.filter.return_value.values = AsyncMock(side_effect=Exception("Database error"))

        hits = [Hit(chunk_id="chunk-1", retrieval_score=0.9)]

        # Should return original hits without enrichment on error
        enriched = await _enrich_hits_with_text(hits)
        assert enriched == hits

    async def test_enrich_hits_empty_input(self):
        """Test hit enrichment with empty input."""
        enriched = await _enrich_hits_with_text([])
        assert enriched == []


class TestBuildGraph:
    """Test cases for the build_graph function."""

    def test_build_graph_compilation(self):
        """Test that the graph compiles successfully."""
        graph = build_graph()
        assert graph is not None

    @patch("app.rag.graph.hybrid_search")
    @patch("app.rag.graph.embed_texts")
    @patch("app.rag.graph._enrich_hits_with_text")
    @patch("app.rag.graph.re_rank_hits")
    @patch("app.rag.graph.run_answerer")
    async def test_graph_full_execution(self, mock_answerer, mock_re_rank, mock_enrich, mock_embed, mock_hybrid):
        """Test complete graph execution with mocked dependencies."""
        # Setup all mocks
        mock_embed.return_value = [np.array([0.1, 0.2, 0.3])]

        mock_hits = [Hit(chunk_id="chunk-1", retrieval_score=0.9, text="Sample text")]
        mock_hybrid.return_value = mock_hits
        mock_enrich.return_value = mock_hits

        re_ranked_hits = [Hit(chunk_id="chunk-1", retrieval_score=0.9, text="Sample text", re_rank_score=0.95)]
        mock_re_rank.return_value = re_ranked_hits

        mock_answerer.return_value = {"answer": "This is the answer", "citations": ["chunk-1"], "confidence": 0.85}

        # Build and run graph
        graph = build_graph()
        initial_state = {"question": "What is AI?", "k": 5}

        # Execute the graph
        result = await graph.ainvoke(initial_state)

        # Verify final response structure
        assert "response" in result
        response = result["response"]
        assert response["answer"] == "This is the answer"
        assert len(response["citations"]) == 1
        assert response["citations"][0]["chunk_id"] == "chunk-1"
        assert response["confidence"] == 0.85

        # Verify all components were called
        mock_embed.assert_called_once()
        mock_hybrid.assert_called_once()
        mock_enrich.assert_called_once()
        mock_re_rank.assert_called_once()
        mock_answerer.assert_called_once()
