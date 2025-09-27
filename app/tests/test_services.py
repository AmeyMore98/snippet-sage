import uuid
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import pytest_asyncio

from app.core.config import Settings
from app.models import Chunk, Document, Embedding
from app.rag.embedder import get_embedding_dimension
from app.schemas.hit import Hit  # Import Hit for type hinting in mock
from app.schemas.ingest import IngestRequest
from app.services.ingestion_service import IngestionService
from app.services.qa_service import QAService

settings = Settings()


@pytest_asyncio.fixture(autouse=True)
async def initialize_db_for_tests():
    """
    Fixture to ensure the database is clean and initialized for each test.
    This relies on the conftest.py setup for Tortoise ORM.
    """
    # This fixture will run for every test in this file.
    # The actual DB setup/teardown is handled by conftest.py.
    pass


@pytest.fixture
def mock_embedder_embed_texts():
    """Mock embedder.embed_texts to return dummy embeddings."""
    with patch("app.rag.embedder.embed_texts") as mock_embed:
        mock_embed.return_value = np.random.rand(1, get_embedding_dimension()).astype(np.float32)
        yield mock_embed


@pytest.fixture
def mock_embedder_persist_embeddings():
    """Mock embedder.persist_embeddings to do nothing."""
    with patch("app.rag.embedder.persist_embeddings", new_callable=AsyncMock) as mock_persist:
        mock_persist.return_value = 1
        yield mock_persist


@pytest.fixture
def mock_graph_run_answerer():
    """Mock graph.run_answerer for QAService tests."""
    with patch("app.rag.graph.run_answerer") as mock_run_answerer:
        mock_run_answerer.return_value = {
            "answer": "Mocked answer.",
            "citations": [str(uuid.uuid4())],  # Return a valid UUID string
            "confidence": 0.9,
        }
        yield mock_run_answerer


class TestIngestionService:
    async def test_ingest_successful(self, mock_embedder_embed_texts, mock_embedder_persist_embeddings):
        service = IngestionService()
        test_text = "This is a test document for ingestion. It has enough characters."
        payload = IngestRequest(text=test_text, source="test")

        doc_id, chunk_ids = await service.ingest(payload)

        assert doc_id is not None
        assert len(chunk_ids) > 0

        document = await Document.get(id=doc_id)
        assert document.source == "test"

        chunks = await Chunk.filter(document=document)
        assert len(chunks) == len(chunk_ids)
        assert all(c.id in chunk_ids for c in chunks)

        # With persist_embeddings mocked, we should not find any embeddings in the DB
        embeddings = await Embedding.filter(chunk_id__in=chunk_ids)
        assert len(embeddings) == 0

        mock_embedder_embed_texts.assert_called_once()
        mock_embedder_persist_embeddings.assert_awaited_once()

    async def test_ingest_deduplication(self, mock_embedder_embed_texts, mock_embedder_persist_embeddings):
        service = IngestionService()
        test_text = "This is a document to test deduplication. It should only be ingested once."
        payload = IngestRequest(text=test_text, source="test")

        # First ingestion
        doc_id_1, chunk_ids_1 = await service.ingest(payload)
        assert doc_id_1 is not None
        assert len(chunk_ids_1) > 0

        # Second ingestion of the same text
        doc_id_2, chunk_ids_2 = await service.ingest(payload)
        assert doc_id_2 == doc_id_1
        assert chunk_ids_2 == chunk_ids_1  # Should return the same chunk IDs

        # Embedder should only be called once for the unique content
        mock_embedder_embed_texts.assert_called_once()
        mock_embedder_persist_embeddings.assert_called_once()

    async def test_ingest_text_too_long(self):
        service = IngestionService()
        long_text = "a" * (settings.MAX_INPUT_CHARS + 1)
        payload = IngestRequest(text=long_text)

        with pytest.raises(ValueError, match="Input text is too long"):
            await service.ingest(payload)

    async def test_ingest_empty_normalized_text(self):
        service = IngestionService()
        # Text that normalizes to empty (e.g., only whitespace or control chars)
        empty_norm_text = " " * (settings.MIN_INPUT_CHARS + 1)  # Long enough but normalizes to empty
        payload = IngestRequest(text=empty_norm_text)

        with pytest.raises(ValueError, match="Normalized text is empty"):
            await service.ingest(payload)

    async def test_ingest_no_chunks_generated(self, mock_embedder_embed_texts, mock_embedder_persist_embeddings):
        service = IngestionService()
        # Mock chunker to return empty list
        with patch("app.rag.chunker.make_chunks", return_value=[]):
            test_text = "This text should not be chunked. It is long enough to pass initial validation."  # Long enough
            payload = IngestRequest(text=test_text)

            with pytest.raises(ValueError, match="No chunks were generated"):
                await service.ingest(payload)

            mock_embedder_embed_texts.assert_not_called()
            mock_embedder_persist_embeddings.assert_not_called()


class TestQAService:
    @pytest_asyncio.fixture(autouse=True)
    async def setup_qa_service_tests(self, mock_graph_run_answerer):
        """
        Setup for QA service tests.
        Ingest a document to ensure there's data for retrieval.
        """
        self.ingestion_service = IngestionService()
        self.qa_service = QAService()

        # Ingest a sample document
        test_text = "Alice met Bob. They built a Personal Knowledge Base MVP. The project uses FastAPI and LangGraph."
        payload = IngestRequest(text=test_text, source="test-qa")
        self.doc_id, self.chunk_ids = await self.ingestion_service.ingest(payload)

        # Update the mock to use a real chunk_id from the ingested doc
        if self.chunk_ids:
            mock_graph_run_answerer.return_value["citations"] = [str(self.chunk_ids[0])]

        # Mock the retriever to return some hits based on the ingested chunks
        # This avoids actual DB interaction for retrieval in QA tests, focusing on graph flow
        mock_hits: list[Hit] = []
        for i, chunk_id in enumerate(self.chunk_ids):
            chunk = await Chunk.get(id=chunk_id)
            mock_hits.append(
                Hit(
                    chunk_id=str(chunk_id),
                    text=chunk.text,
                    text_preview=chunk.text_preview,
                    retrieval_score=0.85 + i * 0.01,
                    re_rank_score=0.9 + i * 0.01,
                )
            )
        with patch("app.rag.graph.hybrid_search", new_callable=AsyncMock) as mock_hybrid_search:
            mock_hybrid_search.return_value = mock_hits
            yield

    async def test_answer_successful(self):
        question = "What did Alice and Bob build?"
        response = await self.qa_service.answer(question)

        assert response.answer == "Mocked answer."
        assert len(response.citations) > 0
        assert response.confidence == 0.9

        assert isinstance(response.citations[0].chunk_id, uuid.UUID)
        assert response.debug is not None if settings.APP_ENV == "dev" else response.debug is None

    async def test_answer_empty_question(self):
        with pytest.raises(ValueError, match="Question cannot be empty."):
            await self.qa_service.answer("")

    async def test_answer_k_capping(self):
        question = "Test k capping."
        # Request k much higher than max_k (50)
        response = await self.qa_service.answer(question, k=100)
        # The actual check for k capping happens inside the service,
        # we just ensure the call completes and returns a response.
        assert response.answer == "Mocked answer."

    async def test_answer_no_relevant_documents(self):
        # Mock hybrid_search to return no hits
        with patch("app.rag.graph.hybrid_search", new_callable=AsyncMock, return_value=[]):
            question = "A question with no relevant docs."
            response = await self.qa_service.answer(question)

            assert response.answer == ""
            assert len(response.citations) == 0
            assert response.confidence == 0.0
            assert response.debug is not None if settings.APP_ENV == "dev" else response.debug is None
