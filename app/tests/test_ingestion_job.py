"""Tests for the async ingestion job."""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.models import Chunk, Document
from app.models.document import DocumentStatus
from app.rag.embedder import get_embedding_dimension
from app.schemas.ingest import IngestRequest
from app.services.ingestion_service import IngestionService


@pytest.fixture
def mock_embedder_embed_texts():
    """Mock embedder.embed_texts to return dummy embeddings."""
    with patch("app.rag.embedder.embed_texts") as mock_embed:
        # Return embeddings that match the number of chunks
        def side_effect(texts):
            return np.random.rand(len(texts), get_embedding_dimension()).astype(np.float32)

        mock_embed.side_effect = side_effect
        yield mock_embed


@pytest.fixture
def mock_dramatiq_message():
    """Mock dramatiq message context for retry testing."""
    with patch("app.jobs.ingestion_job.CurrentMessage.get_current_message") as mock_get_message:
        mock_message = MagicMock()
        mock_message.options = {"retries": 0}
        mock_get_message.return_value = mock_message
        yield mock_message


class TestIngestionJob:
    async def test_process_ingestion_successful(self, mock_embedder_embed_texts):
        """Test that a PENDING document is successfully processed."""
        # Import the core function, not the Dramatiq actor
        from app.jobs.ingestion_job import _process_document_job

        # Create a PENDING document
        service = IngestionService()
        test_text = "This is a test document for async processing. It has enough content."
        payload = IngestRequest(text=test_text, source="test-job")
        result = await service.create_document(payload)
        doc_id = result.document_id

        document = await Document.get(id=doc_id)
        assert document.status == DocumentStatus.PENDING

        # Call the async function directly
        await _process_document_job(str(doc_id))

        # Verify the document was processed
        document = await Document.get(id=doc_id)
        assert document.status == DocumentStatus.COMPLETED
        assert document.processing_errors is None

        # Verify chunks were created
        chunks = await Chunk.filter(document=document)
        assert len(chunks) > 0

        # Verify embedder was called
        mock_embedder_embed_texts.assert_called_once()

    async def test_process_ingestion_document_not_found(self):
        """Test that processing handles non-existent documents gracefully."""
        from app.jobs.ingestion_job import _process_document_job

        fake_id = str(uuid.uuid4())

        # Should not raise an exception, just log and return
        await _process_document_job(fake_id)

        # Verify no document was created
        document = await Document.get_or_none(id=fake_id)
        assert document is None

    async def test_process_ingestion_skips_non_pending_document(self, mock_embedder_embed_texts):
        """Test that the job skips documents not in PENDING status."""
        from app.jobs.ingestion_job import _process_document_job

        # Create a document and manually set it to COMPLETED
        service = IngestionService()
        test_text = "This document is already completed."
        payload = IngestRequest(text=test_text, source="test-job")
        result = await service.create_document(payload)
        doc_id = result.document_id

        document = await Document.get(id=doc_id)
        document.status = DocumentStatus.COMPLETED
        await document.save(update_fields=["status"])

        # Try to process it
        await _process_document_job(str(doc_id))

        # Verify status didn't change
        document = await Document.get(id=doc_id)
        assert document.status == DocumentStatus.COMPLETED

        # Verify no chunks were created
        chunks = await Chunk.filter(document=document)
        assert len(chunks) == 0

        # Verify embedder was not called
        mock_embedder_embed_texts.assert_not_called()

    async def test_process_ingestion_failure_marks_as_failed(self, mock_dramatiq_message):
        """Test that a document is marked FAILED after retries are exhausted."""
        from app.jobs.ingestion_job import _process_document_job

        # Create a PENDING document
        service = IngestionService()
        test_text = "This document will fail processing."
        payload = IngestRequest(text=test_text, source="test-job")
        result = await service.create_document(payload)
        doc_id = result.document_id

        # Set mock to indicate we're on the last retry
        mock_dramatiq_message.options["retries"] = 5

        # Mock the service to raise an exception
        with patch.object(IngestionService, "process_document", side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError, match="Test error"):
                await _process_document_job(str(doc_id))

        # Verify the document was marked as FAILED
        document = await Document.get(id=doc_id)
        assert document.status == DocumentStatus.FAILED
        assert "RuntimeError: Test error" in document.processing_errors

    async def test_process_ingestion_retry_without_marking_failed(self, mock_dramatiq_message):
        """Test that a document stays PROCESSING during intermediate retries."""
        from app.jobs.ingestion_job import _process_document_job

        # Create a PENDING document
        service = IngestionService()
        test_text = "This document will fail but retry."
        payload = IngestRequest(text=test_text, source="test-job")
        result = await service.create_document(payload)
        doc_id = result.document_id

        # Set mock to indicate we're NOT on the last retry
        mock_dramatiq_message.options["retries"] = 2

        # Mock the service to raise an exception
        with patch.object(IngestionService, "process_document", side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError, match="Test error"):
                await _process_document_job(str(doc_id))

        # Verify the document is still in PROCESSING (not FAILED yet)
        document = await Document.get(id=doc_id)
        assert document.status == DocumentStatus.PROCESSING
        assert document.processing_errors is None

    async def test_process_ingestion_failure_only_updates_processing_documents(self, mock_dramatiq_message):
        """Test that failure handler only updates documents in PROCESSING status."""
        from app.jobs.ingestion_job import _process_document_job

        # Create a PENDING document
        service = IngestionService()
        test_text = "This document will be manually marked COMPLETED."
        payload = IngestRequest(text=test_text, source="test-job")
        result = await service.create_document(payload)
        doc_id = result.document_id

        # Set mock to indicate we're on the last retry
        mock_dramatiq_message.options["retries"] = 5

        # Mock the service to raise an exception, but also manually set status to COMPLETED
        async def failing_process_with_status_change(document):
            # Simulate another process completing the document
            doc = await Document.get(id=document.id)
            doc.status = DocumentStatus.COMPLETED
            await doc.save(update_fields=["status"])
            raise RuntimeError("Test error")

        with patch.object(IngestionService, "process_document", side_effect=failing_process_with_status_change):
            with pytest.raises(RuntimeError, match="Test error"):
                await _process_document_job(str(doc_id))

        # Verify the document stayed COMPLETED (not changed to FAILED)
        document = await Document.get(id=doc_id)
        assert document.status == DocumentStatus.COMPLETED
        assert document.processing_errors is None

    async def test_process_ingestion_clears_previous_errors_on_success(self, mock_embedder_embed_texts):
        """Test that successful processing clears any previous error messages."""
        from app.jobs.ingestion_job import _process_document_job

        # Create a document with existing errors
        service = IngestionService()
        test_text = "This document had previous errors."
        payload = IngestRequest(text=test_text, source="test-job")
        result = await service.create_document(payload)
        doc_id = result.document_id

        document = await Document.get(id=doc_id)
        document.processing_errors = "Previous error message"
        await document.save(update_fields=["processing_errors"])

        # Process the document successfully
        await _process_document_job(str(doc_id))

        # Verify errors were cleared
        document = await Document.get(id=doc_id)
        assert document.status == DocumentStatus.COMPLETED
        assert document.processing_errors is None
