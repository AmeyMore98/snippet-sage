"""
Dramatiq actor for async document ingestion processing.

This module defines the background job that processes documents asynchronously,
including chunking, embedding, and error handling with automatic retries.
"""

import logging
from uuid import UUID

import dramatiq
from dramatiq.middleware import CurrentMessage
from tortoise import Tortoise

from app.core import queue  # noqa: F401 - Initialize Dramatiq broker for worker
from app.core.config import TORTOISE_ORM
from app.models import Document
from app.models.document import DocumentStatus
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


async def _process_document_job(document_id: str) -> None:
    """
    Core logic for processing a document ingestion.

    This function contains the actual processing logic and is called by the
    Dramatiq actor. It's separated out to make testing easier.

    Args:
        document_id: The UUID of the document to process (as string)

    Raises:
        Exception: Re-raises exceptions to trigger retry mechanism
    """
    # Initialize Tortoise ORM connection for this worker
    await Tortoise.init(config=TORTOISE_ORM)

    doc_uuid = UUID(document_id)

    try:
        # Fetch the document only if it's in PENDING status
        document = await Document.get_or_none(id=doc_uuid, status=DocumentStatus.PENDING)
        if not document:
            logger.info(f"Document {document_id} not found or not in PENDING status. " f"Skipping processing.")
            return

        # Update status to PROCESSING
        document.status = DocumentStatus.PROCESSING
        await document.save(update_fields=["status"])
        logger.info(f"Processing document {document_id}")

        # Perform the actual ingestion work using IngestionService
        ingestion_service = IngestionService()
        chunk_ids = await ingestion_service.process_document(document)

        # Update status to COMPLETED
        document.status = DocumentStatus.COMPLETED
        document.processing_errors = None  # Clear any previous errors
        await document.save(update_fields=["status", "processing_errors"])
        logger.info(f"Successfully completed processing document {document_id} with {len(chunk_ids)} chunks")

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}", exc_info=True)

        # Check if this is the last retry
        message = CurrentMessage.get_current_message()
        current_retry = message.options.get("retries", 0) if message else 0
        max_retries = 5

        if current_retry >= max_retries:
            # Retries exhausted - mark as FAILED only if still in PROCESSING status
            try:
                document = await Document.get_or_none(id=doc_uuid, status=DocumentStatus.PROCESSING)
                if document:
                    document.status = DocumentStatus.FAILED
                    document.processing_errors = f"{type(e).__name__}: {str(e)}"
                    await document.save(update_fields=["status", "processing_errors"])
                    logger.error(f"Document {document_id} marked as FAILED after {max_retries} retries")
                else:
                    logger.info(f"Document {document_id} not in PROCESSING status, skipping FAILED status update")
            except Exception as save_error:
                logger.error(f"Failed to update document status: {save_error}")

        # Re-raise to let Dramatiq handle the retry
        raise

    finally:
        # Close Tortoise connections
        await Tortoise.close_connections()


@dramatiq.actor(max_retries=5)
async def process_ingestion(document_id: str) -> None:
    """
    Dramatiq actor for processing document ingestion asynchronously.

    This actor wraps the core processing logic and is invoked by the task queue.

    Args:
        document_id: The UUID of the document to process (as string)
    """
    await _process_document_job(document_id)
