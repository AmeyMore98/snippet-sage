import logging
from dataclasses import dataclass
from uuid import UUID

from app.core import hashing, text
from app.core.config import Settings
from app.models import Chunk, Document
from app.rag import chunker, embedder
from app.schemas.ingest import IngestRequest

logger = logging.getLogger(__name__)
settings = Settings()


@dataclass
class DocumentCreationResult:
    """Result of creating or retrieving a document."""

    document_id: UUID
    was_created: bool  # True if newly created, False if already existed


@dataclass
class IngestResult:
    """Result of an async ingestion operation."""

    document_id: UUID
    was_created: bool  # True if newly created and job enqueued, False if duplicate


class IngestionService:
    """
    Service for orchestrating the ingestion of documents into the PKB.
    Handles normalization, hashing, chunking, embedding, and persistence.
    """

    async def create_document(self, payload: IngestRequest) -> DocumentCreationResult:
        """
        Validates, normalizes, and creates a document record.

        This method handles validation, normalization, deduplication, and document creation
        without enqueuing any background jobs. Useful for testing and internal operations.

        Args:
            payload: The IngestRequest containing the text and optional metadata.

        Returns:
            DocumentCreationResult indicating the document ID and whether it was newly created

        Raises:
            ValueError: If the input text is invalid or cannot be normalized
        """
        raw_text = payload.text

        # 1. Validate input text length
        if len(raw_text) > settings.MAX_INPUT_CHARS:
            raise ValueError(f"Input text is too long (max: {settings.MAX_INPUT_CHARS} characters).")

        # 2. Normalize text
        normalized_text = text.normalize(raw_text)
        if not normalized_text:
            raise ValueError("Normalized text is empty after processing.")

        # 3. Hash content for deduplication
        content_sha256 = hashing.sha256(normalized_text)

        # 4. Check for deduplication
        existing_document = await Document.filter(content_sha256=content_sha256).first()
        if existing_document:
            logger.info(f"Document with SHA256 {content_sha256} already exists. Returning existing document.")
            return DocumentCreationResult(document_id=existing_document.id, was_created=False)

        # 5. Create Document record
        document = await Document.create(
            source=payload.source,
            content_sha256=content_sha256,
            raw_text=normalized_text,
        )
        logger.info(f"Created document with ID: {document.id}")

        return DocumentCreationResult(document_id=document.id, was_created=True)

    async def ingest_async(self, payload: IngestRequest) -> IngestResult:
        """
        Validates, normalizes, creates a document record, and enqueues it for async processing.

        This method handles the complete async ingestion workflow: validation,
        normalization, deduplication, document creation, and job enqueueing.
        If the document is new, it enqueues a background job for processing.

        Args:
            payload: The IngestRequest containing the text and optional metadata.

        Returns:
            IngestResult containing the document ID and whether it was newly created

        Raises:
            ValueError: If the input text is invalid or cannot be normalized
        """
        # Import here to avoid circular dependency
        from app.jobs.ingestion_job import process_ingestion

        # Create the document (handles validation, normalization, and deduplication)
        result = await self.create_document(payload)

        # Enqueue background job only for new documents
        if result.was_created:
            process_ingestion.send(str(result.document_id))
            logger.info(f"Enqueued processing job for document {result.document_id}")

        return IngestResult(document_id=result.document_id, was_created=result.was_created)

    async def process_document(self, document: Document) -> list[UUID]:
        """
        Process an existing document by chunking and embedding it.

        This method is used by the async job worker to process documents that
        have already been created and validated. It performs chunking, embedding
        generation, and persistence.

        This method is idempotent - if chunks already exist (e.g., from a previous
        failed attempt), it will reuse them and only regenerate embeddings.

        Args:
            document: The Document instance to process

        Returns:
            A list of chunk IDs that were created or reused

        Raises:
            ValueError: If chunking fails or no chunks are generated
            RuntimeError: If there's a mismatch between chunks and embeddings
        """
        # Check if chunks already exist for this document (idempotency)
        existing_chunks = await Chunk.filter(document=document).all()

        if existing_chunks:
            # Chunks exist - this is a retry after embedding failure
            logger.info(f"Found {len(existing_chunks)} existing chunks for document {document.id}, reusing them")
            chunk_objects = existing_chunks
            chunk_ids = [chunk.id for chunk in chunk_objects]
        else:
            # No existing chunks - perform chunking
            normalized_text = document.raw_text

            # 1. Chunk the normalized text
            text_chunks = chunker.make_chunks(
                normalized_text,
                max_sent=settings.CHUNK_MAX_SENTENCES,
                overlap=settings.CHUNK_OVERLAP_SENTENCES,
            )
            if not text_chunks:
                raise ValueError("No chunks were generated from the document text.")

            # 2. Prepare chunk objects for bulk insertion
            chunk_objects = []
            for chunk_text in text_chunks:
                chunk_sha256 = hashing.sha256(chunk_text)
                chunk_obj = Chunk(
                    document=document,
                    text=chunk_text,
                    chunk_sha256=chunk_sha256,
                )
                chunk_objects.append(chunk_obj)

            # 3. Bulk create chunks
            await Chunk.bulk_create(chunk_objects)
            chunk_ids = [chunk.id for chunk in chunk_objects]
            logger.info(f"Created {len(chunk_objects)} chunks for document {document.id}")

        # 4. Generate and persist embeddings
        chunk_texts_to_embed = [c.text for c in chunk_objects]
        if chunk_texts_to_embed:
            embeddings_np = embedder.embed_texts(chunk_texts_to_embed)
            if embeddings_np.shape[0] != len(chunk_objects):
                raise RuntimeError("Mismatch between number of chunks and generated embeddings.")

            await embedder.persist_embeddings(chunk_ids, embeddings_np)
            logger.info(f"Persisted embeddings for {len(chunk_ids)} chunks.")

        return chunk_ids
