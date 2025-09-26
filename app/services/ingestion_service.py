import logging
from uuid import UUID

from tortoise.transactions import in_transaction

from app.core import hashing, text
from app.core.config import Settings
from app.models import Chunk, Document
from app.rag import chunker, embedder
from app.schemas.ingest import IngestRequest

logger = logging.getLogger(__name__)
settings = Settings()


class IngestionService:
    """
    Service for orchestrating the ingestion of documents into the PKB.
    Handles normalization, hashing, chunking, embedding, and persistence.
    """

    async def ingest(self, payload: IngestRequest) -> tuple[UUID, list[UUID]]:
        """
        Ingests a document by normalizing, hashing, chunking, embedding, and storing it.

        Args:
            payload: The IngestRequest containing the text and optional metadata.

        Returns:
            A tuple of (document_id, list_of_chunk_ids)

        Raises:
            ValueError: If the input text is too short or too long.
            IntegrityError: If a document with the same content_sha256 already exists.
            Exception: For other unexpected errors during ingestion.
        """
        raw_text = payload.text

        # 1. Validate input text length
        if not raw_text or len(raw_text) < settings.MIN_INPUT_CHARS:
            raise ValueError(f"Input text is too short (min: {settings.MIN_INPUT_CHARS} characters).")
        if len(raw_text) > settings.MAX_INPUT_CHARS:
            raise ValueError(f"Input text is too long (max: {settings.MAX_INPUT_CHARS} characters).")

        # 2. Normalize text
        normalized_text = text.normalize(raw_text)
        if not normalized_text:
            raise ValueError("Normalized text is empty after processing.")

        # 3. Hash content for deduplication
        content_sha256 = hashing.sha256(normalized_text)

        # Use a transaction for atomicity
        async with in_transaction() as connection:
            # 4. Check for deduplication
            existing_document = await Document.filter(content_sha256=content_sha256).using_db(connection).first()
            if existing_document:
                logger.info(f"Document with SHA256 {content_sha256} already exists. Skipping ingestion.")
                # Fetch existing chunk IDs for the response
                existing_chunks = await existing_document.chunks.all().using_db(connection)
                return existing_document.id, [chunk.id for chunk in existing_chunks]

            # 5. Create Document record
            document = await Document.create(
                source=payload.source,
                content_sha256=content_sha256,
                raw_text=normalized_text,
                using_db=connection,
            )
            logger.info(f"Created document with ID: {document.id}")

            # 6. Chunk the normalized text
            text_chunks = chunker.make_chunks(
                normalized_text,
                max_sent=settings.CHUNK_MAX_SENTENCES,
                overlap=settings.CHUNK_OVERLAP_SENTENCES,
            )
            if not text_chunks:
                raise ValueError("No chunks were generated from the document text.")

            chunk_ids = []
            chunk_objects = []
            for i, chunk_text in enumerate(text_chunks):
                chunk_sha256 = hashing.sha256(chunk_text)
                chunk_obj = await Chunk.create(
                    document=document,
                    ordinal=i,
                    text=chunk_text,
                    chunk_sha256=chunk_sha256,
                    using_db=connection,
                )
                chunk_objects.append(chunk_obj)
                chunk_ids.append(chunk_obj.id)
            logger.info(f"Created {len(chunk_objects)} chunks for document {document.id}")

            # 7. Embed the chunks
            chunk_texts_to_embed = [c.text for c in chunk_objects]
            if chunk_texts_to_embed:
                embeddings_np = embedder.embed_texts(chunk_texts_to_embed)
                if embeddings_np.shape[0] != len(chunk_objects):
                    raise RuntimeError("Mismatch between number of chunks and generated embeddings.")

                # Persist embeddings
                await embedder.persist_embeddings(chunk_ids, embeddings_np)
                logger.info(f"Persisted embeddings for {len(chunk_ids)} chunks.")

            return document.id, chunk_ids
