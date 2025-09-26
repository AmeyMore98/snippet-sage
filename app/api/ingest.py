from fastapi import APIRouter, HTTPException

from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.ingestion_service import IngestionService

router = APIRouter()
ingestion_service = IngestionService()


@router.post("/ingest", status_code=201, response_model=IngestResponse)
async def ingest(payload: IngestRequest) -> IngestResponse:
    """
    Ingests a document into the personal knowledge base.

    The process involves normalizing the text, hashing it for deduplication,
    chunking it into smaller pieces, generating embeddings, and storing
    everything in the database.
    """
    try:
        doc_id, chunk_ids = await ingestion_service.ingest(payload)
        return IngestResponse(
            document_id=doc_id,
            chunk_ids=chunk_ids,
            chunk_count=len(chunk_ids),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        # For unexpected errors, return a generic 500 response
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}") from e
