from fastapi import APIRouter, HTTPException

from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.ingestion_service import IngestionService

router = APIRouter()
ingestion_service = IngestionService()


@router.post("/ingest", status_code=201, response_model=IngestResponse)
async def ingest(payload: IngestRequest) -> IngestResponse:
    """
    Ingests a document into the personal knowledge base asynchronously.

    This endpoint validates the input, creates a document record, and enqueues
    a background job to process it (if the document is new). The actual chunking
    and embedding generation happens asynchronously.

    Returns the document ID immediately with a 201 Created status.
    """
    try:
        result = await ingestion_service.ingest_async(payload)

        message = "Document accepted for processing" if result.was_created else "Document already exists"

        return IngestResponse(
            document_id=result.document_id,
            message=message,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        # For unexpected errors, return a generic 500 response
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}") from e
