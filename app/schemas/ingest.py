from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text content to ingest.")
    source: str | None = Field(None, description="Optional source of the document (e.g., 'web', 'cli').")
    created_at: datetime | None = Field(None, description="Optional creation timestamp for the document.")


class IngestResponse(BaseModel):
    document_id: UUID = Field(..., description="The ID of the ingested document.")
    chunk_ids: list[UUID] = Field(..., description="A list of IDs for the created chunks.")
    chunk_count: int = Field(..., description="The number of chunks created.")
