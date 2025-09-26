from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    q: str = Field(..., description="The question to answer.")
    k: int | None = Field(None, description="The number of documents to retrieve.")


class Citation(BaseModel):
    chunk_id: UUID = Field(..., description="The ID of the cited chunk.")
    preview: str = Field(..., description="A preview of the cited chunk's text.")
    score: float = Field(..., description="The relevance score of the citation.")


class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The generated answer to the question.")
    citations: list[Citation] = Field(..., description="A list of citations supporting the answer.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="A confidence score for the answer (0.0 to 1.0).")
    debug: Any | None = Field(None, description="Optional debug information, present if APP_ENV=dev.")
