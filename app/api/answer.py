from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.answer import AnswerRequest, AnswerResponse
from app.services.qa_service import QAService

router = APIRouter()
qa_service = QAService()
settings = Settings()


@router.post("/answer", response_model=AnswerResponse)
async def answer(payload: AnswerRequest) -> AnswerResponse:
    """
    Answers a question using the RAG pipeline.

    The process involves parsing the question, retrieving relevant chunks
    from the knowledge base, reranking them, composing an answer, and
    generating citations.
    """
    try:
        result = await qa_service.answer(ques=payload.q, k=payload.k)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}") from e
