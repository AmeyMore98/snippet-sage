import logging

from app.core.config import Settings
from app.rag.graph import build_graph
from app.schemas.answer import AnswerResponse

logger = logging.getLogger(__name__)
settings = Settings()


class QAService:
    """
    Service for orchestrating the question-answering process using LangGraph.
    """

    def __init__(self):
        # Compile the graph once per service instance.
        # Later, this should be handled by a FastAPI startup event
        # or dependency injection to ensure it's only built once.
        self.compiled_graph = build_graph()

    async def answer(self, ques: str, k: int = None, eval_mode: bool = False) -> AnswerResponse:
        """
        Generates an answer to a given question using the RAG pipeline orchestrated by LangGraph.

        Args:
            q: The user's question.
            k: The number of top retrieval results to consider. Defaults to settings.RETRIEVAL_K.
            eval_mode: A boolean indicating whether to run in evaluation mode (e.g., for metrics).

        Returns:
            An AnswerResponse object containing the answer, citations, and confidence.
        """
        if not ques:
            raise ValueError("Question cannot be empty.")

        # Cap k to a reasonable maximum to prevent excessive retrieval
        max_k = 50  # As per architecture.md
        effective_k = min(k if k is not None else settings.RETRIEVAL_K, max_k)

        logger.info(f"Answering question: '{ques}' with k={effective_k}, eval_mode={eval_mode}")

        # Prepare initial state for the graph
        initial_state = {"question": ques, "k": effective_k}

        # Execute the LangGraph
        # The graph's output is expected to be in the 'response' key of the final state
        final_state = await self.compiled_graph.ainvoke(initial_state)

        response_data = final_state.get("response", {})

        # If no answer is generated, return a default empty response
        if not response_data:
            logger.warning(f"LangGraph returned no response for question: '{ques}'")
            return AnswerResponse(answer="", citations=[], confidence=0.0)

        # Construct AnswerResponse from the graph's output
        answer_response = AnswerResponse(
            answer=response_data.get("answer", ""),
            citations=response_data.get("citations", []),
            confidence=response_data.get("confidence", 0.0),
            debug=final_state if settings.APP_ENV == "dev" else None,  # Assuming APP_ENV in settings
        )

        logger.info(f"Answer generated with confidence: {answer_response.confidence}")
        return answer_response
