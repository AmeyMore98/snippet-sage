import ast

import dspy

from .schemas import Hit


class AnswerSignature(dspy.Signature):
    """Answer a question concisely using only the provided context."""

    question = dspy.InputField(desc="natural language query")
    context = dspy.InputField(desc="relevant context with ids")
    answer = dspy.OutputField()
    citations = dspy.OutputField(desc="list of chunk ids used")
    confidence = dspy.OutputField(desc="float 0..1")


# T9.2 - Program
Answerer = dspy.Predict(AnswerSignature)


# T9.3 - Contract wrapper
def run_answerer(question: str, passages: list[Hit]) -> dict:
    """
    Formats passages into a context string, calls the DSPy program,
    and validates the output to enforce the contract.

    Args:
        question: The user's question.
        passages: A list of `Hit` objects

    Returns:
        A dictionary containing the answer, citations, and confidence score.
    """
    context = "\n\n".join(f"[chunk_id={ele.chunk_id} score={ele.re_rank_score:.2f}]\n{ele.text}" for ele in passages)

    # This call requires a configured DSPy language model.
    # e.g., dspy.settings.configure(lm=dspy.OpenAI(model="gpt-3.5-turbo"))
    # For testing without a live model, this call would need to be mocked.
    out = Answerer(question=question, context=context)

    passage_ids = {ele.chunk_id for ele in passages}

    # Post-validate citations: ensure they are a subset of passage IDs
    validated_citations = []
    if hasattr(out, "citations") and out.citations:
        raw_citations = out.citations
        if isinstance(raw_citations, str):
            try:
                # Safely evaluate string representation of a list
                parsed_citations = ast.literal_eval(raw_citations)
                if isinstance(parsed_citations, list):
                    raw_citations = parsed_citations
            except (ValueError, SyntaxError):
                # Fallback for simple comma-separated strings
                raw_citations = [c.strip() for c in raw_citations.split(",")]

        if isinstance(raw_citations, list):
            validated_citations = [c for c in raw_citations if c in passage_ids]

    # Post-validate confidence: clamp to [0, 1]
    validated_confidence = 0.0
    if hasattr(out, "confidence") and out.confidence:
        try:
            validated_confidence = max(0.0, min(1.0, float(out.confidence)))
        except (ValueError, TypeError):
            # Keep default 0.0 if conversion fails
            pass

    return {
        "answer": getattr(out, "answer", ""),
        "citations": validated_citations,
        "confidence": validated_confidence,
    }
