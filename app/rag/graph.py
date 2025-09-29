"""LangGraph orchestration for the Q&A pipeline.

This module implements the stateful graph that orchestrates the question-answering
pipeline: parse → retrieve → re_rank → answer → cite.
"""

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..core.text import normalize
from ..models.chunks import Chunk
from ..schemas.hit import Hit
from .dspy_program import run_answerer
from .embedder import embed_texts
from .re_ranker import re_rank_hits
from .retriever import hybrid_search

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    """State dictionary that flows through the LangGraph nodes."""

    # Input
    question: str
    k: int

    # Intermediate state
    q_norm: str
    query_vector: Any  # np.ndarray
    hits: list[Hit]
    top_passages: list[Hit]
    answer_bundle: dict[str, Any]

    # Output
    response: dict[str, Any]


def parse_node(state: GraphState) -> GraphState:
    """
    Parse and normalize the input question.

    Args:
        state: Current graph state containing the question

    Returns:
        Updated state with normalized question
    """
    question = state["question"]

    # Normalize the question using existing text utility
    q_norm = normalize(question)

    # Basic validation and cleanup
    q_norm = q_norm.strip()
    if not q_norm:
        q_norm = question.strip()  # Fallback to original if normalization fails

    logger.debug(f"Parsed question: '{question}' -> '{q_norm}'")

    return {**state, "q_norm": q_norm}


async def retrieve_node(state: GraphState) -> GraphState:
    """
    Retrieve relevant chunks using hybrid search.

    Args:
        state: Current graph state with normalized question and k parameter

    Returns:
        Updated state with retrieved hits and query vector
    """
    q_norm = state["q_norm"]
    k = state.get("k", 12)

    # Generate query embedding for vector search
    query_vector = embed_texts([q_norm])[0]  # Get single embedding

    # Perform hybrid retrieval
    hits = await hybrid_search(q_norm, query_vector, k=k)

    # Enrich hits with actual text content from database for re_ranking
    if hits:
        enriched_hits = await _enrich_hits_with_text(hits)
    else:
        enriched_hits = []

    logger.debug(f"Retrieved {len(enriched_hits)} hits for question: '{q_norm}'")

    return {**state, "query_vector": query_vector, "hits": enriched_hits}


async def re_rank_node(state: GraphState) -> GraphState:
    """
    Rerank the retrieved hits using cross-encoder scoring.

    Args:
        state: Current graph state with hits

    Returns:
        Updated state with re_ranked top passages
    """
    hits = state.get("hits", [])
    q_norm = state["q_norm"]
    k = state.get("k", 12)

    # Limit to top 6 after re_ranking (as per architecture)
    top_n = min(6, k)

    if not hits:
        logger.warning("No hits to re_rank")
        return {**state, "top_passages": []}

    # Perform re_ranking
    top_passages = await re_rank_hits(q_norm, hits, n=top_n)

    logger.debug(f"Re-ranked {len(hits)} hits down to {len(top_passages)} top passages")

    return {**state, "top_passages": top_passages}


def answer_node(state: GraphState) -> GraphState:
    """
    Generate answer using DSPy program.

    Args:
        state: Current graph state with top passages

    Returns:
        Updated state with answer bundle (answer, citations, confidence)
    """
    q_norm = state["q_norm"]
    top_passages = state.get("top_passages", [])

    if not top_passages:
        logger.warning("No passages available for answering")
        answer_bundle = {"answer": "", "citations": [], "confidence": 0.0}
    else:
        # Use DSPy program to generate answer
        answer_bundle = run_answerer(q_norm, top_passages)

    logger.debug(f"Generated answer with {len(answer_bundle.get('citations', []))} citations")

    return {**state, "answer_bundle": answer_bundle}


async def cite_node(state: GraphState) -> GraphState:
    """
    Format final response with citations containing previews and scores.

    Args:
        state: Current graph state with answer bundle and top passages

    Returns:
        Updated state with formatted response
    """
    answer_bundle = state.get("answer_bundle", {})
    top_passages = state.get("top_passages", [])

    # Create citation lookup map
    passage_map = {hit.chunk_id: hit for hit in top_passages}

    # Format citations with previews and scores
    citations = []
    cited_chunk_ids = answer_bundle.get("citations", [])

    for chunk_id in cited_chunk_ids:
        hit = passage_map.get(chunk_id)
        if hit:
            citation = {"chunk_id": chunk_id, "preview": hit.text_preview, "score": hit.re_rank_score}
            citations.append(citation)

    # Build final response
    response = {
        "answer": answer_bundle.get("answer", ""),
        "citations": citations,
        "confidence": answer_bundle.get("confidence", 0.0),
    }

    logger.debug(f"Formatted response with {len(citations)} citations")

    return {**state, "response": response}


async def _enrich_hits_with_text(hits: list[Hit]) -> list[Hit]:
    """
    Enrich hits with more details from the database.

    Args:
        hits: List of hits with chunk_id and retrieval_score

    Returns:
        List of hits enriched with text content and other details
    """
    if not hits:
        return hits

    chunk_ids = [hit.chunk_id for hit in hits]

    try:
        # Fetch chunks and related documents
        # TODO: Check the underlying query, might not be performant
        chunks_data = (
            await Chunk.filter(id__in=chunk_ids).select_related("document").values("id", "text", "document__created_at")
        )

        # Create lookup map of text content
        text_map = {}
        for chunk_data in chunks_data:
            chunk_id = str(chunk_data["id"])
            text = chunk_data["text"]
            created_at = chunk_data["document__created_at"]
            text_map[chunk_id] = {
                "text": text,
                "text_preview": text[:157] + "..." if len(text) > 160 else text,
                "created_at": created_at.isoformat() if created_at else "",
            }

        # Enrich hits
        enriched_hits = []
        for hit in hits:
            data = text_map.get(hit.chunk_id, {})
            enriched_hit = hit._replace(
                text=data.get("text", ""),
                text_preview=data.get("text_preview", ""),
                created_at=data.get("created_at", ""),
            )
            enriched_hits.append(enriched_hit)

        return enriched_hits

    except Exception as e:
        logger.error(f"Failed to enrich hits with text: {e}")
        return hits  # Return original hits without text content


def build_graph():
    """
    Build and compile the LangGraph for question answering.

    Returns:
        Compiled LangGraph instance ready for execution
    """
    # Create state graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("parse", parse_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("re_rank", re_rank_node)
    graph.add_node("answer", answer_node)
    graph.add_node("cite", cite_node)

    # Set entry point
    graph.set_entry_point("parse")

    # Define the flow
    graph.add_edge("parse", "retrieve")
    graph.add_edge("retrieve", "re_rank")
    graph.add_edge("re_rank", "answer")
    graph.add_edge("answer", "cite")
    graph.add_edge("cite", END)

    # Compile the graph
    compiled_graph = graph.compile()

    logger.info("LangGraph compiled successfully")

    return compiled_graph
