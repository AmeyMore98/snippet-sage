"""Data schemas for the RAG pipeline."""

from typing import NamedTuple


class Hit(NamedTuple):
    """Represents a search hit with chunk ID and retrieval score."""

    chunk_id: str
    retrieval_score: float = 0.0
    text: str = ""
    text_preview: str = ""  # For debugging purposes
    re_rank_score: float = 0.0
    created_at: str = ""
