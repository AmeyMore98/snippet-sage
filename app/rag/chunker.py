"""
Chunking functionality for the RAG pipeline.

This module provides text chunking capabilities including paragraph splitting
and sentence windowing with overlap.
"""

import re


def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs using blank line separation.

    Args:
        text: Input text to split into paragraphs

    Returns:
        List of paragraph strings, with empty paragraphs filtered out

    Test:
        Blank-line separation yields expected paragraphs.
    """
    if not text or not isinstance(text, str):
        return []

    # Split on double newlines (blank lines) and filter out empty paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    return paragraphs


def window_sentences(paragraph: str, max_sentences: int = 6, overlap: int = 1) -> list[str]:
    """
    Split a paragraph into sentence windows with overlap.

    Args:
        paragraph: Input paragraph text to split into sentences
        max_sentences: Maximum number of sentences per window (default: 6)
        overlap: Number of sentences to overlap between windows (default: 1)

    Returns:
        List of sentence windows (strings)

    Test:
        For 7 sentences, returns windows with one-sentence overlap.
    """
    if not paragraph or not isinstance(paragraph, str) or max_sentences <= 0:
        return []

    # Split paragraph into sentences using simple regex
    # This splits on sentence-ending punctuation followed by whitespace
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]

    if not sentences:
        return []

    if len(sentences) <= max_sentences:
        # If paragraph has fewer sentences than max_sent, return as single window
        return [paragraph]

    windows = []
    overlap = max(0, min(overlap, max_sentences - 1))  # Ensure overlap is valid

    # Create sliding windows with overlap
    for i in range(0, len(sentences), max_sentences - overlap):
        window_sentences = sentences[i : i + max_sentences]
        if window_sentences:  # Only add non-empty windows
            window_text = " ".join(window_sentences)
            windows.append(window_text)

        # Stop if we've processed all sentences
        if i + max_sentences >= len(sentences):
            break

    return windows


def make_chunks(text: str, max_sent: int = 6, overlap: int = 1) -> list[str]:
    """
    Main chunk pipeline that combines paragraph splitting and sentence windowing.

    Args:
        text: Input text to chunk
        max_sent: Maximum number of sentences per chunk window (default: 6)
        overlap: Number of sentences to overlap between windows (default: 1)

    Returns:
        List of text chunks (strings)

    Test:
        Deterministic chunks for fixed input; previews built for each.
    """
    if not text or not isinstance(text, str):
        return []

    # First, split into paragraphs
    paragraphs = split_paragraphs(text)

    if not paragraphs:
        return []

    chunks = []

    # For each paragraph, apply sentence windowing
    for paragraph in paragraphs:
        paragraph_windows = window_sentences(paragraph, max_sent, overlap)
        chunks.extend(paragraph_windows)

    return chunks
