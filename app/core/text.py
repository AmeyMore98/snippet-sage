"""Text processing utilities for normalization and preview generation."""

import re
import unicodedata


def normalize(text: str) -> str:
    """
    Normalize text by collapsing whitespace and normalizing unicode quotes.

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    # Normalize unicode characters to NFKC form
    text = unicodedata.normalize("NFKC", text)

    # Further standardize common typographic characters to their basic ASCII
    # equivalents. This ensures consistency for downstream processing like
    # full-text search. While `unicodedata.normalize('NFKC', ...)` handles many
    # of these, this explicit mapping provides clarity and guarantees the
    # desired transformation for specific characters like the em dash.
    #
    # Examples:
    #   - ‘single quotes’ -> 'single quotes'
    #   - “double quotes” -> "double quotes"
    #   - en dash (–) and em dash (—) both become a simple hyphen (-)
    quote_map = {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201c": '"',  # Left double quotation mark
        "\u201d": '"',  # Right double quotation mark
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
    }

    for unicode_char, ascii_char in quote_map.items():
        text = text.replace(unicode_char, ascii_char)

    # Collapse multiple whitespace characters (spaces, tabs, newlines) into single spaces
    text = re.sub(r"\s+", " ", text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text
