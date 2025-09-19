"""Content hashing utilities for deduplication."""

import hashlib


def sha256(text: str) -> str:
    """
    Generate SHA-256 hash of text content.

    Args:
        text: Text to hash

    Returns:
        Hexadecimal string representation of SHA-256 hash
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    # Convert to bytes using UTF-8 encoding
    text_bytes = text.encode("utf-8")

    # Generate SHA-256 hash
    hash_obj = hashlib.sha256(text_bytes)

    # Return hexadecimal representation
    return hash_obj.hexdigest()
