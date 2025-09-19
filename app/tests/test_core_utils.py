"""Tests for core utilities: text normalization and content hashing."""

import pytest

from app.core.hashing import sha256
from app.core.text import normalize


class TestTextNormalize:
    """Test text normalization function."""

    def test_collapses_multiple_spaces(self):
        """Test that multiple spaces are collapsed to single spaces."""
        text = "Hello   world    with     multiple spaces"
        result = normalize(text)
        assert result == "Hello world with multiple spaces"

    def test_collapses_multiple_newlines(self):
        """Test that multiple newlines are collapsed to single spaces."""
        text = "Hello\n\nworld\n\n\nwith\nnewlines"
        result = normalize(text)
        assert result == "Hello world with newlines"

    def test_collapses_mixed_whitespace(self):
        """Test that mixed whitespace (spaces, tabs, newlines) is collapsed."""
        text = "Hello\t\t  world\n\n  \twith\t \n mixed   whitespace"
        result = normalize(text)
        assert result == "Hello world with mixed whitespace"

    def test_normalizes_unicode_quotes(self):
        """Test that unicode quotes are normalized to ASCII."""
        text = "\u201cHello\u201d and \u2018world\u2019"
        result = normalize(text)
        assert result == "\"Hello\" and 'world'"

    def test_normalizes_dashes(self):
        """Test that unicode dashes are normalized to ASCII."""
        text = "en\u2013dash and em\u2014dash"
        result = normalize(text)
        assert result == "en-dash and em-dash"

    def test_strips_leading_trailing_whitespace(self):
        """Test that leading and trailing whitespace is removed."""
        text = "   Hello world   "
        result = normalize(text)
        assert result == "Hello world"

    def test_empty_string(self):
        """Test that empty string returns empty string."""
        assert normalize("") == ""

    def test_none_returns_empty(self):
        """Test that None input returns empty string."""
        assert normalize(None) == ""

    def test_whitespace_only_returns_empty(self):
        """Test that whitespace-only string returns empty string."""
        text = "   \n\n\t  \n  "
        result = normalize(text)
        assert result == ""


class TestSha256Hashing:
    """Test SHA-256 content hashing function."""

    def test_known_input_maps_to_expected_hash(self):
        """Test that known inputs produce expected hash values."""
        # SHA-256 of "test" is known
        result = sha256("test")
        expected = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        assert result == expected

    def test_stable_across_calls(self):
        """Test that same input produces same hash across multiple calls."""
        text = "stable test content"
        hash1 = sha256(text)
        hash2 = sha256(text)
        assert hash1 == hash2

    def test_different_inputs_produce_different_hashes(self):
        """Test that different inputs produce different hash values."""
        hash1 = sha256("content one")
        hash2 = sha256("content two")
        assert hash1 != hash2

    def test_empty_string_hash(self):
        """Test that empty string produces valid hash."""
        result = sha256("")
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert result == expected

    def test_unicode_content(self):
        """Test that unicode content is properly hashed."""
        text = "Hello 世界! 🌍"
        result = sha256(text)
        # Should be consistent and valid hex string
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_returns_hex_string(self):
        """Test that result is a valid hexadecimal string."""
        result = sha256("any content")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in result)

    def test_type_error_for_non_string(self):
        """Test that non-string input raises TypeError."""
        with pytest.raises(TypeError, match="Input must be a string"):
            sha256(123)

        with pytest.raises(TypeError, match="Input must be a string"):
            sha256(None)

        with pytest.raises(TypeError, match="Input must be a string"):
            sha256(["list"])
