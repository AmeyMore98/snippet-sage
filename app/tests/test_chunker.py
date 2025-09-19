"""
Tests for the chunker module.
"""

from app.rag.chunker import make_chunks, split_paragraphs, window_sentences


class TestSplitParagraphs:
    """Test cases for the split_paragraphs function."""

    def test_single_paragraph(self):
        """Test splitting text with a single paragraph."""
        text = "This is a single paragraph with multiple sentences. It has no blank lines."
        result = split_paragraphs(text)
        assert len(result) == 1
        assert result[0] == text

    def test_multiple_paragraphs_double_newline(self):
        """Test splitting text with multiple paragraphs separated by double newlines."""
        text = "First paragraph with some content. Some more content.\n\nSecond paragraph with different content.\n\nThird paragraph here."
        result = split_paragraphs(text)
        assert len(result) == 3
        assert result[0] == "First paragraph with some content. Some more content."
        assert result[1] == "Second paragraph with different content."
        assert result[2] == "Third paragraph here."

    def test_multiple_paragraphs_with_extra_newlines(self):
        """Test splitting with multiple blank lines between paragraphs."""
        text = "First paragraph.\n\n\n\nSecond paragraph.\n\n\n\n\nThird paragraph."
        result = split_paragraphs(text)
        assert len(result) == 3
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."
        assert result[2] == "Third paragraph."

    def test_paragraphs_with_leading_trailing_whitespace(self):
        """Test that leading and trailing whitespace is stripped from paragraphs."""
        text = "  First paragraph with spaces.  \n\n  Second paragraph.  \n\n  Third paragraph.  "
        result = split_paragraphs(text)
        assert len(result) == 3
        assert result[0] == "First paragraph with spaces."
        assert result[1] == "Second paragraph."
        assert result[2] == "Third paragraph."

    def test_empty_string(self):
        """Test splitting an empty string."""
        result = split_paragraphs("")
        assert result == []

    def test_whitespace_only(self):
        """Test splitting whitespace-only text."""
        result = split_paragraphs("   \n\n   \n\n   ")
        assert result == []

    def test_single_newlines_dont_split(self):
        """Test that single newlines don't create paragraph breaks."""
        text = "First line\nSecond line\nThird line\n\nNew paragraph\nAnother line"
        result = split_paragraphs(text)
        assert len(result) == 2
        assert result[0] == "First line\nSecond line\nThird line"
        assert result[1] == "New paragraph\nAnother line"

    def test_paragraphs_with_internal_formatting(self):
        """Test paragraphs containing internal formatting like single newlines."""
        text = "First paragraph\nwith internal line breaks\nand more text.\n\nSecond paragraph\nwith different\nline breaks."
        result = split_paragraphs(text)
        assert len(result) == 2
        assert result[0] == "First paragraph\nwith internal line breaks\nand more text."
        assert result[1] == "Second paragraph\nwith different\nline breaks."

    def test_none_input(self):
        """Test that None input returns empty list."""
        result = split_paragraphs(None)
        assert result == []

    def test_non_string_input(self):
        """Test that non-string input returns empty list."""
        result = split_paragraphs(123)
        assert result == []
        result = split_paragraphs([])
        assert result == []

    def test_paragraph_with_only_blank_lines(self):
        """Test text that contains only blank lines."""
        text = "\n\n\n\n"
        result = split_paragraphs(text)
        assert result == []

    def test_mixed_blank_line_patterns(self):
        """Test various patterns of blank lines and content."""
        text = "\n\nFirst paragraph.\n\n\n\nSecond paragraph.\n\n"
        result = split_paragraphs(text)
        assert len(result) == 2
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."


class TestWindowSentences:
    """Test cases for the window_sentences function."""

    def test_seven_sentences_with_overlap(self):
        """Test splitting 7 sentences with 1-sentence overlap - the main test case."""
        paragraph = (
            "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six. Sentence seven."
        )
        result = window_sentences(paragraph, max_sentences=6, overlap=1)

        # Should create 2 windows: [1-6] and [6-7]
        assert len(result) == 2
        # First window: sentences 1-6
        expected_first = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six."
        assert result[0] == expected_first
        # Second window: sentences 6-7
        expected_second = "Sentence six. Sentence seven."
        assert result[1] == expected_second

    def test_fewer_sentences_than_max(self):
        """Test paragraph with fewer sentences than max_sent."""
        paragraph = "First sentence. Second sentence. Third sentence."
        result = window_sentences(paragraph, max_sentences=6, overlap=1)
        assert len(result) == 1
        assert result[0] == paragraph

    def test_exact_max_sentences(self):
        """Test paragraph with exactly max_sent sentences."""
        paragraph = "One. Two. Three. Four. Five. Six."
        result = window_sentences(paragraph, max_sentences=6, overlap=1)
        assert len(result) == 1
        assert result[0] == paragraph

    def test_no_overlap(self):
        """Test sentence windowing with no overlap."""
        paragraph = "A. B. C. D. E. F. G. H."
        result = window_sentences(paragraph, max_sentences=3, overlap=0)

        # Should create 3 windows: [A,B,C], [D,E,F], [G,H]
        assert len(result) == 3
        assert result[0] == "A. B. C."
        assert result[1] == "D. E. F."
        assert result[2] == "G. H."

    def test_multiple_overlap(self):
        """Test sentence windowing with 2-sentence overlap."""
        paragraph = "A. B. C. D. E. F. G."
        result = window_sentences(paragraph, max_sentences=4, overlap=2)

        # Should create 3 windows with 2-sentence overlap: [A,B,C,D], [C,D,E,F], [E,F,G]
        assert len(result) == 3
        assert result[0] == "A. B. C. D."
        assert result[1] == "C. D. E. F."
        assert result[2] == "E. F. G."

    def test_empty_paragraph(self):
        """Test empty paragraph."""
        result = window_sentences("", max_sentences=6, overlap=1)
        assert result == []

    def test_whitespace_only(self):
        """Test whitespace-only paragraph."""
        result = window_sentences("   ", max_sentences=6, overlap=1)
        assert result == []

    def test_none_input(self):
        """Test None input."""
        result = window_sentences(None, max_sentences=6, overlap=1)
        assert result == []

    def test_non_string_input(self):
        """Test non-string input."""
        result = window_sentences(123, max_sentences=6, overlap=1)
        assert result == []

    def test_zero_max_sent(self):
        """Test invalid max_sent parameter."""
        paragraph = "First sentence. Second sentence."
        result = window_sentences(paragraph, max_sentences=0, overlap=1)
        assert result == []

    def test_negative_max_sent(self):
        """Test negative max_sent parameter."""
        paragraph = "First sentence. Second sentence."
        result = window_sentences(paragraph, max_sentences=-1, overlap=1)
        assert result == []

    def test_different_punctuation(self):
        """Test sentences ending with different punctuation."""
        paragraph = "Question? Exclamation! Regular sentence. Another?"
        result = window_sentences(paragraph, max_sentences=3, overlap=1)

        # Should split correctly on all punctuation types
        assert len(result) == 2
        assert result[0] == "Question? Exclamation! Regular sentence."
        assert result[1] == "Regular sentence. Another?"

    def test_single_sentence(self):
        """Test paragraph with single sentence."""
        paragraph = "Only one sentence here."
        result = window_sentences(paragraph, max_sentences=6, overlap=1)
        assert len(result) == 1
        assert result[0] == paragraph


class TestMakeChunks:
    """Test cases for the make_chunks function."""

    def test_single_paragraph_with_windowing(self):
        """Test chunking single paragraph that requires sentence windowing."""
        text = "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth."
        result = make_chunks(text, max_sent=6, overlap=1)

        # Should create 2 chunks from windowing
        assert len(result) == 2
        assert result[0] == "First. Second. Third. Fourth. Fifth. Sixth."
        assert result[1] == "Sixth. Seventh. Eighth."

    def test_multiple_paragraphs_simple(self):
        """Test multiple paragraphs that don't require windowing."""
        text = "First paragraph. Short one.\n\nSecond paragraph. Also short."
        result = make_chunks(text, max_sent=6, overlap=1)

        # Should return 2 chunks (one per paragraph)
        assert len(result) == 2
        assert result[0] == "First paragraph. Short one."
        assert result[1] == "Second paragraph. Also short."

    def test_multiple_paragraphs_with_windowing(self):
        """Test multiple paragraphs where some require windowing."""
        text = "A. B. C.\n\nD. E. F. G. H. I. J. K."
        result = make_chunks(text, max_sent=4, overlap=1)

        # First paragraph: 1 chunk (3 sentences < 4 max)
        # Second paragraph: 3 chunks with windowing (8 sentences)
        assert len(result) == 4
        assert result[0] == "A. B. C."
        assert result[1] == "D. E. F. G."
        assert result[2] == "G. H. I. J."
        assert result[3] == "J. K."

    def test_empty_text(self):
        """Test empty input text."""
        result = make_chunks("")
        assert result == []

    def test_none_input(self):
        """Test None input."""
        result = make_chunks(None)
        assert result == []

    def test_non_string_input(self):
        """Test non-string input."""
        result = make_chunks(123)
        assert result == []

    def test_whitespace_only_text(self):
        """Test whitespace-only text."""
        result = make_chunks("   \n\n   ")
        assert result == []

    def test_deterministic_output(self):
        """Test that chunking is deterministic for the same input."""
        text = "Deterministic test. Same input. Should produce. Same output. Every time. Always consistent."

        result1 = make_chunks(text, max_sent=4, overlap=1)
        result2 = make_chunks(text, max_sent=4, overlap=1)
        result3 = make_chunks(text, max_sent=4, overlap=1)

        # All results should be identical
        assert result1 == result2 == result3

        # Should have predictable structure
        assert len(result1) == 2  # 6 sentences with max_sent=4, overlap=1
        assert result1[0] == "Deterministic test. Same input. Should produce. Same output."
        assert result1[1] == "Same output. Every time. Always consistent."

    def test_custom_parameters(self):
        """Test make_chunks with different max_sent and overlap parameters."""
        text = "A. B. C. D. E. F. G. H. I."

        # Test with no overlap
        result_no_overlap = make_chunks(text, max_sent=3, overlap=0)
        assert len(result_no_overlap) == 3
        assert result_no_overlap[0] == "A. B. C."
        assert result_no_overlap[1] == "D. E. F."
        assert result_no_overlap[2] == "G. H. I."

        # Test with high overlap
        result_high_overlap = make_chunks(text, max_sent=3, overlap=2)
        assert len(result_high_overlap) == 7  # More windows due to high overlap
