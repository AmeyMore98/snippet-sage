from unittest.mock import MagicMock, patch

from app.rag.dspy_program import run_answerer
from app.rag.schemas import Hit

# Sample passages to be used in tests
SAMPLE_PASSAGES = [
    Hit(chunk_id="id1", text="text for id1", re_rank_score=0.9),
    Hit(chunk_id="id2", text="text for id2", re_rank_score=0.8),
    Hit(chunk_id="id3", text="text for id3", re_rank_score=0.7),
]


def mock_dspy_output(answer, citations, confidence):
    """Helper to create a mock object simulating dspy.Prediction."""
    mock = MagicMock()
    mock.answer = answer
    mock.citations = citations
    mock.confidence = confidence
    return mock


@patch("app.rag.dspy_program.Answerer")
def test_run_answerer_happy_path(mock_answerer):
    """Tests the ideal case with valid citations and confidence."""
    # Arrange
    mock_output = mock_dspy_output(answer="This is a correct answer.", citations=["id1", "id2"], confidence="0.85")
    mock_answerer.return_value = mock_output

    # Act
    result = run_answerer(question="test question", passages=SAMPLE_PASSAGES)

    # Assert
    assert result["answer"] == "This is a correct answer."
    assert result["confidence"] == 0.85
    assert result["citations"] == ["id1", "id2"]
    mock_answerer.assert_called_once()


@patch("app.rag.dspy_program.Answerer")
def test_run_answerer_citations_as_string_repr(mock_answerer):
    """Tests parsing of citations when returned as a string representation of a list."""
    # Arrange
    mock_output = mock_dspy_output(answer="Test answer", citations="['id1', 'id3']", confidence="0.9")
    mock_answerer.return_value = mock_output

    # Act
    result = run_answerer(question="test question", passages=SAMPLE_PASSAGES)

    # Assert
    assert result["citations"] == ["id1", "id3"]


@patch("app.rag.dspy_program.Answerer")
def test_run_answerer_citations_as_comma_separated_string(mock_answerer):
    """Tests parsing of citations when returned as a simple comma-separated string."""
    # Arrange
    mock_output = mock_dspy_output(answer="Test answer", citations="id2, id1", confidence="0.9")
    mock_answerer.return_value = mock_output

    # Act
    result = run_answerer(question="test question", passages=SAMPLE_PASSAGES)

    # Assert
    assert sorted(result["citations"]) == ["id1", "id2"]


@patch("app.rag.dspy_program.Answerer")
def test_run_answerer_filters_invalid_citations(mock_answerer):
    """Tests that citations not present in the source passages are filtered out."""
    # Arrange
    mock_output = mock_dspy_output(answer="Test answer", citations=["id1", "invalid_id", "id3"], confidence="0.9")
    mock_answerer.return_value = mock_output

    # Act
    result = run_answerer(question="test question", passages=SAMPLE_PASSAGES)

    # Assert
    assert result["citations"] == ["id1", "id3"]


@patch("app.rag.dspy_program.Answerer")
def test_run_answerer_clamps_high_confidence(mock_answerer):
    """Tests that confidence scores > 1.0 are clamped to 1.0."""
    # Arrange
    mock_output = mock_dspy_output(answer="Test answer", citations=[], confidence="1.5")
    mock_answerer.return_value = mock_output

    # Act
    result = run_answerer(question="test question", passages=SAMPLE_PASSAGES)

    # Assert
    assert result["confidence"] == 1.0


@patch("app.rag.dspy_program.Answerer")
def test_run_answerer_clamps_low_confidence(mock_answerer):
    """Tests that confidence scores < 0.0 are clamped to 0.0."""
    # Arrange
    mock_output = mock_dspy_output(answer="Test answer", citations=[], confidence="-0.5")
    mock_answerer.return_value = mock_output

    # Act
    result = run_answerer(question="test question", passages=SAMPLE_PASSAGES)

    # Assert
    assert result["confidence"] == 0.0


@patch("app.rag.dspy_program.Answerer")
def test_run_answerer_handles_invalid_confidence(mock_answerer):
    """Tests that non-numeric confidence values default to 0.0."""
    # Arrange
    mock_output = mock_dspy_output(answer="Test answer", citations=[], confidence="not-a-float")
    mock_answerer.return_value = mock_output

    # Act
    result = run_answerer(question="test question", passages=SAMPLE_PASSAGES)

    # Assert
    assert result["confidence"] == 0.0


@patch("app.rag.dspy_program.Answerer")
def test_run_answerer_handles_missing_attributes(mock_answerer):
    """Tests graceful handling of missing 'citations' or 'confidence' in the output."""
    # Arrange
    mock_output = mock_dspy_output(answer="Test answer", citations=None, confidence=None)
    mock_answerer.return_value = mock_output

    # Act
    result = run_answerer(question="test question", passages=SAMPLE_PASSAGES)

    # Assert
    assert result["answer"] == "Test answer"
    assert result["citations"] == []
    assert result["confidence"] == 0.0
