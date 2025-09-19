"""
Tests for the embedder module.
"""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag.embedder import embed_texts, get_embedding_dimension, load_model, persist_embeddings


class TestLoadModel:
    """Test cases for the load_model function."""

    @patch("app.rag.embedder.SentenceTransformer")
    @patch("app.rag.embedder.Settings")
    def test_load_model_success(self, mock_settings, mock_sentence_transformer):
        """Test successful model loading."""
        # Reset global state
        import app.rag.embedder

        app.rag.embedder._model = None
        app.rag.embedder._settings = None

        # Mock the settings
        mock_settings_instance = MagicMock()
        mock_settings_instance.EMBEDDING_MODEL = "test-model"
        mock_settings.return_value = mock_settings_instance

        # Mock the SentenceTransformer
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        # Call function
        result = load_model()

        # Assertions
        mock_sentence_transformer.assert_called_once_with("test-model")
        assert result == mock_model

    @patch("app.rag.embedder.SentenceTransformer")
    def test_load_model_caching(self, mock_sentence_transformer):
        """Test that model is cached and not reloaded."""
        # Reset global state
        import app.rag.embedder

        app.rag.embedder._model = None
        app.rag.embedder._settings = None

        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        # Call function twice
        result1 = load_model()
        result2 = load_model()

        # Should only create model once
        assert mock_sentence_transformer.call_count == 1
        assert result1 == result2 == mock_model

    @patch("app.rag.embedder.load_model")
    def test_get_embedding_dimension(self, mock_load_model):
        """Test getting embedding dimension."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_load_model.return_value = mock_model

        result = get_embedding_dimension()

        assert result == 384
        mock_model.get_sentence_embedding_dimension.assert_called_once()


class TestEmbedTexts:
    """Test cases for the embed_texts function."""

    @patch("app.rag.embedder.load_model")
    def test_embed_texts_single_batch(self, mock_load_model):
        """Test embedding texts without batching."""
        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mock_model.encode.return_value = mock_embeddings
        mock_load_model.return_value = mock_model

        texts = ["First text", "Second text", "Third text"]
        result = embed_texts(texts)

        mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)
        assert np.array_equal(result, mock_embeddings)
        assert result.shape == (3, 3)

    @patch("app.rag.embedder.load_model")
    def test_embed_texts_with_batching(self, mock_load_model):
        """Test embedding texts with batching."""
        mock_model = MagicMock()
        # Mock two batches
        batch1_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        batch2_embeddings = np.array([[0.5, 0.6]])
        mock_model.encode.side_effect = [batch1_embeddings, batch2_embeddings]
        mock_load_model.return_value = mock_model

        texts = ["Text 1", "Text 2", "Text 3"]
        result = embed_texts(texts, batch_size=2)

        # Should be called twice due to batching
        assert mock_model.encode.call_count == 2
        expected_result = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        assert np.array_equal(result, expected_result)
        assert result.shape == (3, 2)

    def test_embed_texts_empty_list(self):
        """Test embedding empty list of texts."""
        result = embed_texts([])

        assert isinstance(result, np.ndarray)
        assert result.size == 0

    @patch("app.rag.embedder.load_model")
    def test_embed_texts_single_text_reshape(self, mock_load_model):
        """Test that single text embedding is properly reshaped."""
        mock_model = MagicMock()
        # Mock single embedding as 1D array
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_embedding
        mock_load_model.return_value = mock_model

        result = embed_texts(["Single text"])

        # Should be reshaped to (1, 3)
        assert result.shape == (1, 3)
        assert np.array_equal(result, mock_embedding.reshape(1, -1))

    @patch("app.rag.embedder.load_model")
    def test_embed_texts_nan_values(self, mock_load_model):
        """Test that NaN values in embeddings raise an error."""
        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, np.nan, 0.3]])
        mock_model.encode.return_value = mock_embeddings
        mock_load_model.return_value = mock_model

        with pytest.raises(ValueError, match="Embeddings contain NaN values"):
            embed_texts(["Text with NaN"])


class TestPersistEmbeddings:
    """Test cases for the persist_embeddings function."""

    @pytest.mark.asyncio
    @patch("app.rag.embedder.get_embedding_dimension")
    @patch("app.rag.embedder.in_transaction")
    @patch("app.models.embeddings.Embedding")
    async def test_persist_embeddings_success(self, mock_embedding_model, mock_transaction, mock_get_dimension):
        """Test successful persistence of embeddings."""
        # Setup mocks
        mock_get_dimension.return_value = 384

        mock_connection = MagicMock()
        mock_transaction.return_value.__aenter__.return_value = mock_connection

        mock_embedding_instance = MagicMock()

        # Make create return an awaitable
        async def mock_create(*args, **kwargs):
            return mock_embedding_instance

        mock_embedding_model.create = mock_create

        # Test data
        chunk_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Call function
        result = await persist_embeddings(chunk_ids, embeddings)

        # Assertions
        assert result == 2

    @pytest.mark.asyncio
    async def test_persist_embeddings_mismatch_length(self):
        """Test error when chunk_ids and embeddings length don't match."""
        chunk_ids = [str(uuid.uuid4())]
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])  # 2 embeddings, 1 chunk_id

        with pytest.raises(ValueError, match="Number of chunk_ids must match number of embeddings"):
            await persist_embeddings(chunk_ids, embeddings)

    @pytest.mark.asyncio
    async def test_persist_embeddings_empty_input(self):
        """Test handling of empty input."""
        result = await persist_embeddings([], np.array([]))
        assert result == 0

    @pytest.mark.asyncio
    @patch("app.rag.embedder.get_embedding_dimension")
    @patch("app.rag.embedder.in_transaction")
    @patch("app.models.embeddings.Embedding")
    async def test_persist_embeddings_partial_failure(self, mock_embedding_model, mock_transaction, mock_get_dimension):
        """Test handling of partial failures during persistence."""
        # Setup mocks
        mock_get_dimension.return_value = 384

        mock_connection = MagicMock()
        mock_transaction.return_value.__aenter__.return_value = mock_connection

        # Mock first create to succeed, second to fail
        call_count = 0

        async def mock_create_with_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock()
            else:
                raise Exception("DB Error")

        mock_embedding_model.create = mock_create_with_failure

        # Test data
        chunk_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Call function
        result = await persist_embeddings(chunk_ids, embeddings)

        # Should return 1 (only one succeeded)
        assert result == 1

    @pytest.mark.asyncio
    @patch("app.rag.embedder.get_embedding_dimension")
    @patch("app.rag.embedder.in_transaction")
    @patch("app.models.embeddings.Embedding")
    async def test_persist_embeddings_uuid_conversion(self, mock_embedding_model, mock_transaction, mock_get_dimension):
        """Test conversion of string chunk_ids to UUID."""
        # Setup mocks
        mock_get_dimension.return_value = 384

        mock_connection = MagicMock()
        mock_transaction.return_value.__aenter__.return_value = mock_connection

        mock_embedding_instance = MagicMock()

        # Make create return an awaitable
        async def mock_create(*args, **kwargs):
            return mock_embedding_instance

        mock_embedding_model.create = mock_create

        # Test with UUID objects and strings
        uuid_obj = uuid.uuid4()
        uuid_str = str(uuid.uuid4())
        chunk_ids = [uuid_obj, uuid_str]
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Call function
        result = await persist_embeddings(chunk_ids, embeddings)

        # Should handle both UUID objects and strings
        assert result == 2
