import pytest

from app.models import Chunk, Document


@pytest.mark.asyncio
async def test_chunk_text_preview():
    """Tests that the text_preview property returns the first 160 characters."""
    doc = await Document.create(content_sha256="abc", raw_text="some text")
    long_text = "a" * 200
    chunk = await Chunk.create(document=doc, text=long_text, chunk_sha256="def")

    print(f"Length of text_preview: {len(chunk.text_preview)}")
    print(f"text_preview: {chunk.text_preview}")
    print(f"Expected: {'a' * 160}")

    assert len(chunk.text_preview) == 160
    assert chunk.text_preview == "a" * 157 + "..."


@pytest.mark.asyncio
async def test_chunk_text_preview_short_text():
    """Tests that the text_preview property returns the full text if it's shorter than 160 characters."""
    doc = await Document.create(content_sha256="abcd", raw_text="some other text")
    short_text = "a" * 100
    chunk = await Chunk.create(document=doc, text=short_text, chunk_sha256="defg")

    assert len(chunk.text_preview) == 100
    assert chunk.text_preview == short_text


@pytest.mark.asyncio
async def test_fts_column_generation(db_session):
    """Tests that the fts column is automatically generated."""
    doc = await Document.create(content_sha256="fts_test_sha", raw_text="some text")
    text = "The quick brown fox jumps over the lazy dog"
    chunk = await Chunk.create(document=doc, text=text, chunk_sha256="fts_chunk_sha")

    # The database automatically populates the 'fts' column.
    # We need to retrieve the object again to see the generated value.
    retrieved_chunk = await Chunk.get(id=chunk.id)

    assert retrieved_chunk.fts is not None
    # The exact representation of the tsvector can vary,
    # but it should contain the lexemes.
    # A simple check is to see if the lexemes are in the string representation.
    expected_lexemes = ["quick", "brown", "fox", "jump", "lazi", "dog"]
    for lexeme in expected_lexemes:
        assert lexeme in retrieved_chunk.fts
