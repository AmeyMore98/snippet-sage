from tortoise import fields

from .base import UUIDModel


class Chunk(UUIDModel):
    document = fields.ForeignKeyField("models.Document", related_name="chunks", on_delete=fields.CASCADE)
    """
    The `fts` column for full-text search is a tsvector generated from the `text` column.
    It is computed automatically at the database level, not in the application,
    to ensure maximum consistency.
    """
    text = fields.TextField()
    chunk_sha256 = fields.CharField(max_length=64, unique=True)

    class Meta:
        table = "chunks"
