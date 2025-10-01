from tortoise import fields
from tortoise.contrib.postgres.fields import TSVectorField

from .base import UUIDModel


class Chunk(UUIDModel):
    document = fields.ForeignKeyField("models.Document", related_name="chunks", on_delete=fields.CASCADE)
    text = fields.TextField()
    chunk_sha256 = fields.CharField(max_length=64, unique=True)

    # PostgreSQL tsvector column for full-text search
    # Generated automatically: to_tsvector('english', text)
    # Note: This field is read-only and computed at the database level for maximum consistency
    fts = TSVectorField(
        null=True, generated=True, description="Full-text search vector (computed)"
    )  # tsvector stored as text

    @property
    def text_preview(self) -> str:
        return self.text[:157] + "..." if len(self.text) > 160 else self.text

    class Meta:
        table = "chunks"
