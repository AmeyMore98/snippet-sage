from tortoise import fields

from .base import UUIDModel


class Chunk(UUIDModel):
    document = fields.ForeignKeyField("models.Document", related_name="chunks", on_delete=fields.CASCADE)
    text = fields.TextField()
    chunk_sha256 = fields.CharField(max_length=64, unique=True)

    class Meta:
        table = "chunks"
