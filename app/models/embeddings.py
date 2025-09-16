from tortoise import fields
from tortoise_vector import VectorField

from .base import UUIDModel


class Embedding(UUIDModel):
    chunk = fields.OneToOneField("models.Chunk", on_delete=fields.CASCADE, related_name="embedding")
    vector = VectorField(dimensions=1536)
    dim = fields.IntField()

    class Meta:
        table = "embeddings"
