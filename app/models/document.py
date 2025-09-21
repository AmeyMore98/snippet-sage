from tortoise import fields

from .base import UUIDModel


class Document(UUIDModel):
    source = fields.CharField(max_length=255, null=True)
    content_sha256 = fields.CharField(max_length=64, unique=True)
    raw_text = fields.TextField()

    class Meta:
        table = "documents"
        table_description = "Documents Table"
