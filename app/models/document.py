from tortoise import fields

from .base import UUIDModel


class DocumentStatus:
    """Document processing status constants."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Document(UUIDModel):
    source = fields.CharField(max_length=255, null=True)
    content_sha256 = fields.CharField(max_length=64, unique=True)
    raw_text = fields.TextField()
    created_at = fields.DatetimeField(auto_now_add=True)
    # Async processing fields
    status = fields.CharField(
        max_length=20,
        default=DocumentStatus.PENDING,
        description="Processing status: PENDING, PROCESSING, COMPLETED, or FAILED",
    )
    processing_errors = fields.TextField(null=True, description="Error messages from failed processing attempts")

    class Meta:
        table = "documents"
        table_description = "Documents Table"
