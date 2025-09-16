import uuid

from tortoise import fields, models


class TimestampedModel(models.Model):
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        abstract = True
        table_description = "Timestamped base model"


class UUIDModel(models.Model):
    id = fields.UUIDField(primary_key=True, default=uuid.uuid4)

    class Meta:
        abstract = True
        table_description = "UUID base model"
