import uuid

from tortoise import fields, models


class UUIDModel(models.Model):
    id = fields.UUIDField(primary_key=True, default=uuid.uuid4)

    class Meta:
        abstract = True
        table_description = "UUID base model"
