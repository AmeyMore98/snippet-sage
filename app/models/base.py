from tortoise import fields, models


class TimestampedModel(models.Model):
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        abstract = True
        table_description = "Timestamped base model"


class UUIDBase(models.Model):
    id = fields.UUIDField(pk=True)

    class Meta:
        abstract = True
        table_description = "UUID base model"
