from app.models.base import TimestampedModel, UUIDModel


class UUIDTestModel(UUIDModel):
    class Meta:
        table = "test_uuid_model"


class TimestampedTestModel(TimestampedModel):
    class Meta:
        table = "test_timestamped_model"
