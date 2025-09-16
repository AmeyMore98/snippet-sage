import asyncio
import uuid
from datetime import datetime

import pytest

from app.tests.test_models import TimestampedTestModel, UUIDTestModel


@pytest.mark.asyncio
async def test_uuid_base_provides_uuid_pk():
    """Tests that the UUIDBase model provides a UUID primary key."""
    obj = await UUIDTestModel.create(name="test")
    assert isinstance(obj.id, uuid.UUID)


@pytest.mark.asyncio
async def test_timestamped_model_creation_timestamps():
    """Tests that created_at and updated_at are set on creation."""
    obj = await TimestampedTestModel.create(name="test")
    assert isinstance(obj.created_at, datetime)
    assert isinstance(obj.updated_at, datetime)


@pytest.mark.asyncio
async def test_timestamped_model_update_timestamp():
    """Tests that updated_at is updated when the model is saved."""
    obj = await TimestampedTestModel.create(name="test")
    first_updated_at = obj.updated_at

    # We need to ensure there's a measurable time difference
    await asyncio.sleep(0.01)

    obj.name = "updated"
    await obj.save()

    assert obj.updated_at > first_updated_at
