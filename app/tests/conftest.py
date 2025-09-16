import pytest
from tortoise import Tortoise


@pytest.fixture(scope="session", autouse=True)
async def init_db():
    await Tortoise.init(db_url="sqlite://:memory:", modules={"models": ["app.tests.test_models"]})
    await Tortoise.generate_schemas()
    yield
    await Tortoise.close_connections()


@pytest.fixture(autouse=True)
async def cleanup_db():
    """Clean up database between tests"""
    from app.tests.test_models import TimestampedTestModel, UUIDTestModel

    yield
    await UUIDTestModel.all().delete()
    await TimestampedTestModel.all().delete()
