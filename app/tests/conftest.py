import pytest
from aerich import Command
from tortoise import Tortoise

from app.core.config import TORTOISE_ORM


@pytest.fixture(autouse=True)
async def db_session():
    await Tortoise.init(config=TORTOISE_ORM)
    cmd = Command(tortoise_config=TORTOISE_ORM, app="models")
    await cmd.init()
    await cmd.upgrade()

    yield
    for app in Tortoise.apps.values():
        for model in app.values():
            await model.all().delete()
    await Tortoise.close_connections()
