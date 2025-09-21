import pytest
from tortoise import Tortoise


@pytest.fixture(autouse=True)
async def db_session():
    db_url = "postgres://root:root@localhost:5432/snippet_sage_test"
    await Tortoise.init(db_url=db_url, modules={"models": ["app.models"]})
    await Tortoise.generate_schemas()
    yield
    from app.models import Chunk, Document

    await Chunk.all().delete()
    await Document.all().delete()
    await Tortoise.close_connections()
