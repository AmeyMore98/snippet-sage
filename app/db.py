from fastapi import FastAPI
from tortoise.contrib.fastapi import register_tortoise

from app.core.config import Settings


def init_db(app: FastAPI, settings: Settings) -> None:
    """Initialize the database connection and register the models."""
    register_tortoise(
        app,
        db_url=settings.DATABASE_URL,
        modules={"models": ["app.models"]},
        generate_schemas=False,
        add_exception_handlers=True,
    )
