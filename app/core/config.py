from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PORT: int = 8080

    # Postgres
    POSTGRES_DSN: str = "postgres://root:root@localhost:5432/snippet_sage"


TORTOISE_ORM = {
    "connections": {"default": "postgres://root:root@localhost:5432/snippet_sage"},
    "apps": {
        "models": {
            "models": ["app.models", "aerich.models"],
            "default_connection": "default",
        },
    },
}
