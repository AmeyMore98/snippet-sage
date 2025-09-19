from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    PORT: int = 8080  # Keep for compatibility

    # Database
    DATABASE_URL: str = "postgres://root:root@localhost:5432/snippet_sage"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


TORTOISE_ORM = {
    "connections": {"default": "postgres://root:root@localhost:5432/snippet_sage"},
    "apps": {
        "models": {
            "models": ["app.models", "aerich.models"],
            "default_connection": "default",
        },
    },
}
