from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    APP_ENV: str = "dev"
    PORT: int = 8080  # Keep for compatibility

    # Database
    DATABASE_URL: str

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384  # Default for all-MiniLM-L6-v2

    # RAG
    RETRIEVAL_K: int = 12
    CHUNK_MAX_SENTENCES: int = 6  # Added based on architecture.md (9.2)
    CHUNK_OVERLAP_SENTENCES: int = 1
    MIN_INPUT_CHARS: int = 40
    MAX_INPUT_CHARS: int = 200_000


TORTOISE_ORM = {
    "connections": {"default": Settings().DATABASE_URL},
    "apps": {
        "models": {
            "models": ["app.models", "aerich.models"],
            "default_connection": "default",
        },
    },
}
