import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from tortoise.contrib.fastapi import RegisterTortoise

from app.api import answer, ingest
from app.core.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info("Starting up the application...")
    # The database connection is managed by register_tortoise's lifespan
    yield
    logger.info("Shutting down the application...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = Settings()
    app = FastAPI(
        title="Personal Knowledge Base (PKB) MVP",
        description="A minimal RAG-based knowledge base with FastAPI.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include API routers
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
    app.include_router(answer.router, prefix="/api/v1", tags=["QA"])

    # Initialize database
    RegisterTortoise(
        app,
        db_url=settings.DATABASE_URL,
        modules={"models": ["app.models", "aerich.models"]},
        generate_schemas=False,  # Let Aerich handle schema generation
    )

    @app.get("/health", tags=["Health"])
    async def health_check():
        """Perform a health check."""
        return {"status": "ok"}

    return app


app = create_app()
