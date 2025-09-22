from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "documents" (
            "id" UUID NOT NULL PRIMARY KEY,
            "source" VARCHAR(255),
            "content_sha256" VARCHAR(64) NOT NULL UNIQUE,
            "raw_text" TEXT NOT NULL
        );
        COMMENT ON TABLE "documents" IS 'Documents Table';
        CREATE TABLE IF NOT EXISTS "chunks" (
            "id" UUID NOT NULL PRIMARY KEY,
            "text" TEXT NOT NULL,
            "chunk_sha256" VARCHAR(64) NOT NULL UNIQUE,
            "document_id" UUID NOT NULL REFERENCES "documents" ("id") ON DELETE CASCADE
        );
        COMMENT ON COLUMN "chunks"."fts" IS 'Full-text search vector (computed)';
        CREATE TABLE IF NOT EXISTS "embeddings" (
            "id" UUID NOT NULL PRIMARY KEY,
            "vector" public.vector(384) NOT NULL,
            "dim" INT NOT NULL,
            "chunk_id" UUID NOT NULL UNIQUE REFERENCES "chunks" ("id") ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS "aerich" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "version" VARCHAR(255) NOT NULL,
            "app" VARCHAR(100) NOT NULL,
            "content" JSONB NOT NULL
        );
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
