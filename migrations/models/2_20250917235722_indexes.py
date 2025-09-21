from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
    -- FTS helper, computed on DB level for maximum consistency
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

    -- Hybrid indices
    CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN (fts);
    CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (vector vector_l2_ops) WITH (lists = 100);

    -- Dedup
    CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_sha ON documents(content_sha256);
    CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_sha ON chunks(chunk_sha256);
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
    -- Drop indices
    DROP INDEX IF EXISTS uq_chunks_sha;
    DROP INDEX IF EXISTS uq_documents_sha;
    DROP INDEX IF EXISTS idx_embeddings_vector;
    DROP INDEX IF EXISTS idx_chunks_fts;

    -- Drop columns
    ALTER TABLE chunks DROP COLUMN IF EXISTS fts;
    """
