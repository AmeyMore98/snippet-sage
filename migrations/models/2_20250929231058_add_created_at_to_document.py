from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "documents" ADD COLUMN IF NOT EXISTS "created_at"
        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "documents" DROP COLUMN "created_at";"""
