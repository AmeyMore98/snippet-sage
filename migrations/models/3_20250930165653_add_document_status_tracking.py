from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "documents" ADD COLUMN IF NOT EXISTS "status" VARCHAR(20) NOT NULL  DEFAULT 'PENDING';
        ALTER TABLE "documents" ADD COLUMN IF NOT EXISTS"processing_errors" TEXT;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "documents" DROP COLUMN "status";
        ALTER TABLE "documents" DROP COLUMN "processing_errors";"""
