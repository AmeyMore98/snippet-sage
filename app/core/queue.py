"""
Queue infrastructure module.

This module provides access to different queue implementations:
- dramatiq_broker: For Dramatiq-based task queue (job processing)

Future queue types can be added.
"""

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import AsyncIO

from app.core.config import Settings

settings = Settings()

# ============================================================================
# Dramatiq Task Queue
# ============================================================================
# Used for async job processing with automatic retries and failure handling
dramatiq_broker = RedisBroker(url=settings.REDIS_URL)

# Add AsyncIO middleware to support async actors
dramatiq_broker.add_middleware(AsyncIO())

dramatiq.set_broker(dramatiq_broker)


# ============================================================================
# Future Queue Types
# ============================================================================
# Add additional queue implementations here as needed:
# - Celery broker
# - RabbitMQ connection
# - AWS SQS client
# - Custom priority queues
# etc.
