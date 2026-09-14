import logging

from redis.asyncio import Redis

logger = logging.getLogger(__name__)
redis_client: Redis | None = None

def set_redis(client: Redis):
    global redis_client
    redis_client = client

def get_redis() -> Redis | None:
    return redis_client
