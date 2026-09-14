import asyncio
import logging

import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.redis import RedisStorage
from redis.asyncio import Redis

from bot.config import get_bot_settings
from bot.handlers import register_routers
from bot.services.alert_drain import drain_alerts
from bot.services.backend_client import BackendClient
from bot.services.http import build_http_client
from bot.services.redis_client import set_redis
from bot.web import build_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("bot")


async def main() -> None:
    settings = get_bot_settings()

    # Создаём Redis-клиент
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    # Проверяем подключение
    try:
        await redis_client.ping()
        log.info("Redis connected successfully")
        # Сохраняем клиент в глобальное хранилище
        set_redis(redis_client)
        storage = RedisStorage(redis=redis_client, ttl=86400)  # 24 часа
    except Exception as e:
        log.warning(f"Redis connection failed: {e}. Falling back to MemoryStorage")
        from aiogram.fsm.storage.memory import MemoryStorage
        storage = MemoryStorage()

    bot = Bot(
        token=settings.bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=storage)

    http = build_http_client(settings)
    backend = BackendClient(
        http, admin_token=settings.admin_token.get_secret_value()
    )
    dp["backend"] = backend

    register_routers(dp)

    api = build_api(bot, settings.internal_token.get_secret_value())
    config = uvicorn.Config(
        api,
        host="0.0.0.0",
        port=settings.bot_api_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    log.info(
        "Bot starting (backend=%s, notify-port=%s, admin_chat_id=%s)",
        settings.backend_url,
        settings.bot_api_port,
        settings.admin_chat_id,
    )
    try:
        await asyncio.gather(
            dp.start_polling(bot),
            server.serve(),
            drain_alerts(bot, backend, settings.admin_chat_id),
        )
    finally:
        # Закрываем Redis-клиент корректно
        if redis_client:
            await redis_client.aclose()
        await backend.aclose()
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
