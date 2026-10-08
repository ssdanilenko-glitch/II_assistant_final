"""Заглушка Itilium: имитирует создание заявки и отправляет webhook
обратно в `/webhook/helpdesk`.

После запуска:
- в Redis появляется `helpdesk:ticket:{ticket_id}` со статусом «В работе»;
- пользователь получает в Telegram уведомление с номером заявки;
- при вопросе агента «какой статус у заявки HD-…» инструмент
  `get_helpdesk_status` вернёт «В работе. Отв: Даниленко С.С.».

Запуск (внутри контейнера app):

    docker compose exec app python -m scripts.fake_helpdesk
    docker compose exec app python -m scripts.fake_helpdesk <correlation_id>

Если correlation_id не передан — берётся последний ключ `helpdesk:corr:*`
из Redis (то есть последняя отправленная через бот заявка).
"""

import argparse
import asyncio
import logging
import random
import sys

import httpx
import redis.asyncio as aioredis

from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fake-helpdesk")

APP_URL = "http://app:8000/webhook/helpdesk"
DELAY_SEC = 5
DEFAULT_STATUS = "in_progress"
DEFAULT_ASSIGNEE = "Иванов И.И."


async def _latest_corr_id(redis_url: str) -> str | None:
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        keys = await r.keys("helpdesk:corr:*")
        if not keys:
            return None
        return keys[0].rsplit(":", 1)[-1]
    finally:
        await r.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Itilium stub: отправка webhook")
    parser.add_argument("correlation_id", nargs="?", default=None)
    parser.add_argument("--delay", type=int, default=DELAY_SEC,
                        help="пауза перед отправкой webhook (сек)")
    parser.add_argument("--status", default=DEFAULT_STATUS,
                        help=f"статус заявки (по умолчанию {DEFAULT_STATUS})")
    parser.add_argument("--assignee", default=DEFAULT_ASSIGNEE,
                        help=f"ответственный (по умолчанию {DEFAULT_ASSIGNEE})")
    parser.add_argument("--ticket-id", default=None,
                        help="явный номер заявки; иначе генерируется HD-2026-NNNNN")
    args = parser.parse_args()

    settings = get_settings()
    token = settings.internal_token.get_secret_value()

    if not token:
        log.error("INTERNAL_TOKEN пуст. Проверь .env и перезапусти app.")
        sys.exit(1)

    corr_id = args.correlation_id
    if not corr_id:
        log.info("correlation_id не передан — ищу в Redis…")
        corr_id = await _latest_corr_id(settings.redis_url)
        if not corr_id:
            log.error(
                "в Redis нет ни одного ключа helpdesk:corr:*. "
                "Сначала отправь письмо через бот (кнопка «✅ Отправить»)."
            )
            sys.exit(1)
        log.info("использую correlation_id=%s", corr_id)

    ticket_id = args.ticket_id or f"HD-2026-{random.randint(10000, 99999)}"

    log.info(
        "Itilium stub: создаю заявку ticket_id=%s status=%s assignee=%s",
        ticket_id, args.status, args.assignee,
    )
    log.info("жду %s сек и отправляю webhook…", args.delay)
    await asyncio.sleep(args.delay)

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            APP_URL,
            json={
                "correlation_id": corr_id,
                "ticket_id": ticket_id,
                "status": args.status,
                "assignee": args.assignee,
            },
            headers={"X-Internal-Token": token},
        )

    if r.status_code == 200:
        log.info("✅ webhook OK: %s", r.text)
        log.info("Теперь спроси в Telegram: «какой статус у заявки %s?»", ticket_id)
    elif r.status_code == 401:
        log.error(
            "❌ webhook 401 Unauthorized. INTERNAL_TOKEN в скрипте не совпадает "
            "с тем, что видит app."
        )
        sys.exit(2)
    elif r.status_code == 503:
        log.error("❌ webhook 503: Redis недоступен на стороне app")
        sys.exit(3)
    else:
        log.error("❌ webhook %s: %s", r.status_code, r.text[:300])
        sys.exit(4)


if __name__ == "__main__":
    asyncio.run(main())