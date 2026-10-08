"""Приём ответов от HelpDesk.

HelpDesk шлёт POST /webhook/helpdesk с телом:
    {"correlation_id": "abc123def456", "ticket_id": "HD-2026-00123",
     "status": "registered", "assignee": "Иванов И.И."}

Мы:
1. Сохраняем статус заявки в Redis (`helpdesk:ticket:{ticket_id}`),
   чтобы инструмент `get_helpdesk_status` мог его прочитать.
2. Находим chat_id по correlation_id и просим бота доставить сообщение.
"""

import json
import logging
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from app.core.config import get_settings

MSK = timezone(timedelta(hours=3))

logger = logging.getLogger("it_assistant")
router = APIRouter(prefix="/webhook", tags=["webhook"])

# TTL статуса заявки: 7 дней. Должен совпадать с HELPDESK_TICKET_TTL
# в app/main.py — там константа используется при чтении.
HELPDESK_TICKET_TTL = 7 * 24 * 3600

STATUS_RU = {
    "registered": "Зарегистрирована",
    "in_progress": "В работе",
    "resolved": "Завершена",
    "closed": "Закрыта",
    "rejected": "Отклонена",
}


class HelpDeskEvent(BaseModel):
    correlation_id: str
    ticket_id: str
    status: str = "registered"
    assignee: str = ""  # опционально: ответственный за заявку


def _ticket_payload(event: HelpDeskEvent) -> dict:
    """Формирует payload для записи в Redis.

    Единая структура с тем, что читает `_get_helpdesk_status` в main.py.
    """
    assignee = event.assignee or "не назначен"
    return {
        "assignee": event.assignee,
        "message": f"{STATUS_RU.get(event.status, event.status)}. Отв: {assignee}",
        "updated_at": datetime.now(MSK).isoformat(),
    }


@router.post("/helpdesk")
async def helpdesk_webhook(
    event: HelpDeskEvent,
    request: Request,
    x_internal_token: str = Header(default=""),
) -> dict:
    settings = get_settings()

    expected = settings.internal_token.get_secret_value()
    if not expected:
        raise HTTPException(status_code=500, detail="internal_token not configured")
    if x_internal_token != expected:
        raise HTTPException(status_code=401, detail="invalid token")

    redis = request.app.state.redis
    if redis is None:
        raise HTTPException(status_code=503, detail="redis unavailable")

    # === 1. Сохраняем статус заявки в Redis — ДО отправки уведомления. ===
    # Даже если уведомление не доставится, `get_helpdesk_status` вернёт
    # актуальный статус.
    try:
        await redis.set(
            f"helpdesk:ticket:{event.ticket_id}",
            json.dumps(_ticket_payload(event), ensure_ascii=False),
            ex=HELPDESK_TICKET_TTL,
        )
        logger.info(
            "[HELPDESK_WEBHOOK] ticket=%s status=%s сохранён (TTL=%ds)",
            event.ticket_id, event.status, HELPDESK_TICKET_TTL,
        )
    except Exception:
        logger.exception(
            "[HELPDESK_WEBHOOK] не удалось сохранить ticket=%s в Redis",
            event.ticket_id,
        )
        # не падаем — продолжаем обработку уведомления

    # === 2. Ищем chat_id и доставляем уведомление. ===
    chat_id = await redis.get(f"helpdesk:corr:{event.correlation_id}")
    if not chat_id:
        # correlation_id не найден — заявка всё равно сохранена,
        # просто уведомлять некого (пользователь не в Telegram-боте
        # или маппинг уже истёк).
        logger.warning(
            "webhook: correlation_id=%s не найден в Redis "
            "(статус заявки ticket=%s сохранён)",
            event.correlation_id, event.ticket_id,
        )
        return {
            "status": "ignored",
            "reason": "unknown correlation_id",
            "ticket_saved": True,
        }

    status_ru = STATUS_RU.get(event.status, event.status)
    text = (
        f"🎫 Заявка создана в HelpDesk.\n"
        f"Номер: <b>{event.ticket_id}</b>\n"
        f"Статус: {status_ru}."
    )
    if event.assignee:
        text += f"\nОтветственный: {event.assignee}"

    # bot слушает на 9000, endpoint /notify
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.post(
                "http://bot:9000/notify",
                json={
                    "chat_id": int(chat_id),
                    "text": text,
                    "parse_mode": "HTML",
                },
                headers={"X-Internal-Token": settings.internal_token.get_secret_value()},
            )
            r.raise_for_status()
        except Exception as exc:
            logger.exception("не удалось доставить в Telegram chat_id=%s", chat_id)
            raise HTTPException(status_code=502, detail=f"bot notify failed: {exc}")

    # Удаляем маппинг correlation_id → chat_id — одна заявка, одно уведомление.
    # Ключ helpdesk:ticket:{ticket_id} НЕ удаляем — он живёт 7 дней и нужен
    # для get_helpdesk_status.
    await redis.delete(f"helpdesk:corr:{event.correlation_id}")

    return {"status": "ok", "ticket_id": event.ticket_id}