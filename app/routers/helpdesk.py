"""Приём ответов от HelpDesk.

HelpDesk шлёт POST /webhook/helpdesk с телом:
    {"correlation_id": "abc123def456", "ticket_id": "HD-2026-00123",
     "status": "registered"}

Мы находим chat_id по correlation_id и просим бота доставить сообщение.
"""

import logging

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger("it_assistant")
router = APIRouter(prefix="/webhook", tags=["webhook"])


class HelpDeskEvent(BaseModel):
    correlation_id: str
    ticket_id: str
    status: str = "registered"


@router.post("/helpdesk")
async def helpdesk_webhook(
    event: HelpDeskEvent,
    request: Request,
    x_internal_token: str = Header(default=""),
) -> dict:
    settings = get_settings()
    if x_internal_token != settings.internal_token:
        raise HTTPException(status_code=401, detail="invalid token")

    redis = request.app.state.redis
    if redis is None:
        raise HTTPException(status_code=503, detail="redis unavailable")

    chat_id = await redis.get(f"helpdesk:corr:{event.correlation_id}")
    if not chat_id:
        logger.warning(
            "webhook: correlation_id=%s не найден в Redis",
            event.correlation_id,
        )
        return {"status": "ignored", "reason": "unknown correlation_id"}

    text = (
        f"🎫 Ваше обращение зарегистрировано в HelpDesk.\n"
        f"Номер: <b>{event.ticket_id}</b>\n"
        f"Статус: {event.status}."
    )

    # bot слушает на 9000, endpoint /notify (у тебя уже есть)
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.post(
                "http://bot:9000/notify",
                json={
                    "chat_id": int(chat_id),
                    "text": text,
                    "parse_mode": "HTML",
                },
                headers={"X-Internal-Token": settings.internal_token},
            )
            r.raise_for_status()
        except Exception:
            logger.exception("не удалось доставить в Telegram chat_id=%s", chat_id)
            raise HTTPException(status_code=502, detail="bot notify failed")

    # Удаляем маппинг — одна заявка, один ответ
    await redis.delete(f"helpdesk:corr:{event.correlation_id}")

    return {"status": "ok", "ticket_id": event.ticket_id}