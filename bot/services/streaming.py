import logging
import uuid
from collections.abc import AsyncIterable
from time import monotonic
from uuid import UUID

import telegramify_markdown
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message

from bot.keyboards.inline import feedback_kb

log = logging.getLogger(__name__)

DRAFT_MIN_INTERVAL_SEC = 0.7


def _to_tg_markdown(text: str) -> str:
    try:
        return telegramify_markdown.markdownify(text)
    except Exception:
        return text


async def stream_to_chat(
    message: Message,
    events: AsyncIterable[dict],
    chat_id: UUID | None = None,
) -> dict:
    """
    Обрабатывает поток событий от агента.
    Возвращает:
      - {'status': 'done', 'text': buffer} – если поток завершён нормально.
      - {'status': 'interrupt', 'thread_id': str, 'payload': dict} – если получен interrupt.
    """
    draft_id = uuid.uuid4().int & 0xFFFFFFFF or 1
    buffer = ""
    assistant_message_id: str | None = None
    last_draft_at = 0.0
    interrupt_data = None

    # Пытаемся использовать sendMessageDraft, иначе fallback
    try:
        await message.bot.send_message_draft(
            chat_id=message.chat.id, draft_id=draft_id, text="",
        )
        last_draft_at = monotonic()
    except AttributeError:
        return await _stream_via_edit_text(message, events, chat_id)
    except TelegramRetryAfter as e:
        log.warning("draft flood on init, falling back to edit_text: retry_after=%s", e.retry_after)
        return await _stream_via_edit_text(message, events, chat_id)

    async for event in events:
        etype = event.get("type")
        if etype == "token":
            buffer += event.get("delta", "")
            if not buffer.strip():
                continue
            now = monotonic()
            if now - last_draft_at < DRAFT_MIN_INTERVAL_SEC:
                continue
            try:
                await message.bot.send_message_draft(
                    chat_id=message.chat.id,
                    draft_id=draft_id,
                    text=buffer,
                )
                last_draft_at = now
            except TelegramRetryAfter as e:
                last_draft_at = now + e.retry_after
            except TelegramBadRequest:
                pass
        elif etype == "message_saved":
            assistant_message_id = event.get("message_id")
        elif etype == "interrupt":
            interrupt_data = {
                "thread_id": event.get("thread_id"),
                "payload": event.get("payload", {}),
            }
            # Прерываем обработку, не отправляем финальное сообщение
            break

    # Если было прерывание – возвращаем данные, не отправляя финальный ответ
    if interrupt_data:
        return {"status": "interrupt", **interrupt_data}

    # Нормальное завершение – отправляем финальное сообщение с кнопками feedback
    if buffer:
        reply_markup = feedback_kb(assistant_message_id) if assistant_message_id else None
        await _send_final(message, buffer, reply_markup)
    return {"status": "done", "text": buffer}


async def _send_final(message: Message, text: str, reply_markup) -> None:
    md = _to_tg_markdown(text)
    try:
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=md,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except TelegramBadRequest as e:
        log.warning("MarkdownV2 parse failed, fallback to plain: %s", e)
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=text,
            reply_markup=reply_markup,
        )


async def _stream_via_edit_text(
    message: Message,
    events: AsyncIterable[dict],
    chat_id: UUID | None = None,
) -> dict:
    """Fallback через edit_text. Возвращает тот же dict."""
    sent = await message.answer("…")
    buffer = ""
    assistant_message_id: str | None = None
    last_edit = monotonic()
    interrupt_data = None

    async for event in events:
        etype = event.get("type")
        if etype == "token":
            buffer += event.get("delta", "")
            if monotonic() - last_edit >= 1.0:
                try:
                    await sent.edit_text(buffer)
                    last_edit = monotonic()
                except TelegramRetryAfter as e:
                    last_edit = monotonic() + e.retry_after
                except TelegramBadRequest:
                    last_edit = monotonic()
        elif etype == "message_saved":
            assistant_message_id = event.get("message_id")
        elif etype == "interrupt":
            interrupt_data = {
                "thread_id": event.get("thread_id"),
                "payload": event.get("payload", {}),
            }
            break

    if interrupt_data:
        return {"status": "interrupt", **interrupt_data}

    if buffer:
        reply_markup = feedback_kb(assistant_message_id) if assistant_message_id else None
        md = _to_tg_markdown(buffer)
        try:
            await sent.edit_text(
                md,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except TelegramBadRequest:
            try:
                await sent.edit_text(buffer, reply_markup=reply_markup)
            except (TelegramBadRequest, TelegramRetryAfter):
                pass
        except TelegramRetryAfter:
            pass

    return {"status": "done", "text": buffer}
