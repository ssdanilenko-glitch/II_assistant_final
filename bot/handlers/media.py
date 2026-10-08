import asyncio
import base64
import json
import logging
from io import BytesIO
import uuid

import httpx
from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.services.backend_client import BackendClient
from bot.services.error_handling import handle_backend_error
from bot.services.redis_client import get_redis
from bot.services.streaming import stream_to_chat
from bot.services.typing import typing_until
from bot.states import ConfirmFlow
from bot.utils.sender import get_sender_info

router = Router(name="media")
log = logging.getLogger(__name__)

MAX_PHOTO_BYTES = 2 * 1024 * 1024   # 2 МБ
MAX_DOC_BYTES = 10 * 1024 * 1024    # 10 МБ
ALLOWED_DOC_EXT = (".jpg",".pdf", ".docx")


async def _download_to_bytes(bot, file_id: str) -> bytes:
    f = await bot.get_file(file_id)
    buf = BytesIO()
    await bot.download_file(f.file_path, destination=buf)
    return buf.getvalue()


async def _process_media_in_backend(data: bytes, mime: str, filename: str) -> str:
    """
    Отправляет файл в бэкенд на /media/process и возвращает извлечённый текст.
    """
    async with httpx.AsyncClient() as client:
        files = {"file": (filename, data, mime)}
        resp = await client.post("http://app:8000/media/process", files=files)
        resp.raise_for_status()
        return resp.json()["text"]


async def _store_media_in_redis(redis_client, thread_id: str, filename: str, data: bytes, mime: str):
    """Сохраняет файл в Redis для последующего прикрепления к письму."""
    key = f"media:{thread_id}"
    raw = await redis_client.get(key)
    files = json.loads(raw) if raw else []
    files.append({
        "filename": filename,
        "data": base64.b64encode(data).decode('utf-8'),
        "mime": mime
    })
    await redis_client.setex(key, 3600, json.dumps(files))  # TTL 1 час


def _pick_photo_size(photos):
    sorted_photos = sorted(photos, key=lambda p: p.file_size or 0, reverse=True)
    for p in sorted_photos:
        if (p.file_size or 0) <= MAX_PHOTO_BYTES:
            return p
    return sorted_photos[-1]


async def _send_media_as_text(
    message: Message,
    backend: BackendClient,
    state: FSMContext,
    data: bytes,
    mime: str,
    filename: str,
    content: str = "",
):
    # thread_id из текущей сессии (создан в /start или предыдущем сообщении).
    state_data = await state.get_data()
    thread_id = state_data.get("thread_id")
    if not thread_id:
        thread_id = f"tg-{message.chat.id}-{uuid.uuid4().hex[:8]}"
        await state.update_data(thread_id=thread_id)
        log.info("media: создана сессия на лету thread_id=%s", thread_id)

    # Сохраняем файл в Redis
    redis = get_redis()
    if redis:
        await _store_media_in_redis(redis, thread_id, filename, data, mime)

    # Получаем текст из медиа
    try:
        media_text = await _process_media_in_backend(data, mime, filename)
    except Exception:
        log.exception("Media processing failed")
        await message.answer("Не удалось обработать файл. Попробуйте позже.")
        return

    sender_info = get_sender_info(message)
    full_text = f"{sender_info}\n\n{content}\n{media_text}" if content else f"{sender_info}\n\n{media_text}"

    # Отправляем агенту
    stop = asyncio.Event()
    typing_task = asyncio.create_task(typing_until(message.bot, message.chat.id, stop))
    try:
        events = backend.send_message(
            content=full_text,
            owner_external_id=str(message.chat.id),
            thread_id=thread_id,
        )
        result = await stream_to_chat(message, events)

        # Если пришёл interrupt – показываем кнопки
        if result.get("status") == "interrupt":
            await state.set_state(ConfirmFlow.waiting_for_decision)
            await state.update_data(thread_id=thread_id)

            preview = result.get("payload", {}).get("preview", {})
            from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="✅ Отправить",
                            callback_data=f"confirm_send_{thread_id}"
                        ),
                        InlineKeyboardButton(
                            text="❌ Отмена",
                            callback_data=f"confirm_cancel_{thread_id}"
                        ),
                    ]
                ]
            )
            await message.answer(
                f"📧 <b>Подтверждение отправки письма</b>\n"
                f"Кому: {preview.get('to', '')}\n"
                f"Тема: {preview.get('subject', '')}\n"
                f"Текст: {preview.get('body', '')[:200]}...",
                reply_markup=kb,
                parse_mode="HTML"
            )
    except Exception as exc:
        await handle_backend_error(message, exc)
    finally:
        stop.set()
        await typing_task

# Обработчики для разных типов медиа

@router.message(F.photo)
async def on_photo(message: Message, backend: BackendClient, state: FSMContext):
    photo = _pick_photo_size(message.photo)
    data = await _download_to_bytes(message.bot, photo.file_id)
    await _send_media_as_text(
        message, backend, state, data, mime="image/jpeg",
        filename="photo.jpg", content=message.caption or ""
    )

@router.message(F.voice)
async def on_voice(message: Message, backend: BackendClient, state: FSMContext):
    data = await _download_to_bytes(message.bot, message.voice.file_id)
    await _send_media_as_text(
        message, backend, state, data, mime="audio/ogg",
        filename="voice.ogg", content=message.caption or ""
    )


@router.message(F.audio)
async def on_audio(message: Message, backend: BackendClient, state: FSMContext):
    data = await _download_to_bytes(message.bot, message.audio.file_id)
    mime = message.audio.mime_type or "audio/mpeg"
    filename = message.audio.file_name or "audio.mp3"
    await _send_media_as_text(
        message, backend, state, data, mime=mime,
        filename=filename, content=message.caption or ""
    )


@router.message(F.document)
async def on_document(message: Message, backend: BackendClient, state: FSMContext):
    doc = message.document
    fname = (doc.file_name or "").lower()
    if not fname.endswith(ALLOWED_DOC_EXT):
        await message.answer(f"Поддерживаются только {', '.join(ALLOWED_DOC_EXT)}.")
        return
    if (doc.file_size or 0) > MAX_DOC_BYTES:
        await message.answer(f"Файл слишком большой (>{MAX_DOC_BYTES // 1024 // 1024} МБ).")
        return
    data = await _download_to_bytes(message.bot, doc.file_id)
    mime = doc.mime_type or "application/octet-stream"
    await _send_media_as_text(
        message, backend, state, data, mime=mime,
        filename=doc.file_name or "document.bin",
        content=message.caption or ""
    )
