import asyncio
import logging
import uuid

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.services.backend_client import BackendClient
from bot.services.error_handling import handle_backend_error
from bot.services.streaming import stream_to_chat
from bot.services.typing import typing_until
from bot.states import ConfirmFlow
from bot.utils.sender import get_sender_info

router = Router(name="text")
log = logging.getLogger(__name__)


# ======== СНАЧАЛА обработчик для состояния подтверждения ========
@router.message(StateFilter(ConfirmFlow.waiting_for_decision), F.text)
async def on_confirm_decision(
        message: Message,
        state: FSMContext,
        backend: BackendClient,
) -> None:
    data = await state.get_data()
    thread_id = data.get("thread_id")
    log.info(
        f"on_confirm_decision вызван для пользователя {message.chat.id}, текст: {message.text}"
    )

    if not thread_id:
        log.info("on_confirm_decision: очистка состояния после успешного resume")
        await state.clear()
        await message.answer("Нет активного запроса подтверждения.")
        return

    text = message.text.lower().strip()
    if text in ("да", "подтверждаю", "отправить", "yes", "ok", "+"):
        decision = True
    elif text in ("нет", "отмена", "cancel", "no", "-"):
        decision = False
    else:
        # Пользователь задал новый вопрос, а не ответил «да/нет».
        # Выходим из режима подтверждения и обрабатываем как обычный текст.
        log.info("on_confirm_decision: пользователь задал новый вопрос — выход из режима")
        await state.clear()
        await on_text(message, backend, state)
        return

    try:
        events = backend.resume(
            thread_id=thread_id,
            decision=decision,
            owner_external_id=str(message.chat.id),
        )
        result = await stream_to_chat(message, events)
        if result.get("status") == "interrupt":
            await message.answer("Произошла ещё одна задержка. Попробуйте снова.")
        else:
            log.info("on_confirm_decision: очистка состояния после успешного resume")
            await state.clear()
            # Убираем жестко заданный текст, теперь ответ придет от модели через стрим
            # await message.answer("✅ Заявка отправлена" if decision else "❌ Отменено")
            pass
    except Exception:
        log.exception("Ошибка при resume")
        log.exception("on_confirm_decision: очистка состояния при ошибке")
        await message.answer("Не удалось обработать решение. Попробуйте позже.")
        await state.clear()

@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message, backend: BackendClient, state: FSMContext) -> None:
    if await state.get_state() == ConfirmFlow.waiting_for_decision:
        return

    # thread_id из текущей сессии (создан в /start или предыдущем сообщении).
    data = await state.get_data()
    thread_id = data.get("thread_id")
    if not thread_id:
        # Защита: пользователь пишет без /start. Создаём сессию на лету.
        thread_id = f"tg-{message.chat.id}-{uuid.uuid4().hex[:8]}"
        await state.update_data(thread_id=thread_id)
        log.info("on_text: создана сессия на лету thread_id=%s", thread_id)

    stop = asyncio.Event()
    typing_task = asyncio.create_task(typing_until(message.bot, message.chat.id, stop))

    try:
        sender_info = get_sender_info(message)
        content_with_sender = f"{sender_info}\n\n{message.text}"

        # ОДИН вызов — один SSE-поток — один проход.
        # thread_id уже определён выше, НЕ перезаписываем его здесь.
        events = backend.send_message(
            content=content_with_sender,
            owner_external_id=str(message.chat.id),
            thread_id=thread_id,
        )

        result = await stream_to_chat(message, events)

        if result.get("status") == "interrupt":
            await state.set_state(ConfirmFlow.waiting_for_decision)
            # thread_id уже в state — просто обновим на всякий случай
            await state.update_data(thread_id=thread_id)

            preview = result.get("payload", {}).get("preview", {})
            sender = preview.get("sender_info", "")
            body = preview.get("body", "")
            text_preview = (
                f"📧 <b>Подтверждение отправки письма</b>\n"
                f"Кому: {preview.get('to', '')}\n"
                f"Тема: {preview.get('subject', '')}\n"
            )
            if sender:
                text_preview += f"От: {sender}\n"
            text_preview += f"\n{body[:200]}..."

            kb = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(
                    text="✅ Отправить",
                    callback_data=f"confirm_send_{thread_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data=f"confirm_cancel_{thread_id}",
                ),
            ]])
            await message.answer(text_preview, reply_markup=kb, parse_mode="HTML")

    except Exception as exc:
        await handle_backend_error(message, exc)
    finally:
        stop.set()
        await typing_task