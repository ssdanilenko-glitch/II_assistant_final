import logging

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from bot.services.backend_client import BackendClient
from bot.services.streaming import stream_to_chat

log = logging.getLogger(__name__)
router = Router(name="confirm")


# Объединяем оба типа callback-запросов (send и cancel) в одном обработчике
@router.callback_query(F.data.startswith(("confirm_send_", "confirm_cancel_")))
async def on_confirm_action(cb: CallbackQuery, backend: BackendClient, state: FSMContext):
    """Обрабатывает нажатие кнопок «Отправить» и «Отмена»."""
    # Сразу отвечаем Telegram — иначе callback протухает за 15 секунд,
    # пока идёт медленный resume с генерацией письма.
    try:
        await cb.answer()
    except Exception:
        pass

    # Извлекаем decision и thread_id по префиксам.
    if cb.data.startswith("confirm_send_"):
        decision = True
        thread_id = cb.data[len("confirm_send_"):]
    elif cb.data.startswith("confirm_cancel_"):
        decision = False
        thread_id = cb.data[len("confirm_cancel_"):]
    else:
        await cb.answer("Неизвестная команда")
        return

    try:
        events = backend.resume(
            thread_id=thread_id,
            decision=decision,
            owner_external_id=str(cb.message.chat.id),
        )

        # stream_to_chat сам разберётся с assistant_text / token / interrupt
        result = await stream_to_chat(cb.message, events)

        await state.clear()
        try:
            await cb.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass

    except Exception:
        log.exception("Ошибка при resume")
        await cb.message.answer("Не удалось обработать решение. Попробуйте позже.")
        await state.clear()