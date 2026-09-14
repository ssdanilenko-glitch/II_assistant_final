import logging

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from bot.services.backend_client import BackendClient
from bot.services.streaming import stream_to_chat

log = logging.getLogger(__name__)
router = Router(name="confirm")


@router.callback_query(F.data.startswith(("confirm_send_", "confirm_cancel_")))
async def on_confirm_action(cb: CallbackQuery, backend: BackendClient, state: FSMContext):
    """Обрабатывает нажатие кнопок «Отправить» и «Отмена»."""
    # Извлекаем thread_id из конца callback_data
    thread_id = cb.data.split("_")[-1]
    decision = "send" in cb.data  # True для отправки, False для отмены

    try:
        # Возобновляем выполнение агента с решением пользователя
        events = backend.resume(
            thread_id=thread_id,
            decision=decision,
            owner_external_id=str(cb.from_user.id)
        )
        result = await stream_to_chat(cb.message, events)

        if result.get("status") == "interrupt":
            await cb.message.answer("Произошла ещё одна задержка. Попробуйте снова.")
        else:
            log.info("on_confirm_action: очистка состояния после успешного resume")
            await state.clear()
            await cb.answer("✅ Заявка отправлена" if decision else "❌ Отменено")
    except Exception:
        log.exception("Ошибка при resume")
        await cb.message.answer("Не удалось обработать решение. Попробуйте позже.")
        await state.clear()
        await cb.answer()

    # Удаляем клавиатуру у сообщения
    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
