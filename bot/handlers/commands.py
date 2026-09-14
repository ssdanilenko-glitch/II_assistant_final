# bot/commands.py

import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.keyboards.inline import topics_kb  # <--- ДОБАВИТЬ ИМПОРТ
from bot.services.backend_client import BackendClient
from bot.states import AskFlow

logger = logging.getLogger(__name__)
router = Router(name="commands")


@router.message(CommandStart())
async def cmd_start(
        message: Message, backend: BackendClient, state: FSMContext
) -> None:
    thread_id = str(message.chat.id)
    try:
        await backend.clear_agent_thread(thread_id)
    except Exception as e:
        logger.warning(f"Could not clear agent thread {thread_id}: {e}")

    await backend.get_or_create_chat(
        owner_external_id=thread_id,
        interface="telegram",
    )

    # Отправляем приветствие С КЛАВИАТУРОЙ
    await message.answer(
        "Привет! Я подключён к chat-сервису. Выберите тему или напишите вопрос.\n"
        "Команды: /help, /ask, /clear, /cancel",
        reply_markup=topics_kb()  # <--- ДОБАВИТЬ ЭТО
    )


@router.callback_query(F.data.startswith("topic:"))
async def on_topic_selected(cb: CallbackQuery, state: FSMContext) -> None:
    """Обработка выбора темы из меню."""
    _, slug = cb.data.split(":", 1)

    if slug == "cancel":
        await cb.answer("Выбор темы отменён")
        await state.clear()
        return

    # Сохраняем выбранную тему в состояние
    await state.set_state(AskFlow.waiting_for_question)
    await state.update_data(topic=slug)

    await cb.answer()
    await cb.message.edit_text(
        f"Вы выбрали тему: **{slug}**\n\nНапишите ваш вопрос:",
        parse_mode="Markdown"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Доступные команды:\n"
        "/start — начать заново\n"
        "/ask — задать вопрос с выбором темы\n"
        "/clear — очистить историю диалога\n"
        "/cancel — отменить текущий сценарий\n"
        "\nДля админов: /stats, /broadcast <текст>"
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    current = await state.get_state()
    if current is None:
        await message.answer("Нечего отменять.")
        return

    if current in (AskFlow.waiting_for_topic, AskFlow.waiting_for_question):
        logger.info("cmd_cancel: очистка состояния")
        await state.clear()
        await message.answer("Сценарий отменён.")
    else:
        await message.answer("Нечего отменять.")


@router.message(Command("clear"))
async def cmd_clear(
        message: Message, backend: BackendClient, state: FSMContext
) -> None:
    logger.info("cmd_clear: очистка состояния")
    await state.clear()
    chat_id = await backend.get_or_create_chat(
        owner_external_id=str(message.chat.id),
        interface="telegram",
    )
    await backend.clear_messages(
        chat_id, owner_external_id=str(message.chat.id)
    )
    await message.answer("История очищена.", reply_markup=topics_kb())
