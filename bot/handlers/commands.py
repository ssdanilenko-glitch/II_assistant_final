# bot/commands.py

import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.keyboards.inline import topics_kb
from bot.services.backend_client import BackendClient

from bot.services.streaming import stream_to_chat
from bot.states import AskFlow, ConfirmFlow

logger = logging.getLogger(__name__)
router = Router(name="commands")


@router.message(CommandStart())
async def cmd_start(
        message: Message, backend: BackendClient, state: FSMContext
) -> None:
    """Приветствие. Кнопки — по команде /ask."""
    await state.clear()
    try:
        await backend.get_or_create_chat(
            owner_external_id=str(message.chat.id),
            interface="telegram",
        )
    except Exception as e:
        logger.warning(f"get_or_create_chat failed: {e}")

    await message.answer(
        "Привет! Я ИТ-ассистент технической поддержки.\n\n"
        "Чтобы задать вопрос — используйте /ask или просто напишите его текстом.\n\n"
        "Команды:\n"
        "/ask — задать вопрос с выбором темы\n"
        "/help — справка по командам\n"
        "/clear — очистить историю диалога\n"
        "/cancel — отменить текущий сценарий"
    )


@router.message(Command("ask"))
async def cmd_ask(message: Message, state: FSMContext) -> None:
    """Показать клавиатуру с темами."""
    await state.set_state(AskFlow.waiting_for_topic)
    await message.answer(
        "Выберите тему из списка или напишите вопрос напрямую:",
        reply_markup=topics_kb(),
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
        f"Вы выбрали тему: **{slug}**\nНапишите ваш вопрос:",
        parse_mode="Markdown"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Доступные команды:\n"
        "/start — приветствие и справка\n"
        "/ask — задать вопрос с выбором темы\n"
        "/clear — очистить историю диалога\n"
        "/cancel — отменить текущий сценарий\n"
        "\nДля админов: /stats, /broadcast «текст»",
        parse_mode=None,
    )
@router.message(Command("cancel"))
async def cmd_cancel(
        message: Message, state: FSMContext, backend: BackendClient
) -> None:
    current = await state.get_state()
    if current is None:
        await message.answer("Нечего отменять.")
        return

    # HIL-подтверждение: отправляем resume(False), чтобы корректно закрыть interrupt
    if current == ConfirmFlow.waiting_for_decision:
        data = await state.get_data()
        thread_id = data.get("thread_id")
        logger.info("cmd_cancel: отмена HIL, thread_id=%s", thread_id)
        if thread_id:
            try:
                events = backend.resume(
                    thread_id=thread_id,
                    decision=False,
                    owner_external_id=str(message.chat.id),
                )
                await stream_to_chat(message, events)
            except Exception:
                logger.exception("cmd_cancel: ошибка при resume")
        await state.clear()
        await message.answer("Сценарий отменён.")
        return

    if current in (AskFlow.waiting_for_topic, AskFlow.waiting_for_question):
        logger.info("cmd_cancel: очистка состояния AskFlow")
        await state.clear()
        await message.answer("Сценарий отменён.")
    else:
        await state.clear()
        await message.answer("Сценарий отменён.")


@router.message(Command("clear"))
async def cmd_clear(
        message: Message, backend: BackendClient, state: FSMContext
) -> None:
    logger.info("cmd_clear: очистка состояния")
    await state.clear()
    await backend.clear_agent_thread(f"tg-{message.chat.id}")
    chat_id = await backend.get_or_create_chat(
        owner_external_id=str(message.chat.id),
        interface="telegram",
    )
    await backend.clear_messages(
        chat_id, owner_external_id=str(message.chat.id)
    )
    await message.answer("История очищена.")
