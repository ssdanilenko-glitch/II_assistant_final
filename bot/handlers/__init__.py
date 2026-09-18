from aiogram import Dispatcher

from . import commands, fsm, text, admin,  handoff,feedback,  media


def register_routers(dp: Dispatcher) -> None:
    # Порядок важен: commands первым (там /start, /ask, topic:),
    # затем fsm (HIL-кнопки), text (общий fallback), остальные.
    dp.include_router(commands.router)
    dp.include_router(admin.router)
    dp.include_router(handoff.router)
    dp.include_router(feedback.router)
    dp.include_router(fsm.router)
    dp.include_router(media.router)
    dp.include_router(text.router)