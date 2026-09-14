# bot/utils/sender.py
from aiogram.types import Message


def get_sender_info(message: Message) -> str:
    """Возвращает строку с информацией об отправителе."""
    user = message.from_user
    if user:
        full_name = user.full_name or "Unknown"
        username = f"@{user.username}" if user.username else "без username"
        return f"Пользователь: {full_name} (ID: {user.id}, {username})"
    return "Пользователь: неизвестен"
