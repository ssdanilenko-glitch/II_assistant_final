"""Утилиты разбора thread_id формата tg-<chat_id>[-<uuid>]."""

import re

_THREAD_RE = re.compile(r"^tg-(\d+)(?:-[0-9a-f]+)?$")


def parse_chat_id(thread_id: str) -> str | None:
    """Возвращает числовой chat_id из thread_id или None, если формат не тот.

    Поддерживает оба варианта:
        tg-443426947
        tg-443426947-cd21bd08
    """
    m = _THREAD_RE.match(thread_id or "")
    return m.group(1) if m else None