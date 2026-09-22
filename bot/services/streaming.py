"""Стриминг ответа агента в Telegram.

Обрабатывает SSE-события от `/agent/stream`:
- `assistant_text` — текст от узлов графа;
- `interrupt` — граф встал на HIL, ждёт resume.

Токены LLM не обрабатываются: модель вызывается через `ainvoke` и полный
текст приходит одним событием `update`.
"""

import logging

logger = logging.getLogger("it_assistant")


async def stream_to_chat(message, events, chat_id=None):
    interrupt_data: dict | None = None
    seen_texts: set[str] = set()

    async for event in events:
        etype = event.get("type")

        if etype == "assistant_text":
            text = (event.get("text") or "").strip()
            if text and text not in seen_texts:
                try:
                    await message.answer(text)
                    seen_texts.add(text)
                except Exception:
                    logger.exception("assistant_text send failed")

        elif etype == "interrupt":
            interrupt_data = {
                "thread_id": event.get("thread_id"),
                "payload": event.get("payload", {}),
            }
            break

    if interrupt_data:
        return {"status": "interrupt", **interrupt_data}
    return {"status": "done"}