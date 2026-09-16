import re
from collections.abc import Awaitable, Callable

from langchain_core.tools import BaseTool, tool


@tool
def multiply(a: int, b: int) -> int:
    """Перемножает два целых числа. Вызывать для любого умножения."""
    return a * b


def build_search_knowledge_base(
    search_fn: Callable[[str], Awaitable[dict]],
) -> BaseTool:
    """Собирает инструмент поиска по базе знаний поверх переданного `search_fn`."""

    @tool
    async def search_knowledge_base(query: str) -> str:
        """Ищет ответ в корпоративной базе знаний по текстовому запросу.

        Вызывать, когда нужен факт из документов компании. Не вызывать для
        арифметики или общих знаний, которые модель знает сама.
        """
        result = await search_fn(query)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        confident = result.get("confident", True)

        if not confident or not answer or answer.startswith("В базе знаний"):
            return "В базе знаний нет ответа на этот вопрос."

        # Убираем inline-цитаты [1 — file.pdf] из тела ответа.
        answer_clean = re.sub(
            r"\s*\[\d+(?:\s*[—–-]\s*[^\]]+)?\]", "", answer
        ).strip()

        # Страховка 1: если LLM продублировала ответ, первая строка встречается
        # дважды. Обрезаем по второму вхождению первой значимой строки.
        lines = answer_clean.split("\n")
        first_line = ""
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 20:
                first_line = stripped
                break
        if first_line:
            idx = answer_clean.find(first_line, 1)
            if idx > 0:
                answer_clean = answer_clean[:idx].rstrip(" \n.,;:-—")

        # Страховка 2: если случайно два блока «Источники:» — оставляем первый.
        parts = re.split(r"\n\s*Источники\s*:", answer_clean, maxsplit=1)
        body = parts[0].strip()

        numbered = "\n".join(
            f"{i}. {s.get('file_name', '')}".strip()
            for i, s in enumerate(sources, start=1)
            if s.get("file_name")
        )
        if not numbered:
            return body
        return f"{body}\n\nИсточники:\n{numbered}"

    return search_knowledge_base