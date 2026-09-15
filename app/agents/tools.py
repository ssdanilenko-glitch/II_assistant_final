"""Инструменты агента.

`multiply` — простой самодостаточный инструмент. `search_knowledge_base` — RAG
как инструмент: обёртка над корпоративной базой знаний. Поиск инжектируется
как async-callable, чтобы инструмент не зависел от инициализации RAG-сервиса
напрямую и легко подменялся в тестах.
"""

from collections.abc import Awaitable, Callable

from langchain_core.tools import BaseTool, StructuredTool, tool


def build_search_knowledge_base(
    search_fn: Callable[[str], Awaitable[dict]],
) -> BaseTool:
    """Собирает инструмент поиска по базе знаний поверх переданного `search_fn`.

    `search_fn(query)` возвращает контракт RAG-сервиса
    `{answer, sources[id, file_name, ...], confident, ...}`.
    """

    async def _search_knowledge_base(query: str) -> str:
        result = await search_fn(query)
        answer = result.get("answer", "")
        confident = result.get("confident", True)
        sources = result.get("sources", [])

        # Если RAG не уверен или вернул пустоту — не отдаём список источников,
        # иначе LLM начнёт их перечислять и пересказывать.
        if not answer or not confident:
            return (
                "В базе знаний нет ответа на этот вопрос. "
                "Не перечисляй источники и не пересказывай их содержимое. "
                "Сообщи пользователю, что информация не найдена."
            )

        cited = ", ".join(
            f"[{s.get('id')}] {s.get('file_name', '')}".strip() for s in sources
        )
        return f"{answer}\nИсточники: {cited}"

    return StructuredTool.from_function(
        coroutine=_search_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Ищет ответ в корпоративной базе знаний по текстовому запросу. "
            "Вызывать, когда нужен факт из документов компании."
        ),
    )


@tool
def multiply(a: int, b: int) -> int:
    """Умножить два числа. Демонстрационный инструмент."""
    return a * b