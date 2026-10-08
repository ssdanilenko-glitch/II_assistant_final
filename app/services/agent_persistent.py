"""Персистентный ReAct-агент: чекпоинтер + человек в цикле на опасном действии.

Инкремент к базовому ReAct-агенту: тот же цикл, но граф компилируется с
чекпоинтером (состояние переживает рестарт), а опасный инструмент `send_email`
проходит через человека — два узла:

- `prepare_email` — idempotent: рендерит payload письма из tool_call, без side-effect;
- `confirm_and_send` — `interrupt()` перед отправкой, реальная отправка ТОЛЬКО после
  `Command(resume=...)`. При роли `full` interrupt пропускается (политика доступа).

Бэкенд чекпоинтера выбирается через `AGENT_CHECKPOINTER`: `memory` | `sqlite` |
`postgres`. Схему чекпоинтера ведёт `setup()`, доменную — Alembic (в `env.py`
таблицы `checkpoint*` исключены из autogenerate через `include_name`).
"""

import operator
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langchain_core.messages import AIMessage
import uuid
import re

logger = logging.getLogger("it_assistant")
MAX_ITERATIONS = 6
DANGEROUS_TOOL = "send_email"
KEEP_LAST = 8
SUMMARY_EVERY = 10

# Реальный side-effect отправки: async-callable, инжектируется в фабрику, чтобы
# в тестах подменяться моком и вызываться ТОЛЬКО после одобрения человеком.
SendEmailFn = Callable[[dict], Awaitable[None]]


class PersistentAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]
    draft: dict | None
    sent: bool
    summary: str | None
    summary_upto: int

# Все невидимые/управляющие символы, которые могут прятаться внутри шапки
_INVISIBLE_RE = re.compile(
    r"[\ufeff\u200b\u200c\u200d\u200e\u200f"
    r"\u202a-\u202e\u2066-\u2069\u00a0]"
)

# Итоговый паттерн: первая буква П/P + «ользователь» + опциональные
# пробелы + двоеточие. IGNORECASE закрывает все варианты регистра.
_SENDER_RE = re.compile(r"^[ПPпp]ользователь\s*:", re.IGNORECASE)


def _looks_like_sender_line(line: str) -> bool:
    """Первая строка: «Пользователь: …» или «Pользователь: …».

    Терпима к невидимым символам (zero-width, bidi, NBSP) между буквами.
    Двоеточие обязательно — иначе это не шапка, а обычный текст.
    """
    if not line:
        return False
    cleaned = _INVISIBLE_RE.sub("", line).strip()
    return bool(_SENDER_RE.match(cleaned))

def _strip_sender_prefix(body: str) -> str:
    body = body.lstrip("\ufeff\u200b\u200e\u200f ")
    while True:
        line, sep, rest = body.partition("\n")
        if not _looks_like_sender_line(line):
            break
        body = rest.lstrip()
    return body

def _extract_sender_info(messages) -> str:
    for m in messages:
        mtype = getattr(m, "type", "")
        if mtype != "human":
            continue
        content = getattr(m, "content", "")
        if isinstance(content, str):
            first, _, _ = content.partition("\n")
            if _looks_like_sender_line(first):
                return first.strip()
    return ""
def _body_contains_sender(body: str) -> bool:
    if not body:
        return False
    head = _INVISIBLE_RE.sub("", body[:200]).lower()
    return "ользователь:" in head

SUMMARY_PROMPT = (
    "Сожми следующую переписку в 3–7 предложениях. Сохрани:\n"
    "- какие вопросы задавал пользователь;\n"
    "- что уже было отвечено;\n"
    "- какие действия предпринимались (заявки в HelpDesk, отправки писем).\n"
    "Не добавляй ничего от себя, только факты из переписки.\n\n"
)
async def _make_summary(
    model: BaseChatModel,
    messages: list[AnyMessage],
    previous_summary: str | None = None,
) -> str:
    """Дешёвый проход: тот же LLM, но без tool-calling."""
    chunks: list[str] = []
    if previous_summary:
        chunks.append(f"[Ранее: {previous_summary}]")
    for m in messages:
        mtype = getattr(m, "type", "")
        if mtype == "tool":
            continue  # tool-ответы не нужны в summary, они длинные
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            continue
        role = {"human": "П", "ai": "А"}.get(mtype, mtype)
        if content.strip():
            chunks.append(f"{role}: {content.strip()[:400]}")

    transcript = "\n".join(chunks)
    # Важно: вызываем на самом model без bind_tools, чтобы не было tool_call
    response = await model.ainvoke([
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=transcript or "(пусто)"),
    ])
    text = getattr(response, "content", "") or ""
    if isinstance(text, list):  # multimodal
        text = " ".join(
            (b.get("text", "") if isinstance(b, dict) else str(b))
            for b in text
        )
    return text.strip()

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Отправляет письмо клиенту. Опасное действие — требует подтверждения человека.

    Вызывать, когда нужно отправить готовый ответ или уведомление наружу.
    """
    # Тело напрямую не исполняется: граф перехватывает вызов и проводит его через
    # HIL-гейт (prepare_email -> confirm_and_send). Реальная отправка — после resume.
    return "queued-for-approval"


def _find_call(message: AnyMessage, name: str) -> dict:
    for call in message.tool_calls:
        if call["name"] == name:
            return call
    raise ValueError(f"в сообщении нет tool_call {name!r}")


def build_agent(
    checkpointer: Any,
    model: BaseChatModel,
    tools: list[BaseTool],
    send_email_fn: SendEmailFn,
    system_prompt: str | None = None,
):
    """Компилирует персистентный ReAct-граф с HIL-гейтом на `send_email`.

    `tools` — безопасные инструменты (search_knowledge_base,
    get_helpdesk_status). Опасный `send_email` добавляется здесь и исполняется
    не в `execute_tool`, а через отдельную ветку с `interrupt`.

    `system_prompt` — если передан, добавляется как SystemMessage перед каждым
    вызовом модели.
    """
    bound_model = model.bind_tools([*tools, send_email])
    tool_by_name = {t.name: t for t in tools}

    async def call_model(state: PersistentAgentState) -> dict:
        messages = state["messages"]
        if not messages:
            logger.warning("[call_model] пустой messages — пропускаю вызов LLM")
            return {
                "messages": [AIMessage(content="Сессия потеряна. Начните заново с /start.")],
                "iteration_count": state.get("iteration_count", 0) + 1,
            }
        if state.get("sent") and isinstance(messages[-1], AIMessage):
            logger.info("[call_model] sent=True, AIMessage уже добавлен — пропускаю LLM")
            return {
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        summary = state.get("summary")
        summary_upto = state.get("summary_upto", 0)

        if len(messages) <= KEEP_LAST + 1:
            head: list[AnyMessage] = []
            tail = messages
        else:
            head = [messages[0]]
            tail = messages[-KEEP_LAST:]

            need_summary = (
                    not summary
                    or (len(messages) - summary_upto) >= SUMMARY_EVERY
            )
            if need_summary:
                to_summarize = messages[1: len(messages) - KEEP_LAST]
                try:
                    summary = await _make_summary(
                        model, to_summarize, previous_summary=summary
                    )
                    summary_upto = len(messages) - KEEP_LAST
                    logger.info(f"[call_model] summary обновлён, покрыто до {summary_upto}")
                except Exception:
                    logger.exception("[call_model] не удалось сделать summary")

        parts: list[AnyMessage] = []
        sys_content = system_prompt or ""
        if summary:
            sys_content = f"{sys_content}\n\n[Ранее в диалоге]\n{summary}".strip()
        if sys_content:
            parts.append(SystemMessage(content=sys_content))

        head_ids = {getattr(m, "id", None) for m in head}
        for m in tail:
            if getattr(m, "id", None) not in head_ids:
                parts.append(m)
        logger.info(
            f"[call_model] state={len(messages)}, to_llm={len(parts)}, "
            f"summary={'есть' if summary else 'нет'}"
        )
        response = await bound_model.ainvoke(parts)
        return {
            "messages": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
            "summary": summary,
            "summary_upto": summary_upto,
        }

    async def execute_tool(state: PersistentAgentState) -> dict:
        last = state["messages"][-1]
        messages: list = []
        results: list[dict] = []
        for call in last.tool_calls:
            if call["name"] == DANGEROUS_TOOL:
                continue  # опасный инструмент идёт через HIL-ветку, не здесь
            if call["name"] not in tool_by_name:
                content = f"error: unknown tool '{call['name']}'"
            else:
                content = str(await tool_by_name[call["name"]].ainvoke(call["args"]))
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
            results.append(
                {"name": call["name"], "args": call["args"], "result": content}
            )
        return {"messages": messages, "tool_results": results}

    async def prepare_email(state: PersistentAgentState) -> dict:
        call = _find_call(state["messages"][-1], DANGEROUS_TOOL)
        args = call["args"]

        raw_body = args.get("body") or ""
        logger.info("[prepare_email] raw_body=%r", raw_body[:120])
        body = _strip_sender_prefix(raw_body)
        logger.info("[prepare_email] after_strip=%r", body[:120])

        sender_info = _extract_sender_info(state["messages"])
        if sender_info and not _body_contains_sender(body):
            body = f"{sender_info}\n\n{body}"

        draft = {
            "to": args.get("to", ""),
            "subject": args.get("subject", ""),
            "body": body,
            "sender_info": sender_info,
            "correlation_id": uuid.uuid4().hex[:12],
            "tool_call_id": call["id"],
        }
        return {"draft": draft}


    async def confirm_and_send(
            state: PersistentAgentState, config: RunnableConfig
    ) -> dict:
        # Защита от повторной отправки
        if state.get("sent", False):
            draft = state.get("draft") or {}
            return {
                "messages": [
                    ToolMessage(
                        content="Письмо уже было отправлено ранее.",
                        tool_call_id=draft.get("tool_call_id", ""),
                    )
                ],
                "tool_results": [
                    {"name": DANGEROUS_TOOL, "args": draft, "result": "already_sent"}
                ],
                "sent": True,
            }
        draft = state["draft"] or {}

        # Добавляем thread_id в draft
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            draft["thread_id"] = thread_id

        role = (config.get("configurable") or {}).get("user_role", "write-with-approve")
        if role == "full":
            decision: Any = True
        else:
            decision = interrupt({"type": "approve_email", "preview": draft})

        approved = decision is True or decision == "approve"

        logger.info(f"[confirm_and_send] decision={decision} (type={type(decision)}), approved={approved}")
        if approved:
            logger.info(f"[confirm_and_send] calling send_email_fn with draft: {draft}")
            await send_email_fn(draft)
            content = f"письмо отправлено: {draft.get('subject', '')}"
            confirmation = (
                f"✅ Заявка отправлена в HelpDesk.\n"
                f"Тема: {draft.get('subject', '')}"
            )
        else:
            content = "отправка отменена пользователем"
            confirmation = "❌ Отправка отменена."

        return {
            "sent": approved,
            "messages": [
                ToolMessage(content=content, tool_call_id=draft.get("tool_call_id", "")),
                AIMessage(content=confirmation),
            ],
            "tool_results": [
                {"name": DANGEROUS_TOOL, "args": draft, "result": content}
            ],
        }
    async def force_finish(state: PersistentAgentState) -> dict:
        logger.info("[force_finish] called")
        return {}

    def route_after_model(
        state: PersistentAgentState,
    ) -> Literal["execute_tool", "prepare_email", "force_finish"]:
        logger.info(
            f"[route_after_model] iteration={state.get('iteration_count', 0)}, tool_calls={getattr(state['messages'][-1], 'tool_calls', None) if state['messages'] else None}")
        if state.get("iteration_count", 0) >= MAX_ITERATIONS:
            return "force_finish"
        if state.get("sent", False):
            return "force_finish"
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None)
        if not calls:
            return "force_finish"
        if any(call["name"] == DANGEROUS_TOOL for call in calls):
            return "prepare_email"
        return "execute_tool"

    builder = StateGraph(PersistentAgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("prepare_email", prepare_email)
    builder.add_node("confirm_and_send", confirm_and_send)
    builder.add_node("force_finish", force_finish)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "execute_tool": "execute_tool",
            "prepare_email": "prepare_email",
            "force_finish": "force_finish",
        },
    )
    builder.add_edge("execute_tool", "call_model")
    builder.add_edge("prepare_email", "confirm_and_send")
    builder.add_edge("confirm_and_send", "call_model")
    #builder.add_edge("confirm_and_send", "force_finish")
    builder.add_edge("force_finish", END)
    return builder.compile(checkpointer=checkpointer)


def _psycopg_uri(database_url: str) -> str:
    """AsyncPostgresSaver работает на psycopg (v3): `postgresql://`, без `+asyncpg`."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://")


@asynccontextmanager
async def agent_lifespan(
    backend: Literal["memory", "sqlite", "postgres"],
    model: BaseChatModel,
    tools: list[BaseTool],
    send_email_fn: SendEmailFn,
    *,
    sqlite_path: str = "",
    postgres_url: str = "",
    system_prompt: str | None = None,
) -> AsyncIterator[Any]:
    """Поднимает нужный чекпоинтер и отдаёт скомпилированный граф.

    `setup()` вызывается ровно один раз здесь — не на каждый запрос.
    """
    if backend == "memory":
        # InMemorySaver не требует setup() и живёт в памяти процесса.
        yield build_agent(InMemorySaver(), model, tools, send_email_fn, system_prompt=system_prompt)
    elif backend == "sqlite":
        from pathlib import Path
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(sqlite_path) as saver:
            await saver.setup()
            logger.info(f"Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True) calling send_email_fn with draft: {send_email_fn}")
            yield build_agent(saver, model, tools, send_email_fn, system_prompt=system_prompt)
    elif backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(_psycopg_uri(postgres_url)) as saver:
            await saver.setup()
            yield build_agent(saver, model, tools, send_email_fn, system_prompt=system_prompt)

    else:
        raise ValueError(f"неизвестный AGENT_CHECKPOINTER: {backend!r}")