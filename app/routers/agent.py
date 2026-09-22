"""Ручки агентного слоя: прогон персистентного ReAct-графа с HIL.
`POST /agent/chat` — один шаг диалога; если агент дошёл до опасного действия,
вернётся `status="interrupted"` с payload для подтверждения.
`POST /agent/resume` — возобновление после подтверждения человеком.
`POST /agent/stream` — SSE-поток прогресса по узлам и токенов LLM.
"""
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel
from app.deps.providers import AgentGraphDep

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger("it_assistant")


def _config(thread_id: str, user_role: str = "write-with-approve") -> dict:
    return {"configurable": {"thread_id": thread_id, "user_role": user_role}}


def _initial_state(message: str) -> dict:
    return {
        "messages": [HumanMessage(message)],
        "iteration_count": 0,
        "tool_results": [],
        "draft": None,
        "sent": False,
    }


class AgentChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class AgentChatResponse(BaseModel):
    status: str  # "done" | "interrupted"
    thread_id: str
    answer: str | None = None
    tool_results: list[dict] = []
    interrupt: dict | None = None


def _to_response(result: dict, thread_id: str) -> AgentChatResponse:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return AgentChatResponse(
            status="interrupted",
            thread_id=thread_id,
            interrupt=payload,
            tool_results=result.get("tool_results", []),
        )
    final = result["messages"][-1]
    return AgentChatResponse(
        status="done",
        thread_id=thread_id,
        answer=final.content or "",
        tool_results=result.get("tool_results", []),
    )


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest, graph: AgentGraphDep) -> AgentChatResponse:
    if graph is None:
        raise HTTPException(status_code=503, detail="агентный граф не инициализирован")
    result = await graph.ainvoke(_initial_state(req.message), _config(req.thread_id))
    return _to_response(result, req.thread_id)


class AgentResumeRequest(BaseModel):
    thread_id: str
    decision: bool | str = True


@router.post("/resume", response_model=AgentChatResponse)
async def agent_resume(
        req: AgentResumeRequest, graph: AgentGraphDep
) -> AgentChatResponse:
    if graph is None:
        raise HTTPException(status_code=503, detail="агентный граф не инициализирован")
    result = await graph.ainvoke(Command(resume=req.decision), _config(req.thread_id))
    return _to_response(result, req.thread_id)


class AgentStreamRequest(BaseModel):
    thread_id: str
    input: dict | None = None  # старт: {"messages": [...]}
    resume: bool | str | None = None  # возобновление после interrupt


def _format_event(stream_type: str, payload: Any) -> dict | None:
    if stream_type == "updates":
        if isinstance(payload, dict) and "__interrupt__" in payload:
            interrupts = payload["__interrupt__"]
            value = interrupts[0].value if interrupts else {}
            return {"type": "interrupt", "payload": value}

        new_messages: list[dict] = []
        for node_name, node_payload in payload.items():
            if not isinstance(node_payload, dict):
                continue
            for m in node_payload.get("messages", []) or []:
                # ToolMessage пользователю не показываем
                mtype = m.get("type", "") if isinstance(m, dict) else getattr(m, "type", "")
                if mtype == "tool":
                    continue
                text = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                if text:
                    new_messages.append({"role": "assistant", "text": text})

        if new_messages:
            return {"type": "update", "nodes": list(payload.keys()), "messages": new_messages}

    if stream_type == "messages":
        chunk, _meta = payload
        from langchain_core.messages import AIMessage
        if not isinstance(chunk, AIMessage):
            return None
        text = getattr(chunk, "content", "") or ""
        return {"type": "token", "text": text} if text else None
    return None

@router.post("/stream")
async def agent_stream(
        req: AgentStreamRequest, graph: AgentGraphDep
) -> StreamingResponse:
   # logger.warning("AGENT STREAM HANDLER v=NEXT thread=%s", req.thread_id)  # ← добавить
    if graph is None:
        raise HTTPException(status_code=503, detail="агентный граф не инициализирован")

    if req.resume is not None:
        graph_input: Any = Command(resume=req.resume)
    elif req.input is not None:
        from langchain_core.messages import (
            AIMessage, HumanMessage, SystemMessage,
        )
        raw = req.input.get("messages") or []
        converted: list = []
        for m in raw:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "user":
                    converted.append(HumanMessage(content=content))
                elif role == "assistant":
                    converted.append(AIMessage(content=content))
                elif role == "system":
                    converted.append(SystemMessage(content=content))
            else:
                converted.append(m)

        graph_input = {
            "iteration_count": 0,
            "tool_results": [],
            "draft": None,
            "sent": False,
            **{k: v for k, v in req.input.items() if k != "messages"},
            "messages": converted,
        }
    else:
        raise HTTPException(status_code=422, detail="нужен input или resume")

    config = _config(req.thread_id)

    async def event_source() -> AsyncIterator[str]:
        logger.info("SSE START thread=%s input_type=%s",
                    req.thread_id, type(graph_input).__name__)
        try:
            async for stream_type, payload in graph.astream(
                    graph_input, config, stream_mode=["updates"]
            ):
                logger.info(
                    "SSE raw type=%s keys=%s",
                    stream_type,
                    list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
                )

                if (
                        stream_type == "updates"
                        and isinstance(payload, dict)
                        and "__interrupt__" in payload
                ):
                    nodes_payload = {k: v for k, v in payload.items() if k != "__interrupt__"}
                    if nodes_payload:
                        ev = _format_event("updates", nodes_payload)
                        if ev is not None:
                            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    ev = _format_event("updates", {"__interrupt__": payload["__interrupt__"]})
                    if ev is not None:
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    continue

                event = _format_event(stream_type, payload)
                if event is None:
                    logger.debug("SSE skip type=%s", stream_type)
                else:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("SSE FAILED thread=%s", req.thread_id)
        yield 'data: {"type": "done"}\n\n'
        logger.info("SSE END thread=%s", req.thread_id)
    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
@router.delete("/thread/{thread_id}")
async def delete_thread(thread_id: str, graph: AgentGraphDep) -> dict:
    """Удаляет состояние (чекпоинт) для указанного thread_id."""
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent graph not available")
    try:
        if not hasattr(graph.checkpointer, "delete_thread"):
            raise HTTPException(status_code=501, detail="Checkpointer does not support deletion")
        if hasattr(graph.checkpointer, "adelete_thread"):
            await graph.checkpointer.adelete_thread(thread_id)
        else:
            await graph.checkpointer.delete_thread(thread_id)
        logger.info(f"Thread {thread_id} deleted successfully")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to delete thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete thread: {e!s}") from e