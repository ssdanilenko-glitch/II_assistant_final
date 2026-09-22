"""Тонкий async-клиент к chat-сервису."""
import json
import logging
import uuid
from collections.abc import AsyncIterator
from uuid import UUID

import httpx

logger = logging.getLogger("it_assistant")


class BackendClient:
    def __init__(self, http: httpx.AsyncClient, admin_token: str = "") -> None:
        self.http = http
        self._admin_token = admin_token

    # --- chat operations -------------------------------------------------
    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        r = await self.http.post(
            "/chats",
            json={"owner_external_id": owner_external_id, "interface": interface},
            headers={"X-Owner-External-Id": owner_external_id},
        )
        r.raise_for_status()
        return UUID(r.json()["chat_id"])

    async def send_message(
        self, content: str, owner_external_id: str, thread_id: str | None = None
    ) -> AsyncIterator[dict]:
        logger.info(f"[send_message] content='{content[:50]}...', owner={owner_external_id}")
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        url = "/agent/stream"
        payload = {
            "thread_id": thread_id,
            "input": {
                "messages": [{"role": "user", "content": content}],
                "iteration_count": 0,
                "tool_results": [],
                "draft": None,
                "sent": False,
            },
        }

        async with self.http.stream(
            "POST", url, json=payload, headers={"X-Owner-External-Id": owner_external_id}
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                    etype = event.get("type")

                    if etype == "done":
                        return

                    # ✅ НОВОЕ: Обработка update от узлов графа (например, prepare_email)
                    if etype == "update":
                        for m in event.get("messages", []) or []:
                            text = (m.get("text") or "").strip()
                            if text:
                                yield {"type": "assistant_text", "text": text}

                    if etype == "token":
                        yield {"type": "token", "delta": event.get("text", "")}

                    if etype == "interrupt":
                        yield {
                            "type": "interrupt",
                            "thread_id": thread_id,
                            "payload": event.get("payload", {}),
                        }
                        return
                except json.JSONDecodeError:
                    pass

    async def clear_messages(self, chat_id: UUID, owner_external_id: str | None = None) -> None:
        headers = {"X-Owner-External-Id": owner_external_id} if owner_external_id else {}
        r = await self.http.delete(f"/chats/{chat_id}/messages", headers=headers)
        r.raise_for_status()

    # --- feedback --------------------------------------------------------
    async def post_feedback(self, chat_id: UUID, message_id: str, owner_external_id: str, value: str) -> None:
        r = await self.http.post(
            f"/chats/{chat_id}/messages/{message_id}/feedback",
            json={"owner_external_id": owner_external_id, "value": value},
            headers={"X-Owner-External-Id": owner_external_id},
        )
        r.raise_for_status()

    # --- admin -----------------------------------------------------------
    def _admin_headers(self) -> dict[str, str]:
        return {"X-Admin-Token": self._admin_token}

    async def get_admin_stats(self, window_hours: int = 24) -> dict:
        r = await self.http.get("/chats/admin/stats", params={"window_hours": window_hours}, headers=self._admin_headers())
        r.raise_for_status()
        return r.json()

    async def broadcast(self, text: str, interface: str = "telegram") -> dict:
        r = await self.http.post("/chats/admin/broadcast", json={"text": text, "interface": interface}, headers=self._admin_headers())
        r.raise_for_status()
        return r.json()

    async def set_handoff_status(self, owner_external_id: str, status: str, interface: str = "telegram") -> dict:
        r = await self.http.post("/chats/admin/handoff", json={"owner_external_id": owner_external_id, "interface": interface, "status": status}, headers=self._admin_headers())
        r.raise_for_status()
        return r.json()

    async def fetch_pending_alerts(self) -> list[dict]:
        r = await self.http.get("/chats/admin/alerts", headers=self._admin_headers())
        r.raise_for_status()
        return r.json()

    async def ack_alert(self, alert_id: int) -> None:
        r = await self.http.post(f"/chats/admin/alerts/{alert_id}/ack", headers=self._admin_headers())
        r.raise_for_status()

    async def resume(self, thread_id: str, decision: bool | str, owner_external_id: str) -> AsyncIterator[dict]:
        logger.info(f"[resume] thread_id={thread_id}, decision={decision}")
        url = "/agent/stream"
        payload = {"thread_id": thread_id, "resume": decision}

        async with self.http.stream(
            "POST", url, json=payload, headers={"X-Owner-External-Id": owner_external_id}
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                    etype = event.get("type")

                    if etype == "done":
                        return

                    # ✅ НОВОЕ: Обработка update в resume (на случай системных сообщений после подтверждения)
                    if etype == "update":
                        for m in event.get("messages", []) or []:
                            text = (m.get("text") or "").strip()
                            if text:
                                yield {"type": "assistant_text", "text": text}

                    if etype == "token":
                        yield {"type": "token", "delta": event.get("text", "")}
                except json.JSONDecodeError:
                    pass

    # --- lifecycle -------------------------------------------------------
    async def aclose(self) -> None:
        await self.http.aclose()

    async def clear_agent_thread(self, thread_id: str) -> None:
        r = await self.http.delete(f"/agent/thread/{thread_id}")
        r.raise_for_status()