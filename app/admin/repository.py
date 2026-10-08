"""AdminRepository: агрегации по LangGraph-checkpoints и таблицам аналитики.

Статистика активности берётся из LangGraph-checkpoints (таблица
`checkpoint_blobs`, канал `messages`, msgpack). Таблицы chat_messages /
chats заполняются только через /chats/* (команды бота), а основной диалог
идёт через /agent/stream и хранится в чекпоинтах.

PII-маскирование применяется только к экспорту, не к сторонним read-API.
Внутренние логи/UI продолжают видеть исходный контент (это полезно для
дебага). Если включить маскировку в /list_messages — пользователь увидит
свои же сообщения с [EMAIL] и не поймёт, что произошло.
"""

import logging
from app.services.thread_id import parse_chat_id
from datetime import UTC, datetime, timedelta
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from sqlalchemy import func, select, text

from app.admin.schemas import ExportItem, ExportResult, StatsOut
from app.chat.repositories.pg_models import RagQueryRow
from app.observability.pii import mask_pii

logger = logging.getLogger(__name__)
_serde = JsonPlusSerializer()


class AdminRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def compute_stats(self, window_hours: int = 24) -> StatsOut:
        """Активность за окно времени.

        Сообщения и активные пользователи — из LangGraph-checkpoints
        (см. `_agent_stats`). Feedback — из `message_feedback` (таблица
        создана моделью `MessageFeedbackRow`, пока пустая).
        """
        if self.session_factory is None:
            return StatsOut(total_messages=0, active_users=0)
        since = datetime.now(UTC) - timedelta(hours=window_hours)
        async with self.session_factory() as s:
            try:
                dau, messages = await self._agent_stats(s, since)
            except Exception as exc:
                logger.warning("compute_stats: agent_stats упал: %s", exc)
                dau, messages = 0, 0

            fb = await s.execute(
                text("""
                    SELECT
                        COUNT(*) FILTER (WHERE value='up') AS up,
                        COUNT(*) FILTER (WHERE value='down') AS down
                    FROM message_feedback
                    WHERE created_at >= :since
                """),
                {"since": since},
            )
            row = fb.first()
            up = (row.up if row else 0) or 0
            down = (row.down if row else 0) or 0
            total_fb = up + down
            ratio = up / total_fb if total_fb > 0 else 0.0
            negative_rate = down / total_fb if total_fb > 0 else 0.0

            rag = (
                await s.execute(
                    text(
                        """
                        SELECT
                            COUNT(*) AS total,
                            COUNT(*) FILTER (WHERE confident = false) AS refused
                        FROM rag_queries
                        WHERE created_at >= :since
                        """
                    ),
                    {"since": since},
                )
            ).first()
            rag_total = (rag.total if rag else 0) or 0
            refused = (rag.refused if rag else 0) or 0
            refusal_rate = refused / rag_total if rag_total > 0 else 0.0

        gaps = await self.knowledge_gaps(limit=10)
        return StatsOut(
            total_messages=messages or 0,
            active_users=dau or 0,
            feedback_ratio=ratio,
            refusal_rate=refusal_rate,
            negative_feedback_rate=negative_rate,
            knowledge_gaps=gaps,
        )

    async def _agent_stats(
        self, session, since: datetime
    ) -> tuple[int, int]:
        """Возвращает (уникальных пользователей, сообщений) из checkpoints.

        Blob'ы `messages` десериализуются через `JsonPlusSerializer` —
        тот же сериализатор, которым LangGraph их писал (msgpack с
        extensions для Pydantic-моделей BaseMessage).
        """
        stmt = text("""
            SELECT DISTINCT ON (cb.thread_id)
                cb.thread_id,
                cb.type,
                cb.blob
            FROM checkpoint_blobs cb
            JOIN checkpoints cp
              ON cp.thread_id = cb.thread_id
             AND cp.checkpoint_ns = cb.checkpoint_ns
            WHERE cb.channel = 'messages'
              AND cb.checkpoint_ns = ''
              AND (cp.checkpoint->>'ts')::timestamptz >= :since
            ORDER BY cb.thread_id, cb.version DESC
        """)
        result = await session.execute(stmt, {"since": since})
        rows = result.all()

        users: set[str] = set()
        total_messages = 0
        for row in rows:
            try:
                data = _serde.loads_typed((row.type, bytes(row.blob)))
            except Exception as exc:
                logger.warning(
                    "agent_stats: не удалось распарсить blob %s (%s): %s",
                    row.thread_id, row.type, exc,
                )
                continue
            if isinstance(data, (list, tuple)) and data:
                chat_id = parse_chat_id(row.thread_id)
                if chat_id:
                    users.add(chat_id)
                total_messages += len(data)

        return len(users), total_messages
    
    async def log_rag_query(
        self, question: str, confident: bool, top_score: float
    ) -> None:
        """Пишет строку лога RAG-запроса. Нормализуем вопрос для группировки пробелов."""
        if self.session_factory is None:
            return
        async with self.session_factory() as s:
            s.add(
                RagQueryRow(
                    question_normalized=question.strip().lower()[:500],
                    confident=confident,
                    top_score=top_score,
                )
            )
            await s.commit()

    async def knowledge_gaps(self, limit: int = 10) -> list[str]:
        """Топ вопросов без уверенного ответа — что добавить в базу знаний.

        Тот же `select().group_by()`, что и весь чат-репозиторий, без сырого SQL.
        """
        if self.session_factory is None:
            return []
        stmt = (
            select(RagQueryRow.question_normalized)
            .where(RagQueryRow.confident.is_(False))
            .group_by(RagQueryRow.question_normalized)
            .order_by(func.count().desc())
            .limit(limit)
        )
        async with self.session_factory() as s:
            rows = (await s.execute(stmt)).scalars().all()
        return list(rows)

    async def list_owner_ids_by_interface(self, interface: str) -> list[int]:
        """Возвращает уникальные owner_external_id для рассылок.

        Для telegram owner_external_id — это str(message.chat.id), поэтому
        корректно интерпретируется как int. Owner'ы, чьи id не приводятся
        к int (другой интерфейс с не-числовым id) — пропускаются.
        """
        if self.session_factory is None:
            return []
        async with self.session_factory() as s:
            rows = (
                await s.execute(
                    text(
                        """
                        SELECT DISTINCT owner_external_id
                        FROM chats
                        WHERE interface = :i
                        """
                    ),
                    {"i": interface},
                )
            ).all()
        out: list[int] = []
        for r in rows:
            try:
                out.append(int(r.owner_external_id))
            except (TypeError, ValueError):
                continue
        return out

    async def export_messages(
        self, after: datetime | None, limit: int
    ) -> ExportResult:
        if self.session_factory is None:
            return ExportResult(items=[], next_after=None)
        async with self.session_factory() as s:
            stmt = text(
                """
                SELECT id, chat_id, role, content, created_at
                FROM chat_messages
                WHERE deleted_at IS NULL
                  AND (CAST(:after AS TIMESTAMPTZ) IS NULL
                       OR created_at > CAST(:after AS TIMESTAMPTZ))
                ORDER BY created_at ASC
                LIMIT :limit
                """
            )
            rows = (
                await s.execute(stmt, {"after": after, "limit": limit})
            ).all()
        items = [
            ExportItem(
                id=str(r.id),
                chat_id=str(r.chat_id),
                role=r.role,
                content=mask_pii(r.content),
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]
        return ExportResult(
            items=items,
            next_after=rows[-1].created_at if rows else None,
        )