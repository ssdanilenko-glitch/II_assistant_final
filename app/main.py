import asyncio
import logging
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.admin.routes import router as admin_router
from app.chat.repositories.pg_models import Base
from app.chat.routes import router as chat_router
from app.core.config import get_settings
from app.core.exceptions import (
    LLMAuthError,
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.observability import setup_tracing
from app.prompts.loader import build_system_prompt
from app.routers import agent, chat, documents, health, models, rag
from app.routers import media as media_router
from app.services.email_service import get_email_service

logger = logging.getLogger("it_assistant")
logging.basicConfig(level=logging.INFO)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Применяем создание таблиц (если их нет) ===
    try:
        # Создаём engine и создаём все таблицы, которых ещё нет
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Таблицы созданы (или уже существуют)")
        app.state.async_engine = engine
        app.state.session_factory = async_sessionmaker(engine, expire_on_commit=False)
    except Exception as e:
        logger.error("Ошибка при создании таблиц: %s", e)
        app.state.async_engine = None
        app.state.session_factory = None

    # === Phoenix-трейсинг ===
    app.state.tracing_enabled = setup_tracing(settings)

    # === LLM Client ===
    app.state.llm = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        base_url=settings.llm.base_url,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )

    # === Redis ===
    app.state.redis = None
    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        app.state.redis = redis_client
    except Exception as e:
        logger.warning("Redis недоступен (%s) — продолжаем без кеша", e)

    # === VectorStore, RAG, Agent (как было) ===
    app.state.vector_store = None
    app.state.embedding_client = None
    app.state.ingestion_service = None
    app.state.rag_service = None
    try:
        from app.services.ingestion import IngestionService
        from app.services.rag import RAGService

        ingestion = IngestionService(settings)
        app.state.ingestion_service = ingestion

        if ingestion.is_collection_empty():
            await asyncio.to_thread(ingestion.ingest_all)

        rag_service = RAGService(settings)
        await asyncio.to_thread(rag_service.build)
        app.state.rag_service = rag_service
        logger.info("RAG-сервис готов")
    except Exception as e:
        logger.warning("RAG/индексация не инициализированы (%s)", e)

    app.state.agent_graph = None
    agent_stack = AsyncExitStack()
    try:
        from langchain_openai import ChatOpenAI

        from app.agents.tools import build_search_knowledge_base
        from app.services.agent_persistent import agent_lifespan

        agent_model = ChatOpenAI(
            model=settings.llm.default_model,
            base_url=settings.llm.base_url,
            temperature=0,
            api_key=settings.llm.openai_api_key.get_secret_value(),
            timeout=settings.llm.request_timeout,
        )

        async def _search_kb(query: str) -> dict:
            if app.state.rag_service is None:
                return {"answer": "База знаний недоступна.", "sources": [], "confident": False}
            return await app.state.rag_service.answer(query)

        async def _send_email(draft: dict) -> None:
            logger.info(f"📧 ПОПЫТКА ОТПРАВКИ ПИСЬМА → {draft.get('to')}: {draft.get('subject')}")

            email_service = get_email_service()
            to = draft.get("to")
            subject = draft.get("subject", "")
            body = draft.get("body", "")

            if not to:
                logger.error("Не указан получатель")
                return

            # Загружаем вложения из Redis, если есть thread_id
            attachments = []
            thread_id = draft.get("thread_id")
            if thread_id and app.state.redis is not None:
                try:
                    redis_key = f"media:{thread_id}"
                    raw = await app.state.redis.get(redis_key)
                    if raw:
                        import json
                        files_data = json.loads(raw)  # список словарей с filename, data_base64, mime
                        for item in files_data:
                            import base64
                            data = base64.b64decode(item["data"])
                            attachments.append((item["filename"], data, item.get("mime", "application/octet-stream")))
                        # Удаляем ключ после загрузки
                        await app.state.redis.delete(redis_key)
                        logger.info(f"Загружено {len(attachments)} вложений для письма")
                except Exception as e:
                    logger.error(f"Ошибка загрузки вложений из Redis: {e}")

            success = await email_service.send_message(
                subject=subject,
                body=body,
                recipient=to,
                is_html=False,
                attachments=attachments if attachments else None
            )

            if not success:
                logger.error(f"[SEND_EMAIL]  Failed to send to {to}")
            else:
                logger.info("[SEND_EMAIL] ✅ Email sent successfully")

        agent_tools = [build_search_knowledge_base(_search_kb)]
        system_prompt =  build_system_prompt("")
        app.state.agent_graph = await agent_stack.enter_async_context(
            agent_lifespan(
                settings.agent_checkpointer,
                agent_model,
                agent_tools,  # ✅ третий аргумент – список инструментов
                _send_email,  # ✅ четвёртый аргумент – функция отправки
                sqlite_path=settings.agent_sqlite_path,
                postgres_url=settings.database_url,
                system_prompt=system_prompt,
            )
        )
        logger.info("Персистентный агент собран")
    except Exception as e:
        app.state.agent_graph = None
        logger.warning("Агентный граф не собран: %s", e)

    yield

    # === Clean-up ===
    await agent_stack.aclose()
    try:
        await app.state.llm.close()
    except Exception:
        logger.exception("ошибка при закрытии LLM-клиента")

    if app.state.redis is not None:
        try:
            await app.state.redis.close()
        except Exception:
            logger.exception("ошибка при закрытии Redis")

    if app.state.async_engine is not None:
        try:
            await app.state.async_engine.dispose()
        except Exception:
            logger.exception("ошибка при остановке engine Postgres")

    if app.state.rag_service is not None:
        try:
            await app.state.rag_service.close()
        except Exception:
            logger.exception("ошибка при закрытии RAG-сервиса")

    if app.state.ingestion_service is not None:
        try:
            app.state.ingestion_service.close()
        except Exception:
            logger.exception("ошибка при закрытии индексатора")



app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="FastAPI-сервис для LLM с кешированием, стримингом и модерацией",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-LLM-Cost-USD"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.llm_cost = 0.0
    request.state.llm_tokens = 0

    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled", extra={"request_id": request.state.request_id})
        raise

    duration_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-LLM-Cost-USD"] = f"{request.state.llm_cost:.6f}"
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request.state.request_id,
    )
    return response


_STATUS_MAP: list[tuple[type[LLMError], int, str]] = [
    (LLMRateLimitError, 429, "llm_rate_limit"),
    (LLMAuthError, 502, "llm_auth_error"),
    (LLMTimeoutError, 504, "llm_timeout"),
    (LLMContentFilterError, 400, "content_filter"),
    (LLMError, 502, "llm_error"),
]


@app.exception_handler(LLMError)
async def handle_llm_error(request: Request, exc: LLMError):
    for cls, status, code in _STATUS_MAP:
        if isinstance(exc, cls):
            return JSONResponse(
                status_code=status,
                content={"error": {"code": code, "message": str(exc)}},
                headers={"X-Request-ID": getattr(request.state, "request_id", "")},
            )
    return JSONResponse(
        status_code=502,
        content={"error": {"code": "llm_error", "message": str(exc)}},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation(request: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "validation_error", "fields": errors}},
        headers={"X-Request-ID": getattr(request.state, "request_id", "")},
    )


app.include_router(chat.router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(models.router)
app.include_router(health.router)
app.include_router(rag.router)
app.include_router(documents.router)
app.include_router(agent.router)
app.include_router(media_router.router)
