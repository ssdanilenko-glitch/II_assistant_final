"""Трейсинг в Phoenix через OpenInference (опциональный runtime-путь).

Включается флагом `PHOENIX_ENABLED=true` и группой зависимостей `tracing`
(`uv sync --extra tracing`). По умолчанию выключено — сервис поднимается
без трейсинга, спаны не пишутся.

Инструменторы подключаются один раз при старте (lifespan):
- LangChain/LangGraph — прогон агентного графа: каждый узел, вызовы инструментов
  с input/output, вызовы LLM и `__interrupt__` становятся дочерними спанами.
- LlamaIndex — вызовы RAG (retrieve, embed, LLM) попадают в спаны автоматически,
  но только если версия инструментатора совместима с установленным llama-index-core.
  При несовместимости инструментатор пропускается с warning, приложение стартует.
"""

import logging

from app.core.config import Settings

logger = logging.getLogger(__name__)


def _try_instrument_llama(provider) -> bool:
    """Пробует подключить LlamaIndex-инструментатор. Возвращает True при успехе."""
    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    except ImportError:
        logger.info("LlamaIndex instrumentor не установлен — пропускаю")
        return False

    try:
        LlamaIndexInstrumentor().instrument(tracer_provider=provider)
    except Exception as exc:
        # Известная проблема: в llama-index-core 0.14+ переехал путь
        # llama_index.core.base.agent.types — старые версии инструментатора
        # (>=3,<4) его не находят. RAG-спаны не критичны для работы сервиса.
        logger.warning(
            "LlamaIndex instrumentor failed (RAG-спаны будут недоступны): %s", exc
        )
        return False

    logger.info("LlamaIndex инструментатор подключён")
    return True


def _try_instrument_langchain(provider) -> bool:
    """Пробует подключить LangChain/LangGraph-инструментатор."""
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError:
        logger.info("LangChain instrumentor не установлен — пропускаю")
        return False

    try:
        # LangGraph построен на LangChain-runnable'ах, поэтому этот же инструментатор
        # покрывает узлы графа, вызовы инструментов и LLM.
        LangChainInstrumentor().instrument(tracer_provider=provider)
    except Exception as exc:
        logger.warning("LangChain instrumentor failed: %s", exc)
        return False

    logger.info("LangChain/LangGraph инструментатор подключён")
    return True


def setup_tracing(settings: Settings) -> bool:
    """Регистрирует инструментаторы RAG и агентного графа → Phoenix.

    Возвращает True, если трейсинг включён и хотя бы один инструментатор поднят.
    Никогда не бросает исключений — падение трейсинга не должно валить приложение.
    """
    if not settings.phoenix_enabled:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning(
            "phoenix_enabled=true, но opentelemetry-пакеты не установлены — "
            "uv sync --extra tracing (%s)",
            exc,
        )
        return False

    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint)
        )
    )
    trace.set_tracer_provider(provider)

    instrumented: list[str] = []
    if _try_instrument_langchain(provider):
        instrumented.append("LangChain/LangGraph")
    if _try_instrument_llama(provider):
        instrumented.append("LlamaIndex")

    if not instrumented:
        logger.warning(
            "Трейсинг включён, но ни один инструментатор не поднялся. "
            "Проверьте совместимость версий в --extra tracing."
        )
        return False

    logger.info(
        "Phoenix-трейсинг включён (%s): %s",
        ", ".join(instrumented),
        settings.phoenix_collector_endpoint,
    )
    return True