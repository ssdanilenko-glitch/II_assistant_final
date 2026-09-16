from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Настройки эмбеддингов (HuggingFace BGE-M3)."""
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    trust_remote_code: bool = True
    dim: int = 1024


class LLMSettings(BaseSettings):
    """Настройки LLM (Ollama / OpenAI-совместимый API)."""
    model_config = SettingsConfigDict(env_prefix="LLM_")

    openai_api_key: SecretStr = SecretStr("ollama")
    base_url: str = "http://ollama:11434/v1"
    default_model: str = "frob/qwen3.5-instruct:4b"
    request_timeout: float = 180.0
    max_retries: int = 3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ===== Приложение =====
    app_name: str = "it-assistant"
    debug: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ===== Redis =====
    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = 86400

    # ===== LLM =====
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # ===== Чат (хранение) =====
    database_url: str = "postgresql+asyncpg://chat:chat@localhost:5432/chat"
    chat_repository: Literal["json", "postgres"] = "json"
    chat_storage_dir: Path = Path("./var/chats")
    chat_context_window: int = 10

    # ===== Безопасность и администрирование =====
    admin_token: SecretStr = SecretStr("change-me-admin")
    internal_token: SecretStr = SecretStr("change-me-internal")
    bot_url: str = "http://bot:9000"
    admin_chat_id: int | None = None
    moderation_use_openai: bool = True
    rate_limit_messages_per_min: int = 15

    # ===== Qdrant =====
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    # ===== Эмбеддинги =====
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    # ===== RAG =====
    rag_data_dir: Path = Path("data/rag-block-03")
    rag_collection: str = "rag_block_04"
    rag_collection_bare: str = "rag_block_04_bare"
    rag_top_k: int = 3
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_score_threshold: float = 0.55
    rag_retrieve_top_k: int = 10
    rag_rerank_top_n: int = 5
    rag_use_reranker: bool = False
    rag_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rag_use_hybrid: bool = False
    rag_sparse_model: str = "Qdrant/bm25"
    rag_restrict_to_internal: bool = False

    # ===== Агент и чекпоинтер =====
    agent_checkpointer: Literal["memory", "sqlite", "postgres"] = "postgres"
    agent_sqlite_path: str = "var/agent_checkpoints.sqlite"

    # ===== Phoenix-трейсинг =====
    phoenix_enabled: bool = False
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"

    # ===== Оценка качества (RAGAS) =====
    anthropic_api_key: SecretStr | None = None
    eval_judge_provider: Literal["anthropic", "openai"] = "openai"
    eval_judge_model: str = "gpt-4o-mini"
    EVAL_JUDGE_URL: str = "https://api.proxyapi.ru/openai/v1"

    # ===== Telegram =====
    BOT_TOKEN: str | None = None
    API_BASE_URL: str = "http://app:8000"
    TELEGRAM_EDIT_DEBOUNCE_MS: int = 800

    # ===== Email (SMTP) =====
    yandex_email: str = ""
    yandex_app_password: SecretStr = SecretStr("")
    exchange_recipient_email: str = ""
    SMTP_HOST: str = "smtp.yandex.ru"
    SMTP_PORT: int = 465

    # ===== Ollama =====
    ollama_url: str = "http://ollama:11434"

    # ===== Vision-модель (описание изображений) =====
    vision_model: str = "frob/qwen3.5-instruct:4b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
