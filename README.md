# ИИ-ассистент технической поддержки АО «УК «БМЗ»

## Общее описание проекта

Проект представляет собой **ИИ-ассистента технической поддержки**, предназначенного для автоматизации обработки обращений в службу УИТ и помощи сотрудникам первой линии сопровождения прикладных систем в решении вопросов 1С:ЕРП компании АО «УК «БМЗ». Это финальный дипломный проект, собранный по модулям 1–6 курса «ИИ-разработчик: от API до агентов».

Система построена как **RAG-агент** с инструментами, работающий поверх векторной базы знаний (Qdrant) и использующий локальную LLM (Ollama + `frob/qwen3.5-instruct:4b`) для генерации ответов на основе релевантных документов. Все ключевые компоненты (LLM, эмбеддинги, векторная база) работают on-premise. Внешний провайдер подключается только как судья RAGAS на
этапе оценки (`EVAL_JUDGE_*`) и как модератор при `MODERATION_USE_OPENAI=true`. Доступ к системе организован через **FastAPI** (основной сервис) и **Telegram-бота** (пользовательский интерфейс).

---

## Структура проекта

```
it_assistant_bmz/
├── app/                    # FastAPI-сервис (подробнее — см. ниже)
├── bot/                    # Telegram-бот (aiogram 3.x)
├── data/                   # Корпус документов (70 файлов)
├── docs/                   # Документация
├── migrations/             # Alembic
├── scripts/                # Утилиты (eval, ingest, verify)
├── tests/                  # Тесты и golden dataset
├── var/                    # Чекпоинты агента
├── compose.yaml            # Docker Compose (7 сервисов)
├── Dockerfile
├── .env.example
├── pyproject.toml
├── uv.lock
├── alembic.ini
└── README.md
```

### Подробное дерево `app/`

```
app/
├── __init__.py
├── main.py                              # Точка входа FastAPI
│
├── admin/                               # Административные маршруты
│   ├── __init__.py
│   ├── routes.py                        # /chats/admin/* (stats, broadcast, export)
│   ├── alerts.py                        # Хранилище алертов
│   ├── broadcaster.py                   # Серийная рассылка пользователям
│   ├── handoff.py                       # Переключение на оператора
│   └── notifier.py                      # Backend → bot /notify
│
├── agents/                              # Агентный слой (LangGraph)
│   ├── __init__.py
│   └── tools.py                         # build_search_knowledge_base
│
├── chat/                                # Логика чата (без агента)
│   ├── __init__.py
│   ├── context.py                       # Подсчёт токенов, fit_to_budget
│   ├── deps.py                          # DI для ChatService
│   ├── domain.py                        # Chat, ChatMessage, SystemPrompt
│   ├── media.py                         # Конвертация медиа в content-part
│   ├── repository.py                    # Protocol ChatRepository
│   ├── routes.py                        # /chats/*
│   ├── service.py                       # ChatService
│   └── repositories/
│       ├── __init__.py
│       ├── json_repo.py                 # JSON-хранилище
│       ├── pg_repo.py                   # Postgres-хранилище
│       └── pg_models.py                 # SQLAlchemy-модели (Chat, ChatMessage, …)
│
├── core/                                # Ядро приложения
│   ├── __init__.py
│   ├── config.py                        # Settings (Pydantic)
│   └── exceptions.py                    # LLMError, LLMAuthError, …
│
├── deps/                                # Глобальные DI
│   ├── __init__.py
│   └── providers.py                     # LLMDep, CacheDep, SessionFactoryDep, …
│
├── eval/                                # Оценка качества (RAGAS)
│   ├── __init__.py
│   └── metrics.py                       # build_judge, build_metrics, eval_row
│
├── moderation/                          # Модерация входящих
│   ├── __init__.py
│   ├── domain.py                        # ModerationResult
│   ├── regex_rules.py                   # Regex-блоклист
│   └── service.py                       # ModerationService (каскад)
│
├── observability/                       # Трейсинг
│   ├── __init__.py
│   └── tracing.py                       # setup_tracing (Phoenix/OTLP)
│
├── prompts/                             # Промпты для LLM и агента
│   ├── __init__.py
│   ├── loader.py                        # build_system_prompt
│   ├── System_prompt_agent.txt          # Промпт агента
│
├── ratelimit/                           # Rate limiting
│   ├── __init__.py
│   ├── dependencies.py                  # enforce_rate_limit
│   └── storage.py                       # Redis-реализация счётчиков
│
├── routers/                             # API-маршруты (кроме chat)
│   ├── __init__.py
│   ├── agent.py                         # /agent/chat, /agent/stream, /agent/resume
│   ├── chat.py                          # /chat (устаревший, обёртка LLMService)
│   ├── documents.py                     # /documents/* (управление корпусом)
│   ├── helpdesk.py                      # POST /webhook/helpdesk
│   ├── health.py                        # /health, /ready
│   ├── media.py                         # /media/process
│   ├── models.py                        # /models
│   └── rag.py                           # /rag/query
│
└── services/                            # Бизнес-логика
    ├── __init__.py
    ├── ingestion.py                     # Индексация в Qdrant (BGE-M3)
    ├── rag.py                           # RAGService (retrieve → rerank → synthesize)
    ├── rag_baremetal.py                 # Демо «RAG руками» (без LlamaIndex)
    ├── vector_store.py                  # Обёртка над Qdrant (async)
    ├── agent_persistent.py              # Персистентный ReAct-агент с HIL
    ├── email_sender.py                  # Низкоуровневая отправка через SMTP
    ├── email_service.py                 # EmailService + вложения
    ├── media_processor.py               # Vision / Whisper / PDF / DOCX
    ├── loader_utils.py                  # stable_id, read_jsonl
    ├── thread_id.py                     # parse_chat_id — единый парсер thread_id
    └── llm.py                           # LLMService (кеш + retry)
```

---

## Сценарии использования

1. **Ответ по базе знаний.**
   - Пользователь задаёт вопрос через Telegram-бота или API (`/agent/chat`).
   - Агент вызывает `search_knowledge_base`.
   - RAG: BGE-M3 → Qdrant (top-10 → top-5) → проверка порога `0.62`.
   - LLM (Ollama + `qwen3.5-instruct:4b`) генерирует ответ с цитатами `[1]`, `[2]`.
   - Бот показывает ответ + блок «Источники:».

2. **Вопрос вне базы → эскалация в HelpDesk (HIL).**
   - RAG возвращает `confident=False` — score ниже порога.
   - Агент формирует `tool_call` `send_email`.
   - Узел `prepare_email` создаёт draft + `correlation_id` (без отправки).
   - `confirm_and_send` вызывает `interrupt()` — граф приостанавливается, state сохраняется в Postgres.
   - Пользователь видит inline-кнопки «✅ Отправить» / «❌ Отмена».
   - После подтверждения письмо уходит через SMTP Yandex, связка `correlation_id → chat_id` кладётся в Redis (TTL 24 ч).

3. **Уведомление от HelpDesk (обратный канал).**
   - HelpDesk отправляет `POST /webhook/helpdesk` с `{correlation_id, ticket_id, status}`.
   - Backend находит `chat_id` в Redis по `correlation_id`.
   - Через `POST bot:9000/notify` пользователь получает сообщение с номером заявки и статусом.

4. **Отправка готового ответа на конкретный email.**
   - Пользователь просит: «Отправь ответ про wifi на `danilenko@ukbmz.ru`».
   - Агент ищет ответ в базе, формирует `send_email` с указанным адресом.
   - В `prepare_email` работает override: email извлекается regex из последнего сообщения и принудительно ставится в `to`.
   - HIL-подтверждение → отправка.

5. **Сброс диалога.**
   - `/start` — новый `thread_id = tg-{chat_id}-{uuid8}`, удаление старого чекпоинта, приветствие.
   - `/clear` — очистка FSM, истории в Postgres и чекпоинта агента.
   - `/cancel` — отмена AskFlow или активного HIL (`resume(decision=False)`).

6. **Команды меню.**
   - `/ask` — inline-клавиатура с 6 темами.
   - `/help` — справка по командам.
   - `/stats` — админ-статистика: сообщения, DAU, feedback ratio, refusal rate.
   - `/broadcast «текст»` — серийная рассылка через `bot:9000/notify`.

7. **Работа с медиа.**
   - Пользователь отправляет голосовое, изображение, PDF или DOCX.
   - `POST /media/process` (multipart) определяет тип и обрабатывает:
     image → Vision (Ollama), audio → faster-whisper, PDF → `pypdf`, DOCX → `python-docx`.
   - Извлечённый текст добавляется к сообщению как `content-part`.

8. **Сжатие контекста длинного диалога.**
   - `KEEP_LAST=8` — последние 8 сообщений сохраняются полностью.
   - `SUMMARY_EVERY=10` — каждые 10 сообщений предыдущая часть сжимается в сводку.
   - Сводка хранится в `state.summary`, граница — `state.summary_upto`.
   - Защита от контекстного переполнения 4B-модели.

9. **Персистентность и восстановление после рестарта.**
   - Все состояния агента — в Postgres (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`).
   - При рестарте backend активный HIL восстанавливается по `thread_id` из checkpoint.
   - Разбор обоих форматов `thread_id` — через `parse_chat_id()` в `app/services/thread_id.py`.
   
10. **Проверка статуса заявки.**
    - Пользователь спрашивает: «какой статус у заявки HD-2026-70127?».
    - Агент вызывает `get_helpdesk_status(ticket_id)` — безопасный
      инструмент без HIL.
    - Инструмент читает `helpdesk:ticket:{ticket_id}` из Redis. Ключ
      наполняется webhook'ом `/webhook/helpdesk` (в production — от
      HelpDesk / 1С:Itilium, для демо — скриптом `scripts/fake_helpdesk.py`).
    - Если записи нет — инструмент возвращает `status="not_found"`,
      и ассистент отвечает, что заявка не найдена. Это честное поведение:
      система не выдумывает статусы для заявок, которых нет в HelpDesk.
    - Пользователь получает ответ с реальным статусом, ответственным
      и временем последнего обновления. 
    - Сценарий проверяется в паре с предыдущим: сначала HIL-форма
      создаёт заявку и письмо уходит на HelpDesk, затем webhook возвращает
      номер `ticket_id`, и пользователь может запросить статус по этому
      номеру.


## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                         Пользователь                            │
│                  (Telegram / API-запросы)                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                     Telegram-бот (aiogram)                      │
│              - обработка сообщений / команд                     │
│              - инлайн-клавиатуры / FSM                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   FastAPI-сервис (app)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                     API-маршруты                          │  │
│  │  /chat  /agent  /rag  /media  /documents  /health  /admin │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  - один агент с инструментами (search_knowledge_base,     │  │
│  │    get_helpdesk_status, send_email)                       │  │
│  │  - персистентность чекпоинтов (Postgres)                  │  │
│  │  - трейсинг (Phoenix)                                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              RAG-сервис (LlamaIndex)                      │  │
│  │  - индексация документов                                  │  │
│  │  - поиск по векторной базе (Qdrant)                       │  │
│  │  - генерация ответа по контексту                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────────────────────┐
│                    Векторная база (Qdrant)                                 │
│              - коллекция документов предметной области                     │
│              - embedding: BAAI/bge-m3 (1024)                               │
└────┬─────────────────┬─────────────────────┬───────────────────────────┬───┘
     │                 │                     │                           │
     ▼                 ▼                     ▼                           ▼
 ┌───────────┐   ┌────────────────┐   ┌────────────────────────┐   ┌─────────────────┐
 │  Qdrant   │   │    Postgres    │   │        Redis           │   │    Phoenix      │
 │ 646 чанк. │   │ чаты +         │   │ • кеш LLM              │   │ • UI трейсинга  │
 │ BGE-M3    │   │ чекпоинты      │   │ • rate-limit           │   │ • OTLP:6006     │
 │ 1024 dim  │   │ агента         │   │ • correlation_id       │   │                 │
 └───────────┘   └────────────────┘   │ • HelpDesk (TTL 24h)   │   └─────────────────┘
                                      └────────────────────────┘
```

## Команда запуска

```bash
# 1. Скопировать и заполнить .env
cp .env.example .env
# отредактировать .env (BOT_TOKEN, YANDEX_EMAIL, YANDEX_APP_PASSWORD и др.)

# 2. Запустить стек
docker compose up -d

# 3. Скачать модель в Ollama (один раз, ~2.5 ГБ)
docker compose exec ollama ollama pull frob/qwen3.5-instruct:4b

# 4. Проверить работоспособность
curl http://localhost:8000/ready
```

Стек поднимается одной командой и включает:

- `app` — FastAPI-сервис (порт 8000)
- `bot` — Telegram-бот (внутренний порт 9000)
- `postgres` — база данных для чатов и чекпоинтов
- `redis` — кеш и rate limiting
- `qdrant` — векторная база данных
- `phoenix` — UI для трейсинга (опционально, порт 6006)
- `ollama` — локальный сервер LLM (порт 11434/11435)

## Переменные окружения

| Переменная | Описание | Пример                                              |
|---|---|-----------------------------------------------------|
| **LLM** | |                                                     |
| `LLM__BASE_URL` | OpenAI-совместимый эндпоинт (агент, vision) | `http://ollama:11434/v1`                            |
| `LLM__OPENAI_API_KEY` | Заглушка для OpenAI-клиента (Ollama игнорирует) | `ollama`                                            |
| `LLM__DEFAULT_MODEL` | Модель для чата и агента | `frob/qwen3.5-instruct:4b`                          |
| `OLLAMA_URL` | Нативный API Ollama (для RAG) | `http://ollama:11434`                               |
| `VISION_MODEL` | Модель для описания изображений | `frob/qwen3.5-instruct:4b`                          |
| **Базы данных** | |                                                     |
| `DATABASE_URL` | PostgreSQL (asyncpg) | `postgresql+asyncpg://chat:chat@postgres:5432/chat` |
| `POSTGRES_PASSWORD` | Пароль БД | `chat`                                              |
| `REDIS_URL` | Redis для кеша, FSM, медиа | `redis://redis:6379/0`                              |
| `QDRANT_URL` | Адрес Qdrant | `http://qdrant:6333`                                |
| **Эмбеддинги** | |                                                     |
| `EMBEDDING__MODEL_NAME` | Модель эмбеддингов (BGE-M3 → dim 1024) | `BAAI/bge-m3`                                       |
| `EMBEDDING__DEVICE` | CPU / GPU | `cpu`                                               |
| `EMBEDDING__TRUST_REMOTE_CODE` | Разрешить кастомный код модели (для BGE-M3 — обязателен) | `true`                                              |
| `EMBEDDING__DIM` | Размерность вектора (справочно) | `1024`                                              |
| **RAG** | |                                                     |
| `RAG_COLLECTION` | Основная коллекция Qdrant | `rag_block_04`                                      |
| `RAG_CHUNK_SIZE` | Размер чанка | `512`                                               |
| `RAG_CHUNK_OVERLAP` | Перекрытие чанков | `64`                                                |
| `RAG_SCORE_THRESHOLD` | Порог релевантности | `0.62`                                              |
| `RAG_RETRIEVE_TOP_K` | Кандидатов из Qdrant | `10`                                                |
| `RAG_RERANK_TOP_N` | Чанков в контексте | `5`                                                 |
| **Telegram** | |                                                     |
| `BOT_TOKEN` | Токен от @BotFather | —                                                   |
| `BOT_ADMIN_IDS` | ID админов через запятую | `123456789,987654321`                               |
| `INTERNAL_TOKEN` | Service-to-service | `openssl rand -hex 32`                              |
| `ADMIN_TOKEN` | Для `/chats/admin/*` | `openssl rand -hex 32`                              |
| **Email** | |                                                     |
| `YANDEX_EMAIL` | Адрес отправителя | `user@yandex.ru`                                    |
| `YANDEX_APP_PASSWORD` | Пароль приложения Яндекс | —                                                   |
| `EXCHANGE_RECIPIENT_EMAIL` | Получатель заявок HelpDesk | `helpdesk@company.com`                              |
| **Агент** | |                                                     |
| `AGENT_CHECKPOINTER` | `memory` / `sqlite` / `postgres` | `postgres`                                          |
| `MODERATION_USE_OPENAI` | Модерация через внешний судья (по аналогии с RAGAS) | `true`                                              |
| `RATE_LIMIT_MESSAGES_PER_MIN` | Лимит сообщений/мин | `15`                                                |
| **Оценка (опционально)** | |                                                     |
| `EVAL_JUDGE_PROVIDER` | Провайдер судьи | `openai` / `anthropic`                              |
| `EVAL_JUDGE_URL` | URL судьи RAGAS | `https://api.proxyapi.ru/openai/v1`                 |
| `EVAL_JUDGE_MODEL` | Модель-судья | `gpt-4o-mini`                                       |
| **HelpDesk webhook** | | |
| `HELPDESK_WEBHOOK_TOKEN` | Токен для входящих вебхуков | `openssl rand -hex 32` |
| `HELPDESK_EMAIL` | Адрес, куда уходят заявки | `helpdesk@company.com` |

> Полный список — в `.env.example`.

## Состав проекта

- **README.md** — этот файл
- **compose.yaml** — стек поднимается одной командой
- **.env.example** — все переменные с заглушками
- **docs/** — архитектура, схема агентного графа, отчёт по качеству поиска, итоговое описание решений
- **tests/** — тесты, написанные по ходу курса
- **Ссылка на видео-демонстрацию** — будет добавлена после записи
- **Репозиторий проекта** — https://github.com/ssdanilenko-glitch/itassistant_final
- **Презентация** — [docs/presentation.pdf](docs/presentation.pdf)

## Документация

- [Схема агентного графа (описание)](docs/agent_graph.md)
- [Mermaid-исходник](docs/agent-graph-custom.mmd)
- ![Граф агента](docs/agent.png)
- [Эксперименты по нарезке документов](docs/chunking_experiment.md)
- [Выбор embedding-модели](docs/embedding.md)
- [RAG-архитектура](docs/rag.md)
- [Выбор векторной базы](docs/vector_store.md)
- [Сравнение метрик Qdrant](docs/metric_comparison.json)
- [Отчёт по качеству RAG](docs/quality_report.md)
- [Итоговое обоснование решений](docs/final_decisions.md)

## Оценка качества

### RAGAS-оценка

Прогон по golden dataset (`tests/eval/golden_dataset.json`, 15 вопросов).
Считаются шесть метрик: faithfulness, answer_relevancy, context_precision,
context_recall, factual_correctness, has_citation.

Запуск:

```bash
uv run --extra eval python scripts/run_eval.py \
    --golden tests/eval/golden_dataset.json \
    --label baseline
```

Результат — CSV в `tests/eval/results/{timestamp}_{label}.csv`.
Детальный анализ — в [docs/quality_report.md](docs/quality_report.md).

| Метрика | 1-й запуск | 3-й запуск | Δ |
|---|---|---|---|
| Faithfulness | 0.876 | **0.933** | +0.06 |
| Answer Relevancy | 0.524 | **0.638** | +0.11 |
| Context Precision | 0.734 | **0.906** | +0.17 |
| Context Recall | 0.667 | **0.933** | +0.27 |
| Factual Correctness | 0.230 | **0.585** | +0.36 |
| Has Citation | — | 15/15 | ✅ |

### Обоснование метрики расстояния Qdrant

Скрипт `scripts/compare_metrics.py` читает точки из текущей RAG-коллекции,
клонирует их в две временные коллекции с метриками `COSINE` и `DOT`,
прогоняет запросы из golden dataset через BGE-M3 и сравнивает top-K.

```bash
docker compose exec app python scripts/compare_metrics.py
```

Результат на актуальной коллекции (`rag_block_04`, 646  точек, 5 запросов):
**5/5 совпадений top-5**. Это ожидаемо, так как BGE-M3 выдаёт
L2-нормализованные векторы. Выбрана метрика `COSINE` как более устойчивая
к возможной смене модели. Артефакт — в
[docs/metric_comparison.json](docs/metric_comparison.json).

## Ограничения

- Система отвечает только по документам, загруженным в базу знаний. На вопросы вне предметной области агент отвечает отказом (если релевантность ниже порога `RAG_SCORE_THRESHOLD`).
- Внешний API-ключ OpenAI для работы не требуется: LLM (Ollama) и эмбеддинги (BGE-M3) работают локально. Прокси-провайдер (proxyapi.ru) подключается только как судья RAGAS на этапе оценки (переменные `EVAL_JUDGE_*`) и как внешний модератор при `MODERATION_USE_OPENAI=true`.
- Трейсинг включён только при установке extra-зависимости `tracing` и флаге `PHOENIX_ENABLED=true`.

## План развития

- **Интеграция с 1С:Itilium (HelpDesk).** REST-коннектор для
  автоматического создания заявок и запроса статуса.
- **Интеграция с корпоративным мессенджером Express.** Второй канал
  взаимодействия помимо Telegram — для сотрудников с ограничениями
  ИБ на внешние мессенджеры.
- **Гибридный поиск** (dense + BM25) для повышения Context Recall.
- **Расширение golden dataset** с 15 до 50 вопросов.
- **Переход на 7B–14B модель** при апгрейде GPU.
- **Мультимодальность в ответе** (TTS через локальные модели).
- **История и поиск по чатам** (полнотекстовый индекс в Postgres).

## Разработка

```bash
# Установка зависимостей (без Docker)
uv sync

# Запуск тестов
pytest

# Линтинг
ruff check .
```

## Автор

**Даниленко С.С.** — итоговая аттестация по курсу «ИИ-разработчик: от API до агентов».
