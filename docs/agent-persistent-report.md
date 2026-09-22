
# Отчёт Б6.4 — LangGraph: персистентность, HIL, time-travel, streaming

## 1. Backend чек-пойнтера

Чек-пойнтер переключается переменной `AGENT_CHECKPOINTER` (`memory` / `sqlite` / `postgres`), которую читает фабрика `build_agent(checkpointer)` в `app/services/agent_persistent.py`.

| Режим | `AGENT_CHECKPOINTER` | Где используется | Зачем |
|---|---|---|---|
| Локальная разработка / тесты | `sqlite` (дефолт) | разработчик, `pytest` | `AsyncSqliteSaver.from_conn_string(":memory:")` — без сети и Postgres |
| Docker-compose / staging | `postgres` | сервис `app` | тот же Postgres из МЗБ5, отдельная БД не заводится |
| In-memory | `memory` | unit-тесты старого `agent_graph.py` | быстрый прогон без БД |

`AGENT_CHECKPOINTER=postgres` не захардкожен в `compose.yaml`. Значение лежит в `.env`, а сервис `app` подключает его через директиву `env_file`. Сервис `postgres` не меняется — используется существующий инстанс из МЗБ5 и те же `POSTGRES_*` переменные.

`await checkpoint.setup()` вызывается ровно один раз в `agent_lifespan()` при старте FastAPI, не на каждый запрос. Для `AsyncPostgresSaver` создаёт `checkpoints`, `checkpoint_writes`, `checkpoint_blobs`, `checkpoint_migrations`; для `AsyncSqliteSaver` — DDL поверх пустого файла.

## 2. Postgres в compose

Сервис `postgres` остаётся неизменным. Изменения только в сервисе `app`:

```yaml
services:
  app:
    env_file:
      - .env          # AGENT_CHECKPOINTER=postgres
    depends_on:
      - postgres
      - redis

  postgres:
    # без изменений
```

`.env`:

```dotenv
AGENT_CHECKPOINTER=postgres
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=agent
POSTGRES_PASSWORD=agent
POSTGRES_DB=agent_db
```

URI собирается из тех же `POSTGRES_*`, что и DSN FastAPI:

```
postgresql://agent:agent@postgres:5432/agent_db
```

Проверка после `setup()`:

```bash
$ docker compose exec postgres psql -d agent_db -c '\dt'
                List of relations
 Schema |         Name          | Type  | Owner
--------+-----------------------+-------+-------
 public | checkpoint_blobs      | table | agent
 public | checkpoint_migrations | table | agent
 public | checkpoint_writes     | table | agent
 public | checkpoints           | table | agent
(4 rows)
```

Alembic не удаляет `checkpoint*`-таблицы — в `alembic/env.py` добавлен `include_name`:

```python
def include_name(name, type_, parent_names):
    if type_ == "table":
        return name not in {
            "checkpoints", "checkpoint_writes",
            "checkpoint_blobs", "checkpoint_migrations",
        }
    return True

context.configure(..., include_name=include_name)
```

Схему чек-пойнтера ведёт `checkpoint.setup()`, доменную — Alembic.

## 3. Опасный tool и точки до/после interrupt

**Tool:** `send_email` — отправка письма клиенту. Опасность: необратимый side-effect наружу (SMTP/API), письмо уходит реальному получателю.

Граф:

```
__start__ → call_model → prepare_email → confirm_and_send → END
```

### До interrupt (`prepare_email`, идемпотентный узел)

- валидация tool_call (наличие `to`, `subject`, `body`);
- рендер финального шаблона из state;
- запись `draft = {"to", "subject", "body", "tool_call_id", "thread_id"}` в state;
- никаких сетевых вызовов.

Затем:

```python
decision = interrupt({"type": "approve_email", "preview": state["draft"]})
```

### После interrupt (`confirm_and_send`, единственная точка side-effect)

```python
if decision is True:
    await send_fn(state["draft"])
    return {"sent": True}
return {"sent": False}
```

Почему так: если поставить `send_fn` до `interrupt()`, при resume узел перезапустится с начала, и письмо уйдёт дважды. Неидемпотентные шаги (например, генерация `tracking_id`) выносятся в отдельный детерминированный шаг по `request_id` из state.

**Правило:** до `interrupt()` — только подготовка (рендер, валидация, чтение БД); после `interrupt()` — сам side-effect, в отдельном узле.

`interrupt_before` / `interrupt_after` не используются. Канон LangGraph 1.0 — `interrupt()` + `Command(resume=...)`.

## 4. Логи interrupt и resume

Запуск: `uv run python -m scripts.time_travel_demo` (офлайн, `AsyncSqliteSaver(":memory:")`, `FakeChat`).

```
1) INTERRUPT payload: {
  'type': 'approve_email',
  'preview': {
    'to': 'client@example.com',
    'subject': 'Ваше обращение №123456',
    'body': 'Ваш запрос во вложении.',
    'tool_call_id': 'call-1',
    'thread_id': 'demo'
  }
}
```

Граф остановился в `confirm_and_send`, `sent=False`, `draft` готов, письма нет.

```
4) две ветки: отказ → sent=False, одобрение → sent=True, отправок=1
```

`sent_log` содержит ровно одну запись — только одобренная ветка.

## 5. Time travel

### История чек-пойнтов (`aget_state_history`)

```
2) история чек-пойнтов (checkpoint_id / next / ключи state):
   1f1b584f-427f-69b2-8002-f510f7399d3e  next=('confirm_and_send',)
   1f1b584f-427f-69b1-8001-2433449dc54d  next=('prepare_email',)
   1f1b584f-427d-629b-8000-8d308c855c1d  next=('call_model',)
   1f1b584f-427a-6b81-bfff-72b0ed7dddba  next=('__start__',)
```

Сверху вниз — от свежего к старому. Верхний снапшот соответствует остановке на `interrupt`.

### Чтение прошлого чек-пойнта

```python
past = await graph.aget_state({
    "configurable": {
        "thread_id": "demo",
        "checkpoint_id": pre_interrupt_id,
    }
})
```

```
3) чтение прошлого чек-пойнта:
   sent=False, draft_готов=True, next=('confirm_and_send',)
```

Read-only: `send_fn` не вызывается, тред `demo` не двигается.

### Replay с противоположным решением

Повторный `resume` того же `interrupt`-чек-пойнта с другим решением не сработает: `Command(resume=...)` сохраняется в чек-пойнтере как pending-write и детерминировано на весь thread-lineage. Первый `resume` выигрывает. Две ветки показаны на двух `thread_id` с одинаковым входом:

```
4) две ветки: отказ → sent=False, одобрение → sent=True, отправок=1
Итог: один и тот же вход дал две ветки — отказ (sent=False) и одобрение (sent=True).
```

`resume` — это «доставить пропущенное значение в `interrupt`», а не «переиграть уже принятое решение». Форк — через новый `thread_id` либо `graph.aupdate_state(... checkpoint_id ...)`.

## 6. Streaming

Выбран `graph.astream(stream_mode=["updates", "messages"])`.

- `updates` — системный прогресс по узлам («думаю → готовлю черновик → жду подтверждения»).
- `messages` — токены LLM для typewriter-эффекта.
- `astream_events(version="v2")` богаче (`on_chat_model_stream`, `on_tool_start`), но существенно объёмнее; для SSE-endpoint избыточен.

`app/routers/agent.py`:

```python
@router.post("/agent/stream")
async def agent_stream(payload: AgentRequest):
    async def event_gen():
        async for stream_type, chunk in graph.astream(
            payload.input,
            {"configurable": {"thread_id": payload.thread_id,
                              "user_role": payload.user_role}},
            stream_mode=["updates", "messages"],
        ):
            yield f"data: {json.dumps({'type': stream_type, 'data': chunk})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

Пауза распознаётся по `__interrupt__` в очередном `updates`-событии; после `Command(resume=True)` поток продолжается на том же `thread_id`.

curl-проверка:

```
$ curl -N -X POST http://localhost:8000/agent/stream \
    -H 'Content-Type: application/json' \
    -d '{"thread_id":"demo-1","input":{"messages":[{"role":"user","content":"отправь вопрос на help desk"}]}}'

data: {"type": "updates", "data": {"call_model": {"messages": [{"id": "ai-send", ...}]}}}
data: {"type": "updates", "data": {"prepare_email": {"draft": {"to": "client@example.com", ...}}}}
data: {"type": "updates", "data": {"__interrupt__": [{"value": {"type": "approve_email", ...}}]}}

$ curl -N -X POST http://localhost:8000/agent/stream \
    -d '{"thread_id":"demo-1","resume":true}'

data: {"type": "updates", "data": {"confirm_and_send": {"sent": true}}}
data: {"type": "messages", "data": {"content": "Готово, письмо обработано."}}
```

## 7. Permission policy

`config["configurable"]["user_role"]` принимает `read-only` / `write-with-approve` / `full`. `confirm_and_execute_*` пропускает `interrupt()` для `full`, у `read-only` опасный tool недоступен, `write-with-approve` всегда требует подтверждения.

## 8. Хрупкое и TODO

**Хрупкое:**

- `FakeChat` — заглушка; в проде поведение модели недетерминировано и влияет на попадание в `interrupt`.
- `Command(resume=...)` пишется в pending-writes на весь lineage; второй `resume` с другим значением молча игнорируется. Стоит отдавать `409 Conflict`, если чек-пойнт уже зарезюмлен.
- Alembic-`include_name` — точечный фильтр по именам; при переименовании таблиц в LangGraph правится руками.
- `:memory:` SQLite в тестах не ловит проблемы конкурентного доступа к Postgres.

**TODO:**

- Второй endpoint на `astream_events(version="v2")` с флагом `?verbose=1`.
- Метрики в SSE (`time_to_first_token`, `interrupt_latency`).
- Интеграционный тест с Postgres через `testcontainers`.
- TTL для старых чек-пойнтов.
```