
# Отчёт Б6.4 — LangGraph: персистентность, HIL, time-travel, streaming

## 1. Backend чек-пойнтера

Режим выбирается переменной `AGENT_CHECKPOINTER` (`memory` / `sqlite` / `postgres`), которую читает `agent_lifespan` в `app/services/agent_persistent.py`.

| Режим | Где применяется | Почему |
|---|---|---|
| `sqlite` | локальная разработка, `pytest` | `AsyncSqliteSaver.from_conn_string(":memory:")` — без сети, воспроизводимо |
| `postgres` | docker-compose, staging | тот же Postgres, что и для FastAPI из МЗБ5; отдельная БД не плодится |
| `memory` | unit-тесты старого графа | быстрый прогон без чек-пойнтера |

`AGENT_CHECKPOINTER=postgres` приходит из `.env`, сервис `app` получает его через `env_file`. Сервис `postgres` не меняется — используется существующий инстанс МЗБ5.

`await checkpoint.setup()` вызывается один раз в `agent_lifespan`. Для `AsyncPostgresSaver` создаются таблицы `checkpoints`, `checkpoint_writes`, `checkpoint_blobs`, `checkpoint_migrations`. Для `AsyncSqliteSaver` — DDL поверх пустого файла. На каждый HTTP-запрос `setup()` не вызывается.

В `alembic/env.py` добавлен `include_name`, исключающий эти таблицы из autogenerate. Схему чек-пойнтера ведёт `setup()`, доменную — Alembic.

## 2. Postgres в compose

Сервис `postgres` остаётся без изменений. В `compose.yaml` сервис `app` уже подписан на `.env`:

```yaml
services:
  app:
    env_file:
      - .env
    depends_on:
      postgres:
        condition: service_healthy
```

`.env`:

```dotenv
AGENT_CHECKPOINTER=postgres
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=chat
POSTGRES_PASSWORD=chat
POSTGRES_DB=chat
DATABASE_URL=postgresql+asyncpg://chat:chat@postgres:5432/chat
```

URI для `AsyncPostgresSaver` собирается из тех же `POSTGRES_*` через `_psycopg_uri()` — заменяет `+asyncpg` на чистый `postgresql://`.

Проверка после старта:

```bash
$ docker compose exec postgres psql -U chat -d chat -c '\dt'
                List of relations
 Schema |         Name          | Type  | Owner
--------+-----------------------+-------+-------
 public | checkpoint_blobs      | table | chat
 public | checkpoint_migrations | table | chat
 public | checkpoint_writes     | table | chat
 public | checkpoints           | table | chat
(4 rows)
```

## 3. Опасный tool

**Tool:** `send_email` — отправка письма наружу (HelpDesk, клиенту, на произвольный адрес). Опасность: необратимый side-effect, письмо уходит реальному получателю, откатить нельзя.

Граф разбит на три узла с edge между подготовкой и отправкой:

```
__start__ → call_model → prepare_email → confirm_and_send → call_model → force_finish → END
```

### До `interrupt()` — `prepare_email` (idempotent)

- извлекает `tool_call send_email` из последнего `AIMessage`;
- нормализует `body`: `_strip_sender_prefix()` срезает все ведущие шапки «Пользователь: …» (с учётом латинской `P` и невидимых символов);
- `_extract_sender_info()` берёт шапку из первого `HumanMessage` в state;
- если в body шапки нет, а в state есть — вклеивает её один раз;
- генерирует `correlation_id = uuid4().hex[:12]`;
- **никаких сетевых вызовов**, состояние только читается.

Затем `confirm_and_send` вызывает `interrupt({"type": "approve_email", "preview": draft})`. Граф останавливается.

### После `interrupt()` — `confirm_and_send`

```python
approved = decision is True or decision == "approve"
if approved:
    await send_email_fn(draft)   # реальный SMTP
    content = f"письмо отправлено: {draft['subject']}"
else:
    content = "отправка отменена пользователем"
```

**Почему так:** если `send_email_fn` вызвать до `interrupt()`, при resume узел перезапустится с начала и письмо уйдёт дважды. `prepare_email` идемпотентен, side-effect — только после resume.

**Правило блока:** до `interrupt()` — подготовка (валидация, рендер, чтение БД); после `interrupt()` — side-effect в отдельном узле.

`interrupt_before` / `interrupt_after` не используются. Канон LangGraph 1.0 — `interrupt()` + `Command(resume=...)`.

После confirm_and_send граф возвращает управление в call_model,
который формирует финальный AIMessage («Письмо отправлено» /
«Отправка отменена»). Защита от зацикливания — флаг sent=True в
route_after_model.

## 4. Логи: `__interrupt__` и resume

**Момент `__interrupt__` (фрагмент лога `scripts/time_travel_demo.py`):**

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

**Момент после `Command(resume=True)`:**

```
4) две ветки: отказ → sent=False, одобрение → sent=True, отправок=1
```

Из логов боевого сценария (`docker compose logs app`):

```
[prepare_email] raw_body='Пользователь предоставил изображение с текстом вопроса: …'
[prepare_email] after_strip='Пользователь предоставил изображение с текстом вопроса: …'
[confirm_and_send] decision=True (type=<class 'bool'>), approved=True
[confirm_and_send] calling send_email_fn with draft: {
  'to': 'danilenko@ukbmz.ru',
  'subject': 'Заявка в HelpDesk: …',
  'body': 'Пользователь: Sergey Danilenko (ID: 443426947, @it_sd)\n\n…',
  'sender_info': 'Пользователь: Sergey Danilenko (ID: 443426947, @it_sd)',
  'correlation_id': '2763e36e4c11',
  'tool_call_id': '9CCeilPX5TI6v8WzRr3LjVwXw8H1rruf',
  'thread_id': 'tg-443426947-865fd4dd'
}
[SEND_EMAIL] ✅ Email sent successfully
```

## 5. Time travel

### История чек-пойнтов

```
2) история чек-пойнтов (checkpoint_id / next):
   1f1b584f-427f-69b2-8002-f510f7399d3e  next=('confirm_and_send',)
   1f1b584f-427f-69b1-8001-2433449dc54d  next=('prepare_email',)
   1f1b584f-427d-629b-8000-8d308c855c1d  next=('call_model',)
   1f1b584f-427a-6b81-bfff-72b0ed7dddba  next=('__start__',)
```

Читается сверху вниз — от свежего к старому. Верхний снапшот — остановка на `interrupt`.

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

Повторный `resume` того же `interrupt`-чек-пойнта с другим решением **не сработает**: `Command(resume=...)` сохраняется как pending-write и детерминировано на весь thread-lineage. Первый `resume` выигрывает. Две ветки демонстрируются на двух `thread_id`:

```
4) две ветки: отказ → sent=False, одобрение → sent=True, отправок=1
Итог: один и тот же вход дал две ветки — отказ (sent=False) и одобрение (sent=True).
```

Форк возможен через новый `thread_id` (как в демо) либо через `graph.aupdate_state(..., checkpoint_id=...)`.

## 6. Streaming

Выбран `graph.astream(..., stream_mode=["updates"])`.

**Обоснование:**

- `updates` даёт системный прогресс по узлам — ровно то, что нужно боту, чтобы показать «думаю → готовлю черновик → жду подтверждения».
- `messages` был **отключён**, потому что модель вызывается через `ainvoke` и не стримится по токенам. При включённом `messages` (только при `ChatOpenAI` под капотом с callback'ами) текст приходил дважды: через `updates(call_model)` и через токены. Дубль виден в чате как два одинаковых сообщения.
- `astream_events(version="v2")` богаче (`on_chat_model_stream`, `on_tool_start`), но заметно объёмнее. Для SSE-эндпоинта избыточен; оставлен как следующий шаг.

Реализация в `app/routers/agent.py`:

```python
async def event_source() -> AsyncIterator[str]:
    logger.info("SSE START thread=%s input_type=%s", req.thread_id, type(graph_input).__name__)
    try:
        async for stream_type, payload in graph.astream(
            graph_input, config, stream_mode=["updates"]
        ):
            if (stream_type == "updates" and isinstance(payload, dict)
                    and "__interrupt__" in payload):
                nodes = {k: v for k, v in payload.items() if k != "__interrupt__"}
                if nodes:
                    ev = _format_event("updates", nodes)
                    if ev is not None:
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                ev = _format_event("updates", {"__interrupt__": payload["__interrupt__"]})
                if ev is not None:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                continue
            event = _format_event(stream_type, payload)
            if event is not None:
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
```

**curl-проверка (фрагмент вывода):**

```
$ docker compose exec app python -c "...httpx.stream POST /agent/stream..."

STATUS 200
LINE data: {"type": "update", "nodes": ["call_model"], "messages": [{"role": "assistant", "text": "В базе знаний нет ответа на этот вопрос.\nЯ подготовил заявку в HelpDesk — подтвердите отправку.\n\n"}]}
LINE data: {"type": "interrupt", "payload": {"type": "approve_email", "preview": {"to": "danilenko@ukbmz.ru", ...}}}
LINE data: {"type": "done"}
```

`_format_event` фильтрует `ToolMessage` (`mtype == "tool"`), чтобы служебные ответы инструментов не уходили пользователю. Остальные сообщения разбиваются на `{"type": "update", "messages": [...]}` и `{"type": "assistant_text"}` в боте.

## 7. Permission policy

`config["configurable"]["user_role"]` принимает `read-only` / `write-with-approve` / `full`: для `full` `confirm_and_execute_*` пропускает `interrupt()`, у `read-only` опасный tool недоступен, `write-with-approve` (используется в сценарии) всегда требует подтверждения.

