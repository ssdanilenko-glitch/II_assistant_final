# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_TORCH_BACKEND=cpu

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

WORKDIR /app
RUN uv venv

COPY pyproject.toml uv.lock ./
COPY app/ ./app/
COPY bot/ ./bot/
COPY migrations/ ./migrations/
COPY alembic.ini ./

# Включаем extras eval для запуска run_eval.py внутри контейнера.
# Если оценка не нужна в образе — уберите --extra eval (образ станет легче на ~300 МБ).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev --extra eval --extra tracing --frozen

# ========== RUNTIME ==========
FROM python:3.12-slim

# Системные пакеты нужны только если используется OCR/PDF-конвертация.
# Если вы не используете OCR, можно удалить tesseract-ocr* и poppler-utils.
RUN apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
        poppler-utils \
        libmagic-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
RUN mkdir -p /app/var /app/data && chown -R appuser:appuser /app

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]