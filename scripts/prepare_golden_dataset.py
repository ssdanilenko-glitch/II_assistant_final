#!/usr/bin/env python
"""Автоматическая генерация golden_dataset.json для оценки RAG.

Как работает:
1. Подключается к Qdrant и берёт случайную выборку чанков из коллекции RAG.
2. Для каждого чанка просит LLM сгенерировать:
   - user_input   — вопрос, на который этот чанк отвечает;
   - reference    — краткий эталонный ответ на основе этого же чанка.
3. Сохраняет результат в tests/eval/golden_dataset.json.

Использование:
    docker compose exec app python scripts/prepare_golden_dataset.py \
        --count 20 --out tests/eval/golden_dataset.json

Опции:
    --count N      сколько пар сгенерировать (по умолчанию 20)
    --out PATH     куда сохранить JSON (по умолчанию tests/eval/golden_dataset.json)
    --seed INT     сид для воспроизводимости выборки (по умолчанию 42)
    --min-len N    минимальная длина чанка в символах (по умолчанию 300)
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from app.core.config import get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


PROMPT_TEMPLATE = """\
Ты — методист, который готовит тестовый набор для оценки RAG-системы технической поддержки.

Ниже — фрагмент документа из базы знаний:
---
{chunk}
---

Сгенерируй ОДИН вопрос пользователя (user_input) и краткий эталонный ответ (reference),
которые полностью покрываются этим фрагментом.

Требования:
- Вопрос должен быть реалистичным: как если бы его задал сотрудник техподдержки.
- Вопрос должен быть на русском языке, короткий (до 15 слов).
- Ответ должен быть на русском, короткий (до 60 слов), и опираться ТОЛЬКО на фрагмент.
- Не упоминай в вопросе название файла или «фрагмента» — вопрос должен звучать естественно.
- Если во фрагменте нет чёткого ответа (шум, оглавление, битый текст) — верни {{"skip": true}}.

Верни СТРОГО валидный JSON без markdown-обёртки:
{{"user_input": "...", "reference": "...", "skip": false}}
"""


async def generate_qa_pair(
    client: AsyncOpenAI,
    model: str,
    chunk: str,
) -> dict | None:
    """Сгенерировать пару (user_input, reference) по одному чанку."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(chunk=chunk[:3000]),
                }
            ],
            temperature=0.3,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        # На случай, если LLM обернёт в ```json ... ```
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        if data.get("skip"):
            return None
        q = (data.get("user_input") or "").strip()
        r = (data.get("reference") or "").strip()
        if not q or not r:
            return None
        return {"user_input": q, "reference": r}
    except json.JSONDecodeError as e:
        logger.warning("LLM вернул невалидный JSON: %s", e)
        return None
    except Exception as e:
        logger.warning("Ошибка генерации пары: %s", e)
        return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=str, default="tests/eval/golden_dataset.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-len", type=int, default=300)
    args = parser.parse_args()

    settings = get_settings()

    # --- 1. Подключение к Qdrant и выборка чанков ---
    qdrant_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key is not None
        else None
    )
    qdrant = QdrantClient(url=settings.qdrant_url, api_key=qdrant_key)

    logger.info("Читаю чанки из коллекции %s...", settings.rag_collection)
    # Берём максимум 5000 чанков для выборки
    scroll_result = qdrant.scroll(
        collection_name=settings.rag_collection,
        limit=5000,
        with_payload=True,
        with_vectors=False,
    )
    points = scroll_result[0]
    qdrant.close()

    if not points:
        logger.error("Коллекция пуста — нечего использовать для генерации.")
        sys.exit(1)

    # Собираем только «содержательные» чанки
    chunks: list[dict] = []
    for p in points:
        payload = p.payload or {}
        text = (
            payload.get("text")
            or payload.get("original_content")
            or payload.get("content")
            or ""
        ).strip()
        if len(text) < args.min_len:
            continue
        chunks.append(
            {
                "text": text,
                "file_name": payload.get("file_name") or payload.get("source") or "unknown",
            }
        )

    logger.info("Готово к генерации: %d чанков (из %d)", len(chunks), len(points))

    if len(chunks) < args.count:
        logger.warning(
            "В корпусе меньше чанков (%d), чем запрошено (%d) — сгенерирую сколько есть.",
            len(chunks),
            args.count,
        )

    # --- 2. Случайная выборка (с фиксированным сидом) ---
    random.seed(args.seed)
    random.shuffle(chunks)
    sample = chunks[: args.count]

    # --- 3. LLM-клиент (используем тот же, что и в агенте) ---
    llm = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        base_url=settings.llm.base_url,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )
    model = settings.llm.default_model

    # --- 4. Генерация пар ---
    dataset: list[dict] = []
    seen_questions: set[str] = set()

    logger.info("Генерирую %d пар (модель: %s)...", len(sample), model)
    for i, chunk in enumerate(sample, start=1):
        pair = await generate_qa_pair(llm, model, chunk["text"])
        if pair is None:
            logger.info("[%d/%d] SKIP (не удалось сгенерировать)", i, len(sample))
            continue
        # Защита от дублей вопросов
        if pair["user_input"].lower() in seen_questions:
            logger.info("[%d/%d] SKIP (дубликат вопроса)", i, len(sample))
            continue
        seen_questions.add(pair["user_input"].lower())
        pair["source"] = chunk["file_name"]
        dataset.append(pair)
        logger.info(
            "[%d/%d] OK (%s) — %s",
            i,
            len(sample),
            chunk["file_name"],
            pair["user_input"],
        )

    await llm.close()

    # --- 5. Сохранение ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Сохранено %d пар в %s", len(dataset), out_path)

    if len(dataset) < 5:
        logger.warning(
            "Получилось мало пар (%d). Проверьте, что в корпусе достаточно "
            "содержательных текстов и что модель отвечает валидным JSON.",
            len(dataset),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
