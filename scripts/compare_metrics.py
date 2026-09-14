"""Сравнение ранжирования cosine vs dot на реальной RAG-коллекции Qdrant.

Скрипт читает точки из базовой коллекции, клонирует их в две временные
коллекции с метриками расстояния COSINE и DOT, прогоняет набор запросов
через локальную BGE-M3 и сравнивает top-K. Различия показывают, насколько
выбор метрики расстояния влияет на ранжирование для текущих данных.

Запросы берутся из tests/eval/golden_dataset.json (первые 5) или из
встроенного fallback-набора.

Запуск:
    uv run python scripts/compare_metrics.py
    uv run python scripts/compare_metrics.py --top-k 10 --keep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from app.core.config import get_settings  # noqa: E402

logger = logging.getLogger("compare")
logging.basicConfig(level=logging.INFO, format="%(message)s")


UPSERT_BATCH = 64
CLIENT_TIMEOUT = 120.0

FALLBACK_QUERIES = [
    "Как создать заявку в 1С:ERP",
    "Как настроить Wi-Fi на корпоративном ноутбуке",
    "Как отправить обращение в службу УИТ",
    "Как проверить статус заявки в HelpDesk",
    "Как восстановить доступ к ЭНТ",
]


def _resolve_collection(settings, override: str | None) -> str:
    if override:
        return override
    for path in (
        "rag.collection",
        "rag_collection",
        "qdrant.collection",
        "qdrant_collection",
    ):
        obj = settings
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, str) and obj:
            return obj
    raise RuntimeError(
        "Не удалось определить имя коллекции. Передайте --collection явно."
    )


def _resolve_qdrant_api_key(settings):
    val = getattr(settings, "qdrant_api_key", None)
    if val is None:
        return None
    if hasattr(val, "get_secret_value"):
        val = val.get_secret_value()
    return val or None


def _build_embedder(settings) -> HuggingFaceEmbedding:
    emb = getattr(settings, "embedding", None)
    if emb is not None:
        model_name = getattr(emb, "model_name", None)
        device = getattr(emb, "device", None)
        trc = getattr(emb, "trust_remote_code", None)
    else:
        model_name = getattr(settings, "embedding_model_name", None)
        device = getattr(settings, "embedding_device", None)
        trc = None

    kwargs: dict = {"model_name": model_name or "BAAI/bge-m3"}
    if device:
        kwargs["device"] = device
    kwargs["trust_remote_code"] = True if trc is None else trc
    return HuggingFaceEmbedding(**kwargs)


def _load_queries(path: Path, n: int = 5) -> list[str]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            queries: list[str] = []
            for row in data:
                q = row.get("user_input") or row.get("question")
                if q:
                    queries.append(q)
                if len(queries) >= n:
                    break
            if queries:
                return queries
        except Exception as exc:
            logger.warning("Не удалось прочитать %s: %s", path, exc)
    return FALLBACK_QUERIES[:n]


async def _scroll_all(client: AsyncQdrantClient, collection: str) -> list[PointStruct]:
    points: list[PointStruct] = []
    offset = None
    while True:
        batch, offset = await client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in batch:
            points.append(
                PointStruct(id=p.id, vector=p.vector, payload=p.payload or {})
            )
        if offset is None:
            break
    return points


async def _vector_size(client: AsyncQdrantClient, collection: str) -> int:
    info = await client.get_collection(collection)
    params = info.config.params.vectors
    if hasattr(params, "size"):
        return int(params.size)
    return int(next(iter(params.values())).size)


async def _clone_collection(
    client: AsyncQdrantClient,
    base: str,
    suffix: str,
    distance: Distance,
    points: list[PointStruct],
    dim: int,
    batch_size: int = UPSERT_BATCH,
) -> str:
    name = f"{base}_{suffix}"
    existing = {c.name for c in (await client.get_collections()).collections}
    if name in existing:
        await client.delete_collection(name)

    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=distance),
    )

    total = len(points)
    for i in range(0, total, batch_size):
        chunk = points[i : i + batch_size]
        is_last = i + batch_size >= total
        await client.upsert(
            collection_name=name,
            points=chunk,
            wait=is_last,
        )
        logger.info(
            "  upsert %s: %d/%d", name, min(i + batch_size, total), total
        )
    return name


async def _top_k(
    client: AsyncQdrantClient, collection: str, vector: list[float], k: int
) -> list[str]:
    result = await client.query_points(
        collection_name=collection,
        query=vector,
        limit=k,
        with_payload=False,
    )
    return [str(p.id) for p in result.points]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение cosine vs dot в Qdrant")
    parser.add_argument("--collection", default=None, help="Имя базовой коллекции")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="docs/metric_comparison.json")
    parser.add_argument(
        "--golden",
        default="tests/eval/golden_dataset.json",
        help="Файл с запросами",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Не удалять временные коллекции после прогона",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=UPSERT_BATCH,
        help=f"Размер батча при upsert (по умолчанию {UPSERT_BATCH})",
    )
    args = parser.parse_args()

    settings = get_settings()
    base = _resolve_collection(settings, args.collection)

    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=_resolve_qdrant_api_key(settings),
        timeout=CLIENT_TIMEOUT,
    )

    try:
        existing = {c.name for c in (await qdrant.get_collections()).collections}
        if base not in existing:
            raise SystemExit(
                f"Коллекция '{base}' не найдена. Сначала проиндексируйте документы."
            )

        dim = await _vector_size(qdrant, base)
        logger.info("Читаю точки из коллекции %s (dim=%d)…", base, dim)
        points = await _scroll_all(qdrant, base)
        if not points:
            raise SystemExit(f"Коллекция '{base}' пуста.")
        logger.info("Загружено %d точек", len(points))

        cos_name = await _clone_collection(
            qdrant, base, "cosine", Distance.COSINE, points, dim,
            batch_size=args.batch_size,
        )
        dot_name = await _clone_collection(
            qdrant, base, "dot", Distance.DOT, points, dim,
            batch_size=args.batch_size,
        )
        logger.info("Временные коллекции: %s, %s", cos_name, dot_name)

        embedder = _build_embedder(settings)
        queries = _load_queries(Path(args.golden), n=5)
        logger.info("Запросов: %d", len(queries))

        query_vectors = await asyncio.to_thread(
            lambda: [embedder.get_query_embedding(q) for q in queries]
        )

        width_q = 48
        width_top = 40
        header = (
            f"{'Запрос':<{width_q}} | {'top-K cosine':<{width_top}} | "
            f"{'top-K dot':<{width_top}} | match"
        )
        logger.info("\n%s", header)
        logger.info("-" * len(header))

        rows = []
        n_match = 0
        for q, qv in zip(queries, query_vectors, strict=True):
            cos_top = await _top_k(qdrant, cos_name, qv, args.top_k)
            dot_top = await _top_k(qdrant, dot_name, qv, args.top_k)
            match = cos_top == dot_top
            n_match += int(match)
            rows.append(
                {
                    "query": q,
                    "cosine": cos_top,
                    "dot": dot_top,
                    "match": match,
                }
            )
            logger.info(
                "%-*s | %-*s | %-*s | %s",
                width_q,
                q[: width_q - 2],
                width_top,
                ",".join(c[:8] for c in cos_top),
                width_top,
                ",".join(d[:8] for d in dot_top),
                "✓" if match else "✗",
            )

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "\nСовпало %d/%d. Результат: %s", n_match, len(queries), out
        )

        if not args.keep:
            await qdrant.delete_collection(cos_name)
            await qdrant.delete_collection(dot_name)
            logger.info("Временные коллекции удалены")
        else:
            logger.info(
                "Временные коллекции сохранены: %s, %s", cos_name, dot_name
            )
    finally:
        await qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())
