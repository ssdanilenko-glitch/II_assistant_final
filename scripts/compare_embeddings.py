"""Сравнение моделей эмбеддингов на реальном корпусе проекта.

Методика:
1. Читает корпус (тексты чанков + source) из Qdrant-коллекции.
2. Загружает golden dataset (user_input + reference + source).
3. Ground truth для вопроса = все чанки из документа, указанного в source
   (document-level retrieval: правильно ли найден нужный документ).
4. Для каждой модели эмбеддингов:
   - считает эмбеддинги чанков и запросов;
   - ранжирует чанки по косинусной близости;
   - считает Recall@K, MRR@K, nDCG@K, Hit@K.

Запуск:
    docker compose exec app python scripts/compare_embeddings.py
    docker compose exec app python scripts/compare_embeddings.py --top-k 1 3 5
    docker compose exec aПроpp python scripts/compare_embeddings.py \
        --models BAAI/bge-m3 intfloat/multilingual-e5-small
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402

logger = logging.getLogger("embeddings")
logging.basicConfig(level=logging.INFO, format="%(message)s")


CANDIDATES: list[dict] = [
    {"name": "BAAI/bge-m3", "qi": None, "ti": None},
    {"name": "intfloat/multilingual-e5-small", "qi": "query: ", "ti": "passage: "},
    {"name": "intfloat/multilingual-e5-large", "qi": "query: ", "ti": "passage: "},
]

UPSERT_TIMEOUT = 120.0
SRC_EXT_RE = re.compile(r"\.(docx?|pdf|txt|md|xlsx?|pptx?)$", re.IGNORECASE)
SRC_CLEAN_RE = re.compile(r"[^\w\u0400-\u04ff]+", re.UNICODE)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ---------- helpers ----------


def _extract_text(payload: dict | None) -> str | None:
    if not payload:
        return None
    for key in ("text", "content", "page_content"):
        v = payload.get(key)
        if isinstance(v, str) and v:
            return v
    # LlamaIndex кладёт JSON-строку в _node_content
    node = payload.get("_node_content")
    if isinstance(node, str):
        try:
            data = json.loads(node)
            for key in ("text", "content"):
                v = data.get(key)
                if isinstance(v, str) and v:
                    return v
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


def _normalize_src(s: str) -> str:
    s = SRC_EXT_RE.sub("", s)
    s = SRC_CLEAN_RE.sub("", s.lower())
    return s


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _resolve_collection(settings, override: str | None) -> str:
    if override:
        return override
    for path in ("rag.collection", "rag_collection"):
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
    return "rag_block_04"


# ---------- corpus / golden ----------


async def _load_corpus(settings, collection: str) -> tuple[list[dict], set[str]]:
    client = AsyncQdrantClient(url=settings.qdrant_url, timeout=UPSERT_TIMEOUT)
    try:
        existing = {c.name for c in (await client.get_collections()).collections}
        if collection not in existing:
            raise SystemExit(f"Коллекция '{collection}' не найдена.")

        chunks: list[dict] = []
        sources_seen: set[str] = set()
        offset = None
        while True:
            batch, offset = await client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in batch:
                text = _extract_text(p.payload)
                src = (p.payload or {}).get("source")
                if text:
                    chunks.append(
                        {
                            "id": str(p.id),
                            "text": text,
                            "source": str(src) if src else None,
                        }
                    )
                if src:
                    sources_seen.add(str(src))
            if offset is None:
                break

        if not chunks:
            raise SystemExit(
                "Не удалось извлечь тексты из payload. "
                "Проверьте структуру: ожидаются поля text / content / page_content / _node_content."
            )

        logger.info("  Чанков с текстом: %d", len(chunks))
        n_with_src = sum(1 for c in chunks if c["source"])
        logger.info("  Чанков с source: %d", n_with_src)
        logger.info("  Уникальных source: %d", len(sources_seen))

        return chunks, sources_seen
    finally:
        await client.close()


def _load_golden(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_ground_truth(
    chunks: list[dict], golden: list[dict]
) -> tuple[list[set[str]], int, list[str]]:
    """Ground truth по source: вопрос → все чанки того же документа."""
    by_source: dict[str, set[str]] = {}
    for c in chunks:
        if c["source"]:
            key = _normalize_src(c["source"])
            by_source.setdefault(key, set()).add(c["id"])

    gt: list[set[str]] = []
    unmatched: list[str] = []
    n_matched = 0

    for row in golden:
        src = row.get("source")
        if not src:
            gt.append(set())
            continue
        key = _normalize_src(str(src))
        if key in by_source:
            gt.append(by_source[key])
            n_matched += 1
            continue
        # fallback: substring
        found = None
        for k, ids in by_source.items():
            if k and (k in key or key in k):
                found = ids
                break
        if found:
            gt.append(found)
            n_matched += 1
        else:
            gt.append(set())
            unmatched.append(str(src))

    return gt, n_matched, unmatched


# ---------- IR-метрики ----------


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for i, rid in enumerate(ranked[:k]):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, rid in enumerate(ranked[:k])
        if rid in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def _hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


# ---------- model evaluation ----------


async def evaluate_model(
    candidate: dict,
    chunks: list[dict],
    queries: list[str],
    gt: list[set[str]],
    k_values: list[int],
) -> dict:

    name = candidate["name"]
    logger.info("Модель: %s — загружаю…", name)
    embedder = HuggingFaceEmbedding(
        model_name=name,
        trust_remote_code=True,
        query_instruction=candidate["qi"] or "",
        text_instruction=candidate["ti"] or "",
    )
    logger.info("  Эмбеддинги для %d чанков…", len(chunks))
    chunk_vecs = await asyncio.to_thread(
        lambda: embedder.get_text_embedding_batch([c["text"] for c in chunks])
    )
    logger.info("  Эмбеддинги для %d запросов…", len(queries))
    # get_query_embedding_batch отсутствует в HuggingFaceEmbedding,
    # используем цикл — 15 запросов обрабатываются мгновенно.
    query_vecs = await asyncio.to_thread(
        lambda: [embedder.get_query_embedding(q) for q in queries]
    )
    chunk_ids = [c["id"] for c in chunks]
    chunk_mat = np.asarray(chunk_vecs, dtype=np.float32)
    query_mat = np.asarray(query_vecs, dtype=np.float32)

    chunk_mat /= np.linalg.norm(chunk_mat, axis=1, keepdims=True) + 1e-12
    query_mat /= np.linalg.norm(query_mat, axis=1, keepdims=True) + 1e-12

    sims = query_mat @ chunk_mat.T
    max_k = min(max(k_values), len(chunk_ids))

    rankings: list[list[str]] = []
    for i in range(len(queries)):
        order = np.argsort(-sims[i])[:max_k]
        rankings.append([chunk_ids[j] for j in order])

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = float(
            np.mean([_recall_at_k(r, g, k) for r, g in zip(rankings, gt)])
        )
        metrics[f"mrr@{k}"] = float(
            np.mean([_mrr_at_k(r, g, k) for r, g in zip(rankings, gt)])
        )
        metrics[f"ndcg@{k}"] = float(
            np.mean([_ndcg_at_k(r, g, k) for r, g in zip(rankings, gt)])
        )
        metrics[f"hit@{k}"] = float(
            np.mean([_hit_at_k(r, g, k) for r, g in zip(rankings, gt)])
        )

    del embedder, chunk_vecs, query_vecs, chunk_mat, query_mat, sims
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return {"model": name, "metrics": metrics}


# ---------- main ----------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение моделей эмбеддингов")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--golden", default="tests/eval/golden_dataset.json")
    parser.add_argument("--out", default="docs/embedding_comparison.json")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Ограничить список моделей (по имени)",
    )
    args = parser.parse_args()

    settings = get_settings()
    collection = _resolve_collection(settings, args.collection)

    logger.info("Читаю корпус из Qdrant: %s", collection)
    chunks, sources_seen = await _load_corpus(settings, collection)
    logger.info("Загружено %d чанков", len(chunks))

    golden_path = Path(args.golden)
    if not golden_path.exists():
        raise SystemExit(f"Golden dataset не найден: {golden_path}")
    golden = _load_golden(golden_path)
    logger.info("Загружено %d вопросов", len(golden))

    queries = [r.get("user_input") or r.get("question") for r in golden]
    if any(q is None for q in queries):
        raise SystemExit("В golden dataset нет поля user_input / question.")

    logger.info("Строю ground truth по source…")
    gt, n_matched, unmatched = _build_ground_truth(chunks, golden)
    logger.info("  Вопросов с ground truth: %d / %d", n_matched, len(gt))
    if unmatched:
        logger.warning("  Не сопоставлено source (%d):", len(unmatched))
        for s in unmatched[:10]:
            logger.warning("    - %s", s)
        if len(unmatched) > 10:
            logger.warning("    ... и ещё %d", len(unmatched) - 10)

    if n_matched == 0:
        logger.error(
            "Ни одного ground truth. Сравните имена source из golden dataset "
            "с payload Qdrant: 'source' в чанках может отличаться."
        )
        logger.info("Уникальные source в Qdrant (первые 20):")
        for s in sorted(sources_seen)[:20]:
            logger.info("    %s", s)
        raise SystemExit("Прерываю: ground truth пуст.")

    candidates = CANDIDATES
    if args.models:
        wanted = set(args.models)
        candidates = [c for c in CANDIDATES if c["name"] in wanted]
        if not candidates:
            raise SystemExit("Ни одна модель не выбрана.")

    results = []
    for c in candidates:
        try:
            results.append(await evaluate_model(c, chunks, queries, gt, args.top_k))
        except Exception as exc:
            logger.error("Ошибка на модели %s: %s", c["name"], exc)

    logger.info("\n%s", "=" * 110)
    header_parts = [f"{'Модель':<38}"]
    for k in args.top_k:
        header_parts.append(f"R@{k:<4}")
        header_parts.append(f"nDCG@{k:<6}")
    logger.info(" | ".join(header_parts))
    logger.info("-" * 110)

    for r in results:
        cells = [f"{r['model']:<38}"]
        for k in args.top_k:
            cells.append(f"{r['metrics'][f'recall@{k}']:.3f} ")
            cells.append(f"{r['metrics'][f'ndcg@{k}']:.3f}   ")
        logger.info(" | ".join(cells))

    # Hit@K — доля вопросов, где правильный документ попал в top-K
    logger.info("\n%s", "=" * 110)
    header_parts = [f"{'Модель':<38}"]
    for k in args.top_k:
        header_parts.append(f"Hit@{k:<4}")
    logger.info(" | ".join(header_parts))
    logger.info("-" * 110)
    for r in results:
        cells = [f"{r['model']:<38}"]
        for k in args.top_k:
            cells.append(f"{r['metrics'][f'hit@{k}']:.3f} ")
        logger.info(" | ".join(cells))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "collection": collection,
                "n_chunks": len(chunks),
                "n_queries": len(queries),
                "n_with_gt": n_matched,
                "top_k": args.top_k,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("\nРезультаты сохранены в %s", out)


if __name__ == "__main__":
    asyncio.run(main())
