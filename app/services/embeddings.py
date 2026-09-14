"""Батчевый клиент embeddings поверх OpenAI-совместимого API."""
import asyncio
import logging
from collections.abc import Sequence

import numpy as np
from openai import APIError, APITimeoutError, AsyncOpenAI

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """Клиент для получения векторных представлений текста."""

    def __init__(
            self,
            client: AsyncOpenAI,
            model: str = "qwen3-embedding-0.6b",  # Имя модели в локальном сервере
            batch_size: int = 64,
            max_retries: int = 2,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Возвращает по вектору на каждый текст в том же порядке.
        Автоматически разбивает входные данные на батчи и обрабатывает ошибки.
        """
        if not texts:
            return []

        out: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = list(texts[i: i + self._batch_size])

            for attempt in range(self._max_retries + 1):
                try:
                    resp = await self._client.embeddings.create(
                        model=self._model,
                        input=batch
                    )
                    out.extend(_normalize_vectors([item.embedding for item in resp.data]))
                    #out.extend([item.embedding for item in resp.data])
                    logger.debug(f"✅ Эмбеддинги получены для батча {i // self._batch_size + 1} ({len(batch)} текстов)")
                    break

                except APITimeoutError as e:
                    logger.warning(f"⏱️ Таймаут при получении эмбеддингов (попытка {attempt + 1}): {e}")
                    if attempt == self._max_retries:
                        raise
                    await asyncio.sleep(2 ** attempt)

                except APIError as e:
                    logger.error(f"❌ Ошибка API при получении эмбеддингов: {e.status_code} - {e.message}")
                    raise

                except Exception as e:
                    logger.exception(f"🔥 Неожиданная ошибка при эмбеддингах: {e}")
                    raise

        return out

    async def embed_one(self, text: str) -> list[float]:
        """Получить эмбеддинг для одного текста."""
        result = await self.embed([text])
        return result[0]


# --- Тестовая функция ---
async def _smoke() -> None:
    """Проверка работоспособности клиента."""
    import os

    # Настройки для локального сервера (Ollama/vLLM)
    base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("EMBEDDING_API_KEY", "ollama")
    model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-0.6b")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    embed_client = EmbeddingsClient(client, model=model)

    try:
        vec = await embed_client.embed_one("Как создать документ реализации в 1С?")
        print(f"✅ Успех! Размерность вектора: {len(vec)}")
        print(f"   Первые 5 значений: {[round(v, 4) for v in vec[:5]]}")
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
    finally:
        await client.close()

def _normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    """L2-нормализация списка векторов."""
    if not vectors:
        return vectors
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Защита от деления на ноль
    norms = np.where(norms == 0, 1.0, norms)
    normalized = arr / norms
    return normalized.tolist()



if __name__ == "__main__":
    asyncio.run(_smoke())
