"""Выводит список имён документов, проиндексированных в Qdrant."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)

    result = client.scroll(collection_name=settings.rag_collection, limit=500)
    names = sorted({
        (p.payload or {}).get("file_name") or (p.payload or {}).get("source") or "unknown"
        for p in result[0]
    })
    print(f"Всего документов: {len(names)}\n")
    for n in names:
        print(f"  - {n}")

    client.close()


if __name__ == "__main__":
    main()
