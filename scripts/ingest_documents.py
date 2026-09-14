#!/usr/bin/env python
"""Скрипт для дозагрузки документов в базу знаний через IngestionService.

Запуск:
    docker compose exec app python scripts/ingest_documents.py --files data/rag-block-03/new1.pdf data/rag-block-03/new2.docx
    docker compose exec app python scripts/ingest_documents.py --reindex
"""

import argparse
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH для импорта app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.services.ingestion import IngestionService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Индексация документов в RAG")
    parser.add_argument(
        "--files",
        nargs="+",
        help="Список файлов для индексации (пути внутри контейнера)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Полная переиндексация (удаляет старую коллекцию и создаёт заново)",
    )
    args = parser.parse_args()

    if not args.files and not args.reindex:
        parser.error("Укажите --files или --reindex")

    settings = get_settings()
    svc = IngestionService(settings)

    try:
        if args.reindex:
            logger.info("Запуск полной переиндексации...")
            count = svc.reindex_all()
            logger.info("Переиндексация завершена. Проиндексировано %d нод", count)
        else:
            # Проверяем существование файлов
            existing_files = []
            for f in args.files:
                p = Path(f)
                if p.exists():
                    existing_files.append(str(p))
                else:
                    logger.warning("Файл не найден: %s", p)
            if not existing_files:
                logger.error("Нет доступных файлов для индексации")
                sys.exit(1)
            logger.info("Индексация файлов: %s", existing_files)
            count = svc.ingest_files(existing_files)
            logger.info("Индексация завершена. Проиндексировано %d нод", count)
    except Exception as e:
        logger.exception("Ошибка при индексации: %s", e)
        sys.exit(1)
    finally:
        svc.close()


if __name__ == "__main__":
    main()
