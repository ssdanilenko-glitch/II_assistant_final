"""Офлайн-контур RAG: парсинг корпуса, обогащение метаданными и индексация."""
import logging
import re
from datetime import date
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader, UnstructuredReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.core.config import Settings as AppSettings

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = [".pdf", ".docx", ".md", ".txt", ".html"]

# Технические поля, исключаемые из эмбеддинга
EXCLUDED_EMBED_KEYS = [
    "file_path", "file_name", "file_type", "file_size",
    "creation_date", "last_modified_date", "doc_type",
    "version", "visibility", "indexed_at",
    "original_content",  # ← добавить
]

class OCRUnstructuredReader(UnstructuredReader):
    """Обёртка, передающая strategy='ocr_only' в load_data."""
    def load_data(self, file, extra_info=None):
        return super().load_data(
            file,
            extra_info=extra_info,
            unstructured_kwargs={
                "strategy": "ocr_only",
                "languages": ["rus", "eng"]  # ← добавьте русский и английский
            }
        )

def clean(text: str) -> str:
    """Снимает типичный шум PDF-экспорта перед чанкингом."""
    text = re.sub(r"Стр\.\s*\d+\s*из\s*\d+", "", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


def department_from_path(path: str) -> str:
    parts = Path(path).parts
    for anchor in ("knowledge_base", "data"):
        if anchor in parts:
            idx = parts.index(anchor)
            return parts[idx + 1] if len(parts) > idx + 1 else "general"
    return "general"


def doc_type_from_path(path: str) -> str:
    return Path(path).suffix.lstrip(".").lower() or "unknown"


def version_from_filename(path: str) -> str:
    match = re.search(r"(20\d{2}(?:[_-]v?\d+)?)", Path(path).stem)
    return match.group(1) if match else "unversioned"


def file_metadata(path: str) -> dict[str, str]:
    return {
        "source": Path(path).name,
        "department": department_from_path(path),
        "doc_type": doc_type_from_path(path),
        "version": version_from_filename(path),
        "visibility": "internal",
        "indexed_at": date.today().isoformat(),
    }


def enrich(documents: list[Document]) -> list[Document]:
    """Чистит текст и помечает технические поля исключёнными из эмбеддинга."""
    for doc in documents:
        doc.metadata["original_content"] = doc.text
        doc.set_content(clean(doc.text))
        doc.excluded_embed_metadata_keys = EXCLUDED_EMBED_KEYS
        doc.excluded_llm_metadata_keys = EXCLUDED_EMBED_KEYS
    return documents

class IngestionService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        qdrant_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
               and settings.qdrant_api_key.get_secret_value()
            else None
        )
        self._data_dir = Path(settings.rag_data_dir)
        self._docstore_path = self._data_dir.parent / f"{settings.rag_collection}_docstore.json"

        self._client = QdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=settings.rag_collection,
            text_key="text",
        )

        # ✅ Создаём коллекцию, если её нет
        self._ensure_collection_exists()

        self._embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
            trust_remote_code=settings.embedding.trust_remote_code,
            embed_batch_size=32,
        )
        self._docstore = self._load_docstore()
        self._pipeline = self._build_pipeline()

    def _ensure_collection_exists(self) -> None:
        """Создаёт коллекцию в Qdrant, если она не существует."""
        from qdrant_client.http.models import Distance, VectorParams
        if not self._client.collection_exists(self._settings.rag_collection):
            self._client.create_collection(
                collection_name=self._settings.rag_collection,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            logger.info(f"Коллекция {self._settings.rag_collection} создана")
    @property
    def docstore_path(self) -> Path:
        return self._docstore_path

    def _build_pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self._settings.rag_chunk_size,
                    chunk_overlap=self._settings.rag_chunk_overlap
                ),
                self._embed_model,
            ],
            docstore=self._docstore,
            vector_store=self._vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )
    def _load_docstore(self) -> SimpleDocumentStore:
        if self._docstore_path.exists():
            return SimpleDocumentStore.from_persist_path(str(self._docstore_path))
        return SimpleDocumentStore()

    def is_collection_empty(self) -> bool:
        if not self._client.collection_exists(self._settings.rag_collection):
            return True
        return self._client.count(self._settings.rag_collection).count == 0

    def _persist_docstore(self) -> None:
        self._docstore_path.parent.mkdir(parents=True, exist_ok=True)
        self._docstore.persist(str(self._docstore_path))

    def _read(self, *, input_files: list[Path] | None = None) -> list[Document]:
        reader = SimpleDirectoryReader(
            input_dir=str(self._data_dir) if input_files is None else None,
            input_files=[str(p) for p in input_files] if input_files else None,
            recursive=input_files is None,
            required_exts=SUPPORTED_EXTS,
            file_metadata=file_metadata,
            filename_as_id=True,
            file_extractor={".pdf": PDFReader()}, #{".pdf": OCRUnstructuredReader()},
        )
        docs = reader.load_data()
        # Фильтруем пустые документы
        docs = [doc for doc in docs if doc.text and doc.text.strip()]
        return enrich(docs)

    def ingest_all(self) -> int:
        documents = self._read()
        nodes = self._pipeline.run(documents=documents, show_progress=False)
        self._persist_docstore()
        logger.info(
            "ingestion: корпус проиндексирован, документов=%d нод=%d",
            len(documents),
            len(nodes),
        )
        return len(nodes)

    def ingest_files(self, paths: list[str]) -> int:
        files = [Path(p) for p in paths if Path(p).exists()]
        if not files:
            return 0
        documents = self._read(input_files=files)
        nodes = self._pipeline.run(documents=documents, show_progress=False)
        self._persist_docstore()
        logger.info("ingestion: точечно проиндексировано файлов=%d нод=%d", len(files), len(nodes))
        return len(nodes)

    def reindex_all(self) -> int:
        if self._client.collection_exists(self._settings.rag_collection):
            self._client.delete_collection(self._settings.rag_collection)
            self._ensure_collection_exists()
        self._docstore = SimpleDocumentStore()
        self._docstore_path.unlink(missing_ok=True)
        self._pipeline = self._build_pipeline()
        logger.info("ingestion: полная переиндексация коллекции %s", self._settings.rag_collection)
        return self.ingest_all()

    def run_for_file(self, path: Path) -> int:
        try:
            count = self.ingest_files([str(path)])
            logger.info("ingestion: файл проиндексирован file=%s нод=%d", path.name, count)
            return count
        except Exception:
            logger.exception("ingestion: файл не проиндексирован file=%s", path.name)
            path.rename(path.with_suffix(path.suffix + ".failed"))
            return 0

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.debug("ошибка при закрытии Qdrant-клиента ingestion", exc_info=True)
