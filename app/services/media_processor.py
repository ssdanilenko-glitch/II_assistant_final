import base64
import logging
from io import BytesIO

from openai import AsyncOpenAI

from app.core.config import get_settings

# Импорт faster-whisper – лучше вынести внутрь метода, чтобы не падать при отсутствии
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)


class MediaProcessor:
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.llm.openai_api_key.get_secret_value(),
            base_url=settings.llm.base_url,
            timeout=settings.llm.request_timeout,
            max_retries=settings.llm.max_retries,
        )
        # Для vision используем модель, поддерживающую изображения.
        # Лучше взять из настроек отдельную переменную или явно указать.
        self.vision_model = settings.vision_model # или settings.llm.vision_model, если добавить
        self.prompt_ru = (
            "Опиши это изображение подробно на русском языке. "
            "Включи все видимые объекты, людей, действия, контекст, а также любой текст, который можно прочитать на изображении. "
            "Если текст есть, перепиши его точно. Ответ должен быть на русском языке."
        )
        # Инициализация Whisper модели один раз (если установлен)
        self.whisper_model = None
        if WhisperModel is not None:
            try:
                self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
                logger.info("Whisper модель загружена")
            except Exception as e:
                logger.warning(f"Не удалось загрузить Whisper модель: {e}")

    async def process_image(self, data: bytes) -> str:
        try:
            b64 = base64.b64encode(data).decode()
            # Можно определить MIME-тип, но для простоты оставим jpeg
            mime_type = "image/jpeg"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt_ru},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"}
                        }
                    ]
                }
            ]
            response = await self.client.chat.completions.create(
                model=self.vision_model,  # явно используем vision-модель
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            description = response.choices[0].message.content.strip()
            return description or "Не удалось получить описание изображения."
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {e}")
            return "Не удалось описать изображение из-за технической ошибки."

    async def _process_audio(self, data: bytes, filename: str) -> str:
        if self.whisper_model is None:
            return "Аудиораспознавание недоступно: модель не загружена."
        try:
            segments, info = self.whisper_model.transcribe(BytesIO(data), language="ru")
            result = " ".join([seg.text for seg in segments])
            return result.strip() or "Речь не распознана."
        except Exception as e:
            logger.error(f"Ошибка распознавания аудио: {e}")
            return "Не удалось распознать аудио."

    def _process_pdf(self, data: bytes) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(data))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text[:30000]
        except Exception as e:
            logger.error(f"Ошибка обработки PDF: {e}")
            return "Не удалось извлечь текст из PDF."

    def _process_docx(self, data: bytes) -> str:
        try:
            from docx import Document
            doc = Document(BytesIO(data))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return text[:30000]
        except Exception as e:
            logger.error(f"Ошибка обработки DOCX: {e}")
            return "Не удалось извлечь текст из DOCX."

    async def process(self, data: bytes, content_type: str, filename: str) -> str:
        if content_type.startswith("image/"):
            return await self.process_image(data)
        elif content_type.startswith("audio/") or content_type == "application/ogg":
            return await self._process_audio(data, filename)
        elif content_type == "application/pdf":
            return self._process_pdf(data)
        elif content_type.endswith("wordprocessingml.document"):
            return self._process_docx(data)
        else:
            raise ValueError(f"Unsupported media type: {content_type}")
