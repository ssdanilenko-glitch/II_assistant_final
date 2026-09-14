from unittest.mock import AsyncMock, patch
import aiosmtplib

import pytest

from app.core.config import Settings
from app.services.email_service import EmailService, get_email_service


@pytest.fixture
def email_service():
    """Создаёт экземпляр EmailService с тестовыми настройками."""
    # Создаём настройки с тестовыми данными
    settings = Settings(
        yandex_email="test@yandex.ru",
        yandex_app_password="test_password",
        exchange_recipient_email="default@example.com"
    )
    # Патчим get_settings, чтобы возвращала наши тестовые настройки
    with patch("app.services.email_service.get_settings", return_value=settings):
        service = EmailService()
        return service


@pytest.mark.asyncio
async def test_send_message_success(email_service):
    """Тест успешной отправки письма."""
    # Мокаем SMTP-сессию
    mock_smtp = AsyncMock()
    mock_smtp.__aenter__.return_value = mock_smtp
    mock_smtp.login = AsyncMock()
    mock_smtp.send_message = AsyncMock()

    with patch("app.services.email_service.aiosmtplib.SMTP", return_value=mock_smtp):
        result = await email_service.send_message(
            subject="Test Subject",
            body="Test Body",
            recipient="recipient@example.com"
        )

    assert result is True
    mock_smtp.login.assert_called_once_with("test@yandex.ru", "test_password")
    mock_smtp.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_send_message_authentication_error(email_service):
    """Тест ошибки аутентификации."""
    mock_smtp = AsyncMock()
    mock_smtp.__aenter__.return_value = mock_smtp
    mock_smtp.login = AsyncMock(
        side_effect=aiosmtplib.errors.SMTPAuthenticationError(535, "Authentication failed")
    )
    with patch("app.services.email_service.aiosmtplib.SMTP", return_value=mock_smtp):
        result = await email_service.send_message(
            subject="Test",
            body="Test",
            recipient="recipient@example.com"
        )

    assert result is False
    mock_smtp.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_general_error(email_service):
    """Тест общей ошибки при отправке."""
    mock_smtp = AsyncMock()
    mock_smtp.__aenter__.return_value = mock_smtp
    mock_smtp.login = AsyncMock()
    mock_smtp.send_message = AsyncMock(side_effect=Exception("Connection error"))

    with patch("app.services.email_service.aiosmtplib.SMTP", return_value=mock_smtp):
        result = await email_service.send_message(
            subject="Test",
            body="Test",
            recipient="recipient@example.com"
        )

    assert result is False


@pytest.mark.asyncio
async def test_send_message_no_recipient(email_service):
    """Тест отправки без указания получателя (используется дефолтный)."""
    # Переопределяем recipient_email в сервисе для этого теста
    email_service.recipient_email = "default@example.com"

    mock_smtp = AsyncMock()
    mock_smtp.__aenter__.return_value = mock_smtp
    mock_smtp.login = AsyncMock()
    mock_smtp.send_message = AsyncMock()

    with patch("app.services.email_service.aiosmtplib.SMTP", return_value=mock_smtp):
        result = await email_service.send_message(
            subject="Test",
            body="Test",
            recipient=None  # Не передаём получателя
        )

    assert result is True
    # Проверяем, что письмо отправлено на дефолтный адрес
    call_args = mock_smtp.send_message.call_args[0][0]
    assert call_args["To"] == "default@example.com"


@pytest.mark.asyncio
async def test_send_message_no_sender_credentials():
    """Тест, когда учётные данные отправителя не заданы."""
    with patch("app.services.email_service.get_settings") as mock_get_settings:
        # Настройки без email и пароля
        mock_get_settings.return_value = Settings(
            yandex_email="",
            yandex_app_password="",
            exchange_recipient_email=""
        )
        service = EmailService()
        result = await service.send_message(
            subject="Test",
            body="Test",
            recipient="recipient@example.com"
        )
    assert result is False


@pytest.mark.asyncio
async def test_send_message_html(email_service):
    """Тест отправки HTML-письма."""
    mock_smtp = AsyncMock()
    mock_smtp.__aenter__.return_value = mock_smtp
    mock_smtp.login = AsyncMock()
    mock_smtp.send_message = AsyncMock()

    with patch("app.services.email_service.aiosmtplib.SMTP", return_value=mock_smtp):
        result = await email_service.send_message(
            subject="HTML Test",
            body="<h1>Hello</h1>",
            recipient="test@example.com",
            is_html=True
        )

    assert result is True
    # Проверяем, что письмо имеет тип text/html
    msg = mock_smtp.send_message.call_args[0][0]
    assert msg.is_multipart() is True
    # Находим HTML-часть
    html_part = None
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            html_part = part
            break
    assert html_part is not None
    payload = html_part.get_payload(decode=True)
    assert payload is not None
    assert payload.decode() == "<h1>Hello</h1>"

def test_singleton():
    """Тест синглтона."""
    service1 = get_email_service()
    service2 = get_email_service()
    assert service1 is service2
