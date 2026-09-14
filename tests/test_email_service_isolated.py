from unittest.mock import AsyncMock, patch

import aiosmtplib
import pytest

from app.core.config import SecretStr, Settings
from app.services.email_service import EmailService


@pytest.fixture
def mock_settings():
    """Фикстура для подмены настроек на тестовые."""
    test_settings = Settings(
        yandex_email="test@yandex.ru",
        yandex_app_password=SecretStr("test_password"),
        exchange_recipient_email="default@example.com"
    )
    with patch("app.services.email_service.get_settings", return_value=test_settings):
        yield test_settings


@pytest.fixture
def empty_settings():
    """Настройки с пустыми учётными данными."""
    test_settings = Settings(
        yandex_email="",
        yandex_app_password=SecretStr(""),
        exchange_recipient_email=""
    )
    with patch("app.services.email_service.get_settings", return_value=test_settings):
        yield test_settings


@pytest.mark.asyncio
async def test_send_message_success(mock_settings):
    service = EmailService()
    mock_smtp = AsyncMock()
    mock_smtp.__aenter__.return_value = mock_smtp
    mock_smtp.login = AsyncMock()
    mock_smtp.send_message = AsyncMock()

    with patch("app.services.email_service.aiosmtplib.SMTP", return_value=mock_smtp):
        result = await service.send_message(
            subject="Test Subject",
            body="Test Body",
            recipient="recipient@example.com"
        )

    assert result is True
    mock_smtp.login.assert_called_once_with("test@yandex.ru", "test_password")
    mock_smtp.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_send_message_authentication_error(mock_settings):
    service = EmailService()
    mock_smtp = AsyncMock()
    mock_smtp.__aenter__.return_value = mock_smtp
    mock_smtp.login = AsyncMock(side_effect=aiosmtplib.errors.SMTPAuthenticationError(535, "Auth failed"))

    with patch("app.services.email_service.aiosmtplib.SMTP", return_value=mock_smtp):
        result = await service.send_message(
            subject="Test",
            body="Test",
            recipient="recipient@example.com"
        )

    assert result is False
    mock_smtp.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_general_error(mock_settings):
    service = EmailService()
    mock_smtp = AsyncMock()
    mock_smtp.__aenter__.return_value = mock_smtp
    mock_smtp.login = AsyncMock()
    mock_smtp.send_message = AsyncMock(side_effect=Exception("Network error"))

    with patch("app.services.email_service.aiosmtplib.SMTP", return_value=mock_smtp):
        result = await service.send_message(
            subject="Test",
            body="Test",
            recipient="recipient@example.com"
        )

    assert result is False


@pytest.mark.asyncio
async def test_send_message_no_recipient_uses_default(mock_settings):
    service = EmailService()
    service.recipient_email = "default@example.com"
    mock_smtp = AsyncMock()
    mock_smtp.__aenter__.return_value = mock_smtp
    mock_smtp.login = AsyncMock()
    mock_smtp.send_message = AsyncMock()

    with patch("app.services.email_service.aiosmtplib.SMTP", return_value=mock_smtp):
        result = await service.send_message(
            subject="Test",
            body="Test",
            recipient=None
        )

    assert result is True
    msg = mock_smtp.send_message.call_args[0][0]
    assert msg["To"] == "default@example.com"


@pytest.mark.asyncio
async def test_send_message_no_sender_credentials(empty_settings):
    """Проверяем, что без учётных данных отправка не происходит."""
    service = EmailService()
    result = await service.send_message(
        subject="Test",
        body="Test",
        recipient="recipient@example.com"
    )
    assert result is False


@pytest.mark.asyncio
async def test_send_message_html(mock_settings):
    service = EmailService()
    mock_smtp = AsyncMock()
    mock_smtp.__aenter__.return_value = mock_smtp
    mock_smtp.login = AsyncMock()
    mock_smtp.send_message = AsyncMock()

    with patch("app.services.email_service.aiosmtplib.SMTP", return_value=mock_smtp):
        result = await service.send_message(
            subject="HTML Test",
            body="<h1>Hello</h1>",
            recipient="test@example.com",
            is_html=True
        )

    assert result is True
    msg = mock_smtp.send_message.call_args[0][0]
    html_part = None
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            html_part = part
            break
    assert html_part is not None
    content = html_part.get_payload(decode=True).decode('utf-8')
    assert content == "<h1>Hello</h1>"


def test_singleton():
    import app.services.email_service as email_module
    from app.services.email_service import get_email_service
    email_module._email_service = None
    s1 = get_email_service()
    s2 = get_email_service()
    assert s1 is s2
