import logging
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib
from pydantic import SecretStr

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self):
        settings = get_settings()
        self.sender_email = settings.yandex_email
        password_attr = settings.yandex_app_password
        if isinstance(password_attr, SecretStr):
            self.sender_password = password_attr.get_secret_value()
        else:
            self.sender_password = password_attr
        self.recipient_email = settings.exchange_recipient_email

        self.smtp_host = "smtp.yandex.ru"
        self.smtp_port = 465
        self.use_tls = True

    async def send_message(
        self,
        subject: str,
        body: str,
        recipient: str | None = None,
        is_html: bool = False,
        attachments: list[tuple[str, bytes, str]] | None = None,  # (filename, data, mime_type)
    ) -> bool:
        if not recipient:
            recipient = self.recipient_email

        if not self.sender_email or not self.sender_password:
            logger.error("SMTP не настроен: проверьте YANDEX_EMAIL и YANDEX_APP_PASSWORD")
            return False

        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = recipient
        msg["Subject"] = subject

        content_type = "html" if is_html else "plain"
        msg.attach(MIMEText(body, content_type))

        # Добавляем вложения
        if attachments:
            for filename, data, mime_type in attachments:
                part = MIMEBase(*mime_type.split('/') if mime_type else ('application', 'octet-stream'))
                part.set_payload(data)
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{filename}"'
                )
                msg.attach(part)

        try:
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port,
                use_tls=self.use_tls
            ) as smtp:
                await smtp.login(self.sender_email, self.sender_password)
                await smtp.send_message(msg)

            logger.info(f"Email sent to {recipient}: {subject}")
            return True
        except aiosmtplib.errors.SMTPAuthenticationError:
            logger.error("SMTP Authentication failed. Check Yandex App Password.")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


# Синглтон остаётся без изменений
_email_service = None

def get_email_service() -> EmailService:
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
