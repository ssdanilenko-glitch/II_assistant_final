# tests/test_llm_unavailable.py
import pytest
from unittest.mock import patch
from openai import APIConnectionError
from httpx import ASGITransport, AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_llm_unavailable_returns_503():
    with patch("app.services.llm.LLMService.complete",
               side_effect=APIConnectionError(request=None)):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/chat", json={...})
    assert resp.status_code == 503