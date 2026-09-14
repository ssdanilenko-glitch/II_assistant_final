"""Проверка доступности Ollama. Запуск: python scripts/check_ollama.py

Не тест — ручная диагностика. Требует либо запущенной Ollama на хосте,
либо проброшенного порта из контейнера (docker compose port ollama 11434).
"""

from __future__ import annotations

import sys

import requests

OLLAMA_URL = "http://localhost:11434"


def main() -> int:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=2)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"❌ Ollama недоступна по {OLLAMA_URL}: {exc}")
        print("   Проверьте: docker compose ps ollama")
        return 1

    print(f"✅ Ollama доступна: {r.json()}")

    payload = {
        "model": "frob/qwen3.5-instruct:4b",
        "messages": [{"role": "user", "content": "Привет!"}],
        "stream": False,
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"❌ Ошибка при запросе к модели: {exc}")
        return 2

    print("Ответ модели:", r.json().get("message", {}).get("content", "—"))
    return 0


if __name__ == "__main__":
    sys.exit(main())