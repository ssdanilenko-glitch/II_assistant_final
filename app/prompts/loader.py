"""Загрузка и сборка промптов для LLM."""

from __future__ import annotations

from importlib import resources
from typing import Any


def _read_prompt_file(filename: str) -> str:
    return resources.files(__package__).joinpath(filename).read_text(encoding="utf-8").strip()


SERVICE_FACTS = _read_prompt_file("service_facts.txt")
SYSTEM_PROMPT_TEMPLATE = _read_prompt_file("Sistem_promt_agent.txt")


def build_system_prompt(service_name: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        service_name=service_name,
        service_facts=SERVICE_FACTS,
    )


def build_classifier_messages(user_message: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT}]
    messages.extend(classifier_few_shot_messages())
    messages.append({"role": "user", "content": user_message})
    return messages
