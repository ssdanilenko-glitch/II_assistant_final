"""Загрузка и сборка промптов для LLM."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def _read_prompt_file(filename: str) -> str:
    return resources.files(__package__).joinpath(filename).read_text(encoding="utf-8").strip()


SYSTEM_PROMPT_TEMPLATE = _read_prompt_file("System_prompt_agent.txt")
CLASSIFIER_SYSTEM_PROMPT = _read_prompt_file("classifier_system_prompt.txt")
CLASSIFIER_FEW_SHOTS = json.loads(_read_prompt_file("classifier_few_shots.json"))

def build_system_prompt(service_name: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        service_name=service_name,
    )

def classifier_few_shot_messages() -> list[dict[str, str]]:
    return CLASSIFIER_FEW_SHOTS.copy()

def build_classifier_messages(user_message: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT}]
    messages.extend(classifier_few_shot_messages())
    messages.append({"role": "user", "content": user_message})
    return messages
