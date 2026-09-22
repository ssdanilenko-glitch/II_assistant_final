"""Загрузка и сборка промптов для LLM."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def _read_prompt_file(filename: str) -> str:
    return resources.files(__package__).joinpath(filename).read_text(encoding="utf-8").strip()


SYSTEM_PROMPT_TEMPLATE = _read_prompt_file("System_prompt_agent.txt")

def build_system_prompt(helpdesk_email: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(helpdesk_email=helpdesk_email)
