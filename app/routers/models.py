from fastapi import APIRouter

from app.deps.providers import SettingsDep
from app.schemas.models import ModelInfo

router = APIRouter(prefix="/models", tags=["models"])

CATALOG: dict[str, ModelInfo] = {
    "frob/qwen3.5-instruct:4b": ModelInfo(
        id="frob/qwen3.5-instruct:4b",
        provider="ollama",
        input_per_1m=0.0,
        output_per_1m=0.0,
        context_window=128_000,
    ),
    "qwen2.5:7b": ModelInfo(
        id="qwen2.5:7b",
        provider="ollama",
        input_per_1m=0.0,
        output_per_1m=0.0,
        context_window=32_768,
    ),
}


@router.get("", response_model=list[ModelInfo])
async def list_models(settings: SettingsDep) -> list[ModelInfo]:
    return list(CATALOG.values())
