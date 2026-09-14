"""Метрики RAGAS 0.4 и сборка строки оценки."""

from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, SecretStr
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import discrete_metric
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from app.core.config import Settings, get_settings


@dataclass
class RagasMetrics:
    faithfulness: Faithfulness
    answer_relevancy: AnswerRelevancy
    context_precision: ContextPrecision
    context_recall: ContextRecall
    factual_correctness: FactualCorrectness


def _resolve_api_key(settings: Settings) -> str:
    key = settings.llm.openai_api_key
    return key.get_secret_value() if isinstance(key, SecretStr) else str(key)


def _resolve_judge_url(settings: Settings) -> str:
    eval_url = getattr(settings, "EVAL_JUDGE_URL", "") or ""
    if eval_url.strip():
        return eval_url.strip()
    return settings.llm.base_url


def build_judge(settings: Settings | None = None) -> tuple[Any, OpenAIEmbeddings]:
    """Судья (LLM) и эмбеддинги для метрик. Эмбеддинги — OpenAI (для RAGAS)."""
    settings = settings or get_settings()

    api_key = _resolve_api_key(settings)
    judge_url = _resolve_judge_url(settings)
    openai_client = AsyncOpenAI(api_key=api_key, base_url=judge_url)

    # Эмбеддинги для AnswerRelevancy (OpenAI-совместимые).
    # Модель можно вынести в настройки, но по умолчанию — text-embedding-3-small.
    embeddings = OpenAIEmbeddings(
        client=openai_client,
        model="text-embedding-3-small",
    )

    provider = getattr(settings, "eval_judge_provider", "openai")
    judge_model = getattr(settings, "eval_judge_model", "gpt-4o-mini")

    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        anthropic_key_attr = getattr(settings, "anthropic_api_key", None)
        anthropic_key = (
            anthropic_key_attr.get_secret_value()
            if isinstance(anthropic_key_attr, SecretStr)
            else anthropic_key_attr
        )
        if not anthropic_key:
            raise ValueError(
                "eval_judge_provider=anthropic, но ANTHROPIC_API_KEY не задан"
            )
        judge = llm_factory(
            judge_model,
            provider="anthropic",
            client=AsyncAnthropic(api_key=anthropic_key),
        )
    else:
        judge = llm_factory(
            judge_model,
            provider="openai",
            client=openai_client,
        )

    return judge, embeddings


def build_metrics(judge: Any, embeddings: OpenAIEmbeddings) -> RagasMetrics:
    return RagasMetrics(
        faithfulness=Faithfulness(llm=judge),
        answer_relevancy=AnswerRelevancy(llm=judge, embeddings=embeddings),
        context_precision=ContextPrecision(llm=judge),
        context_recall=ContextRecall(llm=judge),
        factual_correctness=FactualCorrectness(llm=judge),
    )


class CitationVerdict(BaseModel):
    has_citation: Literal["yes", "no"]


CITATION_PROMPT = (
    "Содержит ли ответ ссылку на источник: маркер вида '[1]'/'[doc_id]', имя "
    "файла, или фразу 'согласно ...', 'в источнике X указано'?\n\n"
    "Ответ: {response}"
)


def make_has_citation(judge: Any):
    @discrete_metric(name="has_citation", allowed_values=["yes", "no"])
    async def has_citation(response: str) -> str:
        verdict = await judge.agenerate(
            CITATION_PROMPT.format(response=response),
            response_model=CitationVerdict,
        )
        return verdict.has_citation

    return has_citation


async def eval_row(rag: Any, row: dict, metrics: RagasMetrics, has_citation: Any) -> dict:
    result = await rag.evaluate_inputs(row["user_input"])
    answer, contexts = result["answer"], result["retrieved_contexts"]
    q, ref = row["user_input"], row["reference"]
    return {
        "user_input": q,
        "faithfulness": (
            await metrics.faithfulness.ascore(
                user_input=q, response=answer, retrieved_contexts=contexts
            )
        ).value,
        "answer_relevancy": (
            await metrics.answer_relevancy.ascore(user_input=q, response=answer)
        ).value,
        "context_precision": (
            await metrics.context_precision.ascore(
                user_input=q, reference=ref, retrieved_contexts=contexts
            )
        ).value,
        "context_recall": (
            await metrics.context_recall.ascore(
                user_input=q, retrieved_contexts=contexts, reference=ref
            )
        ).value,
        "factual_correctness": (
            await metrics.factual_correctness.ascore(response=answer, reference=ref)
        ).value,
        "has_citation": (await has_citation.ascore(response=answer)).value,
    }
