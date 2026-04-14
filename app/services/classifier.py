import json
from functools import lru_cache
from typing import Literal

from openai import OpenAI

from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

Route = Literal["rag", "general"]

CLASSIFIER_SYSTEM_PROMPT = (
    "You classify user queries into one of two routes for a legal assistant:\n"
    "- \"rag\": the query asks about law, cases, statutes, legal procedures, "
    "rights, contracts, or anything that likely requires retrieving reference "
    "material to answer accurately.\n"
    "- \"general\": greetings, small talk, meta questions about the assistant, "
    "or questions fully answerable from common knowledge without references.\n"
    'Respond ONLY with JSON of the form: {"route": "rag" | "general"}. '
    "When unsure, prefer \"rag\"."
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def classify(query: str) -> Route:
    try:
        resp = _client().chat.completions.create(
            model=settings.openai_llm_model,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        route = str(data.get("route", "")).strip().lower()
        if route not in {"rag", "general"}:
            logger.warning("Classifier returned unknown route '%s' — defaulting to rag", route)
            route = "rag"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Classifier failed, defaulting to rag: %s", exc)
        route = "rag"

    logger.info("Query classified as '%s'", route)
    return route  # type: ignore[return-value]
