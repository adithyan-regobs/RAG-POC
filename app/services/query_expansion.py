import json
from functools import lru_cache

from openai import OpenAI

from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

EXPANSION_SYSTEM_PROMPT = (
    "You rewrite user questions to improve information retrieval. "
    "Produce {n} diverse rephrasings that preserve the original intent but vary "
    "in wording, specificity, and keyword choice (include synonyms, legal jargon, "
    "and plain-language forms where appropriate). "
    'Return ONLY a JSON object of the form: {{"queries": ["...", "..."]}}'
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def expand_query(question: str, n: int | None = None) -> list[str]:
    """Return the original question plus up to `n` LLM-generated rephrasings.

    The returned list is deduplicated (case-insensitive) and always starts with
    the original question so retrieval still works if expansion fails.
    """
    n = n or settings.num_query_variations

    variations: list[str] = []
    try:
        resp = _client().chat.completions.create(
            model=settings.openai_llm_model,
            messages=[
                {"role": "system", "content": EXPANSION_SYSTEM_PROMPT.format(n=n)},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        variations = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query expansion failed, using original only: %s", exc)

    seen: set[str] = set()
    result: list[str] = []
    for q in [question, *variations]:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(q.strip())
        if len(result) >= n + 1:
            break

    logger.info("Query expanded into %d queries", len(result))
    return result
