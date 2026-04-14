from functools import lru_cache

import tiktoken

from app.config import settings
from app.core.logger import get_logger
from app.models.schemas import RetrievedChunk

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, model: str) -> int:
    return len(_encoding(model).encode(text))


def _format_block(chunk: RetrievedChunk) -> str:
    meta = chunk.metadata
    header = (
        f"[id={chunk.id} "
        f"source={meta.get('source', 'unknown')} "
        f"area={meta.get('law_area', '')} "
        f"type={meta.get('document_type', '')}]"
    )
    return f"{header}\n{chunk.text.strip()}"


def build_context(
    chunks: list[RetrievedChunk],
    token_budget: int | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    """Pack chunks into a single context string under a token budget.

    Returns the joined context and the subset of chunks that actually fit — the
    caller should surface that subset as `sources` so citations match the input.
    """
    budget = token_budget or settings.context_token_budget
    model = settings.openai_llm_model

    pieces: list[str] = []
    used: list[RetrievedChunk] = []
    consumed = 0

    for chunk in chunks:
        block = _format_block(chunk)
        cost = _count_tokens(block, model)
        if used and consumed + cost > budget:
            break
        pieces.append(block)
        used.append(chunk)
        consumed += cost

    logger.info(
        "Assembled context: %d/%d chunks, ~%d tokens (budget %d)",
        len(used),
        len(chunks),
        consumed,
        budget,
    )
    return "\n\n".join(pieces), used
