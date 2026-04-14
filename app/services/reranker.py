from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.core.logger import get_logger
from app.models.schemas import RetrievedChunk

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    logger.info("Loading cross-encoder '%s'", settings.reranker_model)
    return CrossEncoder(settings.reranker_model)


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    top_k = top_k or settings.top_k_rerank

    pairs = [(query, c.text) for c in chunks]
    scores = _model().predict(pairs)

    reranked = [
        c.model_copy(update={"score": float(s), "source": f"{c.source}+rerank"})
        for c, s in zip(chunks, scores, strict=True)
    ]
    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked[:top_k]
