from collections import defaultdict

from app.config import settings
from app.models.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int | None = None,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion across any number of ranked lists.

    RRF score for a document is sum(1 / (k + rank_i)) over each list where it
    appears (rank_i is 1-indexed). Documents appearing in multiple lists rise.
    """
    k = k or settings.rrf_k
    if not ranked_lists:
        return []

    scores: dict[str, float] = defaultdict(float)
    seen: dict[str, RetrievedChunk] = {}
    origins: dict[str, set[str]] = defaultdict(set)

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.id] += 1.0 / (k + rank)
            origins[chunk.id].add(chunk.source)
            if chunk.id not in seen:
                seen[chunk.id] = chunk

    fused = [
        seen[doc_id].model_copy(
            update={
                "score": score,
                "source": "+".join(sorted(origins[doc_id])) + "+rrf",
            }
        )
        for doc_id, score in scores.items()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused
