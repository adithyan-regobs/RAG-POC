from app.config import settings
from app.core.logger import get_logger
from app.models.schemas import RetrievedChunk

logger = get_logger(__name__)


def _csv_set(value: str) -> set[str]:
    return {v.strip().lower() for v in value.split(",") if v.strip()}


def filter_low_quality(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop chunks that are too short or explicitly marked low-quality."""
    min_len = settings.min_text_length
    kept: list[RetrievedChunk] = []
    for c in chunks:
        text_len = len(c.text.strip())
        if text_len < min_len:
            continue
        if c.metadata.get("low_quality") is True:
            continue
        kept.append(c)
    dropped = len(chunks) - len(kept)
    if dropped:
        logger.info("Filtered %d low-quality chunks (kept %d)", dropped, len(kept))
    return kept


def apply_metadata_boost(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Add a bonus to scores for preferred law areas / document types."""
    boost_areas = _csv_set(settings.boost_law_areas)
    boost_types = _csv_set(settings.boost_document_types)
    boost = settings.boost_value
    if not (boost_areas or boost_types) or boost == 0:
        return chunks

    boosted: list[RetrievedChunk] = []
    for c in chunks:
        delta = 0.0
        if str(c.metadata.get("law_area", "")).lower() in boost_areas:
            delta += boost
        if str(c.metadata.get("document_type", "")).lower() in boost_types:
            delta += boost
        if delta:
            boosted.append(c.model_copy(update={"score": c.score + delta}))
        else:
            boosted.append(c)

    boosted.sort(key=lambda c: c.score, reverse=True)
    return boosted


def process(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return apply_metadata_boost(filter_low_quality(chunks))
