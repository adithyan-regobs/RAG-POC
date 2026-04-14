import uuid
from collections.abc import Iterable, Iterator
from typing import TypeVar

from qdrant_client.http import models as qmodels

from app.config import settings
from app.core.logger import get_logger
from app.db.qdrant_client import get_qdrant_client, init_qdrant
from app.models.schemas import Document
from app.services.bm25_search import bm25_index
from app.services.embedding_service import embed_texts

logger = get_logger(__name__)

T = TypeVar("T")


def _batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _ensure_point_id(doc_id: str) -> str:
    try:
        uuid.UUID(doc_id)
        return doc_id
    except (ValueError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def _to_point(doc: Document, vector: list[float]) -> qmodels.PointStruct:
    return qmodels.PointStruct(
        id=_ensure_point_id(doc.id),
        vector=vector,
        payload={"text": doc.text, "doc_id": doc.id, **doc.metadata},
    )


def index_documents(documents: list[Document]) -> int:
    """Embed, upsert to Qdrant, and rebuild the persisted BM25 corpus.

    Returns the number of documents indexed.
    """
    if not documents:
        logger.warning("No documents provided to index")
        return 0

    init_qdrant()
    client = get_qdrant_client()

    total = len(documents)
    logger.info("Indexing %d documents into '%s'", total, settings.qdrant_collection)

    indexed = 0
    for batch in _batched(documents, settings.embed_batch_size):
        vectors = embed_texts([d.text for d in batch])
        points = [_to_point(d, v) for d, v in zip(batch, vectors, strict=True)]

        for chunk in _batched(points, settings.upsert_batch_size):
            client.upsert(collection_name=settings.qdrant_collection, points=chunk)

        indexed += len(batch)
        logger.info("  ↳ embedded + upserted %d / %d", indexed, total)

    logger.info("Rebuilding BM25 index")
    bm25_index.rebuild(documents)
    bm25_index.save()

    logger.info("Indexing complete: %d documents", indexed)
    return indexed
