import uuid

from qdrant_client.http import models as qmodels

from app.config import settings
from app.db.qdrant_client import get_qdrant_client
from app.models.schemas import Document, RetrievedChunk
from app.services.embedding_service import embed_text, embed_texts


def upsert_documents(documents: list[Document]) -> int:
    if not documents:
        return 0
    vectors = embed_texts([d.text for d in documents])
    points = [
        qmodels.PointStruct(
            id=d.id if _is_valid_id(d.id) else str(uuid.uuid4()),
            vector=vec,
            payload={"text": d.text, "doc_id": d.id, **d.metadata},
        )
        for d, vec in zip(documents, vectors, strict=True)
    ]
    get_qdrant_client().upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )
    return len(points)


def vector_search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k_vector
    vector = embed_text(query)
    hits = get_qdrant_client().search(
        collection_name=settings.qdrant_collection,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    results: list[RetrievedChunk] = []
    for h in hits:
        payload = h.payload or {}
        results.append(
            RetrievedChunk(
                id=str(payload.get("doc_id", h.id)),
                text=payload.get("text", ""),
                score=float(h.score),
                source="vector",
                metadata={k: v for k, v in payload.items() if k not in {"text", "doc_id"}},
            )
        )
    return results


def _is_valid_id(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError):
        return value.isdigit()
