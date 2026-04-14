from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import IngestRequest, IngestResponse
from app.services.bm25_search import bm25_index
from app.services.vector_search import upsert_documents

router = APIRouter()


@router.post("", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    if not payload.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    count = upsert_documents(payload.documents)
    bm25_index.add(payload.documents)

    return IngestResponse(ingested=count, collection=settings.qdrant_collection)
