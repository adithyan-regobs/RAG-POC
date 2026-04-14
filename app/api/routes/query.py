from fastapi import APIRouter

from app.graph.rag_graph import run_rag
from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    return run_rag(payload.question, top_k=payload.top_k)
