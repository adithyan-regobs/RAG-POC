from fastapi import APIRouter

from app.graph.chat_graph import run_chat
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    return run_chat(payload.query, top_k=payload.top_k)
