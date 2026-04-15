from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.core.logger import get_logger
from app.graph.rag_graph import run_rag
from app.models.schemas import ChatResponse, RetrievedChunk
from app.services import classifier, llm_service

logger = get_logger(__name__)


class ChatState(TypedDict, total=False):
    query: str
    top_k: int | None
    route: str
    answer: str
    sources: list[RetrievedChunk]


def _node_classify(state: ChatState) -> ChatState:
    logger.info("─── [chat:classify] query=%r", state["query"])
    route = classifier.classify(state["query"])
    logger.info("─── [chat:classify] → route=%s", route)
    return {"route": route}


def     _node_rag(state: ChatState) -> ChatState:
    logger.info("─── [chat:rag] delegating to RAG sub-graph")
    result = run_rag(state["query"], top_k=state.get("top_k"))
    logger.info(
        "─── [chat:rag] ← RAG returned answer (%d chars), %d sources",
        len(result.answer),
        len(result.sources),
    )
    return {"answer": result.answer, "sources": result.sources}


def _node_general(state: ChatState) -> ChatState:
    logger.info("─── [chat:general] direct LLM (no retrieval)")
    answer = llm_service.generate_direct_answer(state["query"])
    logger.info("─── [chat:general] ← answer (%d chars)", len(answer))
    return {"answer": answer, "sources": []}


def _route(state: ChatState) -> str:
    return "rag" if state.get("route") == "rag" else "general"


@lru_cache(maxsize=1)
def build_chat_graph():
    g = StateGraph(ChatState)
    g.add_node("classify", _node_classify)
    g.add_node("rag", _node_rag)
    g.add_node("general", _node_general)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", _route, {"rag": "rag", "general": "general"})
    g.add_edge("rag", END)
    g.add_edge("general", END)

    return g.compile()


def run_chat(query: str, top_k: int | None = None) -> ChatResponse:
    app = build_chat_graph()
    result: ChatState = app.invoke({"query": query, "top_k": top_k})
    return ChatResponse(
        answer=result.get("answer", ""),
        route=result.get("route", "unknown"),
        sources=result.get("sources", []),
    )
