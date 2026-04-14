from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.core.logger import get_logger
from app.models.schemas import QueryResponse, RetrievedChunk
from app.services import (
    bm25_search,
    context_builder,
    fusion,
    llm_service,
    metadata_processor,
    query_expansion,
    reranker,
    vector_search,
)

logger = get_logger(__name__)


class RAGState(TypedDict, total=False):
    question: str
    top_k: int | None
    expanded_queries: list[str]
    ranked_lists: list[list[RetrievedChunk]]
    fused: list[RetrievedChunk]
    processed: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    context: str
    context_sources: list[RetrievedChunk]
    answer: str


def _node_expand(state: RAGState) -> RAGState:
    queries = query_expansion.expand_query(state["question"])
    return {"expanded_queries": queries}


def _node_retrieve(state: RAGState) -> RAGState:
    """Run dense + sparse retrieval for each expanded query."""
    ranked_lists: list[list[RetrievedChunk]] = []
    for q in state.get("expanded_queries") or [state["question"]]:
        ranked_lists.append(vector_search.vector_search(q))
        ranked_lists.append(bm25_search.bm25_index.search(q))
    non_empty = [r for r in ranked_lists if r]
    logger.info("Retrieved %d non-empty ranked lists", len(non_empty))
    return {"ranked_lists": non_empty}


def _node_fuse(state: RAGState) -> RAGState:
    fused = fusion.reciprocal_rank_fusion(state.get("ranked_lists", []))
    logger.info("RRF fused into %d unique candidates", len(fused))
    return {"fused": fused}


def _node_process_metadata(state: RAGState) -> RAGState:
    return {"processed": metadata_processor.process(state.get("fused", []))}


def _node_rerank(state: RAGState) -> RAGState:
    reranked = reranker.rerank(
        state["question"],
        state.get("processed", []),
        top_k=state.get("top_k"),
    )
    logger.info("Reranked to top %d", len(reranked))
    return {"reranked": reranked}


def _node_build_context(state: RAGState) -> RAGState:
    context, used = context_builder.build_context(state.get("reranked", []))
    return {"context": context, "context_sources": used}


def _node_generate(state: RAGState) -> RAGState:
    answer = llm_service.generate_answer(state["question"], state.get("context", ""))
    return {"answer": answer}


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("expand_query", _node_expand)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("fuse", _node_fuse)
    g.add_node("process_metadata", _node_process_metadata)
    g.add_node("rerank", _node_rerank)
    g.add_node("build_context", _node_build_context)
    g.add_node("generate", _node_generate)

    g.set_entry_point("expand_query")
    g.add_edge("expand_query", "retrieve")
    g.add_edge("retrieve", "fuse")
    g.add_edge("fuse", "process_metadata")
    g.add_edge("process_metadata", "rerank")
    g.add_edge("rerank", "build_context")
    g.add_edge("build_context", "generate")
    g.add_edge("generate", END)

    return g.compile()


def run_rag(question: str, top_k: int | None = None) -> QueryResponse:
    app = build_graph()
    result: RAGState = app.invoke({"question": question, "top_k": top_k})
    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("context_sources", []),
    )
