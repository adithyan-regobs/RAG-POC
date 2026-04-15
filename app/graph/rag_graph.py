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
    logger.info("=== [rag:1/7 expand_query] question=%r", state["question"])
    queries = query_expansion.expand_query(state["question"])
    for i, q in enumerate(queries, start=1):
        logger.info("    expanded[%d]: %s", i, q)
    logger.info("=== [rag:1/7 expand_query] → %d queries", len(queries))
    return {"expanded_queries": queries}


def _node_retrieve(state: RAGState) -> RAGState:
    """Run dense + sparse retrieval for each expanded query."""
    queries = state.get("expanded_queries") or [state["question"]]
    logger.info("=== [rag:2/7 retrieve] running dense+sparse over %d queries", len(queries))
    ranked_lists: list[list[RetrievedChunk]] = []
    for i, q in enumerate(queries, start=1):
        vec = vector_search.vector_search(q)
        bm25 = bm25_search.bm25_index.search(q)
        logger.info("    q[%d] %r → vector=%d, bm25=%d", i, q, len(vec), len(bm25))
        ranked_lists.append(vec)
        ranked_lists.append(bm25)
    non_empty = [r for r in ranked_lists if r]
    logger.info(
        "=== [rag:2/7 retrieve] → %d non-empty ranked lists (of %d total)",
        len(non_empty),
        len(ranked_lists),
    )
    return {"ranked_lists": non_empty}


def _node_fuse(state: RAGState) -> RAGState:
    lists = state.get("ranked_lists", [])
    logger.info("=== [rag:3/7 fuse] RRF over %d lists", len(lists))
    fused = fusion.reciprocal_rank_fusion(lists)
    logger.info("=== [rag:3/7 fuse] → %d unique candidates", len(fused))
    for i, c in enumerate(fused[:5], start=1):
        logger.info("    top[%d] id=%s score=%.4f source=%s", i, c.id, c.score, c.source)
    return {"fused": fused}


def _node_process_metadata(state: RAGState) -> RAGState:
    fused = state.get("fused", [])
    logger.info("=== [rag:4/7 process_metadata] in=%d (filter + boost)", len(fused))
    processed = metadata_processor.process(fused)
    logger.info("=== [rag:4/7 process_metadata] → %d kept", len(processed))
    return {"processed": processed}


def _node_rerank(state: RAGState) -> RAGState:
    processed = state.get("processed", [])
    logger.info("=== [rag:5/7 rerank] cross-encoder over %d candidates", len(processed))
    reranked = reranker.rerank(
        state["question"],
        processed,
        top_k=state.get("top_k"),
    )
    logger.info("=== [rag:5/7 rerank] → top %d", len(reranked))
    for i, c in enumerate(reranked[:5], start=1):
        logger.info("    top[%d] id=%s score=%.4f", i, c.id, c.score)
    return {"reranked": reranked}


def _node_build_context(state: RAGState) -> RAGState:
    reranked = state.get("reranked", [])
    logger.info("=== [rag:6/7 build_context] packing %d reranked chunks", len(reranked))
    context, used = context_builder.build_context(reranked)
    logger.info(
        "=== [rag:6/7 build_context] → %d chunks fit, context=%d chars",
        len(used),
        len(context),
    )
    return {"context": context, "context_sources": used}


def _node_generate(state: RAGState) -> RAGState:
    context = state.get("context", "")
    logger.info("=== [rag:7/7 generate] grounded LLM call (context=%d chars)", len(context))
    answer = llm_service.generate_answer(state["question"], context)
    logger.info("=== [rag:7/7 generate] → answer (%d chars)", len(answer))
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
