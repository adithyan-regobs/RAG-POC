from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.routes import chat, health, ingest, query
from app.config import settings
from app.core.logger import get_logger
from app.db.qdrant_client import init_qdrant
from app.services.bm25_search import bm25_index

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG service")
    init_qdrant()
    bm25_index.load()
    yield
    logger.info("Shutting down RAG service")


app = FastAPI(
    title="RAG Service",
    version="0.1.0",
    description="Hybrid RAG (Qdrant + BM25 + Cross-Encoder) orchestrated by LangGraph",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )
