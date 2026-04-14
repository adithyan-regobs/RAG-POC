from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[Document]


class IngestResponse(BaseModel):
    ingested: int
    collection: str


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
