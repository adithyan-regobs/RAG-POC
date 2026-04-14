# RAG Service

Hybrid Retrieval-Augmented Generation API built with FastAPI, orchestrated by LangGraph.

## Stack

- **FastAPI** — HTTP API
- **Qdrant Cloud** — dense vector store
- **rank-bm25** — in-memory keyword search
- **OpenAI** — embeddings + chat completion
- **sentence-transformers** — cross-encoder reranker
- **LangGraph** — pipeline orchestration

## Structure

```
.
├── main.py                      # FastAPI entrypoint + lifespan
├── requirements.txt
├── .env.example
└── app/
    ├── config.py                # pydantic-settings, loads .env
    ├── core/
    │   └── logger.py
    ├── models/
    │   └── schemas.py           # API request/response models
    ├── db/
    │   └── qdrant_client.py     # Qdrant client + collection init
    ├── services/
    │   ├── embedding_service.py # OpenAI embeddings
    │   ├── llm_service.py       # OpenAI chat (grounded-answer prompt)
    │   ├── vector_search.py     # Qdrant upsert + dense search
    │   ├── bm25_search.py       # Persisted in-memory BM25 index
    │   ├── query_expansion.py   # LLM query rewriting (3-5 variations)
    │   ├── fusion.py            # Reciprocal Rank Fusion
    │   ├── reranker.py          # Cross-encoder reranker
    │   ├── metadata_processor.py# Quality filter + metadata boost
    │   ├── context_builder.py   # Token-budgeted context assembly
    │   └── indexing_service.py  # Bulk index pipeline
    ├── graph/
    │   └── rag_graph.py         # LangGraph pipeline
    └── api/
        └── routes/
            ├── health.py
            ├── ingest.py
            └── query.py
```

## Pipeline

```
expand_query      # OpenAI → 3-5 rephrasings (original always kept)
    ↓
retrieve          # Qdrant (dense) + BM25 (sparse) per sub-query
    ↓
fuse              # Reciprocal Rank Fusion across all ranked lists
    ↓
process_metadata  # Drop low-quality chunks, boost preferred areas/types
    ↓
rerank            # cross-encoder/ms-marco-MiniLM-L-12-v2, keep top 10
    ↓
build_context     # Token-budgeted, citation-ready context blocks
    ↓
generate          # Grounded answer — "only use provided context"
```

All knobs (variation count, RRF k, boost lists/value, rerank top-k, token
budget) are in [app/config.py](app/config.py) / `.env`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in keys
python main.py
```

## API

- `GET  /health`
- `POST /ingest` — `{ "documents": [{ "id": "...", "text": "...", "metadata": {} }] }`
- `POST /query`  — `{ "question": "...", "top_k": 5 }` (always runs the RAG pipeline)
- `POST /chat`   — `{ "query": "...", "top_k": 5 }` (classifier-routed: `rag` vs `general`)

### Chat routing

`/chat` runs a thin LangGraph ([chat_graph.py](app/graph/chat_graph.py)):

```
classify ──► rag      (runs the full RAG graph)
         └─► general  (direct LLM, no retrieval)
```

The response includes the chosen `route` alongside `answer` and `sources`
(empty for the `general` route).

## Bulk indexing

```bash
python scripts/index_data.py data/sample_documents.json
```

Each entry in the JSON file requires `text`; `source`, `law_area`, and
`document_type` are stored as Qdrant payload + BM25 metadata. The script creates
the Qdrant collection if missing, embeds in batches, upserts vectors, and
persists the BM25 corpus to `BM25_INDEX_PATH`. The API rehydrates BM25 from that
file on startup.

## Notes

- The BM25 index is in-memory; for production, persist the corpus and rehydrate on startup via `BM25Index.load()`.
- Qdrant collection is auto-created on startup using `QDRANT_VECTOR_SIZE` (must match your embedding model).
