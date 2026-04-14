from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "rag_documents"
    qdrant_vector_size: int = 1536

    # Retrieval
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_rerank: int = 10
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Query expansion
    num_query_variations: int = 4

    # Metadata processing (comma-separated lists for env-friendliness)
    min_text_length: int = 50
    boost_law_areas: str = "family"
    boost_document_types: str = "judgment"
    boost_value: float = 0.15

    # Context assembly
    context_token_budget: int = 4000

    # Indexing
    embed_batch_size: int = 100
    upsert_batch_size: int = 256
    bm25_index_path: str = "data/bm25_corpus.pkl"


settings = Settings()
