from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=False,
    )


def init_qdrant() -> None:
    client = get_qdrant_client()
    collections = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection in collections:
        logger.info("Qdrant collection '%s' ready", settings.qdrant_collection)
        return

    logger.info("Creating Qdrant collection '%s'", settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qmodels.VectorParams(
            size=settings.qdrant_vector_size,
            distance=qmodels.Distance.COSINE,
        ),
    )
