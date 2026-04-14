import pickle
import re
from pathlib import Path
from threading import Lock

from rank_bm25 import BM25Okapi

from app.config import settings
from app.core.logger import get_logger
from app.models.schemas import Document, RetrievedChunk

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """In-memory BM25 index with pickle-based persistence.

    The indexing script calls `rebuild()` + `save()`. The API calls `load()`
    during startup to rehydrate the index from disk.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._bm25: BM25Okapi | None = None
        self._docs: list[Document] = []

    @property
    def size(self) -> int:
        return len(self._docs)

    def rebuild(self, documents: list[Document]) -> None:
        with self._lock:
            self._docs = list(documents)
            if self._docs:
                self._bm25 = BM25Okapi([_tokenize(d.text) for d in self._docs])
            else:
                self._bm25 = None

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        with self._lock:
            self._docs.extend(documents)
            self._bm25 = BM25Okapi([_tokenize(d.text) for d in self._docs])

    def save(self, path: str | None = None) -> Path:
        target = Path(path or settings.bm25_index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {"docs": [d.model_dump() for d in self._docs]}
        with target.open("wb") as f:
            pickle.dump(payload, f)
        logger.info("Saved BM25 corpus (%d docs) to %s", len(payload["docs"]), target)
        return target

    def load(self, path: str | None = None) -> int:
        source = Path(path or settings.bm25_index_path)
        if not source.exists():
            logger.info("BM25 corpus not found at %s — starting empty", source)
            return 0
        with source.open("rb") as f:
            payload = pickle.load(f)
        docs = [Document(**d) for d in payload.get("docs", [])]
        self.rebuild(docs)
        logger.info("Loaded BM25 corpus (%d docs) from %s", len(docs), source)
        return len(docs)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k_bm25
        with self._lock:
            if self._bm25 is None or not self._docs:
                return []
            scores = self._bm25.get_scores(_tokenize(query))
            docs = self._docs

        ranked = sorted(zip(docs, scores, strict=True), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                id=d.id,
                text=d.text,
                score=float(score),
                source="bm25",
                metadata=d.metadata,
            )
            for d, score in ranked
            if score > 0
        ]


bm25_index = BM25Index()
