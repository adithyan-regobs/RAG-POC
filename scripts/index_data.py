"""Index documents from a JSON file into Qdrant + BM25.

Usage:
    python scripts/index_data.py data/documents.json

Expected JSON format: a list of objects with at least `text`, and optionally
`source`, `law_area`, `document_type`, `id` (any extra keys are preserved as metadata).

    [
      {
        "text": "...",
        "source": "case-law/xyz.pdf",
        "law_area": "contract",
        "document_type": "judgment"
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

# Allow running as `python scripts/index_data.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logger import get_logger  # noqa: E402
from app.models.schemas import Document  # noqa: E402
from app.services.indexing_service import index_documents  # noqa: E402

logger = get_logger("index_data")

REQUIRED_FIELDS = ("text",)
METADATA_FIELDS = ("source", "law_area", "document_type")


def load_documents(path: Path) -> list[Document]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of documents, got {type(raw).__name__}")

    documents: list[Document] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Document #{idx} is not an object")
        for field in REQUIRED_FIELDS:
            if field not in item or not item[field]:
                raise ValueError(f"Document #{idx} missing required field '{field}'")

        doc_id = str(item.get("id") or uuid.uuid4())
        metadata = {
            **{k: item[k] for k in METADATA_FIELDS if k in item},
            **{
                k: v
                for k, v in item.items()
                if k not in {"id", "text", *METADATA_FIELDS}
            },
        }
        documents.append(Document(id=doc_id, text=item["text"], metadata=metadata))

    return documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index documents into Qdrant + BM25")
    parser.add_argument("input", type=Path, help="Path to a JSON file with documents")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    logger.info("Loading documents from %s", args.input)
    documents = load_documents(args.input)
    logger.info("Loaded %d documents", len(documents))

    count = index_documents(documents)
    logger.info("Done. Indexed %d documents.", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
