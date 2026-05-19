"""
Ingest documents from a directory into ChromaDB before running the app.

Usage:
    python ingest.py                          # uses KNOWLEDGE_BASE_DIR from .env
    python ingest.py ./path/to/docs           # ingests from a specific directory
    python ingest.py ./path/to/docs --reset   # clears the DB first, then ingests
"""

import sys
import argparse
from pathlib import Path

from agent.config import (
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR, KNOWLEDGE_BASE_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, KNOWLEDGE_BASE_NAME
)
from agent.rag import RAGAgent

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def ingest(source_dir: str, reset: bool = False):
    path = Path(source_dir)
    if not path.exists():
        print(f"Directory not found: {path}")
        print("Create it and add documents, then run this script again.")
        sys.exit(1)

    files = [f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        print(f"No supported files found in {path}")
        print(f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(0)

    rag = RAGAgent(
        model=EMBEDDING_MODEL,
        name=KNOWLEDGE_BASE_NAME,
        persist_dir=CHROMA_PERSIST_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if reset:
        rag.vector_store.reset_collection()
        print("Knowledge base cleared.\n")

    print(f"Found {len(files)} file(s) in {path}\n")

    ingested, skipped = 0, 0
    for file in files:
        try:
            content = file.read_bytes()
            rag.add_document(content, source=file.name)
            print(f"  [ok] {file.name}")
            ingested += 1
        except Exception as e:
            print(f"  [skip] {file.name} — {e}")
            skipped += 1

    print(f"\nDone. {ingested} file(s) ingested, {skipped} skipped.")
    print(f"Vector store: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=KNOWLEDGE_BASE_DIR,
        help=f"Directory to ingest (default: {KNOWLEDGE_BASE_DIR})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the existing knowledge base before ingesting",
    )
    args = parser.parse_args()
    ingest(args.directory, reset=args.reset)
