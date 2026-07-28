"""Build a Milvus Lite vector index from cleaned archaeology reports.

Reads markdown files from archaeology_model/corpus/facts/clean/,
splits them into heading-level chunks, embeds them with OpenAI,
and stores them in a local Milvus Lite database.

Usage:
    export OPENAI_API_KEY=...
    uv run python scripts/build_milvus_index.py

The resulting database is written to:
    archaeology_model/indices/milvus.db
"""

import os
import re
import sys
import yaml
from pathlib import Path
from typing import Iterator

from openai import OpenAI
from pymilvus import MilvusClient, DataType

CLEAN_DIR = Path("archaeology_model/corpus/facts/clean")
INDEX_DIR = Path("archaeology_model/indices")
DB_PATH = INDEX_DIR / "milvus.db"
COLLECTION_NAME = "archaeology_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_MAX_TOKENS = 8000


def load_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter and return (meta, body)."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
                body = parts[2]
                return meta, body
            except yaml.YAMLError:
                pass
    return {}, text


def split_by_headings(body: str, min_length: int = 50) -> Iterator[tuple[str, str]]:
    """Split markdown body by ## / ### headings.

    Yields (heading, chunk_text) tuples.
    """
    lines = body.splitlines()
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> Iterator[tuple[str, str]]:
        nonlocal current_heading, current_lines
        text = "\n".join(current_lines).strip()
        if len(text) >= min_length:
            yield current_heading, text
        current_heading = ""
        current_lines = []

    for line in lines:
        if re.match(r"^#{2,4}\s+", line):
            yield from flush()
            current_heading = re.sub(r"^#{2,4}\s+", "", line).strip()
        else:
            current_lines.append(line)
    yield from flush()


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def build_index() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    milvus = MilvusClient(uri=str(DB_PATH))

    if milvus.has_collection(COLLECTION_NAME):
        milvus.drop_collection(COLLECTION_NAME)

    schema = milvus.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("source_type", DataType.VARCHAR, max_length=64)
    schema.add_field("region", DataType.VARCHAR, max_length=128)
    schema.add_field("period", DataType.VARCHAR, max_length=128)
    schema.add_field("source_file", DataType.VARCHAR, max_length=512)
    schema.add_field("heading", DataType.VARCHAR, max_length=512)

    index_params = milvus.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    milvus.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    records: list[dict] = []
    chunk_id = 0
    pending_texts: list[str] = []
    pending_records: list[dict] = []

    def flush_batch() -> None:
        nonlocal chunk_id, pending_texts, pending_records
        if not pending_texts:
            return
        embeddings = get_embeddings(openai_client, pending_texts)
        for rec, emb in zip(pending_records, embeddings):
            rec["id"] = chunk_id
            rec["vector"] = emb
            chunk_id += 1
            records.append(rec)
        pending_texts.clear()
        pending_records.clear()

    md_files = sorted(CLEAN_DIR.rglob("*.md"))
    if not md_files:
        print(f"No markdown files found in {CLEAN_DIR}")
        sys.exit(1)

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        meta, body = load_frontmatter(text)
        rel_path = str(path.relative_to(CLEAN_DIR))

        chunks = list(split_by_headings(body))
        if not chunks:
            # If no headings, treat the whole body as one chunk.
            chunks = [("", body.strip())]

        for heading, chunk_text in chunks:
            if len(chunk_text) < 50:
                continue
            record = {
                "text": chunk_text,
                "title": meta.get("title", path.stem) or path.stem,
                "source_type": meta.get("source_type", "unknown"),
                "region": meta.get("region", ""),
                "period": meta.get("period", ""),
                "source_file": rel_path,
                "heading": heading,
            }
            pending_texts.append(chunk_text)
            pending_records.append(record)

            if len(pending_texts) >= 32:
                flush_batch()

    flush_batch()

    if records:
        milvus.insert(collection_name=COLLECTION_NAME, data=records)
        milvus.flush(collection_name=COLLECTION_NAME)
        milvus.load_collection(COLLECTION_NAME)
        print(f"Indexed {len(records)} chunks from {len(md_files)} files into {DB_PATH}")
    else:
        print("No chunks found to index.")


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search the index. Convenience function for imports."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    openai_client = OpenAI(api_key=api_key)
    milvus = MilvusClient(uri=str(DB_PATH))
    embedding = get_embeddings(openai_client, [query])[0]
    results = milvus.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        output_fields=["title", "heading", "text", "source_file", "source_type", "region", "period"],
        limit=top_k,
    )
    return results[0]


if __name__ == "__main__":
    build_index()
