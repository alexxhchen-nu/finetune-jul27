"""Build a Milvus Lite vector index from cleaned archaeology reports.

Reads markdown files from archaeology_model/corpus/facts/clean/
and archaeology_model/corpus/methods/, splits them into heading-level chunks,
embeds them with an OpenAI-compatible endpoint, and stores them in a local
Milvus Lite database.

Interactive usage:
    uv run python scripts/build_milvus_index.py

Non-interactive usage:
    uv run python scripts/build_milvus_index.py \
        --api-key sk-... \
        --base-url https://api.openai.com/v1 \
        --model text-embedding-3-small

The resulting database is written to:
    archaeology_model/indices/milvus.db
"""

import argparse
import os
import re
import sys
import yaml
from pathlib import Path
from typing import Iterator

from openai import OpenAI
from pymilvus import MilvusClient, DataType

SOURCE_DIRS = [
    Path("archaeology_model/corpus/facts/clean"),
    Path("archaeology_model/corpus/methods"),
]
INDEX_DIR = Path("archaeology_model/indices")
DEFAULT_DB_PATH = INDEX_DIR / "milvus.db"
COLLECTION_NAME = "archaeology_chunks"
BATCH_SIZE = 32


def ask(prompt: str, default: str | None = None) -> str:
    if default is not None:
        full = f"{prompt} [{default}]: "
    else:
        full = f"{prompt}: "
    sys.stdout.write(full)
    sys.stdout.flush()
    value = input().strip()
    if not value and default is not None:
        return default
    return value


def select_model(client: OpenAI) -> str:
    try:
        response = client.models.list()
        models = sorted([m.id for m in response.data])
    except Exception as e:
        print(f"Could not list models: {e}")
        models = []

    if models:
        print("\nAvailable models:")
        for i, name in enumerate(models, 1):
            print(f"  {i}. {name}")
        choice = ask("Select model by number (or type a model name)")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
        return choice or models[0]

    return ask("Model name")


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
    """Split markdown body by ## / ### / #### headings.

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


def get_embeddings(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def detect_embedding_dim(client: OpenAI, model: str) -> int:
    """Make a test call to discover the embedding dimension."""
    emb = get_embeddings(client, model, ["test"])[0]
    return len(emb)


def build_index(args: argparse.Namespace) -> None:
    print("\nConfigure embedding endpoint")
    print("-" * 40)
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        base_url = ask("Base URL (e.g. https://api.openai.com/v1 or your proxy)")
        if not base_url:
            print("Error: Base URL is required.")
            sys.exit(1)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or ask("API key", "")
    print(f"Using endpoint: {base_url}\n")

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
    model = args.model or os.getenv("EMBEDDING_MODEL") or select_model(client)
    print(f"Using embedding model: {model}")

    embedding_dim = detect_embedding_dim(client, model)
    print(f"Detected embedding dimension: {embedding_dim}")

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    milvus = MilvusClient(uri=str(db_path))

    if milvus.has_collection(COLLECTION_NAME):
        milvus.drop_collection(COLLECTION_NAME)

    schema = milvus.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=embedding_dim)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("source_type", DataType.VARCHAR, max_length=64)
    schema.add_field("region", DataType.VARCHAR, max_length=128)
    schema.add_field("period", DataType.VARCHAR, max_length=128)
    schema.add_field("source_file", DataType.VARCHAR, max_length=512)
    schema.add_field("heading", DataType.VARCHAR, max_length=512)
    schema.add_field("corpus", DataType.VARCHAR, max_length=32)

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
        embeddings = get_embeddings(client, model, pending_texts)
        for rec, emb in zip(pending_records, embeddings):
            rec["id"] = chunk_id
            rec["vector"] = emb
            chunk_id += 1
            records.append(rec)
        pending_texts.clear()
        pending_records.clear()

    md_files: list[Path] = []
    for src_dir in SOURCE_DIRS:
        if src_dir.exists():
            md_files.extend(src_dir.rglob("*.md"))
    md_files = sorted(set(md_files))
    if not md_files:
        print(f"No markdown files found in {SOURCE_DIRS}")
        sys.exit(1)

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        meta, body = load_frontmatter(text)
        rel_path = str(path)

        chunks = list(split_by_headings(body))
        if not chunks:
            chunks = [("", body.strip())]

        corpus = "facts" if "facts/clean" in rel_path else ("methods" if "methods" in rel_path else "other")
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
                "corpus": corpus,
            }
            pending_texts.append(chunk_text)
            pending_records.append(record)

            if len(pending_texts) >= BATCH_SIZE:
                flush_batch()

    flush_batch()

    if records:
        milvus.insert(collection_name=COLLECTION_NAME, data=records)
        milvus.flush(collection_name=COLLECTION_NAME)
        milvus.load_collection(COLLECTION_NAME)
        print(f"Indexed {len(records)} chunks from {len(md_files)} files into {db_path}")
    else:
        print("No chunks found to index.")


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search the index. Convenience function for imports.

    Reads OPENAI_API_KEY, OPENAI_BASE_URL and EMBEDDING_MODEL from environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    client = OpenAI(api_key=api_key, base_url=base_url)
    milvus = MilvusClient(uri=str(DEFAULT_DB_PATH))
    embedding = get_embeddings(client, model, [query])[0]
    results = milvus.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        output_fields=["title", "heading", "text", "source_file", "source_type", "region", "period", "corpus"],
        limit=top_k,
    )
    return results[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Milvus Lite index from cleaned archaeology reports.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL")
    parser.add_argument("--model", help="Embedding model name")
    parser.add_argument("--db", help="Path to Milvus Lite database (default: archaeology_model/indices/milvus.db)")
    args = parser.parse_args()
    build_index(args)


if __name__ == "__main__":
    main()
