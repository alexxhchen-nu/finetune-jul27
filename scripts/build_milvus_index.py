"""Build a Milvus Lite vector index from cleaned archaeology reports.

Reads markdown files from archaeology_model/corpus/facts/clean/
and archaeology_model/corpus/methods/, splits them into heading-level chunks,
embeds them with an OpenAI-compatible endpoint, and stores them in a local
Milvus Lite database.

Interactive usage:
    uv run python scripts/build_milvus_index.py

Non-interactive usage:
    cp .env.example .env        # add your API key
    source .env
    uv run python scripts/build_milvus_index.py

Or pass the values directly:
    export OPENAI_BASE_URL=https://api.siliconflow.cn/v1
    export OPENAI_API_KEY=sk-...
    export EMBEDDING_MODEL=BAAI/bge-m3
    uv run python scripts/build_milvus_index.py

The resulting database is written to:
    archaeology_model/indices/milvus.db
"""

import argparse
import os
import re
import sys
import time
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
MAX_CHUNK_CHARS = 6000
BATCH_DELAY_SECONDS = 0.5


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


def split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split a long text into smaller pieces at paragraph or sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        stripped = current.strip()
        if stripped:
            pieces.append(stripped)
        current = ""

    for para in paragraphs:
        # If a single paragraph is already too long, split it by sentences.
        while len(para) > max_chars:
            # Take first max_chars chars, try to break at a sentence end.
            chunk = para[:max_chars]
            for delim in "。", "？", "！", ". ", "? ", "! ":
                idx = chunk.rfind(delim)
                if idx > max_chars * 0.5:
                    chunk = para[: idx + len(delim)]
                    break
            pieces.append(chunk.strip())
            para = para[len(chunk):]

        if len(current) + len(para) > max_chars and current:
            flush()
        current = current + "\n\n" + para if current else para

    flush()
    return pieces or [text[:max_chars]]


def split_by_headings(body: str, min_length: int = 50) -> Iterator[tuple[str, str]]:
    """Split markdown body by ## / ### / #### headings.

    Yields (heading, chunk_text) tuples. Long sections are further split by
    paragraph boundaries.
    """
    lines = body.splitlines()
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> Iterator[tuple[str, str]]:
        nonlocal current_heading, current_lines
        text = "\n".join(current_lines).strip()
        if len(text) >= min_length:
            for piece in split_long_text(text, MAX_CHUNK_CHARS):
                yield current_heading, piece
        current_heading = ""
        current_lines = []

    for line in lines:
        if re.match(r"^#{2,4}\s+", line):
            yield from flush()
            current_heading = re.sub(r"^#{2,4}\s+", "", line).strip()
        else:
            current_lines.append(line)
    yield from flush()


def get_embeddings(client: OpenAI, model: str, texts: list[str], max_retries: int = 5) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "rate limit" in err_str.lower() or "429" in err_str:
                wait = 2 ** attempt
                print(f"  rate limited, waiting {wait}s before retry {attempt + 1}/{max_retries}...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise last_error or RuntimeError("embedding request failed")


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
    records_embedded = 0

    def embed_with_retry(texts: list[str]) -> list[list[float] | None]:
        """Embed a batch, retrying with smaller batches on failure."""
        if not texts:
            return []
        try:
            return get_embeddings(client, model, texts)
        except Exception as e:
            if len(texts) == 1:
                print(f"  skipping chunk: {e}")
                return [None]
            print(f"  batch failed ({len(texts)} items): {e}, retrying halves...")
            mid = len(texts) // 2
            left = embed_with_retry(texts[:mid])
            right = embed_with_retry(texts[mid:])
            return left + right

    def flush_batch() -> None:
        nonlocal chunk_id, pending_texts, pending_records, records_embedded
        if not pending_texts:
            return
        embeddings = embed_with_retry(pending_texts)
        for rec, emb in zip(pending_records, embeddings):
            if emb is None:
                continue
            rec["id"] = chunk_id
            rec["vector"] = emb
            chunk_id += 1
            records.append(rec)
            records_embedded += 1
        pending_texts.clear()
        pending_records.clear()
        if BATCH_DELAY_SECONDS > 0:
            time.sleep(BATCH_DELAY_SECONDS)
        print(f"  embedded {records_embedded} chunks", end="\r", flush=True)

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
