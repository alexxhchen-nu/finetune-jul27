"""Search the Milvus Lite archaeology index.

Interactive usage:
    source .env
    uv run python scripts/search_milvus.py

Single query:
    source .env
    uv run python scripts/search_milvus.py \
        --query "竖穴土坑墓 典型随葬品" \
        --top-k 5

Read .env for OPENAI_BASE_URL, OPENAI_API_KEY and EMBEDDING_MODEL.
"""

import argparse
import os
import textwrap
import time

from openai import OpenAI
from pymilvus import MilvusClient

COLLECTION_NAME = "archaeology_chunks"
DB_PATH = "archaeology_model/indices/milvus.db"


def get_embeddings(client: OpenAI, model: str, texts: list[str], max_retries: int = 5) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if any(code in err for code in ("429", "500", "502", "503", "504", "rate limit", "overloaded", "timeout")):
                time.sleep(2 ** attempt)
                continue
            raise
    raise last_error or RuntimeError("embedding request failed")


def search(
    query: str,
    top_k: int = 5,
    corpus: str | None = None,
    region: str | None = None,
    period: str | None = None,
    chunk_type: str | None = None,
) -> list[dict]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    embedding = get_embeddings(client, model, [query])[0]

    milvus = MilvusClient(uri=DB_PATH)
    milvus.load_collection(COLLECTION_NAME)

    filters = []
    if corpus:
        filters.append(f'corpus == "{corpus}"')
    if region:
        filters.append(f'region == "{region}"')
    if period:
        filters.append(f'period == "{period}"')
    if chunk_type:
        filters.append(f'chunk_type == "{chunk_type}"')
    expr = " and ".join(filters) if filters else None

    results = milvus.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        output_fields=[
            "title",
            "heading",
            "text",
            "source_file",
            "source_type",
            "region",
            "period",
            "corpus",
            "chunk_type",
            "chunk_topics",
        ],
        filter=expr,
        limit=top_k,
    )
    return results[0]


def format_hit(hit: dict, index: int) -> str:
    entity = hit["entity"]
    text = entity.get("text", "")
    wrapped = textwrap.fill(text, width=100, initial_indent="        ", subsequent_indent="        ")
    lines = [
        f"[{index + 1}] score={hit['distance']:.4f}",
        f"    type: {entity.get('chunk_type', '')} | topics: {entity.get('chunk_topics', '')}",
        f"    title: {entity.get('title', '')}",
        f"    heading: {entity.get('heading', '')}",
        f"    source: {entity.get('source_file', '')}",
        f"    region: {entity.get('region', '')} | period: {entity.get('period', '')} | corpus: {entity.get('corpus', '')}",
        "    text:",
        wrapped,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the archaeology Milvus index.")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    parser.add_argument("--corpus", choices=["facts", "methods"], help="Filter by corpus")
    parser.add_argument("--region", help="Filter by region metadata")
    parser.add_argument("--period", help="Filter by period metadata")
    parser.add_argument("--chunk-type", help="Filter by chunk_type")
    args = parser.parse_args()

    query = args.query
    if not query:
        query = input("Query: ").strip()

    if not query:
        print("Query is required.")
        return

    hits = search(
        query,
        top_k=args.top_k,
        corpus=args.corpus,
        region=args.region,
        period=args.period,
        chunk_type=args.chunk_type,
    )

    if not hits:
        print("No results found.")
        return

    for i, hit in enumerate(hits):
        print(format_hit(hit, i))
        print()


if __name__ == "__main__":
    main()
