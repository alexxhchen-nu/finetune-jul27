"""Remove noise chunks from the existing Milvus index.

Identifies chunks whose headings look like table of contents, notes, preface,
abstracts, figure plates, etc., and deletes them. Run after building the index.

Usage:
    uv run python scripts/clean_milvus_noise.py
"""

import re
from pymilvus import MilvusClient

COLLECTION_NAME = "archaeology_chunks"
DB_PATH = "archaeology_model/indices/milvus.db"

NOISE_PATTERNS = [
    r"^(目录|插图目录|图版目录|彩版目录|插表目录)$",
    r"^[一二三四五六七八九十]、?(前言|目录|绪论|结语|结论)$",
    r"^(注\s*释|注释)$",
    r"^(后记|前言|Abstract)$",
    r"^(图版|彩版)\s*\w+$",
    r"^[一二三四五六七八九十]$",
]


def is_noise(heading: str) -> bool:
    return any(re.compile(p).match(heading) for p in NOISE_PATTERNS)


def main() -> None:
    milvus = MilvusClient(uri=DB_PATH)
    milvus.load_collection(COLLECTION_NAME)

    print("Querying all chunks for headings...")
    results = milvus.query(
        collection_name=COLLECTION_NAME,
        output_fields=["id", "heading"],
        limit=50000,
    )
    print(f"Loaded {len(results)} chunks")

    noise_ids = [r["id"] for r in results if is_noise(r.get("heading", ""))]
    print(f"Found {len(noise_ids)} noise chunks ({len(noise_ids)/len(results)*100:.1f}%)")

    if not noise_ids:
        print("Nothing to delete.")
        return

    print("Deleting noise chunks...")
    milvus.delete(collection_name=COLLECTION_NAME, ids=noise_ids)
    milvus.flush(collection_name=COLLECTION_NAME)
    print(f"Deleted {len(noise_ids)} chunks")


if __name__ == "__main__":
    main()
