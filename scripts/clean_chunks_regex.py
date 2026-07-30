"""Clean archaeology chunk texts using regex rules.

Reads chunks from Milvus, cleans OCR artifacts with regex, writes back.
Fast, free, deterministic.

Usage:
    source .env
    uv run python scripts/clean_chunks_regex.py
"""

import json
import os
import re
import time
from pathlib import Path

from pymilvus import MilvusClient

COLLECTION_NAME = "archaeology_chunks"
DB_PATH = "archaeology_model/indices/milvus.db"
PROGRESS_PATH = Path("archaeology_model/indices/clean_progress.json")


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def clean_text(text: str) -> str:
    s = text
    # 1. Remove LaTeX commands: \frac{}{}, \underset{}{}, \sqrt{}, \text{}, etc.
    #    Keep the last {} group content (usually the display value)
    s = re.sub(r"\\(?:frac|underset|sqrt|text|mathrm|mathbf|overline|underline|hat|vec|dot)\s*\{[^}]*\}\s*\{([^}]*)\}", r"\1", s)
    # 2. Remove remaining LaTeX commands: \alpha, \beta, \degree, etc.
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    # 3. Remove LaTeX braces
    s = re.sub(r"[{}]", "", s)
    # 4. Remove LaTeX: $...$ → keep inner content stripped of markup
    s = re.sub(r"\$([^$]+)\$", lambda m: re.sub(r"[\\{}^_]", "", m.group(1)).strip(), s)
    # 5. Remove orphan $ signs
    s = s.replace("$", "")
    # 6. Remove figure/plate references: （图三○五）（图一八，4；彩版五一，6）
    s = re.sub(r"[（(]图[^）)]{1,20}[）)]", "", s)
    s = re.sub(r"[（(]彩版[^）)]{1,20}[）)]", "", s)
    # 7. Remove figure numbering: 图一九六：1 / 图版二五：1
    s = re.sub(r"图版?\s*[一二三四五六七八九十百千\d]+\s*[：:]\s*\d+", "", s)
    # 8. Remove HTML tags
    s = re.sub(r"<[^>]+>", "", s)
    # 9. Normalize specimen IDs: 标本 XM18:2 → XM18:2
    s = re.sub(r"标本\s+", "", s)
    # 10. Normalize colons: fullwidth → halfwidth
    s = s.replace("：", ":")
    # 11. Clean whitespace: multiple spaces → one, multiple newlines → two
    s = re.sub(r"[^\S\n]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # 12. Remove leading/trailing whitespace per line
    s = "\n".join(line.strip() for line in s.splitlines())
    # 13. Remove empty lines at start/end
    s = s.strip()
    return s


def main() -> None:
    load_env()

    milvus = MilvusClient(uri=DB_PATH)
    milvus.load_collection(COLLECTION_NAME)

    results = milvus.query(
        collection_name=COLLECTION_NAME,
        output_fields=["id", "text", "vector", "title", "source_type", "region", "period", "source_file", "heading", "corpus", "chunk_type", "chunk_topics"],
        limit=50000,
    )

    done_ids: set[int] = set()
    if PROGRESS_PATH.exists():
        done_ids = set(json.loads(PROGRESS_PATH.read_text()))
        print(f"Resuming: {len(done_ids)} already cleaned")

    pending = [r for r in results if r["id"] not in done_ids]
    print(f"Total {len(results)}, pending {len(pending)}")

    cleaned = 0
    for idx, rec in enumerate(pending):
        old_text = rec["text"]
        new_text = clean_text(old_text)
        if new_text != old_text:
            rec["text"] = new_text
            milvus.upsert(collection_name=COLLECTION_NAME, data=[rec])
        cleaned += 1
        done_ids.add(rec["id"])
        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(pending)} done")
            milvus.flush(collection_name=COLLECTION_NAME)
            PROGRESS_PATH.write_text(json.dumps(list(done_ids)))

    milvus.flush(collection_name=COLLECTION_NAME)
    PROGRESS_PATH.write_text(json.dumps(list(done_ids)))
    print(f"Done. cleaned={cleaned}, total_done={len(done_ids)}")


if __name__ == "__main__":
    main()
