"""Clean archaeology chunk texts using an LLM.

Reads chunks from Milvus, cleans OCR artifacts, and writes cleaned text back.

Usage:
    source .env
    uv run python scripts/clean_chunks.py
"""

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from pymilvus import MilvusClient

COLLECTION_NAME = "archaeology_chunks"
DB_PATH = "archaeology_model/indices/milvus.db"
PROGRESS_PATH = Path("archaeology_model/indices/clean_progress.json")
BATCH_DELAY_SECONDS = 0.3
MAX_RETRIES = 5

CLEAN_PROMPT = """清洗这段考古OCR文本：
1. 删除LaTeX标记（$...$），保留数值（如6°）
2. 删除图版引用（如"（图三○五）"）
3. 删除插图编号标记
4. 统一标本编号（如"标本 XM18:2"→"XM18:2"）
5. 清理多余空格空行
6. 保留所有实质内容：墓葬形制、随葬器物、尺寸数据、年代判断
7. 修复明显OCR错误
8. 不要改写原文意思

只输出清洗后的文本，不要解释。"""


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


def clean_one(client: OpenAI, model: str, text: str, max_retries: int = MAX_RETRIES) -> str:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLEAN_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            if not content:
                return text
            return content.strip()
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if any(code in err for code in ("429", "500", "502", "503", "504", "rate limit", "overloaded", "timeout", "connection")):
                time.sleep(2 ** attempt)
                continue
            raise
    print(f"  clean failed: {last_error}")
    return text


def clean_one(client: OpenAI, model: str, text: str, max_retries: int = MAX_RETRIES) -> str:
    """Clean a single chunk."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLEAN_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            if not content:
                return text
            return content.strip()
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if any(code in err for code in ("429", "500", "502", "503", "504", "rate limit", "overloaded", "timeout", "connection")):
                time.sleep(2 ** attempt)
                continue
            raise
    print(f"  clean failed: {last_error}")
    return text


def main() -> None:
    load_env()
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    clean_model = os.getenv("CLEAN_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0)

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
    failed = 0

    for idx, rec in enumerate(pending):
        new_text = clean_one(client, clean_model, rec["text"])
        try:
            rec["text"] = new_text
            milvus.upsert(
                collection_name=COLLECTION_NAME,
                data=[rec],
            )
            cleaned += 1
            done_ids.add(rec["id"])
        except Exception as e:
            print(f"  upsert failed for {rec['id']}: {e}")
            failed += 1
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(pending)} cleaned ({cleaned} ok, {failed} fail)")
            milvus.flush(collection_name=COLLECTION_NAME)
            PROGRESS_PATH.write_text(json.dumps(list(done_ids)))
        time.sleep(BATCH_DELAY_SECONDS)

    milvus.flush(collection_name=COLLECTION_NAME)
    PROGRESS_PATH.write_text(json.dumps(list(done_ids)))
    print(f"Done. cleaned={cleaned}, failed={failed}, total_done={len(done_ids)}")


if __name__ == "__main__":
    main()
