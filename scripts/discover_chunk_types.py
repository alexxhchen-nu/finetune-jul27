"""Discover a chunk taxonomy from the corpus using heading frequency and an LLM.

Reads markdown files, extracts headings and samples, and asks an LLM to propose
a normalized taxonomy with heading mappings. Writes the result to a JSON file
that can be used by build_milvus_index.py for classification.

Usage:
    source .env
    uv run python scripts/discover_chunk_types.py

Output:
    archaeology_model/glossary/chunk_types.json
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

from openai import OpenAI

SOURCE_DIRS = [
    Path("archaeology_model/corpus/facts/clean"),
    Path("archaeology_model/corpus/methods"),
]
OUTPUT_DIR = Path("archaeology_model/glossary")
OUTPUT_PATH = OUTPUT_DIR / "chunk_types.json"
SAMPLES_PER_HEADING = 1
MAX_HEADINGS = 50


def load_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return {}, parts[2]
    return {}, text


def iter_chunks() -> Iterator[tuple[str, str, Path]]:
    """Yield (heading, body_text, path) chunks."""
    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            continue
        for path in sorted(src_dir.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            _, body = load_frontmatter(text)
            lines = body.splitlines()
            current_heading = ""
            current_lines: list[str] = []

            def flush():
                txt = "\n".join(current_lines).strip()
                if len(txt) >= 50:
                    yield current_heading, txt, path
                current_lines.clear()

            for line in lines:
                if re.match(r"^#{2,4}\s+", line):
                    yield from flush()
                    current_heading = re.sub(r"^#{2,4}\s+", "", line).strip()
                else:
                    current_lines.append(line)
            yield from flush()


def collect_heading_samples() -> tuple[Counter, dict[str, list[str]]]:
    heading_counts: Counter = Counter()
    heading_samples: dict[str, list[str]] = {}

    for heading, text, _ in iter_chunks():
        heading_counts[heading] += 1
        heading_samples.setdefault(heading, []).append(text[:500])

    return heading_counts, heading_samples


def build_prompt(top_headings: list[tuple[str, int]], samples: dict[str, list[str]]) -> str:
    sample_lines = []
    for heading, count in top_headings[:MAX_HEADINGS]:
        for sample in samples[heading][:SAMPLES_PER_HEADING]:
            sample_lines.append(f"heading: {heading}\ncount: {count}\nsample: {sample[:200]}\n---")

    return (
        "你是一位考古学文本分析专家。下面是从一批考古发掘报告中提取的章节标题和对应文本片段。"
        "请基于这些标题和正文内容，归纳出一个稳定的 chunk 分类体系。\n\n"
        "要求：\n"
        "1. 给出 5~12 个基础分类标签（base_types），覆盖墓葬结构、随葬器物、年代分期、发掘方法、背景概述等。\n"
        "2. 建立一个 heading_map：把常见的标题映射到最匹配的基础分类。\n"
        "3. 如果某些标题没有明确对应，归入 '其他'，不要强行分类。\n"
        "4. 返回 JSON 格式，不要任何解释。\n\n"
        "示例输出格式：\n"
        '{"base_types":["墓葬形制","随葬器物","分期断代","葬式葬具","发掘方法","遗址背景","图表数据","其他"],'
        '"heading_map":{"一、墓葬形制":"墓葬形制","三、随葬器物":"随葬器物","附表":"图表数据","前言":"其他","目录":"其他"}}\n\n'
        "数据：\n" + "\n".join(sample_lines)
    )


def call_llm(prompt: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    model = os.getenv("CLASSIFY_MODEL", "Qwen/Qwen2.5-14B-Instruct")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=4096,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty content")
    return json.loads(content)


def main() -> None:
    print("Collecting headings and samples...")
    heading_counts, heading_samples = collect_heading_samples()
    top_headings = heading_counts.most_common(120)

    print(f"Found {len(heading_counts)} unique headings")
    print(f"Top headings: {top_headings[:10]}")

    print("Calling LLM to propose taxonomy...")
    prompt = build_prompt(top_headings, heading_samples)
    taxonomy = call_llm(prompt)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)
    print(f"Wrote taxonomy to {OUTPUT_PATH}")

    print("\nBase types:", taxonomy.get("base_types", []))
    print("Heading map entries:", len(taxonomy.get("heading_map", {})))


if __name__ == "__main__":
    main()
