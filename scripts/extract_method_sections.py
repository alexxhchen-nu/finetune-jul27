"""Extract methodology sections from cleaned reports into corpus/methods/.

Scans archaeology_model/corpus/facts/clean/ for headings that look like
methodology topics (typology, stratigraphy, dating, etc.) and copies those
sections into archaeology_model/corpus/methods/. Run this on demand when you
add new reports.

Usage:
    uv run python scripts/extract_method_sections.py
"""

import re
from pathlib import Path

CLEAN_DIR = Path("archaeology_model/corpus/facts/clean")
METHODS_DIR = Path("archaeology_model/corpus/methods")

# Heading keywords that indicate methodology content.
# Keep this conservative: individual tomb descriptions often contain
# "墓葬形制", so we require stronger method markers.
METHOD_KEYWORDS = [
    "类型学",
    "地层学",
    "断代",
    "分期与年代",
    "文化分期",
    "分期研究",
    "器物描述",
    "田野考古",
    "发掘方法",
    "记录方法",
    "整理方法",
    "研究方法",
    "分析方法",
    "取样方法",
    "测年方法",
    "空间分析",
    "报告编写",
    "编写体例",
    "研究方法与材料",
]


def normalize_filename(text: str) -> str:
    """Make a safe filename stem from a heading."""
    text = text.strip().replace(" ", "_").replace("/", "_")
    text = re.sub(r"[^\w\u4e00-\u9fff_-]", "", text)
    return text[:80] or "section"


def extract_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) sections whose heading matches method keywords."""
    lines = text.splitlines()
    found: list[tuple[str, str]] = []
    current_heading = ""
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_body
        if current_heading and current_body:
            body = "\n".join(current_body).strip()
            if body:
                found.append((current_heading, body))
        current_heading = ""
        current_body = []

    for line in lines:
        if re.match(r"^#{2,4}\s+", line):
            flush()
            current_heading = re.sub(r"^#{2,4}\s+", "", line).strip()
        elif current_heading:
            current_body.append(line)
    flush()

    results = []
    for heading, body in found:
        if any(kw in heading for kw in METHOD_KEYWORDS):
            results.append((heading, body))
    return results


def main() -> None:
    METHODS_DIR.mkdir(parents=True, exist_ok=True)
    extracted = 0

    for path in sorted(CLEAN_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        sections = extract_sections(text)
        if not sections:
            continue

        source_title = path.stem
        for heading, body in sections:
            stem = f"{normalize_filename(source_title)}_{normalize_filename(heading)}"
            out_path = METHODS_DIR / f"{stem}.md"
            counter = 1
            while out_path.exists():
                out_path = METHODS_DIR / f"{stem}_{counter}.md"
                counter += 1

            content = f"# {heading}\n\n> 来源：{path.name}\n\n{body}\n"
            out_path.write_text(content, encoding="utf-8")
            print(f"extracted -> {out_path.relative_to(METHODS_DIR)}")
            extracted += 1

    print(f"\nExtracted {extracted} method section(s) into {METHODS_DIR}")


if __name__ == "__main__":
    main()
