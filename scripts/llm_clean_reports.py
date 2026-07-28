"""LLM-based cleanup pass for OCR-extracted archaeology reports.

Supports any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, etc.).
Install dependency first:

    uv add openai
    # or
    pip install openai
"""

import json
import re
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' package is required. Run: uv add openai")
    sys.exit(1)


RAW_DIR = Path("archaeology_model/corpus/facts/raw")
OUTPUT_DIR = Path("archaeology_model/corpus/facts/clean")
HINTS_FILE = Path(__file__).with_name("extraction_hints.json")

# Safety margin for models with ~128k context. Lower this for smaller context windows.
MAX_TOKENS_PER_CHUNK = 100_000

# Book-like types are split by top-level # chapters; article-like types by ## sections.
BOOK_LIKE_SOURCE_TYPES = {"anthology", "monograph", "book", "publication"}


def load_source_type_hints() -> dict[str, str]:
    """Load source_type keyword mapping from extraction_hints.json."""
    if not HINTS_FILE.exists():
        return {}
    data = json.loads(HINTS_FILE.read_text(encoding="utf-8"))
    return data.get("source_types", {})


def infer_source_type(filename: str) -> str:
    """Infer document type from Chinese keywords in filename."""
    hints = load_source_type_hints()
    for keyword in sorted(hints.keys(), key=len, reverse=True):
        if keyword in filename:
            return hints[keyword]
    return "publication"

_SYSTEM_INTRO = """You are an archaeological text cleanup assistant.

Given an OCR-extracted markdown archaeology report, produce a clean, well-structured markdown version."""

_RULES = """1. Fix obvious OCR errors (wrong Chinese characters, broken punctuation, stray spaces in Chinese text).
2. Remove publisher info, ISBN, cover pages, copyright pages, and "Page N" markers.
3. Normalize headings into a clear hierarchy (# title, ## sections, ### subsections).
4. If the document is an anthology/collection, keep all articles in ONE file but separate them with clear ## headings.
5. Preserve every factual detail: tomb numbers, feature IDs, artifact names, measurements, dates, stratigraphy, references.
6. Convert messy OCR tables into proper Markdown tables when possible.
7. Output ONLY the cleaned markdown. No explanations, no summaries."""

SYSTEM_PROMPT_FIRST = f"""{_SYSTEM_INTRO}

This is the FIRST part of the document. Clean it and add YAML frontmatter at the very top with these fields when inferable:
   - title
   - site
   - region
   - period
   - source_type (excavation_report / site_report / brief / anthology)

Rules:
{_RULES}
"""

SYSTEM_PROMPT_CONTINUE = f"""{_SYSTEM_INTRO}

This is a CONTINUATION part of a larger document. Continue cleaning with the same style as the previous parts. Do NOT add YAML frontmatter. Start directly with the cleaned content, preserving the heading level that matches where this section belongs.

Rules:
{_RULES}
"""


def ask(prompt: str, default: str | None = None) -> str:
    if default is not None:
        full = f"{prompt} [{default}]: "
    else:
        full = f"{prompt}: "
    value = input(full).strip()
    if not value and default is not None:
        return default
    return value


def select_model(client: OpenAI) -> str:
    try:
        response = client.models.list()
    except Exception as e:
        print(f"Failed to list models: {e}")
        sys.exit(1)

    models = sorted([m.id for m in response.data])
    if not models:
        print("No models found at this endpoint.")
        sys.exit(1)

    print("\nAvailable models:")
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")

    while True:
        choice = ask("Select model by number")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
        print("Invalid choice. Try again.")


def estimate_tokens(text: str) -> int:
    # Rough estimate for CJK + Latin mixed text.
    return len(text) // 2


def split_by_headings(text: str, level: int = 2) -> list[str]:
    """Split a markdown file by headings at the given level, keeping headings with their sections."""
    pattern = rf"^#{{{level}}}\s+"
    lines = text.splitlines()
    chunks = []
    current = []

    for line in lines:
        if re.match(pattern, line) and current:
            chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current))

    return chunks


def combine_small_chunks(chunks: list[str], max_tokens: int) -> list[str]:
    """Group consecutive small heading-sections into chunks that fit under max_tokens."""
    combined = []
    current_group = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk)
        if current_group and current_tokens + chunk_tokens > max_tokens:
            combined.append("\n\n".join(current_group))
            current_group = []
            current_tokens = 0
        current_group.append(chunk)
        current_tokens += chunk_tokens

    if current_group:
        combined.append("\n\n".join(current_group))

    return combined


def split_into_chunks(text: str, max_tokens: int, source_type: str = "publication") -> list[str]:
    """Return one chunk if small; otherwise split by headings appropriate to the document type."""
    if estimate_tokens(text) <= max_tokens:
        return [text]

    # Book-like documents usually have meaningful top-level # chapters.
    # Article-like documents usually use ## sections.
    level = 1 if source_type in BOOK_LIKE_SOURCE_TYPES else 2
    heading_chunks = split_by_headings(text, level=level)

    # Group consecutive small sections to reduce API calls.
    combined = combine_small_chunks(heading_chunks, max_tokens)

    # If a single combined chunk is still too large, fall back to naive character split.
    final_chunks = []
    for chunk in combined:
        if estimate_tokens(chunk) <= max_tokens:
            final_chunks.append(chunk)
        else:
            char_limit = max_tokens * 2
            for i in range(0, len(chunk), char_limit):
                final_chunks.append(chunk[i : i + char_limit])

    return final_chunks


def clean_with_llm(client: OpenAI, model: str, text: str, is_first: bool = True) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_FIRST if is_first else SYSTEM_PROMPT_CONTINUE},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def main() -> None:
    print("LLM Report Cleaner (OpenAI-compatible endpoint)")

    api_key = ask("API key", "")
    base_url = ask("Base URL", "http://localhost:11434/v1")

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
    model = select_model(client)

    input_path = Path(ask("Input path (file or directory)", str(RAW_DIR)))
    output_dir = Path(ask("Output directory", str(OUTPUT_DIR)))

    if not input_path.exists():
        print(f"Input path not found: {input_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        if input_path.suffix != ".md":
            print(f"Input file must be a .md file: {input_path}")
            sys.exit(1)
        files = [(input_path, input_path.name)]
    else:
        md_files = sorted(input_path.rglob("*.md"))
        files = [(p, p.relative_to(input_path)) for p in md_files]

    if not files:
        print(f"No .md files found at {input_path}")
        sys.exit(1)

    print(f"\nFound {len(files)} file(s). Processing with {model}...\n")

    for path, relative in files:
        print(f"Processing: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
            source_type = infer_source_type(path.name)
            chunks = split_into_chunks(text, MAX_TOKENS_PER_CHUNK, source_type=source_type)
            print(f"  -> inferred type: {source_type}, split into {len(chunks)} chunk(s)")

            cleaned_parts = []
            for idx, chunk in enumerate(chunks):
                part = clean_with_llm(client, model, chunk, is_first=(idx == 0))
                cleaned_parts.append(part)
                print(f"      chunk {idx + 1}/{len(chunks)} done")

            cleaned = "\n\n".join(cleaned_parts)
            out_path = output_dir / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(cleaned, encoding="utf-8")
            print(f"  -> {out_path}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")


if __name__ == "__main__":
    main()
