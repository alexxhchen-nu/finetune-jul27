"""LLM-based cleanup pass for OCR-extracted archaeology reports.

Supports any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, etc.).
Install dependency first:

    uv add openai
    # or
    pip install openai
"""

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
# Safety margin for models with ~128k context. Lower this for smaller context windows.
MAX_TOKENS_PER_CHUNK = 100_000

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


def split_by_headings(text: str) -> list[str]:
    """Split a large markdown file by second-level headings, keeping headings with their sections."""
    lines = text.splitlines()
    chunks = []
    current = []

    for line in lines:
        if re.match(r"^##\s+", line) and current:
            chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current))

    return chunks


def split_into_chunks(text: str, max_tokens: int) -> list[str]:
    """Return one chunk if small enough; otherwise split by ## headings."""
    if estimate_tokens(text) <= max_tokens:
        return [text]

    heading_chunks = split_by_headings(text)

    # If a single heading chunk is still too large, do a naive character split.
    final_chunks = []
    for chunk in heading_chunks:
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

    input_dir = Path(ask("Input directory", str(RAW_DIR)))
    output_dir = Path(ask("Output directory", str(OUTPUT_DIR)))

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob("*.md"))
    if not files:
        print(f"No .md files found in {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(files)} file(s). Processing with {model}...\n")

    for path in files:
        relative = path.relative_to(input_dir)
        print(f"Processing: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
            chunks = split_into_chunks(text, MAX_TOKENS_PER_CHUNK)
            print(f"  -> split into {len(chunks)} chunk(s)")

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
