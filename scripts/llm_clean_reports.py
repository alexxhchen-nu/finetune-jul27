"""LLM-based cleanup pass for OCR-extracted archaeology reports.

Supports any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, etc.).
Install dependency first:

    uv add openai
    # or
    pip install openai
"""

import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' package is required. Run: uv add openai")
    sys.exit(1)


RAW_DIR = Path("archaeology_model/corpus/facts/reports_raw")
OUTPUT_DIR = Path("archaeology_model/corpus/facts/clean")

SYSTEM_PROMPT = """You are an archaeological text cleanup assistant.

Given an OCR-extracted markdown archaeology report, produce a clean, well-structured markdown version.

Rules:
1. Fix obvious OCR errors (wrong Chinese characters, broken punctuation, stray spaces in Chinese text).
2. Remove publisher info, ISBN, cover pages, copyright pages, and "Page N" markers.
3. Normalize headings into a clear hierarchy (# title, ## sections, ### subsections).
4. If the document is an anthology/collection, keep all articles in ONE file but separate them with clear ## headings.
5. Preserve every factual detail: tomb numbers, feature IDs, artifact names, measurements, dates, stratigraphy, references.
6. Convert messy OCR tables into proper Markdown tables when possible.
7. At the very top of the output, add YAML frontmatter with these fields when inferable:
   - title
   - site
   - region
   - period
   - source_type (excavation_report / site_report / brief / anthology)
8. Output ONLY the cleaned markdown. No explanations, no summaries.
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


def clean_with_llm(client: OpenAI, model: str, text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
            cleaned = clean_with_llm(client, model, text)
            out_path = output_dir / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(cleaned, encoding="utf-8")
            print(f"  -> {out_path}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")


if __name__ == "__main__":
    main()
