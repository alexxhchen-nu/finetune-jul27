import json
import re
import shutil
from pathlib import Path

RAW_DIR = Path("archaeology_model/corpus/facts/reports_raw")
CLEAN_DIR = Path("archaeology_model/corpus/facts/clean")
PROCESSED_DIR = RAW_DIR / "processed"
HINTS_FILE = Path(__file__).with_name("extraction_hints.json")


def load_hints() -> tuple[list[str], list[str]]:
    """Load region/period hints from JSON so the script stays editable without code changes."""
    if not HINTS_FILE.exists():
        return [], []
    data = json.loads(HINTS_FILE.read_text(encoding="utf-8"))
    return data.get("regions", []), data.get("periods", [])


REGIONS, PERIODS = load_hints()


def clean_markdown(text: str) -> str:
    """Light mechanical cleanup. Does not fix OCR misread characters."""
    # Remove image references like ![](images/...)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    lines = text.splitlines()
    cleaned = []
    prev_blank = False
    for line in lines:
        # Normalize whitespace
        stripped = line.strip()
        # Skip lines that are obviously OCR artifact page markers (keep headings)
        if re.match(r"^Page \d+$", stripped):
            continue
        # Collapse multiple blank lines
        if stripped == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
            continue
        cleaned.append(stripped)
        prev_blank = False

    return "\n".join(cleaned).strip()


def extract_title(text: str, filename: str) -> str:
    # First # heading
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Fall back to filename
    name = Path(filename).stem
    name = re.sub(r"\.pdf_by_.+", "", name)
    return name.strip()


def extract_region(title: str) -> str | None:
    for r in REGIONS:
        if r in title:
            return r
    return None


def extract_period(title: str) -> str | None:
    for p in PERIODS:
        if p in title:
            return p
    return None


def build_frontmatter(title: str, region: str | None, period: str | None, source_path: Path) -> str:
    lines = ["---", f'title: "{title}"', f'source_type: excavation_report']
    if region:
        lines.append(f'region: "{region}"')
    if period:
        lines.append(f'period: "{period}"')
    lines.append(f'source_file: "{source_path.name}"')
    lines.append("---")
    return "\n".join(lines)


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw reports dir not found: {RAW_DIR}")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(RAW_DIR.glob("*.md")):
        # Skip the README or any file already in a subfolder.
        if path.parent != RAW_DIR:
            continue

        text = path.read_text(encoding="utf-8")
        cleaned = clean_markdown(text)

        title = extract_title(cleaned, path.name)
        region = extract_region(title)
        period = extract_period(title)

        frontmatter = build_frontmatter(title, region, period, path)
        final = f"{frontmatter}\n\n{cleaned}\n"

        out_path = CLEAN_DIR / path.name
        out_path.write_text(final, encoding="utf-8")

        processed_path = PROCESSED_DIR / path.name
        if processed_path.exists():
            processed_path.unlink()
        shutil.move(str(path), str(processed_path))

        print(f"cleaned -> {out_path}, moved -> {processed_path}")


if __name__ == "__main__":
    main()
