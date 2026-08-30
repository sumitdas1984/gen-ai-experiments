"""OCR benchmark — Docling.

Reads  input/report.pdf
Writes output/docling.md
"""
# Disable Hugging Face symlinks on Windows to avoid WinError 1314
# (requires Developer Mode or admin privileges otherwise).
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from pathlib import Path
from docling.document_converter import DocumentConverter

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input" / "report.pdf"
OUTPUT = HERE / "output" / "docling.md"


def main() -> None:
    converter = DocumentConverter()
    result = converter.convert(str(INPUT))
    markdown = result.document.export_to_markdown()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(markdown):,} chars -> {OUTPUT}")


if __name__ == "__main__":
    main()