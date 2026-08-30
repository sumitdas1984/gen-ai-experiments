"""OCR benchmark — Ollama (local vision model via ollama_ocr).

Reads  input/report.pdf
Writes output/ollama.md
"""
from pathlib import Path
from ollama_ocr import OCRProcessor

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input" / "report.pdf"
OUTPUT = HERE / "output" / "ollama.md"

# Pick whichever local vision model you have pulled. Known-working on
# Ollama 0.31.x: 'llava:13b', 'moondream:latest', 'gemma3:4b'.
# ('llama3.2-vision' is currently broken on Ollama 0.31.1 — issue #16490.)
MODEL_NAME = "qwen2.5vl:3b"
BASE_URL = "http://localhost:11434/api/generate"


def main() -> None:
    ocr = OCRProcessor(model_name=MODEL_NAME, base_url=BASE_URL)
    markdown = ocr.process_image(
        image_path=str(INPUT),
        format_type="markdown",
        custom_prompt="Extract all text faithfully. Preserve structure, headings, tables, and lists.",
        language="English",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(markdown):,} chars -> {OUTPUT}")


if __name__ == "__main__":
    main()