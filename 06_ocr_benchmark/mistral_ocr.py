"""OCR benchmark — Mistral OCR API.

Reads  input/report.pdf
Writes output/mistral.md

Requires MISTRAL_API_KEY in the environment (or .env at repo root).
"""
import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input" / "report.pdf"
OUTPUT = HERE / "output" / "mistral.md"

MODEL = "mistral-ocr-latest"


def main() -> None:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY not set. Add it to your .env or environment.")

    pdf_b64 = base64.b64encode(INPUT.read_bytes()).decode("utf-8")
    document_url = f"data:application/pdf;base64,{pdf_b64}"

    client = Mistral(api_key=api_key)
    response = client.ocr.process(
        model=MODEL,
        document={
            "type": "document_url",
            "document_url": document_url,
        },
        include_image_base64=False,
    )

    # Concatenate each page's markdown into a single document.
    page_markdowns = [page.markdown for page in response.pages]
    markdown = "\n\n".join(page_markdowns)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(markdown, encoding="utf-8")
    print(
        f"Wrote {len(markdown):,} chars across {len(page_markdowns)} pages -> {OUTPUT}"
    )


if __name__ == "__main__":
    main()