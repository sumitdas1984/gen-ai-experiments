# Disable Hugging Face symlinks on Windows to avoid WinError 1314
# (requires Developer Mode or admin privileges otherwise).
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"  # a document via a local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"