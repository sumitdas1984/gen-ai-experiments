from ollama_ocr import OCRProcessor

# Initialize OCR processor
ocr = OCRProcessor(model_name='qwen2.5vl:3b', base_url="http://localhost:11434/api/generate")  # You can use any vision model available on Ollama
# you can pass your custom ollama api

# Process an image
result = ocr.process_image(
    image_path="input/report.pdf", # path to your pdf files "path/to/your/file.pdf"
    format_type="markdown",  # Options: markdown, text, json, structured, key_value
    custom_prompt="Extract all text, focusing on dates and names.", # Optional custom prompt
    language="English" # Specify the language of the text (New! 🆕)
)
print(result)