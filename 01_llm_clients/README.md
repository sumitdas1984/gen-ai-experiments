# LLM Clients

A unified interface for working with multiple Large Language Model (LLM) providers with built-in retry logic and error handling.

## Features

- 🔄 **Unified Interface** - Single API to work with multiple LLM providers
- 🔁 **Automatic Retries** - Built-in retry logic with exponential backoff for transient failures
- 🎯 **Multiple Providers** - Support for OpenAI, Anthropic, Google Gemini, and Groq
- ⚙️ **Configurable** - Easy to customize retry behavior and provider settings
- 📊 **Observable** - Optional logging for debugging and monitoring
- 🧪 **Well-Tested** - Comprehensive test suite included

## Supported Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| **OpenAI** | GPT-4, GPT-4o, GPT-3.5, etc. | `OPENAI_API_KEY` |
| **Anthropic** | Claude 3 (Opus, Sonnet, Haiku) | `ANTHROPIC_API_KEY` |
| **Google** | Gemini 2.5 Flash, Gemini Pro | `GEMINI_API_KEY` |
| **Groq** | LLaMA 3.1, Mixtral, etc. | `GROQ_API_KEY` |

## Installation

1. Clone this repository or copy the files to your project:

```bash
git clone <your-repo-url>
cd 01_llm_clients
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your API keys in a `.env` file in the parent directory:

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

## Quick Start

### Basic Usage

```python
from llm_clients import LLMFactory

# Create factory instance with default settings
factory = LLMFactory()

# Generate response from any provider
response = factory.generate("openai", "Explain quantum computing in simple terms.")
print(response)

# Try different providers
providers = ["openai", "anthropic", "gemini", "groq"]
for provider in providers:
    response = factory.generate(provider, "Hello, world!")
    print(f"{provider}: {response}")
```

### Custom Retry Configuration

```python
from llm_clients import LLMFactory

# Configure more aggressive retry behavior
factory = LLMFactory(
    max_retries=5,    # More retry attempts
    min_wait=2,       # Start with 2 second wait
    max_wait=30       # Cap at 30 seconds
)

response = factory.generate("anthropic", "Your prompt here")
```

### With Logging

```python
from llm_clients import LLMFactory, setup_logging
import logging

# Enable logging to see retry attempts
setup_logging(logging.WARNING)

factory = LLMFactory()
response = factory.generate("gemini", "Test prompt")
```

## Project Structure

```
01_llm_clients/
├── llm_clients/              # Main package
│   ├── __init__.py          # Package exports
│   ├── factory.py           # LLMFactory class
│   ├── config.py            # Logging configuration
│   └── providers/           # Individual provider clients
│       ├── anthropic_client.py
│       ├── openai_client.py
│       ├── gemini_client.py
│       └── groq_client.py
├── examples/                 # Demo scripts
│   └── retry_demo.py        # Retry feature demonstration
├── tests/                    # Test suite
│   ├── test_factory.py      # Factory tests
│   └── __init__.py
├── docs/                     # Documentation
│   └── RETRY_FEATURE.md     # Detailed retry documentation
├── requirements.txt          # Python dependencies
├── pytest.ini               # Test configuration
└── README.md                # This file
```

## Examples

### Example 1: Compare Provider Responses

```python
from llm_clients import LLMFactory

factory = LLMFactory()
prompt = "What is the capital of France?"

for provider in ["openai", "anthropic", "gemini", "groq"]:
    try:
        response = factory.generate(provider, prompt)
        print(f"\n{provider.upper()}:")
        print(response)
    except Exception as e:
        print(f"{provider} error: {e}")
```

### Example 2: Error Handling

```python
from llm_clients import LLMFactory

factory = LLMFactory()

try:
    response = factory.generate("openai", "Hello!")
    print(response)
except ValueError as e:
    print(f"Invalid provider: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Running Examples

```bash
# Run the retry demonstration
python examples/retry_demo.py
```

## Running Tests

```bash
# Run all tests
python tests/test_factory.py

# Or use pytest
pytest

# Run with verbose output
pytest -v
```

## Retry Mechanism

The package includes automatic retry logic for handling transient failures:

### What Gets Retried
- ✅ Rate limit errors (429)
- ✅ Network/connection errors
- ✅ Server errors (5xx)
- ✅ Temporary API failures

### What Doesn't Get Retried
- ❌ Invalid API keys (401, 403)
- ❌ Invalid requests (400, 422)
- ❌ Unsupported providers
- ❌ Content policy violations

### Retry Strategy
- Uses **exponential backoff** (1s, 2s, 4s, 8s, ...)
- Default: **3 attempts** with 1-10 second wait range
- Fully **configurable** per factory instance

See [docs/RETRY_FEATURE.md](docs/RETRY_FEATURE.md) for detailed documentation.

## API Reference

### LLMFactory

```python
class LLMFactory(max_retries=3, min_wait=1, max_wait=10)
```

**Parameters:**
- `max_retries` (int): Maximum number of retry attempts (default: 3)
- `min_wait` (int): Minimum wait time in seconds for exponential backoff (default: 1)
- `max_wait` (int): Maximum wait time in seconds for exponential backoff (default: 10)

**Methods:**

#### `generate(provider, prompt)`

Generate response from selected provider.

**Parameters:**
- `provider` (str): Provider name ("openai", "anthropic", "gemini", or "groq")
- `prompt` (str): The prompt/question to send to the LLM

**Returns:**
- `str`: The model's response

**Raises:**
- `ValueError`: If provider is not supported
- Various API errors if all retry attempts fail

### setup_logging

```python
def setup_logging(level=logging.INFO)
```

Configure logging for LLM clients.

**Parameters:**
- `level`: Logging level (default: logging.INFO)

## Dependencies

- `python-dotenv` - Environment variable management
- `tenacity` - Retry logic
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `google-genai` - Google Gemini API client
- `groq` - Groq API client

## Contributing

This is a learning/experimental project. Feel free to:
- Add new providers
- Improve error handling
- Add more examples
- Enhance documentation

## License

This project is for educational purposes.

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running scripts from the project root:

```bash
cd 01_llm_clients
python examples/retry_demo.py
python tests/test_factory.py
```

### API Key Errors

Ensure your `.env` file is in the parent directory (`../.env` relative to the project root) and contains valid API keys.

### Rate Limiting

If you hit rate limits frequently, increase the retry configuration:

```python
factory = LLMFactory(max_retries=5, min_wait=2, max_wait=60)
```

## Further Reading

- [Retry Feature Documentation](docs/RETRY_FEATURE.md)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [Groq API Docs](https://console.groq.com/docs)
