# Automatic Retry Feature

The LLMFactory now includes automatic retry logic to handle transient API failures gracefully.

## Quick Start

### Basic Usage (Default Settings)
```python
from llm_factory import LLMFactory

factory = LLMFactory()  # Default: 3 retries, 1-10s backoff
response = factory.generate("openai", "Hello!")
```

### Custom Configuration
```python
factory = LLMFactory(
    max_retries=5,    # More retry attempts
    min_wait=2,       # Start with 2s wait
    max_wait=30       # Cap at 30s wait
)
```

## What Gets Retried

The retry mechanism automatically handles:

- ✅ **Rate limit errors (429)** - When you hit API rate limits
- ✅ **Network errors** - Connection timeouts, DNS failures
- ✅ **Server errors (5xx)** - Temporary server issues
- ✅ **Transient API errors** - Other temporary failures

## What Doesn't Get Retried

- ❌ **Invalid API keys (401, 403)** - Permanent authentication issues
- ❌ **Invalid requests (400, 422)** - Malformed requests
- ❌ **Unsupported providers** - ValueError for wrong provider names
- ❌ **Content policy violations** - Blocked content

## Retry Strategy

Uses **exponential backoff**:

| Attempt | Wait Time | Total Time Elapsed |
|---------|-----------|-------------------|
| 1st     | 0s        | 0s                |
| 2nd     | ~1s       | ~1s               |
| 3rd     | ~2s       | ~3s               |
| 4th     | ~4s       | ~7s               |

The wait time doubles after each retry, up to `max_wait` seconds.

## Configuration Options

```python
LLMFactory(
    max_retries=3,   # Number of retry attempts (default: 3)
    min_wait=1,      # Minimum wait between retries in seconds (default: 1)
    max_wait=10      # Maximum wait between retries in seconds (default: 10)
)
```

## Logging

Enable logging to see retry attempts:

```python
from logging_config import setup_logging
import logging

setup_logging(logging.WARNING)  # Shows retry attempts
# or
setup_logging(logging.INFO)     # Normal operation
# or
setup_logging(logging.DEBUG)    # Detailed debugging
```

When retries occur, you'll see:
```
2025-03-16 21:30:45 - llm_factory - WARNING - Retrying after 1.0s
2025-03-16 21:30:46 - llm_factory - WARNING - Retrying after 2.0s
```

## Examples

### Example 1: Resilient API Calls
```python
from llm_factory import LLMFactory

factory = LLMFactory()

# Even if there's a temporary network issue or rate limit,
# the call will automatically retry up to 3 times
response = factory.generate("openai", "What is AI?")
print(response)
```

### Example 2: Production Settings
```python
# For production: more retries, longer waits
factory = LLMFactory(max_retries=5, min_wait=2, max_wait=60)

response = factory.generate("anthropic", "Analyze this data...")
```

### Example 3: Fast-Fail for Testing
```python
# For testing: fail fast
factory = LLMFactory(max_retries=1, min_wait=0, max_wait=1)

response = factory.generate("gemini", "Quick test")
```

## Run Demos

See the retry feature in action:

```bash
# Run comprehensive demo
python demo_retry.py

# Run test suite
python tests/test_llm_factory.py
```

## Benefits

1. 🛡️ **Resilience** - Handles temporary failures automatically
2. ⚡ **Better UX** - Reduces failed requests
3. 🔄 **Consistent** - Same retry behavior across all providers
4. ⚙️ **Configurable** - Adjust to your needs
5. 📊 **Observable** - Logging shows what's happening

## Implementation Details

- Uses [tenacity](https://tenacity.readthedocs.io/) library for robust retry logic
- Retry logic is applied at the `LLMFactory.generate()` level
- All providers (OpenAI, Anthropic, Gemini, Groq) benefit from the same retry protection
- Exponential backoff prevents overwhelming APIs during incidents
