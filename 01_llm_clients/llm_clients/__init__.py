"""
LLM Clients - A unified interface for multiple LLM providers.

This package provides a factory pattern for working with different LLM providers
through a consistent interface with built-in retry logic and error handling.

Supported providers:
- OpenAI (GPT models)
- Anthropic (Claude models)
- Google Gemini
- Groq (LLaMA and other models)

Basic usage:
    >>> from llm_clients import LLMFactory
    >>> factory = LLMFactory()
    >>> response = factory.generate("openai", "Hello, world!")
    >>> print(response)

For more examples, see the examples/ directory.
"""

from llm_clients.factory import LLMFactory
from llm_clients.config import setup_logging

__version__ = "0.1.0"
__all__ = ["LLMFactory", "setup_logging"]
