"""
Individual LLM provider clients.

Each provider module exports a function that takes a question string and returns
a response string. All providers support optional model selection.

Available providers:
- ask_claude: Anthropic Claude API
- ask_openai: OpenAI GPT API
- ask_gemini: Google Gemini API
- ask_groq: Groq-hosted models API
"""

from llm_clients.providers.anthropic_client import ask_claude
from llm_clients.providers.openai_client import ask_openai
from llm_clients.providers.gemini_client import ask_gemini
from llm_clients.providers.groq_client import ask_groq

__all__ = ["ask_claude", "ask_openai", "ask_gemini", "ask_groq"]
