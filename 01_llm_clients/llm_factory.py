from openai_client import ask_openai
from anthropic_client import ask_claude
from gemini_client import ask_gemini
from groq_client import ask_groq


class LLMFactory:
    """
    Factory class to call different LLM providers
    with a unified interface.
    """

    def __init__(self):
        self.providers = {
            "openai": ask_openai,
            "anthropic": ask_claude,
            "gemini": ask_gemini,
            "groq": ask_groq
        }

    def generate(self, provider: str, prompt: str) -> str:
        """
        Generate response from selected provider.

        Args:
            provider (str): openai | anthropic | gemini | groq
            prompt (str): user prompt

        Returns:
            str: model response
        """

        provider = provider.lower()

        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")

        return self.providers[provider](prompt)