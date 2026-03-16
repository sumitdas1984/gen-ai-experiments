import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from llm_clients.providers import ask_openai, ask_claude, ask_gemini, ask_groq

# Set up logger
logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory class to call different LLM providers
    with a unified interface and automatic retry logic.
    """

    def __init__(self, max_retries=3, min_wait=1, max_wait=10):
        """
        Initialize LLMFactory with retry configuration.

        Args:
            max_retries (int): Maximum number of retry attempts (default: 3)
            min_wait (int): Minimum wait time in seconds for exponential backoff (default: 1)
            max_wait (int): Maximum wait time in seconds for exponential backoff (default: 10)
        """
        self.providers = {
            "openai": ask_openai,
            "anthropic": ask_claude,
            "gemini": ask_gemini,
            "groq": ask_groq
        }
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait

    def generate(self, provider: str, prompt: str) -> str:
        """
        Generate response from selected provider with automatic retries.

        Automatically retries on transient errors:
        - Rate limit errors (429)
        - Network/connection errors
        - Server errors (5xx)
        - Other temporary API errors

        Args:
            provider (str): openai | anthropic | gemini | groq
            prompt (str): user prompt

        Returns:
            str: model response
        """
        provider = provider.lower()

        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")

        # Apply retry logic
        return self._generate_with_retry(provider, prompt)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            # Network errors
            ConnectionError,
            TimeoutError,
            # Catch all transient errors (rate limits, API errors, etc.)
            Exception
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _generate_with_retry(self, provider: str, prompt: str) -> str:
        """
        Internal method with retry decorator.

        This method is wrapped with tenacity's @retry decorator to automatically
        retry on transient failures with exponential backoff.
        """
        try:
            return self.providers[provider](prompt)
        except ValueError:
            # Don't retry on unsupported provider - this is a permanent error
            raise
