"""
Logging configuration for LLM clients.

This module provides simple logging setup for tracking API calls,
retries, and errors across different LLM providers.
"""

import logging


def setup_logging(level=logging.INFO):
    """
    Configure logging for LLM clients.

    Args:
        level: Logging level (default: logging.INFO)
               Use logging.DEBUG for more detailed output
               Use logging.WARNING for only warnings and errors

    Example:
        >>> from llm_clients import setup_logging
        >>> setup_logging(logging.DEBUG)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Suppress overly verbose logs from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


if __name__ == "__main__":
    # Demo logging at different levels
    setup_logging(logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
