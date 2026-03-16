"""
Demo script showing automatic retry functionality in LLMFactory.

This demonstrates:
1. Default retry configuration
2. Custom retry configuration
3. How retries work with different providers
"""

import logging
from llm_clients import LLMFactory, setup_logging


def demo_default_retry():
    """Demo with default retry settings."""
    print("=" * 70)
    print("DEMO 1: Default Retry Configuration")
    print("=" * 70)

    factory = LLMFactory()  # Uses defaults: max_retries=3, min_wait=1, max_wait=10

    print(f"\nFactory configuration:")
    print(f"  Max retries: {factory.max_retries}")
    print(f"  Min wait: {factory.min_wait}s")
    print(f"  Max wait: {factory.max_wait}s")

    print("\nMaking API call with automatic retry protection...")
    try:
        response = factory.generate("openai", "What is 5+5?")
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"\nFailed after all retries: {e}")


def demo_custom_retry():
    """Demo with custom retry settings."""
    print("\n" + "=" * 70)
    print("DEMO 2: Custom Retry Configuration")
    print("=" * 70)

    # More aggressive retry: 5 attempts with longer waits
    factory = LLMFactory(max_retries=5, min_wait=2, max_wait=30)

    print(f"\nFactory configuration:")
    print(f"  Max retries: {factory.max_retries}")
    print(f"  Min wait: {factory.min_wait}s")
    print(f"  Max wait: {factory.max_wait}s")

    print("\nMaking API call with custom retry protection...")
    try:
        response = factory.generate("anthropic", "What is the capital of France?")
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"\nFailed after all retries: {e}")


def demo_multi_provider():
    """Demo retry behavior across multiple providers."""
    print("\n" + "=" * 70)
    print("DEMO 3: Retry Across Multiple Providers")
    print("=" * 70)

    factory = LLMFactory(max_retries=3)

    providers = ["openai", "anthropic", "gemini", "groq"]
    prompt = "Say 'Hello' in one word."

    print(f"\nTesting all providers with prompt: '{prompt}'")
    print("Each call is protected by automatic retry logic\n")

    for provider in providers:
        try:
            print(f"{provider.upper()}:", end=" ")
            response = factory.generate(provider, prompt)
            print(response)
        except Exception as e:
            print(f"ERROR - {e}")


def main():
    """Run all demos."""
    # Set up logging to see retry attempts
    # Use logging.WARNING to see retry messages
    # Use logging.INFO for normal operation
    setup_logging(logging.WARNING)

    print("\n" + "=" * 70)
    print("LLM FACTORY RETRY MECHANISM DEMO")
    print("=" * 70)
    print("\nAutomatic retry handles:")
    print("  • Rate limit errors (429)")
    print("  • Network/connection errors")
    print("  • Server errors (5xx)")
    print("  • Temporary API failures")
    print("\nRetry strategy: Exponential backoff")
    print("  First retry: ~1 second")
    print("  Second retry: ~2 seconds")
    print("  Third retry: ~4 seconds")
    print("  (and so on...)")

    demo_default_retry()
    demo_custom_retry()
    demo_multi_provider()

    print("\n" + "=" * 70)
    print("DEMO COMPLETED")
    print("=" * 70)
    print("\nNote: If you see retry messages, it means the retry logic")
    print("is working to handle transient errors automatically!")


if __name__ == "__main__":
    main()
