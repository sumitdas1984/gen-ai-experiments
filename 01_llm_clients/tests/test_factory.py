"""
Test script for LLMFactory.

This script demonstrates:
1. Basic usage with all providers
2. Error handling for unsupported providers
3. Case-insensitivity of provider names
4. Comparison of responses from different providers
5. Retry configuration and functionality
"""

from llm_clients import LLMFactory


def test_basic_usage():
    """Test basic usage with a simple prompt."""
    print("=" * 60)
    print("TEST 1: Basic Usage")
    print("=" * 60)

    factory = LLMFactory()
    prompt = "What is 2+2?"

    providers = ["openai", "anthropic", "gemini", "groq"]

    for provider in providers:
        try:
            print(f"\n{provider.upper()}:")
            response = factory.generate(provider, prompt)
            print(response)
        except Exception as e:
            print(f"Error with {provider}: {e}")


def test_case_insensitivity():
    """Test that provider names are case-insensitive."""
    print("\n" + "=" * 60)
    print("TEST 2: Case Insensitivity")
    print("=" * 60)

    factory = LLMFactory()
    prompt = "Say 'Hello' in one word."

    test_cases = ["OpenAI", "ANTHROPIC", "Gemini", "GROQ"]

    for provider in test_cases:
        try:
            print(f"\n{provider}:")
            response = factory.generate(provider, prompt)
            print(response)
        except Exception as e:
            print(f"Error with {provider}: {e}")


def test_error_handling():
    """Test error handling for unsupported providers."""
    print("\n" + "=" * 60)
    print("TEST 3: Error Handling")
    print("=" * 60)

    factory = LLMFactory()
    prompt = "Hello"

    invalid_providers = ["gpt4", "mistral", "llama", "invalid"]

    for provider in invalid_providers:
        try:
            print(f"\nTrying unsupported provider: {provider}")
            response = factory.generate(provider, prompt)
            print(f"Unexpected success: {response}")
        except ValueError as e:
            print(f"[PASS] Correctly raised ValueError: {e}")
        except Exception as e:
            print(f"[FAIL] Unexpected error type: {type(e).__name__}: {e}")


def test_comparison():
    """Compare responses from all providers on the same prompt."""
    print("\n" + "=" * 60)
    print("TEST 4: Provider Comparison")
    print("=" * 60)

    factory = LLMFactory()
    prompt = "Explain what an LLM is in one sentence."

    print(f"\nPrompt: {prompt}\n")

    providers = ["openai", "anthropic", "gemini", "groq"]

    for provider in providers:
        try:
            print(f"{provider.upper()}:")
            print("-" * 60)
            response = factory.generate(provider, prompt)
            print(response)
            print()
        except Exception as e:
            print(f"Error: {e}\n")


def test_available_providers():
    """Display all available providers."""
    print("\n" + "=" * 60)
    print("TEST 5: Available Providers")
    print("=" * 60)

    factory = LLMFactory()

    print("\nAvailable providers:")
    for provider in factory.providers.keys():
        print(f"  - {provider}")


def test_retry_configuration():
    """Test that factory has retry configuration."""
    print("\n" + "=" * 60)
    print("TEST 6: Retry Configuration")
    print("=" * 60)

    # Test default configuration
    factory_default = LLMFactory()
    print("\nDefault configuration:")
    print(f"  - Max retries: {factory_default.max_retries}")
    print(f"  - Min wait: {factory_default.min_wait}s")
    print(f"  - Max wait: {factory_default.max_wait}s")

    # Test custom configuration
    factory_custom = LLMFactory(max_retries=5, min_wait=2, max_wait=30)
    print("\nCustom configuration:")
    print(f"  - Max retries: {factory_custom.max_retries}")
    print(f"  - Min wait: {factory_custom.min_wait}s")
    print(f"  - Max wait: {factory_custom.max_wait}s")

    print("\nRetry mechanism automatically handles:")
    print("  - Rate limit errors (429)")
    print("  - Network/connection errors")
    print("  - Server errors (5xx)")
    print("  - Other transient API errors")
    print("\nRetries use exponential backoff strategy")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LLM FACTORY TEST SUITE")
    print("=" * 60)

    # Run tests
    test_available_providers()
    test_retry_configuration()
    test_basic_usage()
    test_case_insensitivity()
    test_error_handling()
    test_comparison()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
