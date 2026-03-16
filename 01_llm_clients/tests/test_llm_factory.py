"""
Test script for LLMFactory.

This script demonstrates:
1. Basic usage with all providers
2. Error handling for unsupported providers
3. Case-insensitivity of provider names
4. Comparison of responses from different providers
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_factory import LLMFactory


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


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LLM FACTORY TEST SUITE")
    print("=" * 60)

    # Run tests
    test_available_providers()
    test_basic_usage()
    test_case_insensitivity()
    test_error_handling()
    test_comparison()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
