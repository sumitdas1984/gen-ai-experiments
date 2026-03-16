import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Load .env from project root (3 levels up from this file)
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=api_key)


def ask_claude(question: str, model: str = "claude-3-haiku-20240307") -> str:
    """
    Send a prompt to Anthropic Claude model and return the response.
    """

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "user", "content": question}
        ]
    )

    return response.content[0].text


if __name__ == "__main__":
    answer = ask_claude("Explain what an LLM is in simple terms.")
    print(answer)
