import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root (3 levels up from this file)
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


def ask_openai(question: str, model: str = "gpt-4o-mini") -> str:
    """
    Send a prompt to OpenAI model and return the response.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    answer = ask_openai("Explain what an LLM is in simple terms.")
    print(answer)
