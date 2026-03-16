import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load .env from project root (3 levels up from this file)
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)


def ask_groq(question: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Send a prompt to Groq-hosted model and return the response.
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
    answer = ask_groq("Explain what an LLM is in simple terms.")
    print(answer)
