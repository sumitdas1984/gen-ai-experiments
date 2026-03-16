import os
from dotenv import load_dotenv
from groq import Groq

# Load .env from parent directory
load_dotenv("../.env")

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