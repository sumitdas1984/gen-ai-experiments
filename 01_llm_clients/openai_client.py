import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from parent directory
load_dotenv("../.env")

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