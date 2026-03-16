import os
from dotenv import load_dotenv
from google import genai

# Load .env from parent directory
load_dotenv("../.env")

api_key = os.getenv("GEMINI_API_KEY")

# Initialize the client
client = genai.Client(api_key=api_key)


def ask_gemini(question: str, model: str = "gemini-2.5-flash") -> str:
    """
    Send a prompt to Gemini model and return the response.
    """

    response = client.models.generate_content(
        model=model,
        contents=question
    )

    return response.text


if __name__ == "__main__":
    answer = ask_gemini("Explain what an LLM is in simple terms.")
    print(answer)