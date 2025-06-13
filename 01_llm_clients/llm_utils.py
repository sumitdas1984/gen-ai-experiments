import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv("../.env")

def get_groq_response(messages: list) -> str:
    """
    Get response from Groq API using a list of messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    
    Returns:
        String response from the model
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        max_tokens=100,
        temperature=0.7,
    )
    
    return chat_completion.choices[0].message.content


def get_openai_response(messages: list) -> str:
    """
    Get response from OpenAI API using a list of messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    
    Returns:
        String response from the model
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.7,
    )
    
    return chat_completion.choices[0].message.content


# Example usage
if __name__ == "__main__":

    # example 1: Simple question
    messages_1 = [
        {
            "role": "user", 
            "content": "What is Python?"
        }
    ]

    # example 2: With system context
    messages_2 = [
        {
            "role": "system", 
            "content": "You are a coding mentor. Give practical advice."
        },
        {
            "role": "user", 
            "content": "How can I improve my Python skills?"
        }
    ]

    # example 3: Conversation flow
    messages_3 = [
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user", 
            "content": "Can you explain what a Python decorator is?"
        },
        {
            "role": "assistant", 
            "content": "Sure! A Python decorator is a function that modifies the behavior of another function or method. It is often used to add functionality in a clean and reusable way."
        },
        {
            "role": "user", 
            "content": "Can you give me an example?"
        },
    ]

    llm_name = "groq"  # or "openai"
    messages = messages_1  # Change this to messages_2 or messages_3 as needed

    if llm_name == "groq":
        response = get_groq_response(messages)
    elif llm_name == "openai":
        response = get_openai_response(messages)
    else:
        raise ValueError("Unsupported LLM name. Use 'groq' or 'openai'.")
    
    print(f"Response from {llm_name}:\n{response}")