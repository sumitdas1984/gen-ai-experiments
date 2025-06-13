from llm_client_exploration.llm_utils import get_groq_response

def test_get_groq_response():
    """Test the get_groq_response function with a simple query."""
    messages = [
        {
            "role": "user", 
            "content": "What is Python?"
        }
    ]
    
    response = get_groq_response(messages)
    
    assert isinstance(response, str)
    assert len(response) > 0
