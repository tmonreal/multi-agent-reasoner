import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def load_llm():
    """
    Loads and returns the ChatGroq LLM client using environment variables.
    
    Returns:
        ChatGroq: An instance of the ChatGroq LLM client.
    
    Raises:
        ValueError: If the API key is not set in the environment variables.
    """
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0
    )