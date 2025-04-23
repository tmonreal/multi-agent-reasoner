import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

def load_llm():
    load_dotenv()
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0
    )