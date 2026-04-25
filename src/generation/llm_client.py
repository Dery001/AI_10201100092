# ============================================================
# llm_client.py – Ollama Cloud client for response generation
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

import os
from ollama import Client
from dotenv import load_dotenv

# Read the .env file
load_dotenv()

_client = None

def _get_client() -> Client:
    global _client
    if _client is None:
        api_key = os.getenv("OLLAMA_API_KEY")
        base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        
        # If API key is provided, we send it in headers.
        if api_key:
            _client = Client(
                host=base_url,
                headers={'Authorization': f'Bearer {api_key}'}
            )
        else:
            _client = Client(host=base_url)
            
    return _client


def generate_response(
    prompt: str,
    model: str = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Send a prompt to Ollama Cloud and return the response text.
    Temperature is kept low (0.2) to reduce hallucination in factual tasks.
    """
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
        
    client = _get_client()
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"[LLM Error] {e}"


def generate_pure_llm_response(
    query: str,
    model: str = None,
    max_tokens: int = 512,
) -> str:
    """
    Generate a response WITHOUT any retrieved context.
    Used for RAG vs pure-LLM comparison in evaluation.
    """
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
        
    prompt = (
        "You are a knowledgeable assistant. Answer the following question "
        "using your own general knowledge:\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return generate_response(prompt, model=model, max_tokens=max_tokens, temperature=0.7)
