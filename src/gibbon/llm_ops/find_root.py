import asyncio

from ollama import AsyncClient

async def send_to_llm(combined_text: str, ollama_url: str, model: str) -> dict:
    """Send draft text with prompt to ollama and return the response."""
    client = AsyncClient(host=ollama_url)

    # Build the full prompt
    full_prompt = combined_text

    print(f"\nSending to {model} at {ollama_url}...")
    print(f"Prompt length: {len(full_prompt)} chars")

    messages = [{'role': 'user', 'content': full_prompt}]

    response = await client.chat(
        model=model,
        messages=messages,
        format='json',  # Request JSON output format
        options={
            'temperature': 0,  # Deterministic responses
        }
    )
    return response
