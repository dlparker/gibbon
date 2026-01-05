import asyncio

from ollama import AsyncClient

tools = [
    {
        "type": "function",
        "function": {
            "name": "submit_category_matches",
            "description": "Submit the ranked category matches for the transcript.",
            "parameters": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "description": "List of matching categories, sorted by confidence descending",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category_name": {
                                    "type": "string",
                                    "description": "Exact key from the category list (e.g. 'new_topic', 'copper')"
                                },
                                "category_description": {
                                    "type": "string",
                                    "description": "Exact description from the category list"
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Confidence score from 0.0 to 1.0"
                                }
                            },
                            "required": ["category_name", "category_description", "confidence"]
                        }
                    }
                },
                "required": ["matches"]
            }
        }
    }
]
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
        tools=tools,
        # format='json',  # Removed: conflicts with tool calling
        options={
            'temperature': 0,  # Deterministic responses
        }
    )
    return response
