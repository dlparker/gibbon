import asyncio
from typing import Optional

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
                                },
                                "matching_excerpt": {
                                    "type": "string",
                                    "description": "The exact words or phrase from the transcript that led to this category match"
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
async def send_to_llm(prompts:dict,  ollama_url: str, model: str, tools:Optional[list[dict]]=None) -> dict:
    client = AsyncClient(host=ollama_url)

    # Handle both dict (new) and string (old) formats for backwards compatibility
    system_msg = prompts.get('system', '')
    if system_msg == '' and tools:
        system_msg = "You are a function-calling assistant. You MUST respond using tool calls, never with direct JSON output."
    user_msg = prompts.get('user', '')

    print(f"\nSending to {model} at {ollama_url}...")
    print(f"System prompt length: {len(system_msg)} chars")
    print(f"User prompt length: {len(user_msg)} chars")
    print(f"Tool calling: {'enabled' if tools else 'disabled'}")

    # Build messages array with system and user
    messages = [
        {'role': 'system', 'content': system_msg},
        {'role': 'user', 'content': user_msg}
    ]

    # Build chat parameters
    chat_params = {
        'model': model,
        'messages': messages,
        'options': {
            'temperature': 0,  # Deterministic responses
        }
    }

    # Only include tools if tool calling is enabled
    if tools:
        chat_params['tools'] = tools

    response = await client.chat(**chat_params)
    return response
