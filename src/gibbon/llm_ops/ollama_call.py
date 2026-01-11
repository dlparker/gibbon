import asyncio
import logging
from typing import Optional

logger = logging.getLogger("OllamaCall")

from ollama import AsyncClient

async def send_to_ollama(prompts:dict,  ollama_url: str, model: str, tools:Optional[list[dict]]=None) -> dict:
    client = AsyncClient(host=ollama_url)

    # Handle both dict (new) and string (old) formats for backwards compatibility
    system_msg = prompts.get('system', '')
    if system_msg == '' and tools:
        system_msg = "You are a function-calling assistant. You MUST respond using tool calls, never with direct JSON output."
    user_msg = prompts.get('user', '')

    logger.debug(f"\nSending to {model} at {ollama_url}...")
    logger.debug(f"System prompt length: {len(system_msg)} chars")
    logger.debug(f"User prompt length: {len(user_msg)} chars")
    logger.debug(f"Tool calling: {'enabled' if tools else 'disabled'}")

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
