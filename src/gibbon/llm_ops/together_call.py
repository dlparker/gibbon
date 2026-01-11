import os
from pathlib import Path

from dotenv import load_dotenv
from together import Together
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

models = {
    "mistral-3:14B": "mistralai/Ministral-3-14B-Instruct-2512",
    "mistral-small:24B": "mistralai/Mistral-Small-24B-Instruct-2501",
    "llama-3.1:70B": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
}
async def send_to_together_ai(prompts, model: str, tools:list[dict]=None) -> dict:
    client = Together()
    messages = [
            {
                "role": "system",
                "content": prompts['system']
            },
            {
                "role": "user",
                "content": prompts['user']
            }
        ]
        
    response = client.chat.completions.create(
        model="mistralai/Ministral-3-14B-Instruct-2512",
        max_tokens=2000,
        temperature=0.1,  # Low temperature for more consistent JSON
        tools=tools,
        messages=messages)
    return response
