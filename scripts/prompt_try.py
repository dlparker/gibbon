import asyncio
import json
import re
from gibbon.llm_ops.find_root import send_to_llm

url = 'http://192.168.100.242:11434'
model = 'llama3.1:8b-instruct-q4_K_M'


def parse_llm_response(response) -> list[dict]:
    """
    Parse LLM response to extract category matches.

    Handles various formats:
    - Single JSON object: {...}
    - JSON array: [{...}, {...}]
    - Markdown code blocks: ```json ... ```
    - Extra text before/after JSON

    Returns:
        List of match dicts with keys: category_ID, category_name, confidence
        Empty list if parsing fails
    """
    # Extract content string from response object
    if hasattr(response, 'message') and hasattr(response.message, 'content'):
        content = response.message.content
    elif isinstance(response, str):
        content = response
    else:
        content = str(response)

    # Remove markdown code blocks if present
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)

    # Try to find JSON in the content (array or object)
    # Look for [...] or {...}
    json_match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)

    if not json_match:
        print(f"Warning: No JSON found in response: {content[:100]}")
        return []

    json_str = json_match.group(1)

    try:
        parsed = json.loads(json_str)

        # Normalize to list format
        if isinstance(parsed, dict):
            # Single object, wrap in list
            matches = [parsed]
        elif isinstance(parsed, list):
            matches = parsed
        else:
            print(f"Warning: Unexpected JSON type: {type(parsed)}")
            return []

        # Validate structure
        valid_matches = []
        for match in matches:
            if isinstance(match, dict) and 'category_ID' in match and 'category_name' in match:
                # Add confidence if missing
                if 'confidence' not in match:
                    match['confidence'] = 0.0
                valid_matches.append(match)
            else:
                print(f"Warning: Invalid match format: {match}")

        return valid_matches

    except json.JSONDecodeError as e:
        print(f"Warning: JSON parse error: {e}")
        print(f"Content: {json_str[:200]}")
        return []

cats = """
Categories (hierarchical - child categories are more specific than parents):

- Todo lists (ID=2, parent: none)
  - grocery shopping list (ID=3, parent: 2)
  - taxes (ID=5, parent: 2)
- Project Notes (ID=4, parent: none)
  - Customize kanban board app for software projects (ID=6, parent: 4)
  - Plan moving out of house (ID=7, parent: 4)
  - Prepare house for sale (ID=10, parent: 4)
"""
inst_no_context = f"""
Given these categories:
{cats}

**Confidence Scoring Guidelines**:

- **0.9-1.0 (Very High)**:
  * Direct keyword match in first 3 words ("add to grocery list", "grocery item")
  * Exact category name mentioned ("taxes", "todo")
  * Example: "Add milk to list" → "grocery shopping list" = 0.95

- **0.7-0.9 (High)**:
  * Clear semantic match with category-specific terms
  * Common synonyms or related actions ("buy food", "shopping", "get groceries")
  * Specific items that clearly belong to one category ("get steaks", "buy milk")
  * Example: "Get some steaks" → "grocery shopping list" = 0.75

- **0.5-0.7 (Moderate)**:
  * Implied or inferred intent requiring interpretation
  * Generic verbs that could apply to multiple categories ("add", "note", "remember")
  * Match requires context or domain knowledge
  * Example: "Remember to check prices" → "grocery shopping list" = 0.6

- **Below 0.5 (Low)**:
  * Weak match, highly speculative
  * Don't return these unless no better options exist

**Additional Rules**:
- Matches in first 3 words: +0.1 to confidence
- Ignore filler words: "um", "uh", "like", "you know"
- Always prefer the most SPECIFIC category (child over parent when both match)

Review the following Voice to Text transcript and pick the best category matches.

CRITICAL INSTRUCTIONS:
1. Evaluate and score ALL categories in the list first - do not stop early
2. After scoring all categories, rank them by confidence score
3. Return the top 2 highest-scoring matches
4. Only include matches with confidence >= 0.5
5. If fewer than 2 matches score >= 0.5, return only those above 0.5

"""
return_spec = """
Return EXACTLY 2 matches in this JSON array format (no other text):
[
  {"category_ID": <integer>, "category_name": "<string>", "confidence": <float>},
  {"category_ID": <integer>, "category_name": "<string>", "confidence": <float>}
]
"""

context_weighting = """
**CONTEXT WEIGHTING - CRITICAL**:
The previous draft created this context. Users typically stay on the same topic
for multiple consecutive drafts. Give STRONG preference to categories related to
the context below. Only choose a different category if the new transcript has
CLEAR, EXPLICIT keywords that strongly indicate a topic change.

Context weight: +0.2 to confidence for categories related to the context below.

"""
def form_prompt(draft_text, context=None):
    text = f"{inst_no_context}"
    text += return_spec
    if context:
        text += context_weighting
        text += "\n ---- Context begins -----\n"
        text += context
        text += "\n ---- Context ends -----\n\n"
    text += "\n ---- Transcript begins -----\n"
    text += draft_text
    text += "\n ---- Transcript ends -----\n"
    return text

async def prompt1(print_prompts=False, use_context=True):

    drafts = [
        "Get some steaks",
        "get moving boxes",
        "pack up kitchen",
        "email accountant",
        ]

    previous_context = None

    for i, draft in enumerate(drafts):
        # Build context from previous match if available
        context = None
        if previous_context and use_context:
            context = "Previous draft matched to: " \
                f"'{previous_context['category_name']}' " \
                f" (ID={previous_context['category_ID']}," \
                f" confidence={previous_context['confidence']:.2f})"

        text = form_prompt(draft, context=context)
        if print_prompts:
            print(text)

        res = await send_to_llm(text, ollama_url=url, model=model)

        # Parse the response
        matches = parse_llm_response(res)

        # Display results
        print(f"\n{'='*60}")
        print(f"Draft #{i+1}: '{draft}'")
        if context:
            print(f"Context: {context}")
        print(f"Matches found: {len(matches)}")
        for j, match in enumerate(matches, 1):
            print(f"  {j}. Category: {match['category_name']} (ID={match['category_ID']})")
            print(f"     Confidence: {match['confidence']:.2f}")

        if not matches:
            print("  (No valid matches found)")
        else:
            # Store first match as context for next iteration
            previous_context = matches[0]

        print(f"{'='*60}")

    
if __name__=="__main__":
    asyncio.run(prompt1())
    
