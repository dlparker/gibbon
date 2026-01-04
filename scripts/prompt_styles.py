import asyncio
import re
import json
import yaml
from arborist import Arborist
from gibbon.llm_ops.find_root import send_to_llm

url = 'http://192.168.100.242:11434'
model = 'llama3.1:8b-instruct-q4_K_M'


return_spec = """
**OUTPUT FORMAT - CRITICAL**:

Respond with EXACTLY a JSON array of 0 to 2 objects. No extra text, no markdown.

Valid structures:
[]
[
  {
    "category_name": "some_category_key",
    "category_path": ["roots", "parent1", "parent2", "leaf"],
    "category_description": "The description text for this category",
    "confidence": 0.91
  }
]
[
  { ... first match ... },
  { ... second match ... }
]

Fields:
- "category_name": exact key from the category list
- "category_path": full array of keys from root to this category
- "category_description": exact description from the category
- "confidence": float 0.5 to 1.0
"""
context_weighting = """
**CONTEXT WEIGHTING - CRITICAL**:
The previous draft created this context. Users typically stay on the same topic
for multiple consecutive drafts. Give STRONG preference to categories related to
the context below. Only choose a different category if the new transcript has
CLEAR, EXPLICIT keywords that strongly indicate a topic change.

Context weight: +0.2 to confidence for categories related to the context below.

"""

p_template_yaml_cats1 = """
Categories (hierarchical - child categories are more specific than parents):
Given these categories:
{cats_yaml}
Review the following Voice to Text transcript and pick the best category matches.

First decide: task_board vs book_background.
Then within the chosen branch, decide next level, and so on.
**REASONING PROCESS - FOLLOW THIS EXACTLY**:

Think step-by-step down the hierarchy:

1. Look at the transcript and decide which TOP-LEVEL root it belongs to:
   - "task_board" → if about projects, tasks, tools, selection, tracking
   - "book_background" → if about world-building, resources, technology, production, transport

2. Once you choose a root, ONLY consider categories inside that branch from now on.
   Do NOT go back or consider the other root.

3. Within the chosen branch, repeat: pick the best child category based on specificity.
   Always prefer deeper, more specific matches when possible.

4. After traversing, evaluate confidence using the scoring guidelines below.

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


CRITICAL INSTRUCTIONS:
1. Evaluate and score ALL categories in the list first - do not stop early
2. After scoring all categories, rank them by confidence score
3. Return the top 2 highest-scoring matches
4. Only include matches with confidence >= 0.5
5. If fewer than 2 matches score >= 0.5, return only those above 0.5

"""

p_template_flat_yaml_1 =  """
Categories (hierarchical - child categories are more specific than parents):

Review the following Voice to Text transcript and pick the best category matches.

Categories (flat list of full paths):
{flat_yaml}

**REASONING PROCESS**:
1. Scan the transcript for keywords/semantic matches.
2. Score EVERY path based on how well the transcript matches its full path + description.
3. Prefer deeper paths (longer strings) for specificity.
4. Apply confidence guidelines.

Additional Rules:
- Only match if the transcript aligns with the ENTIRE path context (e.g., ignore 'iron' in transport if path is mining-specific).

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


CRITICAL INSTRUCTIONS:
1. Evaluate and score ALL categories in the list first - do not stop early
2. After scoring all categories, rank them by confidence score
3. Return the top 2 highest-scoring matches
4. Only include matches with confidence >= 0.5
5. If fewer than 2 matches score >= 0.5, return only those above 0.5

"""

def make_style_yaml_cats_1(draft, yaml_path):
    template = p_template_yaml_cats1
    with open(yaml_path) as f:
        cats_yaml = f.read()
    text = template.format(draft=draft, cats_yaml=cats_yaml)
    return text

def make_style_flat_yaml_1(draft, yaml_path):
    arbor = Arborist(yaml_path)
    flattened = arbor.flatten_paths()  # Or prune first, then flatten
    flat_yaml = yaml.dump({'categories': flattened}, sort_keys=False)  # Number them? Use list indices.

    template = p_template_flat_yaml_1
    text = template.format(draft=draft, flat_yaml=flat_yaml)
    return text

def make_prompt(draft, yaml_path, style, context=None):
    if style == "yaml_cats_1":
        text = make_style_yaml_cats_1(draft, yaml_path)
    elif style == "flat_yaml_1":
        text = make_style_flat_yaml_1(draft, yaml_path)
    else:
        raise Exception(f'no style {style}')
    text += return_spec
    if context:
        text += context_weighting
        text += "\n ---- Context begins -----\n"
        text += context
        text += "\n ---- Context ends -----\n\n"
    text += "\n ---- Transcript begins -----\n"
    text += draft
    text += "\n ---- Transcript ends -----\n"
    return text

def parse_llm_response(response) -> list[dict]:
    """
    Parse LLM response to extract category matches.

    Handles various formats:
    - Single JSON object: {...}
    - JSON array: [{...}, {...}]
    - Markdown code blocks: ```json ... ```
    - Extra text before/after JSON

    Returns:
        List of match dicts with keys: category_name, category_desciption, confidence
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
            if isinstance(match, dict) and 'category_name' in match:
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

async def do_draft_prompt(draft, yaml_path, style, use_context=None):
    context = None
    if use_context:
        context = "Previous draft matched to: " \
            f"'{use_context['category_name']}' " \
            f" (name={use_context['category_name']}," \
            f" confidence={use_context['confidence']:.2f})"
    prompt = make_prompt(draft, yaml_path, style=style, context=context)
    
    full_res = await send_to_llm(prompt, ollama_url=url, model=model)
    
    # Parse the response
    matches = parse_llm_response(full_res)
    
    return {
        'prompt': prompt,
        'context': use_context,
        'full_res': full_res,
        'matches': matches
    }
