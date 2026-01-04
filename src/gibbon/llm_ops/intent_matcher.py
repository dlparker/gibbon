"""
LLM-based intent matching utilities.

Functions for building prompts, calling LLM, and parsing responses.
"""
import json
import re
import logging
from typing import Optional

from gibbon.store.models import IBranchStore, IBranch
from gibbon.llm_ops.find_root import send_to_llm

logger = logging.getLogger("IntentMatcher")


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
    json_match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)

    if not json_match:
        logger.warning(f"No JSON found in response: {content[:100]}")
        return []

    json_str = json_match.group(1)

    try:
        parsed = json.loads(json_str)

        # Normalize to list format
        if isinstance(parsed, dict):
            matches = [parsed]
        elif isinstance(parsed, list):
            matches = parsed
        else:
            logger.warning(f"Unexpected JSON type: {type(parsed)}")
            return []

        # Validate structure
        valid_matches = []
        for match in matches:
            if isinstance(match, dict) and 'category_ID' in match and 'category_name' in match:
                if 'confidence' not in match:
                    match['confidence'] = 0.0
                valid_matches.append(match)
            else:
                logger.warning(f"Invalid match format: {match}")

        return valid_matches

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        logger.debug(f"Content: {json_str[:200]}")
        return []


def format_categories_for_prompt(ibranch_store: IBranchStore) -> str:
    """Format IBranches as text for LLM prompt.

    Only includes active category trees.
    """
    lines = ["Categories (hierarchical - child categories are more specific than parents):", ""]

    with ibranch_store.session() as session:
        roots = IBranch.get_roots(session, active_only=True)

        for root in roots:
            lines.append(f"- {root.name} (ID={root.id}, parent: none)")
            for child in root.children:
                lines.append(f"  - {child.name} (ID={child.id}, parent: {root.id})")

    return "\n".join(lines)


def build_prompt(draft_text: str, categories: str, context_text: Optional[str] = None) -> str:
    """
    Build the full prompt for LLM intent matching.

    Args:
        draft_text: The VTT transcription to match
        categories: Formatted category list
        context_text: Optional context from previous match

    Returns:
        Complete prompt string
    """
    prompt = f"""
Given these categories:
{categories}

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

    if context_text:
        prompt += """
**CONTEXT WEIGHTING - CRITICAL**:
The previous draft created this context. Users typically stay on the same topic
for multiple consecutive drafts. Give STRONG preference to categories related to
the context below. Only choose a different category if the new transcript has
CLEAR, EXPLICIT keywords that strongly indicate a topic change.

Context weight: +0.2 to confidence for categories related to the context below.

"""
        prompt += "\n ---- Context begins -----\n"
        prompt += context_text
        prompt += "\n ---- Context ends -----\n\n"

    prompt += """
Return EXACTLY 2 matches in this JSON array format (no other text):
[
  {"category_ID": <integer>, "category_name": "<string>", "confidence": <float>},
  {"category_ID": <integer>, "category_name": "<string>", "confidence": <float>}
]

"""
    prompt += "\n ---- Transcript begins -----\n"
    prompt += draft_text
    prompt += "\n ---- Transcript ends -----\n"

    return prompt
