"""
Run LLM-based intent matching with context management.

Loads data from database, calls LLM for matches, manages context.
"""
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Optional

from gibbon.store.models import (
    Draft,
    DraftStore,
    IBranch,
    IBranchStore,
    ContextEvent,
    ContextStore,
)
from gibbon.llm_ops.find_root import send_to_llm


# LLM Configuration
OLLAMA_URL = 'http://192.168.100.242:11434'
MODEL = 'llama3.1:8b-instruct-q4_K_M'


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
        print(f"Warning: No JSON found in response: {content[:100]}")
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
            print(f"Warning: Unexpected JSON type: {type(parsed)}")
            return []

        # Validate structure
        valid_matches = []
        for match in matches:
            if isinstance(match, dict) and 'category_ID' in match and 'category_name' in match:
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


async def run_intent_matching(
    store_dir: Path,
    use_context: bool = True,
    stack_name: str = "default"
):
    """
    Run intent matching on all drafts in the database.

    Args:
        store_dir: Directory containing the databases
        use_context: Whether to use context from previous matches
        stack_name: Name of context stack to use
    """
    # Initialize stores
    draft_store = DraftStore(store_dir)
    ibranch_store = IBranchStore(store_dir)
    context_store = ContextStore(store_dir)

    # Get formatted categories
    categories = format_categories_for_prompt(ibranch_store)

    # Get all drafts
    drafts = draft_store.get_all_drafts()

    print("=" * 60)
    print(f"Running intent matching on {len(drafts)} drafts")
    print(f"Context enabled: {use_context}")
    print(f"Stack: {stack_name}")
    print("=" * 60)

    # Track previous context
    previous_context = None
    if use_context:
        previous_context = context_store.get_current_context(stack_name)

    for i, draft in enumerate(drafts, 1):
        # Build context text from previous match
        context_text = None
        if use_context and previous_context:
            context_text = previous_context.match_result

        # Build prompt
        prompt = build_prompt(draft.full_text, categories, context_text)

        # Call LLM
        print(f"\nSending draft #{i} to LLM...")
        print(f"Prompt length: {len(prompt)} chars")
        response = await send_to_llm(prompt, ollama_url=OLLAMA_URL, model=MODEL)

        # Parse results
        matches = parse_llm_response(response)

        # Display results
        print(f"\n{'='*60}")
        print(f"Draft #{i}: '{draft.full_text}'")
        if context_text:
            print(f"Context: {context_text}")
        print(f"Matches found: {len(matches)}")

        for j, match in enumerate(matches, 1):
            print(f"  {j}. Category: {match['category_name']} (ID={match['category_ID']})")
            print(f"     Confidence: {match['confidence']:.2f}")

        if not matches:
            print("  (No valid matches found)")
        else:
            # Store first match as context event
            top_match = matches[0]
            match_result_text = (
                f"Previous draft matched to: '{top_match['category_name']}' "
                f"(ID={top_match['category_ID']}, confidence={top_match['confidence']:.2f})"
            )

            event = ContextEvent(
                intent_name=top_match['category_name'],
                draft_id=draft.draft_id,
                timestamp=time.time(),
                stack_name=stack_name,
                match_result=match_result_text
            )
            await context_store.add_event(event)
            previous_context = event

        print(f"{'='*60}")


async def main():
    """Run intent matching on test data."""
    store_dir = Path("/tmp/gibbon_test")

    if not store_dir.exists():
        print(f"Error: Test data not found at {store_dir}")
        print("Run generate_models.py first to create test data")
        return

    await run_intent_matching(store_dir, use_context=True)


if __name__ == "__main__":
    asyncio.run(main())
