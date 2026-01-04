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

Respond with EXACTLY a JSON array of objects. No extra text, no markdown.

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
for multiple consecutive drafts. Give preference to categories related to
the context below. Only choose a different category if the new transcript has
CLEAR, EXPLICIT keywords that strongly indicate a topic change.

Context weight: +0.15 to confidence for categories related to the context below.

IMPORTANT: Context preference means choosing MORE SPECIFIC children WITHIN the context branch.
If the context is "metals", prefer "mining" or "refining" children, not staying at "metals".
The context bonus should help you choose the right branch, then GO DEEPER to the leaf.

"""

p_template_yaml_cats_2 = """
Categories (hierarchical - child categories are more specific than parents):
{cats_yaml}

**PATH CONSTRUCTION - CRITICAL**:

The category_path array MUST be built by following the YAML hierarchy exactly:
1. Start with "roots"
2. Use the "name" field value when present (e.g., "book_background" not "root2")
3. If no "name" field, use the YAML key itself
4. Each subsequent element must be a direct child of the previous
5. All paths have meaning, not just the leaf nodes

Example from the YAML above:
- To reach "iron" under mining: ["roots", "book_background", "splits", "resource_grid", "metals", "mining", "iron"]
- NOT: ["roots", "book_background", "world_building", "splits", ...] ← world_building is not on this path!

VALIDATION: Trace your path element-by-element through the YAML to verify each step.

Compare each category with the transcript provide below. For each category,
compute a score of how well the first 10 words of the transcript match
the current category. In these  example category paths ending in the same leaf:

["roots", "book_background", "splits", "resource_grid", "metals",]
["roots", "book_background", "splits", "resource_grid", "metals", "mining",]
["roots", "book_background", "splits", "resource_grid", "metals", "mining", "copper"]

The transcript "Iron mining" would have a high score for levels "metals" and "mining", but a lower score for
the leaf "copper".

Compile a list of confidence scores for each level and leaf of the entire category tree
and return it according to the output format specification. Do not quit scoring when a high
score is found.

**Confidence Scoring Guidelines**:

- **0.9-1.0 (Very High)**:
  * Direct keyword match in first 3 to 5 words ("add to grocery list", "grocery item")
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


CRITICAL INSTRUCTIONS:
1. Evaluate and score ALL categories in the list - do not stop early
2. After scoring all categories, rank them by confidence score
3. Construct a list of all categories with scores according to specified output format

"""
p_template_yaml_cats_1 = """
Categories (hierarchical - child categories are more specific than parents):
Given these categories:
{cats_yaml}
Review the following Voice to Text transcript and pick the best category matches.

First decide: task_board vs book_background.
Then within the chosen branch, decide next level, and so on.

**PATH CONSTRUCTION - CRITICAL**:

The category_path array MUST be built by following the YAML hierarchy exactly:
1. Start with "roots"
2. Use the "name" field value when present (e.g., "book_background" not "root2")
3. If no "name" field, use the YAML key itself
4. Each subsequent element must be a direct child of the previous
5. The path ends at the leaf node you select

Example from the YAML above:
- To reach "iron" under mining: ["roots", "book_background", "splits", "resource_grid", "metals", "mining", "iron"]
- NOT: ["roots", "book_background", "world_building", "splits", ...] ← world_building is not on this path!

VALIDATION: Trace your path element-by-element through the YAML to verify each step.

**REASONING PROCESS - FOLLOW THIS EXACTLY**:

Think step-by-step down the hierarchy:

1. **MANDATORY SPECIFICITY RULE**: Always select the DEEPEST, MOST SPECIFIC leaf node that matches.
   Never stop at a parent category if any of its children match the transcript.
   This is a hard requirement - violating it is an error.

2. Look at the transcript and decide which TOP-LEVEL root it belongs to:
   - "task_board" → if about projects, tasks, tools, selection, tracking
   - "book_background" → if about world-building, resources, technology, production, transport

3. Once you choose a root, ONLY consider categories inside that branch from now on.
   Do NOT go back or consider the other root.

4. Within the chosen branch, traverse to the DEEPEST matching child:
   - At each level, examine the TRANSCRIPT for keywords that match ANY child category
   - If a child category name or key word appears in the transcript, you MUST select that child
   - Example: "Iron Mine" contains "Iron" → must select the "iron" child under "mining", not stop at "mining"
   - Keep going deeper until you reach a leaf node with no children
   - A leaf node is one that has no "children:" field in the YAML

5. After selecting the deepest match, evaluate confidence using the scoring guidelines below.

   REMEMBER: The goal is to reach a LEAF NODE (no children). Before finalizing your selection:
   - Check: does this category have children in the YAML?
   - If YES: scan the transcript for keywords matching any child
   - If a child matches: you MUST select that child instead

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
- CRITICAL: If you stop at a parent when a child matches, this is an ERROR
- Leaf nodes are preferred: if choosing a parent, confidence should be lower unless children clearly don't match


CRITICAL INSTRUCTIONS:
1. Evaluate and score ALL categories in the list first - do not stop early
2. After scoring all categories, rank them by confidence score
3. Return the top 2 highest-scoring matches
4. Only include matches with confidence >= 0.5
5. If fewer than 2 matches score >= 0.5, return only those above 0.5

**FINAL VALIDATION CHECKLIST - MANDATORY**:

Before outputting your answer, verify EACH selected category:

□ Is this category a LEAF NODE (has no "children:" in the YAML)?
□ If NOT a leaf: Did I check all its children for keyword matches in the transcript?
□ Does the transcript contain any word that matches a child category name?
  - If YES: I MUST select that child instead
  - Example: "Iron" in transcript + "iron:" child exists → select "iron", not parent
□ Is my category_path valid (traced element-by-element through the YAML)?

If any check fails, revise your selection to go deeper.

"""

p_template_flat_yaml_1 =  """
Categories (hierarchical - child categories are more specific than parents):

Review the following Voice to Text transcript and pick the best category matches.

Categories (flat list of full paths):
{flat_yaml}

**PATH CONSTRUCTION - CRITICAL**:

Each category in the flat list above includes its full path. When you select a match:
1. Use the EXACT path shown in the category entry
2. The path must be copied precisely - do not modify or truncate it
3. Verify the path matches the description and category name

**REASONING PROCESS**:
1. **MANDATORY SPECIFICITY RULE**: Always select the DEEPEST, MOST SPECIFIC path that matches.
   Prefer longer paths (more specific) over shorter paths (more general).
2. Scan the transcript for keywords/semantic matches.
3. Score EVERY path based on how well the transcript matches its full path + description.
4. After scoring, rank by specificity (depth) and confidence.
5. Apply confidence guidelines.

Additional Rules:
- Only match if the transcript aligns with the ENTIRE path context (e.g., ignore 'iron' in transport if path is mining-specific).
- Longer paths are more specific and should be preferred when both match.

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
- CRITICAL: If you stop at a parent when a child matches, this is an ERROR
- Leaf nodes are preferred: if choosing a parent, confidence should be lower unless children clearly don't match


CRITICAL INSTRUCTIONS:
1. Evaluate and score ALL categories in the list first - do not stop early
2. After scoring all categories, rank them by confidence score
3. Return the top 2 highest-scoring matches
4. Only include matches with confidence >= 0.5
5. If fewer than 2 matches score >= 0.5, return only those above 0.5

**FINAL VALIDATION CHECKLIST - MANDATORY**:

Before outputting your answer, verify EACH selected category:

□ Is this category a LEAF NODE (has no "children:" in the YAML)?
□ If NOT a leaf: Did I check all its children for keyword matches in the transcript?
□ Does the transcript contain any word that matches a child category name?
  - If YES: I MUST select that child instead
  - Example: "Iron" in transcript + "iron:" child exists → select "iron", not parent
□ Is my category_path valid (traced element-by-element through the YAML)?

If any check fails, revise your selection to go deeper.

"""

def make_style_yaml_cats_1(draft, yaml_path):
    template = p_template_yaml_cats_1
    with open(yaml_path) as f:
        cats_yaml = f.read()
    text = template.format(draft=draft, cats_yaml=cats_yaml)
    return text

def make_style_yaml_cats_2(draft, yaml_path):
    template = p_template_yaml_cats_2
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

def make_style_topics_only(draft, yaml_path, keywords_file=None):
    """Only score nodes marked as topics."""
    arbor = Arborist(yaml_path)
    topic_paths = arbor.get_topic_paths(keywords_file=keywords_file)

    topics_yaml = yaml.dump({'topics': topic_paths}, sort_keys=False)

    template = """
**TOPIC-BASED CATEGORIZATION**:

These are the actual topics in the hierarchy (structural nodes removed):

{topics_yaml}

**SCORING INSTRUCTIONS**:

1. For each topic, score how well it matches the transcript
2. If keywords are provided, use them as HINTS, but semantic matches are equally valid
   - Keywords help with common phrasings
   - LLM should also match synonyms, related terms, semantic meaning
3. Consider the full path for context
4. Return top 5 topics with confidence >= 0.5

**Scoring Guidelines**:
- 0.9-1.0: Direct keyword match OR strong semantic match
- 0.7-0.9: Related terms or clear semantic relation
- 0.5-0.7: Indirect match or inferred intent
- < 0.5: Don't return

**CRITICAL - SPECIFICITY MATTERS**:
- PRIORITIZE specific/unique words over generic ones
- "copper production" → "copper" is SPECIFIC (discriminator), "production" is GENERIC (common)
- Match on SPECIFIC words first: copper, iron, steel, coal, plastic, oil
- Generic words (production, tools, resources, ore) are weak signals alone
- If transcript has BOTH specific and generic words, match primarily on the specific word
- Example: "copper production" should match "Copper ore mining" (0.9+), NOT "Plastics production" (0.3)

**Path Matching**:
- Check if ANY element in the path matches transcript words
- Path "root2 -> splits -> resource_grid -> metals -> mining -> copper" matches "copper" strongly
- If keywords exist, they are suggestions, not requirements
- "Iron mine" should match topics with "iron" in path/description at 0.95+
- "Dig for metal ore" should ALSO match mining topics at 0.8+ (semantic understanding)
- Transcription variations are OK: "mind" → "mine", "iran" → "iron"

**OUTPUT**: Return top 5 matches as JSON array (same format as before)
"""

    return template.format(topics_yaml=topics_yaml)

def make_prompt(draft, yaml_path, style, context=None, keywords_file=None):
    if style == "yaml_cats_1":
        text = make_style_yaml_cats_1(draft, yaml_path)
    elif style == "yaml_cats_2":
        text = make_style_yaml_cats_2(draft, yaml_path)
    elif style == "flat_yaml_1":
        text = make_style_flat_yaml_1(draft, yaml_path)
    elif style == "topics_only":
        text = make_style_topics_only(draft, yaml_path, keywords_file=keywords_file)
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
