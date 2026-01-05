import asyncio
import re
import json
import yaml
from arborist import Arborist
from gibbon.llm_ops.find_root import send_to_llm



main_template = """
**CATEGORIZATION**:
**AVAILABLE CATEGORIES** (exactly {num_topics} — use ONLY these):
Each category has a fixed "key" that you MUST use for "category_name".

{topics_yaml}

**CRITICAL RULES FOR category_name**:
- ALWAYS use the exact "key" field (e.g., "new_topic", "record_essay", "edit_essay")
- NEVER use the path, description, or any invented version like "create_new_topic"
- NEVER snake_case the description — copy the "key" exactly

Compare the "transcription" below with the categories above. Calculate
a "match score" for each category. Score all the categories,
don't just stop when one gets a high score.

**SCORING INSTRUCTIONS**:

1. For each category, score how well it matches the first few words of the transcript
2. If keywords are provided, use them as HINTS, but semantic matches are equally valid
   - Keywords help with common phrasings
   - LLM should also match synonyms, related terms, semantic meaning
3. Consider the full path for context
4. Create a ranking of results by category from high score to low.
5. Return ranked categories up to a total of five, in the specified output format. 

**CRITICAL INSTRUCTION!**
DO NOT invent categories! Return results only for categories actually found in the above list


**Scoring Guidelines**:
- 0.9-1.0: Direct keyword match OR strong semantic match
- 0.7-0.9: Related terms with clear semantic relation
- 0.5-0.7: Indirect match or inferred intent
- < 0.5: No useful match or different intent

**CRITICAL - BE STRICT**:
- Different actions (create vs edit vs select) should score LOW unless both match
- Generic word overlap alone is not enough for high scores
- Example: "Create topic" matches "Create a new topic" (0.95+), NOT "Edit essay" (0.2)

**SPECIFICITY MATTERS**:
- PRIORITIZE specific/unique words over generic ones
- "copper production" → "copper" is SPECIFIC (discriminator), "production" is GENERIC (common)
- Match on SPECIFIC words first: copper, iron, steel, coal, plastic, oil
- Generic words (production, tools, resources, ore) are weak signals alone
- Generic action words (create, edit, select, record) must match the category's action
- If transcript has BOTH specific and generic words, match primarily on the specific word
- Example: "copper production" should match "Copper ore mining" (0.9+), NOT "Plastics production" (0.3)

**Path Matching**:
- Check if ANY element in the path matches transcript words
- Path "root2 -> splits -> resource_grid -> metals -> mining -> copper" matches "copper" strongly
- If keywords exist, they are suggestions, not requirements
- "Iron mine" should match subjects with "iron" in path/description at 0.95+
- "Dig for metal ore" should ALSO match mining subjects at 0.8+ (semantic understanding)
- Transcription variations are OK: "mind" → "mine", "iran" → "iron"

First, silently score all {num_topics} categories.
Then, apply the ≥0.50 rule plus best-match exception.
Then, output only the qualifying ones in the exact JSON format above.

----- Transcription begins ------
{draft}
----- Transcription ends ------

"""


context_template= """
**CONTEXT WEIGHTING - CRITICAL**:
The previous draft created this context. Users typically stay on the same category
for multiple consecutive drafts. Give preference to categories related to
the context below. Only choose a different category if the new transcript has
CLEAR, EXPLICIT keywords that strongly indicate a category change.

Context weight: +0.15 to confidence for categories related to the context below.

IMPORTANT: Context preference means choosing MORE SPECIFIC children WITHIN the context branch.
If the context is "metals", prefer "mining" or "refining" children, not staying at "metals".
The context bonus should help you choose the right branch, then GO DEEPER to the leaf.

---- Context begins -----
{context}
---- Context ends -----

"""
output_spec1 = """
**OUTPUT FORMAT - CRITICAL**:
Return all matches up to a total of 5 as JSON array. No extra text, no markdown.
Should look like this but with actual category names and descriptions:
[
  {
    "category_name": "some_category_key",
    "category_description": "This category is really close",
    "confidence": 0.91
  },
  {
    "category_name": "second_category_key",
    "category_description": "This category is pretty close",
    "confidence": 0.70
  },
  {
    "category_name": "another_category_key",
    "category_description": "Another category ",
    "confidence": 0.60
  }
  {
    "category_name": "fifth_category_key",
    "category_description": "not much like transcript",
    "confidence": 0.2
  },
  {
    "category_name": "fourth_category_key",
    "category_description": "no match at all",
    "confidence": 0.0
  },
]


Fields:
- "category_name": exact key from the category list
- "category_description": exact description from the category
- "confidence": float 0.0 to 1.0
"""

output_spec = """
**OUTPUT FORMAT - MUST FOLLOW EXACTLY**:
- Return ONLY a valid JSON array. No outer object, no "categories" wrapper.
- No extra text, no explanations, no markdown, no trailing commas.
- Array must contain 1 to 3 objects (since there are only 3 categories).
- Each object must have exactly these three fields:
  {
    "category_name": "<exact key from the list above, e.g. record_essay>",
    "category_description": "<exact description from the list above>",
    "confidence": <float between 0.0 and 1.0>
  }
- Sort by confidence descending.

**GOOD EXAMPLE (multiple matches)**:
[
  {
    "category_name": "record_essay",
    "category_description": "Record a topic essay",
    "confidence": 0.95
  },
  {
    "category_name": "edit_essay",
    "category_description": "Edit existing essay",
    "confidence": 0.75
  }
]

**GOOD EXAMPLE (single match)**:
[
  {
    "category_name": "record_essay",
    "category_description": "Record a topic essay",
    "confidence": 1.0
  }
]

**BAD EXAMPLES (do not do these)**:
{"categories": [ ... ]}   ← no outer object
[ { "path": "..." } ]     ← wrong field names
"""

output_spec_tool_call = """
**FINAL OUTPUT RULE**:
You MUST call the tool submit_category_matches with the ranked results.
Do not output JSON directly — use the tool.

Score all categories silently.
Include:
- All categories with confidence ≥ 0.50
- Plus the highest one even if below 0.50
Sort by confidence descending.
Use exact key and description from the list.
"""
class TopicsOnly:

    @staticmethod
    def make_prompt(draft, yaml_path, context=None, keywords_file=None, use_tool_calling=True):
        arbor = Arborist(yaml_path)
        topic_paths = arbor.get_topic_paths(keywords_file=keywords_file)
        topics_yaml = yaml.dump({'subjects': topic_paths}, sort_keys=False)
        prompt = main_template.format(draft=draft, topics_yaml=topics_yaml, num_topics=len(topic_paths))
        if context:
            prompt += context_part.format(context=context)
        # Use tool calling or direct JSON based on flag
        prompt += output_spec_tool_call if use_tool_calling else output_spec
        return prompt

    @staticmethod
    def parse_llm_response(response)  -> list[dict]:
        """
        Parse LLM response to extract category matches.

        Handles various formats:
        - Single JSON object: {...}
        - JSON array: [{...}, {...}]
        - Markdown code blocks: ```json ... ```
        - Extra text before/after JSON

        Returns:
            List of match dicts with keys: category_name, category_description, confidence
            Empty list if parsing fails
        """
        # Check for native tool calling FIRST (highest priority)
        if hasattr(response, 'message') and hasattr(response.message, 'tool_calls') and response.message.tool_calls:
            tool_call = response.message.tool_calls[0]
            if tool_call.function.name == "submit_category_matches":
                # Arguments might be a dict or a JSON string
                args = tool_call.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                matches = args["matches"]
                return matches

        # Extract content string from response object for fallback parsing
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            content = response.message.content
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)

        print("\n\nTool calling not working, using fallback parser...\n\n")

        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)

        # Try to find JSON arrays in the content
        # Use a more robust approach: find all potential JSON arrays and try each
        # Starting from the end (cleaner results typically come last)
        array_matches = re.findall(r'\[\s*\{[^\[\]]*\}\s*(?:,\s*\{[^\[\]]*\}\s*)*\]', content, re.DOTALL)

        # Try parsing from the last match first (most likely to be clean output)
        parsed = None
        for json_str in reversed(array_matches):
            try:
                parsed = json.loads(json_str)
                # If it parses and is a list, use it
                if isinstance(parsed, list):
                    break
            except json.JSONDecodeError:
                continue

        # Fallback to the old greedy approach if the new one didn't work
        if parsed is None:
            json_match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)
            if not json_match:
                print(f"Warning: No JSON found in response: {content[:100]}")
                return []
            json_str = json_match.group(1)
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON: {e}")
                print(f"Content: {json_str[:200]}")
                return []

        # Check if parsed is a tool call format: [{"name": "...", "arguments": {"matches": [...]}}]
        if isinstance(parsed, list) and len(parsed) > 0:
            first_item = parsed[0]
            if isinstance(first_item, dict) and "name" in first_item and "arguments" in first_item:
                if first_item.get("name") == "submit_category_matches":
                    args = first_item["arguments"]
                    if "matches" in args:
                        return args["matches"]

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

        #
        # could be
        type_1 = """ [{'categories': [{'category_name': 'new_topic', 'category_path': ['gibbon_taxonomy', 'new_topic'], 'category_description': 'Create a new topic', 'confidence': 0.8}, {'category_name': 'record_essay', 'category_path': ['gibbon_taxonomy', 'record_essay'], 'category_description': 'Create a topical essay', 'confidence': 0.7}]}]"""
        # could be
        type_2 = """[{'category_name': 'new_topic', 'category_path': ['gibbon_taxonomy', 'new_topic'], 'category_description': 'Create a new topic', 'confidence': 0.8}, {'category_name': 'record_essay', 'category_path': ['gibbon_taxonomy', 'record_essay'], 'category_description': 'Create a topical essay', 'confidence': 0.7}]"""
        # could be
        type_3 = """{'category_name': 'new_topic', 'category_path': ['gibbon_taxonomy', 'new_topic'], 'category_description': 'Create a new topic', 'confidence': 0.8}"""
        # could be
        type_4 = """[{'category_name': 'new_topic', 'category_path': ['gibbon_taxonomy', 'new_topic'], 'category_description': 'Create a new topic', 'confidence': 0.8}]"""

        
        if len(matches) == 0:
            return []
        top_list = []
        if isinstance(matches, dict):
            if "categories" in matches:
                top_list = matches['categories']
            elif "results" in matches:
                top_list = matches['results']
            else:
                top_list.append(matches)
        elif isinstance(matches, list):
            if len(matches[0]) == 1:
                top_dict = matches[0]
                if "categories" in top_dict:
                    top_list = top_dict['categories']
                elif "results" in top_dict:
                    top_list = top_dict['results']
            else:
                top_list = matches    
                
        for cat in top_list:
            if 'category_name' in cat:
                if 'confidence' not in cat:
                    cat['confidence'] = 0.0
                valid_matches.append(cat)
            else:
                print(f"Warning: Invalid cat format: {cat}")

        if len(valid_matches) == 0:
            print("\nno matches ----------------")
            from pprint import pprint
            pprint(matches)
            print("\n---------------------------")
            
        return valid_matches

        
        
