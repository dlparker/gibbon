from typing import Optional
import logging
import json
import os
from pprint import pformat
from dataclasses import dataclass, asdict, field
from palaver_shared.text_events import TextEvent
from palaver_shared.draft_events import Draft
from gibbon.llm_ops.ollama_call import send_to_ollama
from gibbon.llm_ops.together_call import send_to_together_ai

logger = logging.getLogger("AimSelect")


@dataclass
class MatchResult:
    intent_key: str
    confidence: float
    key_phrase: str
    tool: 'AimTool'
    excerpt_pos: Optional[int] = -1
    
        
@dataclass
class AimDef:
    unique_name: str
    description: str
    examples: str

    
@dataclass
class AimToolResponse:
    success: bool
    can_continue: bool # if draft not finished, can accept more text, if draft finished, can accept unmatched next draft
    reprocess_partial: Optional[tuple[int,int]] = None
    reprocess_tool: Optional['AimTool'] = None
    
class AimTool:

    def __init__(self, simple_target=None):
        self.simple_target = simple_target
        self.draft = None
        
    def get_aim_defs(self):
        raise Exception('child must implement')

    def start_draft(self, draft):
        self.draft = draft
        return AimToolResponse(success=True, can_continue=True)

    def end_draft(self, draft):
        if self.simple_target:
            success = self.simple_target(draft, self)
            self.draft = None
            return AimToolResponse(success=True, can_continue=False)
        
    def continuation_draft(self, draft):
        return AimToolResponse(success=False, can_continue=False)

        
class ClaudeCodeTool(AimTool):


    def __init__(self):
        self.draft = None
        self.ops = []
        self.current_level = 0
        
    def get_aim_defs(self):
        return [
            AimDef(unique_name = 'claude_code_request',
                   description =  'User wants to do claude code planning - author design documents, run tech research, create development stories.',
                   examples = '"Why is this code needed?", "How can I use this python library", ' \
                   '"Create story to for code for palaver project", "Create coding tasks based on story 10", "How can I get this property from a GNUCash file"'
                   )
            ]
    
    def start_draft(self, draft):
        self.draft = draft
        return AimToolResponse(success=True, can_continue=True)

    def end_draft(self, draft):
        self.ops.append(draft.full_text)
        return AimToolResponse(success=True, can_continue=True)

    def continuation_draft(self, draft):
        if not draft.end_text:
            self.draft = draft
            return AimToolResponse(success=False, can_continue=True)
        self.ops.append(draft.full_text)
        return AimToolResponse(success=True, can_continue=True)

class MetaAimTool(AimTool):

    def __init__(self):
        self.draft = None
        self.ops = []
        
    def get_aim_defs(self):
        return [
            AimDef(unique_name = 'post_mark_drafts',
                   description =  'User is giving instructions about how to handle a previous draft or modify system behavior.',
                   examples = "'Change last draft to project management category', 'Reprocess last draft with this prefix, add code to palaver"
                   )
            ]

    def start_draft(self, draft):
        self.draft = draft
        return AimToolResponse(success=True, can_continue=True)

    def end_draft(self, draft):
        self.ops.append(draft.full_text)
        return AimToolResponse(success=True, can_continue=True)

    def continuation_draft(self, draft):
        if not draft.end_text:
            self.draft = draft
            return AimToolResponse(success=False, can_continue=True)
        self.ops.append(draft.full_text)
        return AimToolResponse(success=True, can_continue=True)
    

class AimToolbox:

    def __init__(self, url:str, model:str, tools:list[AimTool], call_mode:Optional[str]=None):
        self.url = url
        self.model = model
        self.tools = tools
        if call_mode:
            self.call_mode = call_mode
        else:
            self.call_mode = os.environ.get("LLM_CALL_CODE", "ollama")

    def get_categories(self):
        blocks = []
        for tool in self.tools:
            for aim_def in tool.get_aim_defs():
                blocks.append(asdict(aim_def))
        return blocks

    def make_prompts(self, transcript, system_type="hard"):
        intent_categories = "Intent Categories List:\n"
        for index, block in enumerate(self.get_categories()):
            intent_categories += f"{index+1}. intent_key: {block['unique_name']}\n"
            intent_categories += f"   description: {block['description']}\n"
            intent_categories += f"   examples: {block['examples']}\n"
            intent_categories += "\n"

            
        system_prompt_hard = (
            "You are an intent classification system. You MUST use the classify_intent tool to respond.\n\n"
            "CRITICAL: You MUST call the classify_intent tool. Do NOT return JSON in text. Use the tool calling mechanism.\n\n"
            "MANDATORY FIELDS - ALL 4 ARE REQUIRED:\n"
            "1. intent_key - REQUIRED\n"
            "2. confidence - REQUIRED\n"
            "3. key_phrase - REQUIRED\n\n"
            "CRITICAL RULES for the key_phrase field:\n"
            "- The key_phrase field is MANDATORY - you MUST always include it, even if it is None\n"
            "- Extract the FIRST MINIMAL phrase from the transcript that triggers the intent, then STOP\n"
            "- This key_phrase is an extract from the transcript, a direct quote,\n"
            "- Do NOT include proper names, pronouns, or trailing words after the key phrase\n"
            "- Do NOT use the intent description for the key phrase, use the actual words in the transcript.\n"
            "- Strip articles (the, a, an, in, at, on) from the beginning and end\n"
            "- Exclude any words before or after the core phrase\n"
            "- Typical length: 2-8 words maximum\n"
            "- This is the EXACT text that will be removed before further processing\n\n"
            "Examples of CORRECT key_phrase:\n"
            "- Transcript: 'mumble mumble In the house moving project, do this thing'\n"
            "  ✓ CORRECT: 'house moving project'\n"
            "  ✗ WRONG: 'In the house moving project, do this thing'\n"
            "  ✗ WRONG: 'In the house moving project'\n\n"
            "- Transcript: 'Name a new task Bob. More information to follow about it.'\n"
            "  ✓ CORRECT: 'Name a new task' (STOP after the action phrase, exclude 'Bob')\n"
            "  ✗ WRONG: 'Name a new task Bob'\n"
            "  ✗ WRONG: 'Name a new task Bob. More information to follow about it.'\n\n"
            "- Transcript: 'Show me the project status for palaver'\n"
            "  ✓ CORRECT: 'project status' (core phrase only)\n"
            "  ✗ WRONG: 'Show me the project status for palaver'\n\n"
            "REMEMBER: You MUST include key_phrase in every response. Do not skip it.\n"
            "Example of result when no match found:\n"
            "1. intent_key - None\n"
            "2. confidence - 0.0\n"
            "3. key_phrase - None\n\n"
            
        )
            
        system_prompt_soft = (
            "You are an intent classification system. You detecting whether the transcript has any\n"
            "phrases that match one of of the intent descriptions. No match is a possible correct asnwer.\n"
            "You MUST use the classify_intent tool to respond.\n\n"
            "CRITICAL: You MUST call the classify_intent tool. Do NOT return JSON in text. Use the tool calling mechanism.\n\n"
            "1. intent_key - REQUIRED\n"
            "2. confidence - REQUIRED\n"
            "3. key_phrase - REQUIRED (DO NOT OMIT THIS FIELD)\n\n"
            "REMEMBER: the key_phrase must be a direct quote from the transcript\n"
            "Example of a valid result when no match found:\n"
            "1. intent_key - None\n"
            "2. confidence - 0.0\n"
            "3. key_phrase - None\n\n"
        )
        prompt  = "Search this transcript for key phrases indicating the speaker's intent\n"
        prompt += "\n"
        prompt += "------ TRANSCRIPT BEGINS -------"
        prompt += f"\n{transcript}\n"
        prompt += "------ TRANSCRIPT ENDS -------"
        prompt += "\n"
        prompt +=  "This is the list of intent categories that can be matched by the transcript\n"
        prompt += "\n"
        prompt += f"\n{intent_categories}\n"
        prompt += "\n"
        prompt += "Your task is to **detect** whether this transcript contains language that expresses one of the available intents categries.\n"
        prompt += "\n"
        prompt += "1. Examine the descriptions in the intent category list.\n"
        prompt += "2. Search the transcript to see if any phrase strongly correlates with any description.\n"
        prompt += "3. Use the examples in the intent category list to help you understand the intent.\n"
        prompt += "4. Do not use the examples in the intent category list in key_phrase reporting, only\n"
        prompt += "   use transcript words and phrases.\n"
        prompt += "\n"
        prompt += "Only report a match if you see a transcript phrase that reasonably belongs to one of the categories. \n"
        prompt += "If there is no good match that is a valid result.\n"
        prompt += "\n"
        prompt += "Return:\n"
        prompt += "• the first reasonably matching phrase, it must be an exact quote from the transcript\n"
        prompt += "• the corresponding intent category key\n"
        prompt += "• a realistic confidence score\n"
        prompt += "\n"
        prompt += "\nUse the classify_intent tool to return your analysis."

        if system_type == "hard":
            system_prompt = system_prompt_hard
        else:
            system_prompt = system_prompt_soft
        logger.debug("System Prompt:\n%s\n", system_prompt)
        logger.debug("User Prompt:\n%s\n", prompt)
        return {'system': system_prompt, 'user': prompt}

    async def ollama_match_call(self, transcript):
        prompts = self.make_prompts(transcript, "soft")
        logger.info("Calling ollama %s %s", self.url, self.model)
        response = await send_to_ollama(prompts, self.url, self.model, level_1_llm_tools)
        logger.debug("%s", pformat(response))
        result = None
        if not response.message:
            logger.warning("Reponse from llm not workable, no message")
            return
        if not response.message.tool_calls:
            logger.warning("Reponse from llm not workable, tool_calls is None")
            return
        tool_call = None
        for tc in response.message.tool_calls:
            if tc.function.name == "classify_intent":
                tool_call = tc
                break

        if tool_call is None:
            logger.warning("Reponse from llm not workable, classify_intent tool call not in result")
            return

        res_dict = tc.function.arguments
        if not res_dict:
            logger.warning("Reponse from llm not workable, classify_intent tool call missing arguments attribute")
            return

        # Check for required fields
        if 'key_phrase' not in res_dict:
            logger.warning("Response from llm missing required 'key_phrase' field. Arguments: %s", res_dict)
            return
        if 'intent_key' not in res_dict:
            logger.warning("Response from llm missing required 'intent_key' field")
            return

        target_def = None
        target_tool = None
        for tool in self.tools:
            for aim_def in tool.get_aim_defs():
                if aim_def.unique_name == res_dict['intent_key']:
                    target_tool = tool
                    target_def = aim_def

        result = MatchResult(intent_key=res_dict['intent_key'],
                             confidence=res_dict.get('confidence', 0.0),
                             key_phrase=res_dict['key_phrase'],
                             tool=tool)
                        
        logger.info("returning %s",result)
        return result

    async def tai_match_call(self, transcript):
        prompts = self.make_prompts(transcript)
        from gibbon.llm_ops.together_call import models
        model_name = self.call_mode
        if model_name not in models:
            raise Exception(f"invalid model_name {model_name}, not int {list(models.keys())}")
        logger.info("Calling together AI model %s", model_name)
        response = await send_to_together_ai(prompts, model_name, level_1_llm_tools)
        result = None
        if not hasattr(response, 'choices'):
            logger.warning("Reponse from llm not workable, no choices")
            return
        if len(response.choices) != 1:
            logger.warning(f"Reponse from llm not workable, len of choices not 1, {len(choices)}")
            return
        message = response.choices[0].message
        if not message:
            logger.warning("Reponse from llm not workable, no message")
            return
        if not message.tool_calls:
            logger.warning("Reponse from llm not workable, tool_calls is None, content = \n%s",
                           message.content)
            return
        tool_call = None
        for tc in message.tool_calls:
            if tc.function.name == "classify_intent":
                tool_call = tc
                break

        if tool_call is None:
            logger.warning("Reponse from llm not workable, classify_intent tool call not in result")
            return

        res_dict = json.loads(tc.function.arguments)
        if not res_dict:
            logger.warning("Reponse from llm not workable, classify_intent tool call missing arguments attribute")
            return

        # Check for required fields
        if 'key_phrase' not in res_dict:
            logger.warning("Response from llm missing required 'key_phrase' field. Arguments: %s", res_dict)
            return
        if 'intent_key' not in res_dict:
            logger.warning("Response from llm missing required 'intent_key' field")
            return

        target_def = None
        target_tool = None
        for tool in self.tools:
            for aim_def in tool.get_aim_defs():
                if aim_def.unique_name == res_dict['intent_key']:
                    target_tool = tool
                    target_def = aim_def

        result = MatchResult(intent_key=res_dict['intent_key'],
                             confidence=res_dict.get('confidence', 0.0),
                             key_phrase=res_dict['key_phrase'],
                             tool=tool)
                        
        logger.info("returning %s",result)
        return result

    async def match_call(self, transcript):
        if self.call_mode == "ollama":
            return await self.ollama_match_call(transcript)
        else:
            return await self.tai_match_call(transcript)
        
level_1_llm_tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_intent",
            "description": "Classify the user's high-level intent to route to appropriate handler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent_key": {
                        "type": "string",
                        "description": "The unique key identifying the intent category"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence score from 0.0 to 1.0"
                    },
                    "key_phrase": {
                        "type": "string",
                        "description": "The MINIMAL key phrase from the transcript that identifies this intent."
                    }
                },
                "required": ["intent_key", "confidence", "key_phrase"]
            }
        }
    }
]


"""
result_example = {
    model='mistral:7b-instruct'
    created_at='2026-01-09T18:20:43.982702469Z'
    done=True done_reason='stop'
    total_duration=1575226232
    load_duration=26363135
    prompt_eval_count=527
    prompt_eval_duration=42591777
    eval_count=77
    eval_duration=1481660420
    message=Message(role='assistant',
                    content='',
                    thinking=None,
                    images=None,
                    tool_name=None,
                    tool_calls=[
                        ToolCall(function=
                                 Function(name='classify_intent',
                                          arguments={'confidence': 0.95,
                                                     'intent_key': 'claude_code_request',
                                                     'key_phrase': 'Voice to text code',
                                          )
                                 )
                    ]
                    )
}
"""
