from typing import Optional
import logging
from pprint import pformat
from dataclasses import dataclass, asdict
from palaver_shared.text_events import TextEvent
from palaver_shared.draft_events import Draft
from gibbon.llm_ops.single_call import send_to_llm

logger = logging.getLogger("AimSelect")

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
        
    def get_aim_def(self):
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

        
class KBoardTool(AimTool):

    def __init__(self):
        self.draft = None
        self.ops = []
        
    def get_aim_def(self):
        return AimDef(
            unique_name = 'project_management',
            description =  'User wants to interact with tasks or projects - create, search, modify, or query status of tasks/projects.',
            examples = '"Add task to fix door", "What tasks mention garage?", "Show project status", "Mark call as done"'
            )

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

class ClaudeCodeTool(AimTool):


    def __init__(self):
        self.draft = None
        self.ops = []
        
    def get_aim_def(self):
        return AimDef(
            unique_name = 'claude_code_request',
            description =  'User wants to do claude code planning - author design documents, run tech research, create development stories.',
            examples = '"Why is this code needed?", "How can I use this python library", ' \
            '"Create story to for code for palaver project", "Create coding tasks based on story 10", "How can I get this property from a GNUCash file"'
            )
    
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
        
    def get_aim_def(self):
        return AimDef(
            unique_name = 'post_mark_drafts',
            description =  'User is giving instructions about how to handle a previous draft or modify system behavior.',
            examples = "'Change last draft to project management category', 'Reprocess last draft with this prefix, add code to palaver"
            )

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
    

class AimLevel:

    def __init__(self, url:str, model:str, tools:list[AimTool], parent:Optional['AimLevel']=None):
        self.url = url
        self.model = model
        self.tools = tools
        self.parent = parent

    def get_categories(self):
        blocks = []
        for tool in self.tools:
            blocks.append(asdict(tool.get_aim_def()))
        return blocks
    
    async def try_match(self, text_events:list[TextEvent], draft:Draft):
        if draft.end_text:
            transcript = draft.full_text
        else:
            transcript = ' '
            for te in text_events:
                if not transcript[-1].isspace() and not te.text[0].isspace():
                    transcript += ' '
                transcript += te.text
        intent_categories = "Available Intent Categories:\n"
        for index, block in enumerate(self.get_categories()):
            intent_categories += f"{index+1} {block['unique_name']}\n"
            intent_categories += f"  {block['description']}\n"
            intent_categories += f"  {block['examples']}\n"
            intent_categories += "\n"

        
        system_prompt = "You are an intent classification system. You must analyze transcripts and classify them using the classify_intent tool. " \
                       "IMPORTANT: You must include all required fields in your response: intent_key, confidence, reasoning, and matched_excerpt. " \
                       "The matched_excerpt field should contain the specific portion of the transcript that led to your classification decision."

        prompt =  "Here are the available intent categories:\n"
        prompt += f"\n{intent_categories}\n"
        prompt += "Please classify this voice to text transcript to identify the high-level intent:\n"
        prompt += f"\n{transcript}\n"
        prompt += "\nUse the classify_intent tool to return your analysis."
        logger.debug("%s", prompt)
        if not logger.isEnabledFor(logging.DEBUG):
            logger.info("sending prompt to llm")
        response = await send_to_llm({'system': system_prompt, 'user':prompt}, self.url, self.model, level_1_llm_tools)
        logger.debug("%s", pformat(response))
        if not logger.isEnabledFor(logging.DEBUG):
            logger.info("%s", response)
        result = None
        if response.message:
            if response.message.tool_calls:
                for tc in response.message.tool_calls:
                    if tc.function.arguments:
                        result = tc.function.arguments
                        
        logger.info("returning %s",result)
        return result
            
        
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
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this intent was selected"
                    },
                    "matched_excerpt": {
                        "type": "string",
                        "description": "The portion of the transcript that was matched to make this classification. This will be stripped off before further processing by the attached tool."
                    }
                },
                "required": ["intent_key", "confidence", "reasoning", "matched_excerpt"]
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
                                                     'matched_excerpt': 'Voice to text code',
                                                     'reasoning': "The user is asking for voice to text code, which falls under the 'claude_code_request' category."}
                                          )
                                 )
                    ]
                    )
}
"""
