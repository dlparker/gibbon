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
from flow_types import AimTool, AimDef, AimToolResponse, MatchResult, DraftContext


class MetaAimTool(AimTool):

    def __init__(self):
        self.draft = None
        self.ops = []
        
    def get_aim_defs(self):
        return [
            AimDef(unique_name = 'post_mark_drafts',
                   description =  'User is giving instructions about how to handle a previous draft or modify system behavior.',
                   examples = "'Change last draft to project management category', 'Reprocess last draft with this prefix, add code to palaver",
                   preferred_words = ['draft',]
                   )
            ]

    async def note_match(self, draft_context:DraftContext, match_res:MatchResult):
        pass
    

class AimToolbox:

    def __init__(self, url:str, model:str, tools:list[AimTool], call_mode:Optional[str]=None):
        self.url = url
        self.model = model
        self.tools = tools
        if call_mode:
            self.call_mode = call_mode
        else:
            self.call_mode = os.environ.get("LLM_CALL_CODE", "mistral-small:24B")

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
            if block['preferred_words']:
                intent_categories += f"   preferred words: {','.join(block['preferred_words'])}\n"
            intent_categories += "\n"

            
        system_prompt_hard = (
            "You are an intent classification system. You MUST use the classify_intent tool to respond.\n\n"
            "CRITICAL: You MUST call the classify_intent tool. Do NOT return JSON in text. Use the tool calling mechanism.\n\n"
            "MANDATORY FIELDS - ALL 4 ARE REQUIRED:\n"
            "1. intent_key - REQUIRED\n"
            "2. confidence - REQUIRED\n"
            "3. matched_phrase - REQUIRED\n\n"
            "CRITICAL RULES for the matched_phrase field:\n"
            "- The matched_phrase field is MANDATORY - you MUST always include it, even if it is None\n"
            "- Extract the FIRST MINIMAL phrase from the transcript that triggers the intent, then STOP\n"
            "- This matched_phrase is an extract from the transcript, a direct quote,\n"
            "- Do NOT include proper names, pronouns, or trailing words after the key phrase\n"
            "- Do NOT use the intent description for the key phrase, use the actual words in the transcript.\n"
            "- Strip articles (the, a, an, in, at, on) from the beginning and end\n"
            "- Exclude any words before or after the core phrase\n"
            "- Typical length: 2-8 words maximum\n"
            "- This is the EXACT text that will be removed before further processing\n\n"
            "Examples of CORRECT matched_phrase:\n"
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
            "REMEMBER: You MUST include matched_phrase in every response. Do not skip it.\n"
            "Example of result when no match found:\n"
            "1. intent_key - None\n"
            "2. confidence - 0.0\n"
            "3. matched_phrase - None\n\n"
            
        )
            
        system_prompt_soft = (
            "You are an intent classification system. You detecting whether the transcript has any\n"
            "phrases that match one of of the intent descriptions. No match is a possible correct asnwer.\n"
            "You MUST use the classify_intent tool to respond.\n\n"
            "CRITICAL: You MUST call the classify_intent tool. Do NOT return JSON in text. Use the tool calling mechanism.\n\n"
            "1. intent_key - identified match from intent categories\n"
            "2. confidence - score from 0.0 to 10.0\n"
            "3. matched_phrase - direct quote from transcript of matching identified intent\n\n"
            "REMEMBER: the matched_phrase must be a direct quote from the transcript\n"
            "Example of a valid result when no match found:\n"
            "1. intent_key - None\n"
            "2. confidence - 0.0\n"
            "3. matched_phrase - None\n\n"
        )
        prompt  = "Search this voice to text transcript of for key phrases indicating the speaker's intent\n"
        prompt += "\n"
        prompt += "------ TRANSCRIPT BEGINS -------"
        prompt += f"\n{transcript}\n"
        prompt += "------ TRANSCRIPT ENDS -------"
        prompt += "\n"
        prompt +=  "This is the list of intent categories that can be matched by the transcript:\n"
        prompt += "\n"
        prompt += f"\n{intent_categories}\n"
        prompt += "\n"
        prompt += "Your task is to whether the speaker clearly used language that expresses one of the available intents categries, or was talking about something else.\n"
        prompt += "\n"
        prompt += "1. Make a determination if there is a strong match between the first few words of the\n"
        prompt += "   transcript draft and one of the listed intent categories.\n"
        prompt += "2. If there was a strong match between one of the intent descriptions and the transcript,\n"
        prompt += "   then identify the intent_key, the matching phrase in the transcript and a score for how well they match.\n"
        prompt += "3. if there was no strong match, then set the intent_key to None, the matching phrase to None and the confidence to 0.0\n"
        prompt += "4. The examples in the intent category list to help you understand the intent, do not treat them as part of the transcript.\n"
        prompt += "5. If the intent category list item includes a list of 'preferred words', lower the score\n"
        prompt += "   by .25 for any potential match that does not contain one of these words.\n"
        prompt += "\n"
        prompt += "Prepare and return a result using the classify_intent tool:\n"
        prompt += "• the first reasonably matching phrase, it must be an exact quote from the transcript\n"
        prompt += "• the corresponding intent category key\n"
        prompt += "• a realistic confidence score\n"
        prompt += "\n"
        prompt += "\nCRITICAL: Use the classify_intent tool to return your analysis."

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
        if 'matched_phrase' not in res_dict:
            logger.warning("Response from llm missing required 'matched_phrase' field. Arguments: %s", res_dict)
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
                    break

        result = MatchResult(intent_key=res_dict['intent_key'],
                             confidence=res_dict.get('confidence', 0.0),
                             matched_phrase=res_dict['matched_phrase'],
                             tool=tool)
                        
        logger.info("returning %s",result)
        return result

    async def match_call(self, transcript:str) -> [None | MatchResult]:
        prompts = self.make_prompts(transcript)
        if self.call_mode == "ollama":
            logger.info("Calling ollama %s %s", self.url, self.model)
            response = await send_to_ollama(prompts, self.url, self.model, level_1_llm_tools)
        else:
            from gibbon.llm_ops.together_call import models
            model_name = self.call_mode
            if model_name not in models:
                raise Exception(f"invalid model_name {model_name}, not int {list(models.keys())}")
            logger.info("Calling together AI model %s", model_name)
            response = await send_to_together_ai(prompts, model_name, level_1_llm_tools)
        logger.debug("%s", pformat(response))
        if not hasattr(response, 'choices'):
            logger.warning("Reponse from llm not workable, no choices")
            return None
        if len(response.choices) != 1:
            logger.warning(f"Reponse from llm not workable, len of choices not 1, {len(choices)}")
            return None
        message = response.choices[0].message
        if not message:
            logger.warning("Reponse from llm not workable, no message")
            return None
        if message.tool_calls:
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
                return None
        else:
            # sometimes it follows the tool call orders but does not say it used the tool call. sigh
            # have seen this:
            """
            content = 
            [TOOL_CALLS]classify_intent[ARGS]{"intent_key": None, "confidence": 0.0, "matched_phrase": None}
            """
            # and this:
            """
            content = 
            classify_intent{"intent_key": "project_management", "confidence": 0.95, "matched_phrase": "house moving project"}
            """
            logger.warning("Reponse from llm not tool_calls is None, trying parse of content = \n%s",
                           message.content)
            res_dict = None
            start_pattern1 = "[TOOL_CALLS]classify_intent[ARGS]"
            start_pattern2 = 'classify_intent{"intent_key"'
            cstring = message.content.strip()
            cstring = cstring.replace("None", "null")
            if cstring.lower().startswith(start_pattern1.lower()):
                cstring = cstring[len(start_pattern1):]
                res_dict = json.loads(cstring)
            elif cstring.lower().startswith(start_pattern2.lower()):
                index = cstring.find("{")
                cstring = cstring[index:]
                res_dict = json.loads(cstring)
            if not res_dict:
                logger.warning("Reponse from llm not workable, tool_calls is None, content = \n%s",
                               message.content)
                return None

        # Check for required fields
        if 'matched_phrase' not in res_dict:
            logger.warning("Response from llm missing required 'matched_phrase' field. Arguments: %s", res_dict)
            return None
        if 'intent_key' not in res_dict:
            logger.warning("Response from llm missing required 'intent_key' field")
            return None

        target_tool = None
        for tool in self.tools:
            for aim_def in tool.get_aim_defs():
                if aim_def.unique_name == res_dict['intent_key']:
                    target_tool = tool
                    break

        result = MatchResult(intent_key=res_dict['intent_key'],
                             confidence=res_dict.get('confidence', 0.0),
                             matched_phrase=res_dict['matched_phrase'],
                             tool=target_tool)
                        
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
                    "matched_phrase": {
                        "type": "string",
                        "description": "The MINIMAL key phrase from the transcript that identifies this intent."
                    }
                },
                "required": ["intent_key", "confidence", "matched_phrase"]
            }
        }
    }
]

