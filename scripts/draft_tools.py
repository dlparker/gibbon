from typing import Optional
import logging
import json
import os
import string
from pprint import pformat
from dataclasses import dataclass, asdict, field
from palaver_shared.text_events import TextEvent
from palaver_shared.draft_events import Draft
from gibbon.llm_ops.ollama_call import send_to_ollama
from gibbon.llm_ops.together_call import send_to_together_ai

from aim_select import AimToolbox
from flow_types import DraftContext

logger = logging.getLogger("DraftTools")
    
    
class DraftMatcher:
    
    def __init__(self, toolbox: AimToolbox):
        self.toolbox = toolbox
        self.last_match = None
        self.draft_context = None
        self.past_drafts = []

    @property
    def current_draft(self):
        if self.draft_context:
            return self.draft_context.draft
        return None
    
    @property
    def text_events(self):
        if self.draft_context:
            return self.draft_context.text_events
        return None
    
    def new_draft(self, draft:Draft):
        if self.draft_context and self.draft_context.draft != draft:
            self.finish_draft_context()
        self.draft_context = DraftContext(draft)
        return self.draft_context

    def new_text_event(self, text_event:TextEvent):
        if self.draft_context:
            self.draft_context.text_events.append(text_event)
            tbuff = self.draft_context.text
            if tbuff != '' and not tbuff[-1].isspace() and not text_event.text.isspace():
                tbuff += " "
            tbuff += text_event.text
            self.draft_context.text= tbuff
        return self.draft_context

    async def run_matcher(self, start=None, end=None):
        if start is None:
            start = self.draft_context.search_pos
        if end is None:
            end = len(self.draft_context.text)
        transcript = self.draft_context.text[start:end]
        tdict = {ord(c): None for c in string.punctuation}
        tmp = transcript.strip().translate(tdict)
        if tmp.strip() == "":
            return None,None
        ctxt = self.draft_context
        ctxt.last_check_text_index = len(ctxt.text_events) - 1 
        match_res = await self.toolbox.match_call(transcript)
        if not match_res:
            return None,None
        self.draft_context.matches.append(match_res)
        # find the position in the text of the latest_match. First find
        # it in the sent transcript so as to avoid matching previously
        # matched text
        if match_res.confidence < 0.5:
            return None,None
        pos = transcript.find(match_res.matched_phrase)
        # add the offset where we started the transcript
        pos += start
        match_res.excerpt_pos = pos
        pos += len(match_res.matched_phrase)
        # update the seach position to pass the matched text
        self.draft_context.search_pos = pos
        tool_res = await match_res.tool.note_match(ctxt, match_res)
        return match_res,tool_res
        
    async def try_match(self):
        if not self.draft_context:
            return None
        match_res,tool_res = await self.run_matcher()
        if not match_res or match_res.intent_key in ['', None]:
            return None
        result = [dict(match_res=match_res,
                       tool_res=tool_res),]
        while tool_res and tool_res.reprocess_partial:
            start, end = tool_res.reprocess_partial
            match_res,tool_res = await self.run_matcher(start, end)
            if match_res:
                result.append(dict(match_res=match_res,
                                   tool_res=tool_res))
        return result

    def finish_draft_context(self):
        res = self.draft_context
        if self.draft_context:
            # more to do later
            self.past_drafts.append(self.draft_context)
            self.draft_context = None
        return res
        
