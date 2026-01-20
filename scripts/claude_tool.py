from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field
import jsonlines

from palaver_shared.draft_events import Draft
from gibbon.flow_types import AimTool, AimDef, AimToolResponse, MatchResult, DraftContext

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
                   '"Create story to for code for palaver project", "Create coding tasks based on story 10", "How can I get this property from a GNUCash file"',
                   )
            ]
    
    async def note_match(self, draft_context:DraftContext, match_res:MatchResult):
        pass

