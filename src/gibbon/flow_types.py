from dataclasses import dataclass
from typing import Optional
from palaver_shared.draft_events import Draft

@dataclass
class MatchResult:
    intent_key: str
    confidence: float
    matched_phrase: str
    tool: 'AimTool'
    excerpt_pos: Optional[int] = -1
    
        
@dataclass
class AimDef:
    unique_name: str
    description: str
    examples: str
    preferred_words: Optional[list[str]] = None

    
@dataclass
class AimToolResponse:
    success: bool
    can_continue: bool # if draft not finished, can accept more text, if draft finished, can accept unmatched next draft
    reprocess_partial: Optional[tuple[int,int]] = None
    reprocess_tool: Optional['AimTool'] = None
    
class DraftContext:

    def __init__(self, draft:Draft):
        self.draft = draft
        self.text = ""
        self.search_pos = 0
        self.matches = []
        self.text_events = []
        self.last_check_text_index = -1

    def try_needed(self):
        if self.last_check_text_index < len(self.text_events) -1:
            tmp = self.text[self.search_pos:]
            if len(tmp.split()) < 3:
                return False
            return True
        return False

class AimTool:

    def get_aim_defs(self):
        raise Exception('child must implement')

    async def note_match(self, draft_context:DraftContext, match_res:MatchResult) -> AimToolResponse:
        raise Exception('child must implement')
    


