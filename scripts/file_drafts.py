#!/usr/bin/env python3
import asyncio
from dataclasses import asdict
from pprint import pprint
import fire

from palaver_shared.draft_events import Draft, DraftStartEvent, DraftEndEvent
from palaver_shared.text_events import TextEvent
from aim_select import AimToolbox, MetaAimTool
from kboard_tool import KBoardTool
from claude_tool import ClaudeCodeTool
from draft_tools import DraftMatcher

from loggers import setup_logging
OLLAMA_URL = "http://192.168.100.242:11434"
MODEL = "mistral:7b-instruct"


class FileDraftRunner:

    def __init__(self):
        self.toolbox = AimToolbox(OLLAMA_URL, MODEL,
                             [KBoardTool(), ClaudeCodeTool(), MetaAimTool()])
        self.draft_matcher = DraftMatcher(self.toolbox)

    async def one_file(self, file_path: str):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        draft = Draft(start_text="Freddy take this down")
        ctxt = self.draft_matcher.new_draft(draft)

        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                text_event = TextEvent(text=line)
                ctxt = self.draft_matcher.new_text_event(text_event)

        await self.draft_matcher.try_match()
        ctxt = self.draft_matcher.draft_context
        for match_res in ctxt.matches:
            pprint(match_res)
        
        
        
if __name__ == "__main__":
    setup_logging(more_loggers=[],
                  info_loggers=['TogetherAPI', 'AimSelect', 'DraftTools',],
                  debug_loggers=[])
    fire.Fire(FileDraftRunner())
