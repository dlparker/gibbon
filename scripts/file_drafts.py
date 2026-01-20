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

    async def direct_bad(self):
        draft = Draft(start_text="Freddy take this down")
        ctxt = self.draft_matcher.new_draft(draft)
        for part in [
                "mumble mumble mumble",
                "In the house moving project,"
                " do this thing.",
                ]:
            text_event = TextEvent(text=part)
            ctxt = self.draft_matcher.new_text_event(text_event)
        await self.draft_matcher.try_match()
        ctxt = self.draft_matcher.draft_context
        assert len(ctxt.matches) == 1
        top_match = ctxt.matches[0]
        pprint(top_match)
        for part in [
                "mumble mumble mumble",
                "Name a new task Bob.",
                " Move stuff from garage to shed.",
                ]:
            text_event = TextEvent(text=part)
            ctxt = self.draft_matcher.new_text_event(text_event)
        await self.draft_matcher.try_match()
        ctxt = self.draft_matcher.finish_draft_context()
        assert len(ctxt.matches) == 2
        task_match = ctxt.matches[1]
        pprint(task_match)
        
    async def direct_good(self):
        draft = Draft(start_text="Freddy take this down")
        ctxt = self.draft_matcher.new_draft(draft)
        for part in [
                "mumble mumble mumble",
                "In the house moving project,"
                " do this thing.",
                ]:
            text_event = TextEvent(text=part)
            ctxt = self.draft_matcher.new_text_event(text_event)
        await self.draft_matcher.try_match()
        ctxt = self.draft_matcher.draft_context
        assert len(ctxt.matches) == 1
        top_match = ctxt.matches[0]
        pprint(top_match)
        for part in [
                "mumble mumble mumble",
                "Name a new task Bob.",
                " I'll add info to that now",
                ]:
            text_event = TextEvent(text=part)
            ctxt = self.draft_matcher.new_text_event(text_event)
        await self.draft_matcher.try_match()
        ctxt = self.draft_matcher.finish_draft_context()
        assert len(ctxt.matches) == 2
        task_match = ctxt.matches[1]
        pprint(task_match)

    async def direct_great(self):
        draft = Draft(start_text="Freddy take this down")
        ctxt = self.draft_matcher.new_draft(draft)
        for part in [
                "mumble mumble mumble",
                "In the house moving project,"
                " do this thing.",
                ]:
            text_event = TextEvent(text=part)
            ctxt = self.draft_matcher.new_text_event(text_event)
        await self.draft_matcher.try_match()
        ctxt = self.draft_matcher.draft_context
        assert len(ctxt.matches) == 1
        top_match = ctxt.matches[0]
        pprint(top_match)
        for part in [
                "mumble mumble mumble",
                "Make a new task. ",
                " I'll add info to that now.",
                " When it is finished name it Last Task",
                ]:
            text_event = TextEvent(text=part)
            ctxt = self.draft_matcher.new_text_event(text_event)
        await self.draft_matcher.try_match()
        ctxt = self.draft_matcher.finish_draft_context()
        assert len(ctxt.matches) == 2
        task_match = ctxt.matches[1]
        pprint(task_match)
        

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
                  info_loggers=['TogetherAPI', 'AimSelect',],
                  debug_loggers=[ 'DraftTools', ])
    fire.Fire(FileDraftRunner())
