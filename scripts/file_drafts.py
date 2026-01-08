import asyncio
import fire
from palaver_shared.draft_events import Draft, DraftStartEvent, DraftEndEvent
from palaver_shared.text_events import TextEvent

from l1_server import Listener, L1Matcher
from loggers import setup_logging


class FileDraftRunner:

    def __init__(self):
        self.matcher = L1Matcher()
        self.listener = Listener(self.matcher)

    async def direct(self):
        draft = Draft(start_text="Freddy take this down")
        draft_start =  DraftStartEvent(draft=draft)
        await self.listener.on_draft_event(draft_start)
        text_1 = TextEvent(text="need a new task")
        await self.listener.on_text_event(text_1)
        draft_end =  DraftEndEvent(draft=draft)
        await self.listener.on_draft_event(draft_end)

    async def one_file(self, file_path: str):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        draft = Draft(start_text="Freddy take this down")
        draft_start = DraftStartEvent(draft=draft)
        await self.listener.on_draft_event(draft_start)

        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                text_event = TextEvent(text=line)
                await self.listener.on_text_event(text_event)

        draft_end = DraftEndEvent(draft=draft)
        await self.listener.on_draft_event(draft_end)
        
        
        
if __name__ == "__main__":
    setup_logging(more_loggers=[],
                  info_loggers=[],
                  debug_loggers=["AimSelect",])
    fire.Fire(FileDraftRunner())
