import asyncio
import time
from pprint import pprint
import logging
from palaver_client.api import PalaverEventListener
from palaver_client.rest_client import PalaverRestClient
from palaver_client.websocket_client import PalaverWebSocketClient
from palaver_shared.top_error import TopErrorHandler, TopLevelCallback
from palaver_shared.text_events import TextEvent
from palaver_shared.draft_events import DraftEvent, DraftStartEvent, DraftEndEvent, Draft
from palaver_shared.audio_events import AudioEvent, AudioChunkEvent
from aim_select import AimLevel, KBoardTool, ClaudeCodeTool, MetaAimTool

from loggers import setup_logging

palaver = "http://localhost:8000"

OLLAMA_URL = "http://192.168.100.242:11434"
MODEL = "mistral:7b-instruct"

logger = logging.getLogger('L1Server')

class L1Matcher:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.level = AimLevel(OLLAMA_URL,
                              MODEL,
                              [KBoardTool(), ClaudeCodeTool(), MetaAimTool()])
        
    async def try_match(self, texts:list[TextEvent], draft:Draft):
        await self.level.try_match(texts, draft)
    
class Listener(PalaverEventListener):

    def __init__(self, l1_matcher):
        super().__init__()
        self.l1_matcher = l1_matcher
        self.current_draft = None
        self.matched_level = None
        self.in_draft_text_events = []
        self.done_drafts = []
        self.last_speech_time = None
        self.last_try_time = None
        
    async def on_draft_event(self, event:DraftEvent):
        if isinstance(event, DraftStartEvent):
            if self.current_draft:
                logger.error("out of order arrival of new draft, didn't get end of last one")
            self.current_draft = event.draft
        if isinstance(event, DraftEndEvent):
            # If we never tried to match this draft, do it now before ending
            if not self.matched_level and len(self.in_draft_text_events) > 0:
                logger.info('Draft ended without match attempt, trying match on %s', self.in_draft_text_events)
                self.matched_level = await self.l1_matcher.try_match(self.in_draft_text_events,
                                                                     event.draft)
            self.done_drafts.append(event.draft)
            self.current_draft = None
            self.in_draft_text_events = []
            self.matched_level = None
            self.last_speech_time = None
            self.last_try_time = None

    async def on_text_event(self, event: TextEvent):
        if not self.current_draft:
            logger.debug(event)
            return
        self.in_draft_text_events.append(event)

    async def on_audio_event(self, event: AudioEvent):
        if isinstance(event, AudioChunkEvent):
            if event.in_speech:
                self.last_speech_time = event.timestamp
                
        if self.current_draft and len(self.in_draft_text_events) > 0:
            if self.last_speech_time and time.time() - self.last_speech_time >= 2.0 and not self.matched_level:
                if not self.last_try_time or self.last_speech_time > self.last_try_time:
                    logger.info('Trying match on %s', self.in_draft_text_events)
                    self.matched_level = await self.l1_matcher.try_match(self.in_draft_text_events,
                                                                         self.current_draft)
                    self.last_try_time = time.time()

async def main_loop():

    rest_cli = PalaverRestClient(palaver)
    async with PalaverRestClient("http://localhost:8000") as client:
        drafts = await client.fetch_all_drafts(limit=1)
        if drafts:
            logger.debug(drafts[0])
        else:
            logger.debug("No draft found")
        async with PalaverWebSocketClient(listener=Listener(L1Matcher()),
                                          palaver_url="http://localhost:8000") as ws_client:
            ws_client.start_listening()
            while True:
                await asyncio.sleep(1)

async def main():

    setup_logging(more_loggers=[logger,],
                  info_loggers=[logger.name,],
                  debug_loggers=["AimSelect",])
    background_error_dict = None
    class ErrorCallback(TopLevelCallback):
        async def on_error(self, error_dict: dict):
            nonlocal background_error_dict
            background_error_dict = error_dict
            
    handler = TopErrorHandler(top_level_callback=ErrorCallback())
    await handler.async_run(main_loop)

if __name__ == "__main__":
    asyncio.run(main())
    
