import asyncio
import time
from pprint import pprint
import logging
from pathlib import Path
from palaver_client.api import PalaverEventListener
from palaver_client.rest_client import PalaverRestClient
from palaver_client.websocket_client import PalaverWebSocketClient
from palaver_shared.top_error import TopErrorHandler, TopLevelCallback, get_error_handler
from palaver_shared.text_events import TextEvent
from palaver_shared.draft_events import DraftEvent, DraftStartEvent, DraftEndEvent, Draft
from palaver_shared.audio_events import AudioEvent, AudioChunkEvent
from aim_select import AimToolbox, MetaAimTool
from claude_tool import ClaudeCodeTool
from draft_tools import DraftMatcher
from kboard_tool import KBoardTool

from loggers import setup_logging

palaver = "http://localhost:8000"

OLLAMA_URL = "http://192.168.100.242:11434"
MODEL = "mistral:7b-instruct"

signals_dir = Path("../src/gibbon/signal_sounds")

logger = logging.getLogger('L1Server')

class Listener(PalaverEventListener):

    def __init__(self, palaver_url: str, toolbox: AimToolbox, voice_feedback=False, do_signals=False):
        super().__init__(event_types="no_audio")
        self.palaver_url = palaver_url
        self.toolbox = toolbox
        self.voice_feedback = voice_feedback
        self.do_signals = do_signals
        self.rest_client = None
        self.done_drafts = []
        self.draft_matcher = DraftMatcher(self.toolbox)
        
    async def on_draft_event(self, event:DraftEvent):
        if isinstance(event, DraftStartEvent):
            if self.draft_matcher.current_draft:
                logger.error("out of order arrival of new draft, didn't get end of last one")
            self.draft_matcher.new_draft(event.draft)
        if isinstance(event, DraftEndEvent):
            await self.try_match()
            if self.draft_matcher.current_draft:
                logger.info("finishing draft")
                ctxt = self.draft_matcher.finish_draft_context()
                if ctxt:
                    logger.info("draft result %s", ctxt)

    async def on_text_event(self, event: TextEvent):
        self.draft_matcher.new_text_event(event)
        logger.info(event)
        async def check_try(event):
            await asyncio.sleep(2)
            await self.try_match()
        get_error_handler().wrap_task(lambda event=event: check_try(event))

    async def try_match(self):
        if self.draft_matcher.draft_context and self.draft_matcher.draft_context.try_needed():
            logger.info('Trying match on %s', self.draft_matcher)
            if self.do_signals:
                if self.rest_client is None:
                    logger.info("making connection to %s", self.palaver_url)
                    self.rest_client = PalaverRestClient("http://localhost:8000")
                    await self.rest_client.connect()
                await self.rest_client.play_signal_sound('working')
            match_data = await self.draft_matcher.try_match()
            if not match_data:
                logger.info("no match on try_match call")
                if self.voice_feedback:
                    for_speech = f"No match found for draft "
                    logger.info("Sending speech text %s to palaver", for_speech)
                    await self.rest_client.text_to_speech(for_speech)
                return None
            for res_item in match_data:
                match_res = res_item['match_res']
                logger.info("matched %s", match_res.intent_key)
                if self.voice_feedback:
                    for_speech = f"Good match! key was, {" ".join(match_res.intent_key.split('_'))}"
                    logger.info("Sending speech text %s to palaver", for_speech)
                    await self.rest_client.text_to_speech(for_speech)

async def main_loop():

    async with PalaverRestClient("http://localhost:8000") as client:
        drafts = await client.fetch_all_drafts(limit=1)
        if drafts:
            logger.debug(drafts[0])
        else:
            logger.debug("No draft found")
        toolbox = AimToolbox(OLLAMA_URL, MODEL,
                             [KBoardTool(), ClaudeCodeTool(), MetaAimTool()])
        listener = Listener("http://localhost:8000", toolbox)
        async with PalaverWebSocketClient(listener=listener,
                                          palaver_url="http://localhost:8000") as ws_client:
            logger.info("Starting websocket listener")
            ws_client.start_listening()
            while True:
                await asyncio.sleep(1)

async def main():

    setup_logging(more_loggers=[logger,],
                  info_loggers=[logger.name,"AimSelect", "DraftTools",],
                  debug_loggers=[])
    background_error_dict = None
    class ErrorCallback(TopLevelCallback):
        async def on_error(self, error_dict: dict):
            nonlocal background_error_dict
            background_error_dict = error_dict
            
    handler = TopErrorHandler(top_level_callback=ErrorCallback())
    await handler.async_run(main_loop)

if __name__ == "__main__":
    asyncio.run(main())
    
