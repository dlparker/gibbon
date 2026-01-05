"""
Draft listener server - connects to palaver and processes new drafts.

Listens for new drafts from palaver via WebSocket, runs LLM intent matching,
and stores results in gibbon's database.
"""
import asyncio
import logging
import time
import json
import signal
from pathlib import Path
from typing import Optional

from gibbon.palaver_client import (
    PalaverWebSocketClient,
    PalaverRestClient,
    convert_palaver_draft,
)
from gibbon.store.models import (
    Draft,
    DraftStore,
    ContextEvent,
    ContextStore,
)
from gibbon.llm_ops.find_root import send_to_llm
from prompt_tools.topic_cats import TopicsOnly

# Configuration
PALAVER_URL = "http://localhost:8000"
GIBBON_STORE_DIR = Path("./data")
OLLAMA_URL = "http://192.168.100.242:11434"
#MODEL = "llama3.1:8b-instruct-q4_K_M"
MODEL = "mistral:7b-instruct"
CONTEXT_STACK = "palaver_live"
SYNC_LOOKBACK_HOURS = 5  # How many hours back to sync on startup (0 = disable sync)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DraftListener")

url = 'http://192.168.100.242:11434'
model = 'llama3.1:8b-instruct-q4_K_M'

class DraftListenerServer:
    """Server that listens for drafts from palaver and processes them."""

    def __init__(self,  palaver_url: str,  category_yaml: Path, store_dir: Path, context_stack:str):
        self.palaver_url = palaver_url
        self.category_yaml = category_yaml
        self.store_dir = Path(store_dir)
        self.context_stack = context_stack
        self.prompt_tool = TopicsOnly
        self.ollama_url = url
        self.model = model

        # Initialize stores
        self.draft_store = DraftStore(self.store_dir)
        self.context_store = ContextStore(self.store_dir)

        # WebSocket client
        self.ws_client = PalaverWebSocketClient(
            base_url=palaver_url,
            auto_reconnect=True,
            reconnect_delay=5.0
        )

        # Track last sync timestamp
        self.last_sync_timestamp: Optional[float] = None

        # Shutdown flag
        self.running = False
        self.listen_task: Optional[asyncio.Task] = None

    async def sync_missed_drafts(self, lookback_hours: float) -> int:
        """
        Sync any drafts that might have been missed while offline.

        Args:
            lookback_hours: How many hours back to look for drafts.
                           0 = disable sync, use last context timestamp only

        Returns:
            Number of drafts synced
        """
        if lookback_hours == 0:
            logger.info("Draft sync disabled (lookback_hours=0)")
            return 0

        logger.info("Syncing missed drafts from palaver...")

        # Calculate lookback timestamp
        lookback_seconds = lookback_hours * 3600
        lookback_timestamp = time.time() - lookback_seconds

        # Get last processed timestamp from context
        last_context = self.context_store.get_current_context(self.context_stack)
        if last_context:
            # Use the more recent of: last context or lookback limit
            since_timestamp = max(last_context.timestamp, lookback_timestamp)
            logger.info(f"Last processed draft at {last_context.timestamp}")
            logger.info(f"Lookback limit: {lookback_hours} hours ({lookback_timestamp})")
            logger.info(f"Using since_timestamp: {since_timestamp}")
        else:
            # No previous context, use lookback limit
            since_timestamp = lookback_timestamp
            logger.info(f"No previous context, fetching drafts from last {lookback_hours} hour(s)")

        # Fetch drafts from palaver
        async with PalaverRestClient(self.palaver_url) as client:
            drafts, total = await client.fetch_drafts_since(
                since_timestamp=since_timestamp,
                limit=1000,
                order="asc"  # Process oldest first
            )

        logger.info(f"Found {len(drafts)} drafts to sync")

        # Process each draft
        synced = 0
        for draft_dict in drafts:
            # Check if we already have this draft
            existing = self.draft_store.get_draft_by_uuid(draft_dict["draft_id"])
            if existing:
                logger.debug(f"Draft {draft_dict['draft_id']} already exists, skipping")
                continue

            # Convert and save draft
            draft = convert_palaver_draft(draft_dict)
            await self.draft_store.add_draft(draft)

            # Run intent matching
            await self.process_draft(draft)
            synced += 1

        logger.info(f"Synced {synced} new drafts")
        return synced

    async def process_draft(self, draft: Draft) -> None:
        """
        Process a draft: run LLM intent matching and store results.

        Args:
            draft: Draft to process
        """
        logger.info(f"Processing draft: '{draft.full_text[:50]}...'")
        
        # Get current context
        previous_context = self.context_store.get_current_context(self.context_stack)
        context_text = previous_context.match_result if previous_context else None
        prompt = self.prompt_tool.make_prompt(draft.full_text, self.category_yaml, context=context_text)
            

        # Parse the response
        # Call LLM
        logger.debug(f"Calling LLM (prompt length: {len(prompt)} chars)")
        full_res = await send_to_llm(prompt, ollama_url=url, model=model)
        matches = self.prompt_tool.parse_llm_response(full_res)

        async def process_result():
            
            if matches:
                top_match = matches[0]
                try:
                    logger.info(
                        f"Matched to: {top_match['category_name']} "
                        f"(ID={top_match['category_name']}, conf={top_match['confidence']:.2f})"
                    )
                except Exception as e:
                    breakpoint()
                    import traceback
                    traceback.print_exc()
                    from pprint import pprint
                    pprint(matches)
                    pprint(full_result.__dict__)
            
                # Create context event
                match_result_text = (
                    f"Previous draft matched to: '{top_match['category_name']}' "
                    f"(ID={top_match['category_name']}, confidence={top_match['confidence']:.2f})"
                )
            
                event = ContextEvent(
                    intent_name=top_match['category_name'],
                    draft_id=draft.draft_id,
                    timestamp=time.time(),
                    stack_name=self.context_stack,
                    match_result=match_result_text
                )
                await self.context_store.add_event(event)
            
                logger.info(f"Stored context event for draft {draft.draft_id}")
            else:
                logger.warning(f"No matches found for draft {draft.draft_id}")
                            
            await process_result()
        result = {
            'context': context_text,
            'matches': matches
        }
        llm_op_record = await self.draft_store.add_draft_llm_op(draft, prompt, json.dumps(result))
        logger.info(f"Stored llm_topic_op {llm_op_record.id} for draft {draft.draft_id}")
            
    prompt_text: str
    async def handle_draft_event(self, event_data: dict) -> None:
        """
        Handle a DraftEndEvent from palaver.

        Args:
            event_data: Event dictionary from WebSocket
        """
        event_class = event_data.get("event_class", "")
        logger.debug(f"Received event: {event_class}")

        # Only process DraftEndEvent
        if event_class != PalaverWebSocketClient.DRAFT_END_EVENT:
            return

        # Extract draft data
        draft_dict = event_data.get("draft", {})
        draft_id = draft_dict.get("draft_id")

        if not draft_id:
            logger.warning("Received DraftEndEvent without draft_id")
            return

        # Check if we already have this draft
        existing = self.draft_store.get_draft_by_uuid(draft_id)
        if existing:
            logger.info(f"Draft {draft_id} already processed, skipping")
            return

        # Convert palaver draft to gibbon draft
        from gibbon.palaver_client.draft_converter import convert_palaver_event_draft
        draft = convert_palaver_event_draft(draft_dict)

        # Save draft
        await self.draft_store.add_draft(draft)
        logger.info(f"Saved new draft: {draft.draft_id}")

        # Process draft (LLM matching)
        
        await self.process_draft(draft)

    async def start(self, sync_lookback_hours: float = 1.0) -> None:
        """
        Start the draft listener server.

        Args:
            sync_lookback_hours: How many hours back to sync on startup (0 = disable)
        """
        logger.info("="*60)
        logger.info("Starting Draft Listener Server")
        logger.info(f"Palaver URL: {self.palaver_url}")
        logger.info(f"Store directory: {self.store_dir}")
        logger.info(f"LLM: {self.model} at {self.ollama_url}")
        logger.info(f"Context stack: {self.context_stack}")
        logger.info(f"Sync lookback: {sync_lookback_hours} hour(s)")
        logger.info("="*60)

        self.running = True

        # Sync any missed drafts first
        try:
            await self.sync_missed_drafts(sync_lookback_hours)
        except Exception as e:
            logger.error(f"Error syncing missed drafts: {e}", exc_info=True)

        # Start listening for new drafts
        logger.info("Starting WebSocket listener...")
        self.listen_task = self.ws_client.start_listening(
            event_handler=self.handle_draft_event,
            event_types=[PalaverWebSocketClient.DRAFT_END_EVENT]
        )

        logger.info("Draft listener server running. Press Ctrl+C to stop.")

        # Wait for shutdown
        try:
            await self.listen_task
        except asyncio.CancelledError:
            logger.info("Listen task cancelled")

    async def stop(self) -> None:
        """Stop the draft listener server."""
        logger.info("Stopping draft listener server...")
        self.running = False

        await self.ws_client.disconnect()

        logger.info("Draft listener server stopped")


async def main(category_yaml):
    """Main entry point."""
    # Create store directory if needed
    GIBBON_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Create server
    server = DraftListenerServer(
        palaver_url=PALAVER_URL,
        store_dir=GIBBON_STORE_DIR,
        category_yaml=category_yaml,
        context_stack=CONTEXT_STACK
    )

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(server.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Start server
    try:
        await server.start(sync_lookback_hours=SYNC_LOOKBACK_HOURS)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await server.stop()


if __name__ == "__main__":
    category_yaml = Path(__file__).parent / "essay.yaml"
    asyncio.run(main(category_yaml))
