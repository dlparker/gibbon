"""
WebSocket client for listening to real-time draft events from palaver.
"""
import asyncio
import json
import logging
from typing import Optional, Callable, Awaitable
import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger("PalaverWebSocketClient")


class PalaverWebSocketClient:
    """Client for palaver's WebSocket event stream."""

    # Event type constants
    DRAFT_START_EVENT = "<class 'palaver.scribe.draft_events.DraftStartEvent'>"
    DRAFT_END_EVENT = "<class 'palaver.scribe.draft_events.DraftEndEvent'>"
    DRAFT_REVISION_EVENT = "<class 'palaver.scribe.draft_events.DraftRevisionEvent'>"
    ALL_EVENTS = "all"
    ALL_BUT_CHUNKS = "all_but_chunks"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0
    ):
        """
        Initialize palaver WebSocket client.

        Args:
            base_url: Base URL of palaver server (default: http://localhost:8000)
            auto_reconnect: Automatically reconnect on disconnect (default: True)
            reconnect_delay: Seconds to wait before reconnecting (default: 5.0)
        """
        # Convert http:// to ws://
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = ws_url.rstrip('/') + "/events"

        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay

        self._websocket: Optional[WebSocketClientProtocol] = None
        self._event_handler: Optional[Callable[[dict], Awaitable[None]]] = None
        self._subscribed_events: list[str] = []
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None

    async def connect(self, event_types: list[str]) -> None:
        """
        Connect to palaver and subscribe to event types.

        Args:
            event_types: List of event type strings to subscribe to.
                Common values:
                - PalaverWebSocketClient.DRAFT_END_EVENT
                - PalaverWebSocketClient.DRAFT_START_EVENT
                - PalaverWebSocketClient.ALL_BUT_CHUNKS
                - PalaverWebSocketClient.ALL_EVENTS

        Raises:
            websockets.WebSocketException: If connection fails
        """
        logger.info(f"Connecting to palaver at {self.ws_url}")
        self._websocket = await websockets.connect(self.ws_url)

        # Send subscription request
        subscription = {"subscribe": event_types}
        await self._websocket.send(json.dumps(subscription))
        self._subscribed_events = event_types

        logger.info(f"Subscribed to events: {event_types}")

    async def disconnect(self) -> None:
        """Disconnect from palaver."""
        self._running = False

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            logger.info("Disconnected from palaver")

    async def listen(
        self,
        event_handler: Callable[[dict], Awaitable[None]],
        event_types: Optional[list[str]] = None
    ) -> None:
        """
        Listen for events and call handler for each event received.

        This method will run indefinitely until disconnect() is called or
        an unrecoverable error occurs.

        Args:
            event_handler: Async function to call for each event.
                Receives event dict with keys:
                - event_class (str)
                - draft (dict)
                - timestamp (float)
                - event_id (str)
                - author_uri (str)
            event_types: Event types to subscribe to. If None, subscribes to
                DRAFT_END_EVENT only.

        Raises:
            websockets.WebSocketException: If connection fails and auto_reconnect is False
        """
        if event_types is None:
            event_types = [self.DRAFT_END_EVENT]

        self._event_handler = event_handler
        self._running = True

        while self._running:
            try:
                # Connect if not connected
                if not self._websocket or self._websocket.closed:
                    await self.connect(event_types)

                # Listen for events
                async for message in self._websocket:
                    if not self._running:
                        break

                    try:
                        event_data = json.loads(message)
                        logger.debug(f"Received event: {event_data.get('event_class', 'unknown')}")

                        # Call handler
                        if self._event_handler:
                            await self._event_handler(event_data)

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse event JSON: {e}")
                    except Exception as e:
                        logger.error(f"Error handling event: {e}", exc_info=True)

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")

                if not self.auto_reconnect or not self._running:
                    break

                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)

            except Exception as e:
                logger.error(f"Error in listen loop: {e}", exc_info=True)

                if not self.auto_reconnect or not self._running:
                    break

                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)

        logger.info("Stopped listening for events")

    def start_listening(
        self,
        event_handler: Callable[[dict], Awaitable[None]],
        event_types: Optional[list[str]] = None
    ) -> asyncio.Task:
        """
        Start listening for events in the background.

        Returns a Task that can be awaited or cancelled.

        Args:
            event_handler: Async function to call for each event
            event_types: Event types to subscribe to (default: [DRAFT_END_EVENT])

        Returns:
            asyncio.Task running the listen loop
        """
        self._listen_task = asyncio.create_task(
            self.listen(event_handler, event_types)
        )
        return self._listen_task

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
