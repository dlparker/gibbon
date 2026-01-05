from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class DraftContext:
    """Context information about the draft and categorization."""
    draft_text: str
    category_name: str
    category_path: list[str]
    category_description: str
    confidence: float
    timestamp: str
    session_id: Optional[str] = None

@dataclass
class SinkResponse:
    """Response from a DraftSink after processing."""
    status: str  # "continue", "done", "error"
    message: Optional[str] = None
    data: Optional[dict] = None

class DraftSink(ABC):
    """
    Abstract base class for tools that process categorized drafts.

    DraftSinks register themselves with categories they can handle.
    When a draft matches a claimed category, the sink takes over processing.
    """

    @abstractmethod
    def can_handle(self, category_path: list[str]) -> bool:
        """
        Returns True if this sink wants to handle drafts for this category.

        Called by the registry to determine which sink(s) claim a category.
        """
        pass

    @abstractmethod
    def starting_draft(self, context: DraftContext) -> SinkResponse:
        """
        Handle the first draft in a potential multi-draft session.

        The sink can:
        1. Process the draft immediately (return "done")
        2. Start a session and wait for more drafts (return "continue")
        3. Report an error (return "error")

        Common pattern: strip aim signal from draft_text, extract intent,
        begin processing.
        """
        pass

    @abstractmethod
    def continue_draft(self, context: DraftContext) -> SinkResponse:
        """
        Handle subsequent drafts in the same session.

        Only called if:
        1. Previous draft returned "continue"
        2. This draft matches the same category

        Otherwise draft is "orphaned" and goes to starting_draft.
        """
        pass

    def get_name(self) -> str:
        """Return human-readable name for this sink."""
        return self.__class__.__name__

class SinkRegistry:
    """
    Manages registration and dispatch of DraftSinks.

    Usage:
        registry = SinkRegistry()
        registry.register(TopicEssayBuilder())
        registry.register(TaskListBuilder())

        # When draft is categorized:
        response = registry.dispatch(draft_context)
    """

    def __init__(self):
        self._sinks: list[DraftSink] = []
        self._active_sessions: dict[str, DraftSink] = {}

    def register(self, sink: DraftSink):
        """Register a DraftSink."""
        self._sinks.append(sink)

    def find_sink(self, category_path: list[str]) -> Optional[DraftSink]:
        """Find a sink that can handle this category."""
        for sink in self._sinks:
            if sink.can_handle(category_path):
                return sink
        return None

    def dispatch(self, context: DraftContext) -> SinkResponse:
        """
        Dispatch a categorized draft to appropriate sink.

        Handles session management (starting_draft vs continue_draft).
        """
        session_id = context.session_id

        # Check if this is a continuation of existing session
        if session_id and session_id in self._active_sessions:
            sink = self._active_sessions[session_id]
            response = sink.continue_draft(context)

            # Clean up session if done
            if response.status in ("done", "error"):
                del self._active_sessions[session_id]

            return response

        # New session - find appropriate sink
        sink = self.find_sink(context.category_path)
        if not sink:
            return SinkResponse(
                status="error",
                message=f"No sink registered for category: {context.category_path}"
            )

        response = sink.starting_draft(context)

        # Track session if continuing
        if response.status == "continue" and session_id:
            self._active_sessions[session_id] = sink

        return response
    
