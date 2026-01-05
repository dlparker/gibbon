import yaml
import re
from pathlib import Path
from typing import List, Optional
from ..draft_sink import DraftSink, DraftContext, SinkResponse

class TopicEssayBuilder(DraftSink):
    """
    DraftSink for building topic essays via voice dictation.

    Handles two modes:
    1. "add topic <name>" - Create new topic entry
    2. "topic essay" - Add/edit essay for current topic

    Multi-draft flow:
    - "add topic copper mining" → prompts for essay
    - "topic essay. Copper mining extracts..." → stores essay
    - "done" → finishes session
    """

    def __init__(self, essays_file: str = "topo_essays.yaml"):
        self.essays_file = Path(essays_file)
        self._ensure_essays_file()

        # Session state
        self._current_topic: Optional[str] = None
        self._essay_parts: List[str] = []

    def _ensure_essays_file(self):
        """Create essays file if it doesn't exist."""
        if not self.essays_file.exists():
            self.essays_file.write_text("# Topic Essays\n{}\n")

    def can_handle(self, category_path: List[str]) -> bool:
        """
        Handle categories under taxonomy:
        - taxonomy -> add_topic
        - taxonomy -> topic_essay
        """
        # Look for "taxonomy" in path and check leaf
        if "taxonomy" in category_path:
            leaf = category_path[-1]
            return leaf in ("add_topic", "topic_essay")
        return False

    def starting_draft(self, context: DraftContext) -> SinkResponse:
        """Handle initial draft for topic/essay creation."""
        draft_text = context.draft_text.strip()
        leaf = context.category_path[-1]

        if leaf == "add_topic":
            return self._handle_add_topic(draft_text)
        elif leaf == "topic_essay":
            return self._handle_topic_essay(draft_text, is_start=True)

        return SinkResponse(
            status="error",
            message=f"Unknown taxonomy action: {leaf}"
        )

    def continue_draft(self, context: DraftContext) -> SinkResponse:
        """Handle continuation drafts in essay building session."""
        draft_text = context.draft_text.strip()

        # Check for completion signals
        if draft_text.lower() in ("done", "finish", "complete"):
            return self._finish_essay()

        # Add to essay
        self._essay_parts.append(draft_text)

        return SinkResponse(
            status="continue",
            message=f"Essay for '{self._current_topic}': {len(self._essay_parts)} parts collected"
        )

    def _handle_add_topic(self, draft_text: str) -> SinkResponse:
        """Extract topic name from 'add topic <name>' draft."""
        # Strip aim signal (e.g., "add topic" or "add top")
        pattern = r'^(?:add\s+)?top(?:ic)?\s+(.+)$'
        match = re.match(pattern, draft_text, re.IGNORECASE)

        if not match:
            return SinkResponse(
                status="error",
                message=f"Could not parse topic name from: {draft_text}"
            )

        topic_name = match.group(1).strip()
        self._current_topic = topic_name
        self._essay_parts = []

        return SinkResponse(
            status="continue",
            message=f"Topic '{topic_name}' started. Dictate essay content or say 'done'."
        )

    def _handle_topic_essay(self, draft_text: str, is_start: bool) -> SinkResponse:
        """Handle essay content."""
        # Strip "topic essay" prefix if present
        pattern = r'^topic\s+essay\.?\s*(.*)$'
        match = re.match(pattern, draft_text, re.IGNORECASE)

        if match:
            content = match.group(1).strip()
        else:
            content = draft_text

        if not self._current_topic and is_start:
            return SinkResponse(
                status="error",
                message="No current topic. Use 'add topic <name>' first."
            )

        if content:
            self._essay_parts.append(content)

        return SinkResponse(
            status="continue",
            message=f"Essay content added. Say 'done' to finish."
        )

    def _finish_essay(self) -> SinkResponse:
        """Save essay and finish session."""
        if not self._current_topic or not self._essay_parts:
            return SinkResponse(
                status="error",
                message="No topic or essay content to save."
            )

        # Join essay parts
        essay_text = " ".join(self._essay_parts)

        # Load existing essays
        essays = {}
        if self.essays_file.exists():
            with open(self.essays_file, 'r') as f:
                content = f.read()
                if content.strip() and content.strip() != '{}':
                    essays = yaml.safe_load(content) or {}

        # Add/update essay
        essays[self._current_topic] = {
            'essay': essay_text,
            'created_at': str(context.timestamp) if hasattr(self, 'context') else None
        }

        # Save
        with open(self.essays_file, 'w') as f:
            yaml.dump(essays, f, default_flow_style=False, allow_unicode=True)

        topic = self._current_topic

        # Clear session
        self._current_topic = None
        self._essay_parts = []

        return SinkResponse(
            status="done",
            message=f"Essay for '{topic}' saved successfully!",
            data={'topic': topic, 'essay': essay_text}
        )
