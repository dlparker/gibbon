"""
Utilities to convert palaver draft data to gibbon Draft models.
"""
from datetime import datetime
from gibbon.store.models import Draft


def convert_palaver_draft(palaver_draft: dict) -> Draft:
    """
    Convert a palaver draft dictionary to a gibbon Draft model.

    Args:
        palaver_draft: Dictionary from palaver API with fields:
            - draft_id (str): UUID
            - timestamp (float): Unix timestamp
            - full_text (str): Transcribed text
            - created_at (str, optional): ISO datetime string
            - classname (str, ignored)
            - directory_path (str, ignored)
            - parent_draft_id (str, ignored for now)

    Returns:
        Draft model instance ready to save to gibbon database
    """
    # Extract core fields
    draft = Draft(
        draft_id=palaver_draft["draft_id"],
        timestamp=palaver_draft["timestamp"],
        full_text=palaver_draft["full_text"],
    )

    # Parse created_at if present
    if "created_at" in palaver_draft and palaver_draft["created_at"]:
        try:
            draft.created_at = datetime.fromisoformat(palaver_draft["created_at"])
        except (ValueError, TypeError):
            # If parsing fails, default to now (will be set by field default)
            pass

    return draft


def convert_palaver_event_draft(event_draft: dict) -> Draft:
    """
    Convert a draft from a palaver event to a gibbon Draft model.

    Event drafts have a slightly different structure than REST API drafts.

    Args:
        event_draft: Draft dictionary from WebSocket event with fields:
            - draft_id (str): UUID
            - timestamp (float): Unix timestamp
            - full_text (str): Transcribed text
            - start_text (str, ignored)
            - end_text (str, ignored)
            - parent_draft_id (str, ignored)
            - audio_start_time (float, ignored)
            - audio_end_time (float, ignored)

    Returns:
        Draft model instance ready to save to gibbon database
    """
    return Draft(
        draft_id=event_draft["draft_id"],
        timestamp=event_draft["timestamp"],
        full_text=event_draft["full_text"],
    )
