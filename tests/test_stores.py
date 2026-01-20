import asyncio
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from gibbon.store.models import (
    Draft,
    DraftStore,
)


async def test_draft_stores():

    # Use temp directory for test databases
    with TemporaryDirectory() as tmpdir:
        store_dir = Path(tmpdir)

        # Initialize all three stores
        draft_store = DraftStore(store_dir)
        # Verify we have three separate database files
        assert (store_dir / "drafts.db").exists()

        # Add a draft (incoming VTT transcription)
        draft = Draft(
            draft_id="test-uuid-123",
            timestamp=time.time(),
            full_text="Add milk to grocery list"
        )
        await draft_store.add_draft(draft)

        

if __name__ == "__main__":
    asyncio.run(test_three_stores())
