import asyncio
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from gibbon.store.models import (
    Draft,
    DraftStore,
    IBranch,
    IBranchStore,
    ContextEvent,
    ContextStore,
)


async def test_three_stores():
    """Test all three stores working independently but coordinated."""

    # Use temp directory for test databases
    with TemporaryDirectory() as tmpdir:
        store_dir = Path(tmpdir)

        # Initialize all three stores
        draft_store = DraftStore(store_dir)
        ibranch_store = IBranchStore(store_dir)
        context_store = ContextStore(store_dir)

        # Verify we have three separate database files
        assert (store_dir / "drafts.db").exists()
        assert (store_dir / "ibranches.db").exists()
        assert (store_dir / "context.db").exists()

        print("✓ Three separate databases created")

        # Add some intent branches (tree structure)
        with ibranch_store.session() as session:
            # Create a root branch
            root = IBranch(
                name="grocery list",
                description="Manage grocery shopping list"
            )
            session.add(root)
            session.commit()
            session.refresh(root)

            # Create a child branch
            child = IBranch(
                name="add grocery item",
                description="Add an item to the grocery list",
                parent_id=root.id
            )
            session.add(child)
            session.commit()
            session.refresh(child)

            print(f"✓ Created root branch: '{root.name}' (id={root.id})")
            print(f"✓ Created child branch: '{child.name}' (id={child.id}, parent_id={child.parent_id})")

        # Query the tree structure
        with ibranch_store.session() as session:
            roots = IBranch.get_roots(session)
            assert len(roots) == 1
            assert roots[0].name == "grocery list"

            # Access children via relationship
            assert len(roots[0].children) == 1
            assert roots[0].children[0].name == "add grocery item"

            print(f"✓ Root '{roots[0].name}' has {len(roots[0].children)} child(ren)")

        # Add a draft (incoming VTT transcription)
        draft = Draft(
            draft_id="test-uuid-123",
            timestamp=time.time(),
            full_text="Add milk to grocery list"
        )
        await draft_store.add_draft(draft)
        print(f"✓ Added draft: '{draft.full_text}'")

        # Log a context event (loose coupling - references IBranch by name!)
        event = ContextEvent(
            intent_name="add grocery item",  # String reference, no FK
            draft_id="test-uuid-123",
            timestamp=time.time(),
            stack_name="default"
        )
        await context_store.add_event(event)
        print(f"✓ Logged context event: intent='{event.intent_name}'")

        # Check current context
        current = context_store.get_current_context()
        assert current is not None
        assert current.intent_name == "add grocery item"
        assert current.draft_id == "test-uuid-123"
        print(f"✓ Current context: intent='{current.intent_name}', draft='{current.draft_id}'")

        # Add another event to test history
        event2 = ContextEvent(
            intent_name="grocery list",
            draft_id=None,  # Could be triggered by something other than a draft
            timestamp=time.time() + 1,
            stack_name="default"
        )
        await context_store.add_event(event2)

        # Get context history
        history = context_store.get_context_history(limit=5)
        assert len(history) == 2
        assert history[0].intent_name == "grocery list"  # Most recent first
        assert history[1].intent_name == "add grocery item"
        print(f"✓ Context history has {len(history)} events")

        # Demonstrate loose coupling - context references branches by name
        print("\n=== Loose Coupling Demo ===")
        print(f"Context event references IBranch by name: '{current.intent_name}'")
        print("No foreign key relationship - each store is independent!")

        print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_three_stores())
