import asyncio
import time
import uuid
from pathlib import Path
import shutil
from gibbon.store.models import (
    Draft,
    DraftStore,
    IBranch,
    IBranchStore,
    ContextEvent,
    ContextStore,
)

async def main():
    # Set up storage
    ddir = Path("/tmp/gibbon")
    if ddir.exists():
        shutil.rmtree(ddir)
    ddir.mkdir()

    # Initialize all three stores
    draft_store = DraftStore(ddir)
    ibranch_store = IBranchStore(ddir)
    context_store = ContextStore(ddir)

    # Create intent tree structure using SQLModel
    with ibranch_store.session() as session:
        # Create root branch (no parent)
        root1 = IBranch(
            name="todos",
            description="Lists of items that need doing"
        )
        session.add(root1)
        session.commit()
        session.refresh(root1)
        print(f"Created root: '{root1.name}' (id={root1.id})")

        # Create child branches
        todo_b1 = IBranch(
            name="coding",
            description="Stuff to code",
            parent_id=root1.id
        )
        session.add(todo_b1)
        session.commit()
        session.refresh(todo_b1)
        print(f"Created branch: '{todo_b1.name}' (id={todo_b1.id}, parent_id={todo_b1.parent_id})")

        todo_b2 = IBranch(
            name="shop",
            description="Stuff to buy",
            parent_id=todo_b1.id
        )
        session.add(todo_b2)
        session.commit()
        session.refresh(todo_b2)
        print(f"Created branch: '{todo_b2.name}' (id={todo_b2.id}, parent_id={todo_b2.parent_id})")

    # Query and display the tree
    with ibranch_store.session() as session:
        roots = IBranch.get_roots(session)
        print(f"\nFound {len(roots)} root(s):")
        for root in roots:
            print(f"  Root: {root.name}")
            print(f"    Children: {[c.name for c in root.children]}")
            for child in root.children:
                print(f"      Grandchildren of {child.name}: {[gc.name for gc in child.children]}")

    # Add drafts
    d1 = Draft(
        draft_id=str(uuid.uuid4()),
        timestamp=time.time(),
        full_text="New todo. Review source code for kboard"
    )
    await draft_store.add_draft(d1)
    print(f"\nAdded draft: '{d1.full_text}'")

    d2 = Draft(
        draft_id=str(uuid.uuid4()),
        timestamp=time.time(),
        full_text="New todo. Pay property tax"
    )
    await draft_store.add_draft(d2)
    print(f"Added draft: '{d2.full_text}'")

    # Log context events (matching drafts to intents)
    event1 = ContextEvent(
        intent_name="coding",
        draft_id=d1.draft_id,
        timestamp=time.time()
    )
    await context_store.add_event(event1)
    print(f"\nLogged context: draft matched to intent '{event1.intent_name}'")

    # Show current context
    current = context_store.get_current_context()
    print(f"Current context: {current.intent_name}")

asyncio.run(main())
