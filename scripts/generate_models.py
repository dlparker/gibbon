"""
Generate test data for intent matching experiments.

Creates IBranches (intent categories) and Drafts (VTT transcriptions)
in the database for testing LLM matching.
"""
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
)


def create_test_branches(store_dir: Path) -> IBranchStore:
    """
    Create the test intent tree structure.

    Returns IBranchStore with populated branches.
    """
    ibranch_store = IBranchStore(store_dir)

    with ibranch_store.session() as session:
        # Root: Todo lists
        todo_root = IBranch(
            name="Todo lists",
            description="Lists of items that need doing"
        )
        session.add(todo_root)
        session.commit()
        session.refresh(todo_root)

        # Child: grocery shopping
        grocery = IBranch(
            name="grocery shopping list",
            description="Items to purchase at grocery store",
            parent_id=todo_root.id
        )
        session.add(grocery)
        session.commit()
        session.refresh(grocery)

        # Child: taxes
        taxes = IBranch(
            name="taxes",
            description="Tax-related tasks and deadlines",
            parent_id=todo_root.id
        )
        session.add(taxes)
        session.commit()
        session.refresh(taxes)

        # Root: Project Notes
        projects_root = IBranch(
            name="Project Notes",
            description="Notes and tasks for various projects"
        )
        session.add(projects_root)
        session.commit()
        session.refresh(projects_root)

        # Child: kanban board
        kanban = IBranch(
            name="Customize kanban board app for software projects",
            description="Development work on kanban board customization",
            parent_id=projects_root.id
        )
        session.add(kanban)
        session.commit()
        session.refresh(kanban)

        # Child: moving out
        moving = IBranch(
            name="Plan moving out of house",
            description="Planning and tasks for moving to new residence",
            parent_id=projects_root.id
        )
        session.add(moving)
        session.commit()
        session.refresh(moving)

        # Child: house sale prep
        house_sale = IBranch(
            name="Prepare house for sale",
            description="Tasks to prepare current house for sale",
            parent_id=projects_root.id
        )
        session.add(house_sale)
        session.commit()
        session.refresh(house_sale)

        # Count branches
        from sqlmodel import select, func
        count = session.exec(select(func.count(IBranch.id))).one()
        print(f"Created {count} IBranches")

    return ibranch_store


async def create_test_drafts(store_dir: Path) -> DraftStore:
    """
    Create test draft transcriptions.

    Returns DraftStore with populated drafts.
    """
    draft_store = DraftStore(store_dir)

    test_texts = [
        "Get some steaks",
        "get moving boxes",
        "pack up kitchen",
        "email accountant",
    ]

    for text in test_texts:
        draft = Draft(
            draft_id=str(uuid.uuid4()),
            timestamp=time.time(),
            full_text=text
        )
        await draft_store.add_draft(draft)
        print(f"Created draft: '{text}'")

    return draft_store


def format_branches_for_prompt(ibranch_store: IBranchStore) -> str:
    """
    Format IBranches as text for LLM prompt.

    Returns formatted category list.
    """
    lines = ["Categories (hierarchical - child categories are more specific than parents):", ""]

    with ibranch_store.session() as session:
        roots = IBranch.get_roots(session)

        for root in roots:
            parent_str = "none"
            lines.append(f"- {root.name} (ID={root.id}, parent: {parent_str})")

            for child in root.children:
                lines.append(f"  - {child.name} (ID={child.id}, parent: {root.id})")

    return "\n".join(lines)


async def main():
    """Generate all test data."""
    # Set up clean test directory
    store_dir = Path("/tmp/gibbon_test")
    if store_dir.exists():
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating test data...")
    print("=" * 60)

    # Create branches
    ibranch_store = create_test_branches(store_dir)

    # Create drafts
    draft_store = await create_test_drafts(store_dir)

    # Show formatted categories
    print("\n" + "=" * 60)
    print("Formatted categories for prompt:")
    print("=" * 60)
    categories_text = format_branches_for_prompt(ibranch_store)
    print(categories_text)

    print("\n" + "=" * 60)
    print(f"Test data created in: {store_dir}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
