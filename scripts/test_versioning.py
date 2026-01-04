"""
Test script to demonstrate IBranch versioning.

Shows:
1. Querying active categories (default behavior)
2. Creating a new version of a category tree
3. Marking old version as inactive
4. Querying historical versions
"""
import asyncio
from pathlib import Path

from gibbon.store.models import IBranch, IBranchStore


async def main():
    store_dir = Path("./data")
    ibranch_store = IBranchStore(store_dir)

    print("=" * 60)
    print("IBranch Versioning Test")
    print("=" * 60)

    # Show current active categories
    with ibranch_store.session() as session:
        active_roots = IBranch.get_roots(session, active_only=True)
        print(f"\nActive category trees (version 1):")
        for root in active_roots:
            print(f"  - {root.name} (ID={root.id}, version={root.version})")
            for child in root.children:
                print(f"    - {child.name} (ID={child.id})")

    # Simulate creating a new version of "Todo lists" tree
    print("\n" + "-" * 60)
    print("Creating version 2 of 'Todo lists' category tree...")
    print("-" * 60)

    with ibranch_store.session() as session:
        from sqlmodel import select

        # Mark old version as inactive
        statement = select(IBranch).where(IBranch.name == "Todo lists")
        old_root = session.exec(statement).first()
        old_root.is_active = False
        session.add(old_root)
        session.commit()
        print(f"✓ Marked old 'Todo lists' (ID={old_root.id}) as inactive")

        # Create new version of root
        new_root = IBranch(
            name="Todo lists",
            description="Lists of items that need doing (revised)",
            version=2,
            is_active=True
        )
        session.add(new_root)
        session.commit()
        session.refresh(new_root)
        print(f"✓ Created new 'Todo lists' (ID={new_root.id}, version=2)")

        # Create updated children
        grocery_v2 = IBranch(
            name="grocery shopping list",
            description="Items to purchase at grocery store (improved)",
            parent_id=new_root.id
        )
        session.add(grocery_v2)

        household_v2 = IBranch(
            name="household tasks",
            description="General household chores and maintenance",
            parent_id=new_root.id
        )
        session.add(household_v2)

        session.commit()
        session.refresh(grocery_v2)
        session.refresh(household_v2)
        print(f"✓ Created 2 children under new root")

    # Show active categories (should show v2 only)
    print("\n" + "=" * 60)
    print("Active category trees (after creating v2):")
    print("=" * 60)

    with ibranch_store.session() as session:
        active_roots = IBranch.get_roots(session, active_only=True)
        for root in active_roots:
            print(f"\n- {root.name} (ID={root.id}, version={root.version}, active={root.is_active})")
            for child in root.children:
                print(f"  - {child.name} (ID={child.id})")

    # Show all roots (including inactive)
    print("\n" + "=" * 60)
    print("All category trees (including inactive):")
    print("=" * 60)

    with ibranch_store.session() as session:
        all_roots = IBranch.get_roots(session, active_only=False)
        for root in all_roots:
            status = "ACTIVE" if root.is_active else "INACTIVE"
            print(f"\n- {root.name} (ID={root.id}, version={root.version}, {status})")
            for child in root.children:
                print(f"  - {child.name} (ID={child.id})")

    print("\n" + "=" * 60)
    print("Versioning test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
