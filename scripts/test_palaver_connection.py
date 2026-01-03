"""
Quick test to verify palaver connection and show recent drafts.
"""
import asyncio
from gibbon.palaver_client import PalaverRestClient


async def main():
    palaver_url = "http://localhost:8000"

    print("=" * 60)
    print("Testing Palaver Connection")
    print(f"URL: {palaver_url}")
    print("=" * 60)

    try:
        async with PalaverRestClient(palaver_url) as client:
            # Fetch recent drafts (last 5, summary only)
            print("\nFetching recent drafts (summary)...")
            drafts, total = await client.fetch_all_drafts(limit=5, order="desc")

            print(f"\nFound {total} total drafts")
            print(f"Showing {len(drafts)} most recent:\n")

            for i, draft in enumerate(drafts, 1):
                print(f"{i}. Draft ID: {draft['draft_id']}")
                print(f"   Timestamp: {draft['timestamp']}")
                print(f"   Text: {draft['full_text'][:60]}...")
                print()

            print("✅ Connection successful!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
