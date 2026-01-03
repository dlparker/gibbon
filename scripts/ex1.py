import asyncio
import time
import uuid
from pathlib import Path
import shutil
from gibbon.itree.intent_tree import IRoot, IBranch
from gibbon.store.models import Draft, DraftStore

async def main():

    root1 = IRoot("todos", "Lists of items that need doing")
    todo_b1 = IBranch("coding", "Stuff to code", parent=root1)
    todo_b2 = IBranch("shop", "Stuff to buy", root=root1, parent=todo_b1)
    from pprint import pprint
    pprint(root1)

    ddir = Path("/tmp/gibbon")
    if ddir.exists():
        shutil.rmtree(ddir)
    ddir.mkdir()
    ds = DraftStore(ddir)

    d1 = dict(draft_id=uuid.uuid4(), timestamp=time.time(),
              full_text = "New todo. Review source code for kboard")
    draft1 = Draft(**d1)
    await ds.add_draft(draft1)
    
    d2 = dict(draft_id=uuid.uuid4(), timestamp=time.time(),
              full_text = "New todo. Pay property tax")

    await ds.add_draft(Draft(**d2))
    

asyncio.run(main())
