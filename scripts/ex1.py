import asyncio
from gibbon.itree.intent_tree import TreeRoot, Branch

async def main():

    root1 = TreeRoot("todos", "Lists of items that need doing")
    todo_b1 = Branch("coding", "Stuff to code", parent=root1)
    from pprint import pprint
    pprint(root1)

asyncio.run(main())
