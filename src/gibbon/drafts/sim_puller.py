

class SimPuller:

    def __init__(self, drafts=None):
        if drafts is None:
            drafts = []
        self.drafts = drafts

    def add_draft(self, draft):
        self.drafts.append(draft):

    async def next_draft(self):
        index = 0
        while True:
            if index < len(item_list):
                item = item_list[index]
                index += 1
                yield item
            else:
                await asyncio.sleep(0.01)            

