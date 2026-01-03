from dataclasses import dataclass, field

@dataclass
class Branch:
    name: str
    description: str
    parent: 'Branch'
    branches: list['Branch'] = field(default_factory=list['Branch'])

    def __post_init__(self):
        if self.parent and self not in self.parent.branches:
            self.parent.branches.append(self)

@dataclass(kw_only=True)
class TreeRoot(Branch):
    parent: Branch = None
