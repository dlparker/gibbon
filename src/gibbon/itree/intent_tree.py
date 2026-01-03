from dataclasses import dataclass, field
from typing import Optional

@dataclass
class IBranch:
    name: str
    description: str
    parent: 'IBranch'
    root: Optional['IRoot'] = None
    branches: list['IBranch'] = field(default_factory=list['IBranch'])

    def __post_init__(self):
        if self.parent and self not in self.parent.branches:
            self.parent.branches.append(self)
        if self.root is None:
            self.root = self.parent

@dataclass(kw_only=True)
class IRoot(IBranch):
    parent: IBranch = None
    root: 'IRoot' = None
