from typing import Iterable
import numpy as np
import json 



class BinPattern:

    def __init__(self, items:Iterable) -> None:
        self.items = tuple(sorted(items))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinPattern):
            return NotImplemented
        return self.items == other.items

    def __hash__(self):
        return hash(";".join(map(str, self.items)))

    def __contains__(self, item):
        return item in self.items

    def comp(self, target):
        common, ldiff, rdiff = [], [], []
        if isinstance(target, BinPattern):
            target = target.items
        target = sorted(target)
        i, j = 0, 0

        while i < len(self.items) and j < len(target):
            if self.items[i] == target[j]:
                common.append(self.items[i])
                i += 1
                j += 1

            elif self.items[i] < target[j]:
                ldiff.append(self.items[i])
                i += 1
            else:
                rdiff.append(target[j])
                j += 1

        ldiff += self.items[i:]
        rdiff += target[j:]

        # consider a pattern comparing with bin:
        # - common: packed 
        # - ldiff: in pattern but not in bin, empty slots
        # - rdiff: in bin but not in pattern, items not in plan
        return common, ldiff, rdiff

    def __repr__(self):
        return self.items.__repr__()

    def to_np(self, item_order):
        counts = {i: self.items.count(i) for i in item_order}
        arr = np.zeros(len(item_order), dtype=int)
        for i in counts:
            arr[i] = counts[i]
        return arr

    def to_json(self) -> str:
        return json.dumps({i: self.items.count(i) for i in set(self.items)})
