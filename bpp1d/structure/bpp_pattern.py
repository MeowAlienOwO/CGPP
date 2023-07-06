from typing import Iterable, List, Tuple
import numpy as np
import json 



class BinPattern:

    def __init__(self, items:Iterable[int]) -> None:
        self.items = tuple(sorted(items))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinPattern):
            return NotImplemented
        return self.items == other.items

    def __hash__(self):
        return hash(";".join(map(str, self.items)))

    def __contains__(self, item) -> bool:
        return item in self.items

    def comp(self, target: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """compare with target bin, to see how many items in common

        Args:
            target List[int]: target bin

        Returns:
            Tuple[List[int], List[int], List[int]]: return a tuple of comparison results: (common, ldiff, rdiff),
            where: 
                - common: packed items
                - ldiff: in pattern but not in bin, i.e. empty slots
                - rdiff: in bin but not in pattern, i.e. items not in plan
        """

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
