from typing import List, Tuple
import numpy as np
from numpy.typing import ArrayLike
import json

from .bpp_pattern import BinPattern



class BppBin(object):
    """Bin representation
    """
    def __init__(self, capacity:int, items: List | None = None) -> None:
        self.items = items if items is not None else []
        self.capacity = capacity

    def __len__(self) -> int:
        return len(self.items)

        
    def __setitem__(self, key:int, value:int) -> None:
        self.items[key] = value

    def __getitem__(self, key:int) -> int:
        return self.items[key]


    def __iter__(self):
        return iter(self.items)

    def to_np(self) -> ArrayLike:
        return np.array(self.items)

    @property
    def empty_space(self) -> int:
        return self.capacity - self.filled_space

    @property
    def filled_space(self) -> int:
        return sum(self.items)

    @property
    def empty(self) -> bool:
        return not self.items

    @property
    def full(self) -> bool:
        return sum(self.items) == self.capacity

    def pack(self, item: int) -> None:
        """Pack a item to the bin

        Args:
            item (int): item to be packed

        Raises:
            ValueError: non-positive or exceed bin capacity
        """
        if not isinstance(item, int):
            raise ValueError("Item should be integer")

        if item > 0 and item <= self.empty_space:
            self.items.append(item)
        elif item < 0:
            raise ValueError("Item should be positive integer")
        else:
            raise ValueError("Item exceed bin capacity.\nItem:{}\nBin:{}".format(item, self.__repr__()))

    def append(self, item: int) -> None:
        """[Deprecated]synonyms of pack, for legacy compatibility

        Args:
            item (int): item to be packed
        """
        self.pack(item)
    
    def __repr__(self) -> str:
        return self.items.__repr__()
    
    def to_json(self) -> str:
        return json.dumps(self.items)
    

class BinWithPattern(BppBin):
    def __init__(self, capacity: int, pattern: BinPattern, items: List | None = None):
        super().__init__(capacity=capacity, items=items)
        self.pattern = pattern
    
    def check(self)-> Tuple[List[int], List[int], List[int]]:
        """return pattern checking result.

        Returns:
            Tuple[List[int], List[int], List[int]]: same as BinPattern.comp
        """
        return self.pattern.comp(self.items)
    
    def __repr__(self) -> str:
        return str({
            "items": self.items,
            "pattern": self.pattern
        })

    def to_json(self) -> str:
        return json.dumps({
            "items": self.items,
            "pattern": self.pattern
        })


