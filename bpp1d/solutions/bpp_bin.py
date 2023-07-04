from typing import List
import numpy as np
from numpy.typing import ArrayLike



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

    def append(self, item) -> None:
        if item <= self.empty_space:
            self.items.append(item)
        else:
            raise ValueError("Item exceed bin capacity.\nItem:{}\nBin:{}".format(item, self.__repr__()))

    def __repr__(self) -> str:
        return str(self.items)

