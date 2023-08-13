from typing import Dict


class Solution:
    """Stores the solution of 1d bpp
    """

    @property
    def waste(self) -> int:
        return NotImplemented
    
    @property
    def metrics(self) -> Dict:
        return NotImplemented
    
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def num_bins(self) -> int:
        return NotImplemented
    
    @capacity.setter
    def capacity(self, value: int) -> None:
        if(value < 0):
            raise ValueError("Capacity should be greater than zero")
        self._capacity = value


    def to_json(self) -> str:
        return NotImplemented
    
