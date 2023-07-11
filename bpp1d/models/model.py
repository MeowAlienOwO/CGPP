from abc import abstractmethod
from typing import Any, Dict, Sequence, Tuple

from bpp1d.structure import Solution



class Model:
    def __init__(self,capacity:int, instance:Sequence[int], name: str = 'model'):
        self.name = name
        self.instance = instance
        self.capacity = capacity

    def build(self) -> Any:
        """build the model
        """
        pass

    @abstractmethod
    def solve(self) -> Tuple[Solution, Dict | None]:
        """solve problem instance
        """
        raise NotImplementedError
