from abc import abstractmethod
import enum
from typing import Any, Dict, Sequence, Tuple

from bpp1d.structure import Solution



class ModelStatus(enum.Enum):
    WAITING = 0
    SOLVING  = 1
    FINISHED = 2
    ERROR = -1

class Model:
    def __init__(self,capacity:int, instance:Sequence[int], name: str = 'model'):
        self.name = name
        self.instance = instance
        self.capacity = capacity
        self.status = ModelStatus.WAITING

    def build(self) -> Any:
        """build the model
        """
        pass

    @abstractmethod
    def solve(self) -> Tuple[Solution, Dict | None]:
        """solve problem instance
        """
        raise NotImplementedError
