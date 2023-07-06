from abc import abstractmethod
from typing import Any, Dict, Sequence, Tuple

from bpp1d.structure import Solution



class Model:
    def __init__(self, name: str = 'model'):
        self.name = name

    def build(self, *args, **kwargs) -> Any:
        """build the model
        """
        pass

    @abstractmethod
    def solve(self, instance:Sequence[int]) -> Tuple[Solution, Dict | None]:  # noqa: F821
        """solve problem instance
        """
        raise NotImplementedError
