import numpy as np
from typing import Dict, Any, Tuple
from .model import Model
from .mip.column_generation import ColumnGeneration
from bpp1d.structure import BppPlan, BppBin, Solution
from bpp1d.utils.anyfit import b


class CGFit(Model):
    def __init__(self, name: str = 'cg_fit', max_iter: int = 1000):
        super().__init__(name)
        self.max_iter= max_iter


    def build(self, *args, **kwargs) -> Any:
        # items:Iterable[int] = np.array(kwargs['items'])
        demands:Dict[int, int] = np.array(kwargs['demands'])
        capacity:int = kwargs['capacity']
        
        cg = ColumnGeneration(capacity, demands)
        plan = cg.solve()
        return BppPlan(plan)
    
    def solve(self, problem, *args, **kwargs)  -> Tuple[Solution, Dict | None]:
        pass
