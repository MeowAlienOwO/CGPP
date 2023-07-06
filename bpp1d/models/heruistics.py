from typing import Dict, List, Sequence, Tuple
from bpp1d.structure import BinSolution, BppBin, Solution
from bpp1d.utils.anyfit import HeuristicChoiceFn, best_fit_choice
from .model import Model

class HeuristicModel(Model):

    def __init__(self, capacity: int, 
                    choice_fn: HeuristicChoiceFn | None = None, name='best_fit'):
        super().__init__(name)
        self.capacity = capacity
        self.choice_fn = choice_fn if choice_fn is not None else best_fit_choice

    def solve(self, instance:Sequence[int]) -> Tuple[Solution, Dict | None]:
        bins: List[BppBin] = []
        for item in instance:
            choice = self.choice_fn(item, bins)
            if choice < 0:
                bins.append(BppBin(self.capacity, [item]))
            else:
                bins[choice].pack(item)
        return BinSolution(self.capacity, bins), None