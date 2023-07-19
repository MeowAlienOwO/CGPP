from typing import Dict, List, Sequence, Tuple
from bpp1d.structure import BinSolution, BppBin, Solution
from bpp1d.utils.heuristic_choice import HeuristicChoiceFn, best_fit_choice
from .model import Model, ModelStatus

class HeuristicModel(Model):


    def __init__(self, capacity: int, instance: Sequence[int], name: str = 'model',
                    choice_fn: HeuristicChoiceFn | None = None):
        super().__init__(capacity, instance, name)
        self.choice_fn = choice_fn if choice_fn is not None else best_fit_choice
        self.bins: List[BppBin] = []

    def solve(self) -> Tuple[Solution, Dict | None]:
        self.status = ModelStatus.SOLVING
        for item in self.instance:
            choice = self.choice_fn(item, self.bins)
            if choice < 0:
                self.bins.append(BppBin(self.capacity, [item]))
            else:
                self.bins[choice].pack(item)
        
        self.status = ModelStatus.FINISHED
        return BinSolution(self.capacity, self.bins), None