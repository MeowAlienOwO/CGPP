from typing import Dict, Any, Sequence, Tuple, List
from bpp1d.structure.bpp_bin import BinWithPattern

from bpp1d.structure.bpp_plan import BinPlanExecutor
from .model import Model
from .mip.column_generation import ColumnGeneration
from bpp1d.structure import BppPlan, Solution, BinSolution
from bpp1d.utils.anyfit import best_fit_choice


class CGFit(Model):

    def __init__(self, capacity: int, instance: Sequence[int], 
                    demands: Dict[int, int], name: str = 'cg_fit'):
        super().__init__(capacity, instance, name)
        self.demands = demands

    def build(self) -> Any:
        cg = ColumnGeneration(self.capacity, self.demands)
        result = cg.solve()
        if result is not None:
            self.plan = BppPlan(result, self.capacity)

    def solve(self)  -> Tuple[Solution, Dict | None]:
        assert self.plan is not None
        bins: List[BinWithPattern] = []
        plan_executor = BinPlanExecutor(self.plan, self.capacity, bins)
        for item in self.instance:
            plan_executor.put(item, best_fit_choice)

        return BinSolution(self.capacity, bins), {"plan": self.plan}
