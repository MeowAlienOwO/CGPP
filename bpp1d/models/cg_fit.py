from typing import Dict, Any, Sequence, Tuple, List
from bpp1d.structure.bpp_bin import BinWithPattern

from bpp1d.structure.bpp_plan import BinPlanExecutor
from .model import Model, ModelStatus
# from .mip.column_generation import ColumnGeneration
from .mip.column_generation_scipy import CGScipy
from bpp1d.structure import BppPlan, Solution, BinSolution
from bpp1d.utils.heuristic_choice import best_fit_choice


class CGFit(Model):

    def __init__(self, capacity: int, instance: Sequence[int], 
                    demands: Dict[int, int], name='cg_fit'):
        super().__init__(capacity, instance, name)
        self.demands = demands
        self.bins: List[BinWithPattern] = []

    def build(self) -> Any:
        # cg = ColumnGeneration(self.capacity, self.demands)
        cg = CGScipy(self.capacity, self.demands)
        result = cg.solve()
        if result is not None:
            self.plan = BppPlan(result, self.capacity)
        else:
            raise ValueError("infeasible solution")
        return super().build()

    def solve(self)  -> Tuple[Solution, Dict | None]:
        assert self.plan is not None
        self.plan_executor = BinPlanExecutor(self.plan, self.capacity, self.bins, shall_rebalance=False)
        self.status = ModelStatus.SOLVING
        for i, item in enumerate(self.instance):
            if i / len(self.instance) > 0.9:
                self.plan_executor.heuristic_put(item, best_fit_choice)
            else:
                self.plan_executor.put(item, best_fit_choice)
        
        self.status = ModelStatus.FINISHED

        return BinSolution(self.capacity, self.bins), {} #{"plan": self.plan}
