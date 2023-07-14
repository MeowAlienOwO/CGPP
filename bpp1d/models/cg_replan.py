from typing import Any, Dict, List, Sequence, Tuple
from bpp1d.models.mip.column_generation import ColumnGeneration
from bpp1d.models.model import Model, ModelStatus
from bpp1d.structure.bin_solution import BinSolution
from bpp1d.structure.bpp_bin import BinWithPattern
from bpp1d.structure.bpp_plan import BinPlanExecutor, BppPlan, OutOfPlanException
from bpp1d.structure.solution import Solution
from bpp1d.utils.anyfit import best_fit_choice
from bpp1d.utils.demand_estimator import generate_discrete_demand_estimator
from bpp3d_dataset.utils.distributions import Discrete



class CGReplan(Model):
    def __init__(self, capacity: int, instance: Sequence[int], distribution: Discrete,
                    init_demands: Dict[int, int], name: str = 'cg_replan', consider_opened_bins=True):
        super().__init__(capacity, instance, name)
        self.demands = init_demands
        self.distribution = distribution
        self.history_plan = {}
        self.history_demands = {}
        self.bins: List[BinWithPattern] = []
        self.consider_opened_bins = consider_opened_bins
        
    def build(self) -> Any:
        cg = ColumnGeneration(self.capacity, self.demands)
        result = cg.solve()
        if result is not None:
            self.plan = BppPlan(result, self.capacity)       



    def _replan(self) -> BppPlan | None:
        estimator = generate_discrete_demand_estimator(consider_opened_bins=self.consider_opened_bins)
        self.demands = estimator(self.distribution, self.remain_count, 
                                    [b for b in self.bins if not b.full])
        cg = ColumnGeneration(self.capacity, self.demands)
        result = cg.solve()
        if result is not None:
            return BppPlan(result, self.capacity)
        else:
            return None

    def _check_plan(self):
        # only run after trigger replan with considering the opened bins
        opened_bins = [b for b in self.bins if not b.full]
        for b in opened_bins:
            if b.pattern in self.plan and self.plan[b.pattern] > 1:
                self.plan[b.pattern] -= 1
    



    def solve(self) -> Tuple[Solution, Dict | None]:
        assert self.plan is not None
        # self.bins: List[BinWithPattern] = []
        self.plan_executor = BinPlanExecutor(self.plan, self.capacity, self.bins)
        self.remain_count = len(self.instance)
        for i, item in enumerate(self.instance):
            self.remain_count -= 1

            try:
                self.plan_executor.put(item, None)
            except OutOfPlanException:



                result = self._replan()
                if result is None:
                    self.plan_executor.put(item, best_fit_choice)
                else:
                    # record new plan
                    self.history_plan.add(self.plan)
                    self.history_demands.add(self.demands)

                    self.plan = BppPlan(result, self.capacity)
                    if self.consider_opened_bins:
                        self._check_plan()
                    self.plan_executor.plan = self.plan
                    self.plan_executor.put(item, None)

        return BinSolution(self.capacity, self.bins), {"history_plan": self.history_plan}

