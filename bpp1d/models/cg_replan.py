from typing import Any, Dict, List, Sequence, Tuple
# from bpp1d.models.mip.column_generation import ColumnGeneration
from bpp1d.models.mip.column_generation_scipy import CGScipy
from bpp1d.models.model import Model, ModelStatus
from bpp1d.structure.bin_solution import BinSolution
from bpp1d.structure.bpp_bin import BinWithPattern
from bpp1d.structure.bpp_plan import BinPlanExecutor, BppPlan, OutOfPlanException
from bpp1d.structure.solution import Solution
from bpp1d.utils.heuristic_choice import best_fit_choice
from bpp1d.utils.demand_estimator import generate_discrete_demand_estimator
from bpp3d_dataset.utils.distributions import Discrete



class CGReplan(Model):
    def __init__(self, capacity: int, instance: Sequence[int], distribution: Discrete,
                        consider_opened_bins=False, 
                        shall_rebalance = True,
                        name='cg_replan', end_heuristic_theshold: float =0.1):
        super().__init__(capacity, instance, name)
        self.distribution = distribution
        self.history_plan: Dict[int, BppPlan] = {}
        self.history_demands:Dict[int, Dict[int, int]] = {}
        self.bins: List[BinWithPattern] = []
        self.consider_opened_bins = consider_opened_bins
        self.estimator = generate_discrete_demand_estimator(consider_opened_bins)
        self.demands = self.estimator(self.distribution, len(instance), [])
        self.replan_count = 0
        self.end_heuristic_theshold = end_heuristic_theshold
        self.shall_rebalance = shall_rebalance
        
    def build(self) -> Any:
        # cg = ColumnGeneration(self.capacity, self.demands)
        cg = CGScipy(self.capacity, self.demands)
        result = cg.solve()
        if result is not None:
            self.plan = BppPlan(result, self.capacity)       

        return super().build()


    def _replan(self) -> BppPlan | None:
        self.replan_count += 1
        # print(f"replanning, count{self.replan_count}" )
        self.demands = self.estimator(self.distribution, self.remain_count, 
                                    [b for b in self.bins if not b.full])
        # cg = ColumnGeneration(self.capacity, self.demands)
        for i, d in self.plan_executor.extra_demands.items():
            self.demands[i] = max(self.demands.get(i, 0), d)
        self.plan_executor.extra_demands = {}
        cg = CGScipy(self.capacity, self.demands)
        result = cg.solve()

        # print("replan finished")
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
        self.plan_executor = BinPlanExecutor(self.plan, self.capacity, self.bins, shall_rebalance= self.shall_rebalance)
        self.remain_count = len(self.instance)
        self.status = ModelStatus.SOLVING
        for i, item in enumerate(self.instance):
            # print(f'step {i}')

            try:
                self.plan_executor.put(item, None)

            except OutOfPlanException:

                result = self._replan()
                if (result is None
                    or 1 - i / len(self.instance) < self.end_heuristic_theshold):
                    self.plan_executor.put(item, best_fit_choice)
                else:
                    # record new plan
                    self.history_plan[i] = self.plan.copy()
                    self.history_demands[i] = self.demands.copy()

                    # self.plan = BppPlan(result, self.capacity)
                    # print(self.demands)
                    self.plan = result
                    if self.consider_opened_bins:
                        self._check_plan()
                    self.plan_executor.plan = self.plan
                    self.plan_executor.put(item, None)

            self.remain_count -= 1

        self.status = ModelStatus.FINISHED

        return BinSolution(self.capacity, self.bins), {"replan_count": self.replan_count}

