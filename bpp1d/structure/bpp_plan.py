from typing import  Dict, List, Tuple, Sequence
import json
import numpy as np
import math
from .bpp_bin import BppBin, BinWithPattern
from .bin_pattern import BinPattern

from bpp1d.utils.heuristic_choice import HeuristicChoiceFn, best_fit_choice



class BppPlan:
    def __init__(self, pattern_dict:Dict[BinPattern, int], capacity: int) -> None:
        self.capacity = capacity
        self.plan_dict = pattern_dict.copy()


    def __setitem__(self, key: Tuple | BinPattern, val: int) -> None:
        if not isinstance(key, BinPattern):
            key = BinPattern(key)
        self.plan_dict[key] = val

    def __getitem__(self, key) -> int:
        return self.plan_dict[key]

    def __repr__(self):
        return self.plan_dict.__repr__()

    def __contains__(self, key):
        return key in self.plan_dict.keys()
    

    def to_np(self, item_order):
        pattern_arrs = np.stack([p.to_np(item_order) for p in self.plan_dict])

        pattern_counts = np.array(list(self.plan_dict.values()))
        return pattern_arrs, pattern_counts

    def to_json(self) -> str:
        return json.dumps({
            k.items: v
            for k, v in self.plan_dict.items()
        })

    def copy(self) -> 'BppPlan':
        return BppPlan(self.plan_dict, self.capacity)

    def keys(self):
        return self.plan_dict.keys()

    def values(self):
        return self.plan_dict.values()

    def items(self):
        return self.plan_dict.items()
    
    def metrics(self):
        return {
            'exp_fill_rate': sum([self.plan_dict[k] 
                                        for k in self.keys() if sum(k.items) == self.capacity ]) / sum(self.values()),
            'exp_bins': sum(self.values()),
        }
    def check_plan_execution(self, bins: Sequence[BppBin]) -> float:
        res = {}
        for b in bins:
            pattern = tuple(sorted(b.items))
            if pattern in self.plan_dict:
                res[pattern] = res.get(pattern, 0) + 1
        
        for k, v in self.items():
            res[k] = res.get(k, 0) - v 
        
        return sum([v for v in res.values() if v > 0])

    
    def useone(self, pattern: BinPattern) -> BinWithPattern:
        """use one of pattern, this will alter the plan dictionary
        Args:
            pattern (BinPattern): pattern to be used

        Raises:
            ValueError: If pattern not in plan or remained usage is empty

        Returns:
            BinWithPattern : Create an empty bin with pattern, 
        """

        if pattern in self.plan_dict and self.plan_dict[pattern] > 0:
            self.plan_dict[pattern] -= 1
            return BinWithPattern(self.capacity, pattern)
        else:
            raise ValueError("Pattern not in plan")

    def match(self, item: int) -> BinPattern | None:
        """match a pattern in the plan, returns the pattern with max remain count

        Args:
            item (int): item to be searched

        Returns:
            BinPattern | None: target pattern, None if no such pattern in plan
        """

        candidates = {k: v for k, v in self.plan_dict.items() if item in k}
        if not candidates or all(v == 0 for v in candidates.values()):
            return None
        
        # mypy compatible method
        return max(candidates, key=lambda k: candidates[k])

    
def new_bin_with_pattern_callback(capacity: int, pattern: BinPattern):
    return BinWithPattern(capacity, pattern)

class OutOfPlanException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class BinPlanExecutor:
    def __init__(self, plan: BppPlan, capacity: int, # demands: Dict[int, int],
                    bins:List[BinWithPattern] | None=None, 
                    shall_rebalance = True,
                    balance_empty_threshold=0.2,
                    balance_k_bins = 200) -> None:
        self.bins = bins if bins is not None else []
        self.capacity = capacity
        self.plan = plan
        # self.demands = demands
        self.extra_demands = {} # stores items that been replaced out
        self.balance_empty_threshold = balance_empty_threshold
        self.balance_k_bins = balance_k_bins
        self.shall_rebalance = shall_rebalance
        self.fallback_count = 0
        # self.empty_slots = {}



    @property
    def plan(self) -> BppPlan:
        return self._plan

    @plan.setter
    def plan(self, value: BppPlan):
        self._plan = value.copy()

    def is_fit(self, item: int) -> bool:
        for pattern, count in self.plan.items():
            if item in pattern and count > 0:
                return True
        
        check_bin_fit = [b.check()[1] for b in self.bins]
        any_bin_fit = any(item in c for c in check_bin_fit)
        return any_bin_fit

    def heuristic_put(self, item: int, fallback: HeuristicChoiceFn) -> int:
        choice = fallback(item, self.bins)

        if choice < 0:
            self.bins.append(BinWithPattern(self.capacity, pattern=BinPattern([item, self.capacity-item]), items=[item]))
            return choice
        else:
            # adjust pattern and record replacement
            target_bin = self.bins[choice]
            # find the replacement of patterns with current items
            packed, non_packed, _ = target_bin.check()
            non_packed = sorted(non_packed)

            replacement_idx = 0
            for i in range(len(non_packed)):
                if sum(non_packed[:i]) >= item:
                    replacement_idx = i
                    break
            replacement = non_packed[:replacement_idx]
            new_pattern = packed + [item] + non_packed[replacement_idx:]
            if sum(new_pattern) < target_bin.capacity:
                new_pattern.append(target_bin.capacity - sum(new_pattern))
            target_bin.pattern = BinPattern(new_pattern)
            # update extra demands, so that in next plan we will consider 
            # the items that involved in current plan
            for i in replacement:
                self.extra_demands[i] = self.extra_demands.get(i, 0)+ 1
                # self.empty_slots[i] -= 1

            target_bin.pack(item)
            return choice       
        


    def put(self, item: int, fallback: HeuristicChoiceFn | None = None) -> int:
        matched_bins = [b for b in self.bins if item in b.check()[1] and b.empty_space >= item]
        # number of bins whose filled space less than threshold
        # num_nonfill_bins = len([ b for b in self.bins 
                                # if b.filled_space / b.capacity < self.balance_empty_threshold])
        

        if matched_bins:
            choice = self.bins.index(matched_bins[0])
            matched_bins[0].pack(item)
            # self.empty_slots[item] -= 1
            return choice
        # elif (num_nonfill_bins >= self.balance_k_bins 
        #         and self.shall_rebalance):
        #     # force fill bins 
        #     choice = self.heuristic_put(item, fallback if fallback is not None else best_fit_choice)
        #     self.fallback_count += 1
        #     return choice
        else:
            pattern = self.plan.match(item)
            if pattern is not None:
                # find a pattern can be put into plan
                newbin = self.plan.useone(pattern)
                newbin.pack(item)
                # _, _, empty_items = newbin.check()
                # for i in empty_items:
                #     self.empty_slots[i] = self.empty_slots.get(i, 0) + 1
                self.bins.append(newbin)
                choice = -1
                return choice

            else:

                if fallback is None:
                    raise OutOfPlanException(f"item: {item} plan:{str(self._plan)}")
                else:
                    choice = self.heuristic_put(item, fallback)
                    self.fallback_count += 1
                    return choice
