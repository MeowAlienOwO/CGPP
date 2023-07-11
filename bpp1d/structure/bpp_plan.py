from typing import  Dict, List, Tuple
import json
import numpy as np
from .bpp_bin import BinWithPattern
from .bin_pattern import BinPattern
from bpp1d.utils.anyfit import HeuristicChoiceFn, best_fit_choice

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

class BinPlanExecutor:
    def __init__(self, plan: BppPlan, capacity: int, bins:List[BinWithPattern] | None=None) -> None:
        self.bins = bins if bins is not None else []
        self.capacity = capacity
        self.plan = plan.copy()
        # self.force_match_pattern = force_match_pattern
    
    def put(self, item: int, fallback: HeuristicChoiceFn | None = None) -> int:
        matched_bins = [b for b in self.bins if item in b.check()[1] and b.empty_space >= item]
        if matched_bins:
            choice = self.bins.index(matched_bins[0])
            matched_bins[0].pack(item)
            return choice
        else:
            pattern = self.plan.match(item)
            if pattern is not None:
                newbin = self.plan.useone(pattern)
                newbin.pack(item)
                self.bins.append(newbin)
                choice = -1
                return choice
            else:
                fallback = fallback if fallback is not None else best_fit_choice
                choice = fallback(item, self.bins)
                if choice < 0:
                    self.bins.append(BinWithPattern(self.capacity, pattern=BinPattern([item]), items=[item]))
                else:
                    self.bins[choice].pack(item)
                return choice