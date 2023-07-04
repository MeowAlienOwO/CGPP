from typing import  Dict, List, Tuple
import numpy as np
from .bpp_bin import BppBin
from .bpp_pattern import BinPattern

class BinPlan:
    def __init__(self, pattern_dict:Dict[BinPattern, int]) -> None:
        self.plan_dict = pattern_dict.copy()


    def __setitem__(self, key, val) -> None:
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

    def copy(self):
        return BinPlan(self.plan_dict)

    def keys(self):
        return self.plan_dict.keys()

    def values(self):
        return self.plan_dict.values()


class BinPlanExecutor:
    def __init__(self, plan, capacity, bins_with_pattern:List=[], force_match_pattern=False) -> None:
        self.bins_with_pattern:List[Tuple[BppBin, BinPattern]] = bins_with_pattern
        self.capacity = capacity
        self.plan_dict = plan.plan_dict
        self.force_match_pattern = force_match_pattern

    def put(self, item):
        def _put_new_bin(item, pattern):
            self.bins_with_pattern.append((BppBin(self.capacity, [item]), pattern))
        def _put_exist_bin(item, bin):
            bin.append(item)

        # bins with matched empty slot and not matched but empty slot
        matched_available_bins, nonmatched_available_bins = (
            [(b, p) for b, p in self.bins_with_pattern if item in p.comp(b)[1] and b.empty_space >= item],
            [(b, p) for b, p in self.bins_with_pattern if item not in p.comp(b)[1] and b.empty_space >= item]
        )
        # print(item, matched_available_bins, nonmatched_available_bins)
        if matched_available_bins:
            # print("match available bin")
            b, _ = matched_available_bins[0]
            _put_exist_bin(item, b)
        elif nonmatched_available_bins:
            pattern, matched = self.match_pattern_in_plan(item)
            if matched:
                _put_new_bin(item, pattern)
                # print("match available pattern")
                self.plan_dict[pattern] -= 1
            elif self.force_match_pattern:
                # print("put into dummy")
                _put_new_bin(item, pattern)
                self.plan_dict[pattern] = 0
            else:
                # print("put by anyfit")
                b, _ = nonmatched_available_bins[0]

                _put_exist_bin(item, b)
        else:
            pattern, matched = self.match_pattern_in_plan(item)
            if matched:
                # print("match available pattern")
                self.plan_dict[pattern] -= 1
            else:
                # add new dummy and minus it
                # print("put into dummy")
                self.plan_dict[pattern] = 0
            _put_new_bin(item, pattern)
        # print(item, self.bins_with_pattern)

    def match_pattern_in_plan(self, item):
        available_patterns = [pattern for pattern in self.plan_dict if item in pattern and self.plan_dict[pattern] > 0]
        if available_patterns:
            return max(available_patterns, key=lambda x: self.plan_dict[x]), True
        else:
            # no matching pattern
            # dummy pattern: (0,0,..item=1, 0, 0,...)
            dummy_pattern = BinPattern((item, ))
            return dummy_pattern, False