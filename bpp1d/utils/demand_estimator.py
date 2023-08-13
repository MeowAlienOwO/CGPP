
from typing import Callable, Dict, Sequence

from bpp1d.structure.bpp_bin import BinWithPattern
import math

from bpp3d_dataset.utils.distributions import Discrete

# DemandEstimator = Callable[[Discrete, int, Sequence[BinWithPattern]], Dict[int, int]]


class DiscreteDemandEstimator:
    def __init__(self, consider_opened_bins: bool = False, smooth=False):
        self.consider_opened_bins = consider_opened_bins
        self.smooth=smooth

    def __call__(self, distribution: Discrete, remain_items: int, 
                    opened_bins: Sequence[BinWithPattern]) -> Dict[int, int]:
        if self.consider_opened_bins:
            return _opened_bins_estimator(distribution, remain_items, opened_bins, self.smooth)
        else:
            return _simple_estimator(distribution, remain_items, opened_bins, self.smooth)


def generate_discrete_demand_estimator(consider_opened_bins: bool = False, smooth=False):
    return DiscreteDemandEstimator(consider_opened_bins, smooth)
    
def _opened_bins_estimator(distribution: Discrete, remain_items: int, 
                            opened_bins: Sequence[BinWithPattern], smooth: bool=False) -> Dict[int, int]:
    if smooth:
        estimate_remains = {i: max(math.ceil(remain_items * p), 1) for i, p in distribution.prob_dict.items()}
    else:
        estimate_remains = {i: math.ceil(remain_items * p) for i, p in distribution.prob_dict.items()}
    for b in opened_bins:
        # for item in b:
        #     estimate_remains[item] = estimate_remains.get(item, 0) + 1
        _, _, nonfilled = b.check()
        # nonfilled_dict = { for i in set(nonfilled)}
        for i in nonfilled:
            estimate_remains[i] = max(estimate_remains[i] - 1, 0)

        
    return estimate_remains

def _simple_estimator(distribution: Discrete, remain_items: int, 
                        opened_bins: Sequence[BinWithPattern], smooth: bool=False) -> Dict[int, int]:
    if smooth:
        estimate_remains = {i: max(math.ceil(remain_items * p), 1) for i, p in distribution.prob_dict.items()}
    else:
        estimate_remains = {i: math.ceil(remain_items * p) for i, p in distribution.prob_dict.items()}
    return estimate_remains

