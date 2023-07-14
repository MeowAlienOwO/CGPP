
from typing import Callable, Dict, List, Sequence
from bpp1d.structure.bin_pattern import BinPattern

from bpp1d.structure.bpp_bin import BinWithPattern
import math

from bpp3d_dataset.utils.distributions import Discrete

DemandEstimator = Callable[[Discrete, int, Sequence[BinWithPattern]], Dict[int, int]]

def generate_discrete_demand_estimator(consider_opened_bins: bool = False):
    if consider_opened_bins:
        return _opened_bins_estimator
    else:
        return _simple_estimator

    
def _opened_bins_estimator(distribution: Discrete, remain_items: int, 
                            opened_bins: Sequence[BinWithPattern]) -> Dict[int, int]:
    estimate_remains = {i: math.ceil(remain_items * p) for i, p in distribution.prob_dict.items()}
    for b in opened_bins:
        for item in b:
            estimate_remains[item] = estimate_remains.get(item, 0) + 1
    return estimate_remains

def _simple_estimator(distribution: Discrete, remain_items: int, 
                        opened_bins: Sequence[BinWithPattern]) -> Dict[int, int]:
    estimate_remains = {i: math.ceil(remain_items * p) for i, p in distribution.prob_dict.items()}
    return estimate_remains

