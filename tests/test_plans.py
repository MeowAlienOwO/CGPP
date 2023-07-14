from typing import List
import pytest
from bpp1d.structure.bin_pattern import BinPattern
from bpp1d.structure.bpp_plan import BinPlanExecutor, BppPlan
from bpp1d.utils.anyfit import best_fit_choice
import random


TEST_CAPACITY = 10

TEST_CASES_PERFECT = [
    {
        "capacity": 10,
        "instance": [5, 4, 4, 3, 2, 2],
        "patterns": [(5, 3, 2), (4, 4, 2)],
        "plan": {
            (5, 3, 2): 1,
            (4, 4, 2): 1
        }

    }, 
    {
        "capacity": 10,
        "instance": [5, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2],
        "patterns": [(5, 3, 2), (4, 3, 3), (2, 2, 2, 2, 2)],
        "plan": {
            (5, 3, 2): 1,
            (4, 3, 3): 2,
            (2, 2, 2, 2, 2): 1
        }
    }
]


TEST_CASES_NOT_PERFECT = [
    {
        "capacity": 10,
        "instance": [5, 4, 4, 3, 2, 2, 2, 3],
        "patterns": [(5, 3, 2), (4, 4, 2)],
        "plan": {
            (5, 3, 2): 1,
            (4, 4, 2): 1
        },
        "expected": [[5, 3, 2], [4, 4, 2], [2, 3]]

    }, 

    {
        "capacity": 10,
        "instance": [5, 4, 2, 3, 4, 3, 2, 2,],
        "patterns": [(5, 3, 2), (4, 4, 2)],
        "plan": {
            (5, 3, 2): 1,
            (4, 4, 2): 1
        },
        "expected": [[5, 2, 3], [4, 4, 2], [3, 2]]

    }, 
]


def _plan_creation(plan_dict: dict) -> BppPlan:
    return BppPlan({ BinPattern(k): v for k, v in plan_dict.items()}, TEST_CAPACITY)

@pytest.mark.parametrize(
    ('plan', 'items'), [
        (_plan_creation(test_case["plan"]), test_case['instance'])
        for test_case in TEST_CASES_PERFECT
    ]
    
)
def test_plan_perfect(plan: BppPlan, items: List[int]):
    random.shuffle(items)
    executor = BinPlanExecutor(plan, TEST_CAPACITY, [])
    
    for i in items:
        executor.put(i)
    
    for bin in executor.bins:
        _, ldiff, rdiff = bin.check()
        assert ldiff == [] and rdiff == []


@pytest.mark.parametrize(
    ('plan', 'items', 'expected'), [
        (_plan_creation(test_case["plan"]), test_case['instance'], test_case['expected'])
        for test_case in TEST_CASES_NOT_PERFECT
    ]
    
)
def test_plan_not_perfect(plan: BppPlan, items: List[int], expected: List[List[int]]):
    executor = BinPlanExecutor(plan, TEST_CAPACITY, [])

    for i in items:
        executor.put(i, best_fit_choice)

    assert len(executor.bins) == len(expected)

    
    for bin, exp in zip(executor.bins, expected):
        assert bin.items == exp
