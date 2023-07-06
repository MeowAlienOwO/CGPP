from typing import List, Sequence
import pytest

from bpp1d.models import HeuristicModel

TEST_CASES_BF = [
    {
        "capacity": 10,
        "instance": [5, 4, 4, 3, 2, 2],
        "expected": [[5,4], [4,3,2], [2]]
    }, 
    {
        "capacity": 10,
        "instance": [5, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2],
        "expected": [[5,4], [4,3,3], [3, 3, 3], [2,2,2,2,2,], [2]]
    }
]

@pytest.mark.parametrize(
    ('capacity', 'instance', 'expected'), [
        (test_case['capacity'], test_case['instance'], test_case['expected'])
        for test_case in TEST_CASES_BF
    ]
    
)
def test_best_fit(capacity: int, instance: Sequence[int], expected: List[List[int]]):
    model = HeuristicModel(capacity)
    solution, _ = model.solve(instance)

    assert len(solution.bins) == len(expected)
    for bin, exp in zip(solution.bins, expected):
        assert bin.items == exp
