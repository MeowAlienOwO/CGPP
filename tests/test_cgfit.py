from typing import Dict, Sequence
import pytest

from bpp1d.models.cg_fit import CGFit

TEST_CASES = [
    {
        "capacity": 10,
        "instance": [5, 4, 4, 3, 2, 2],
        "demands":{
            5: 1,
            4: 2, 
            2: 1
        }
    }, 
    {
        "capacity": 10,
        "instance": [5, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2],
        "demands": {
            5:1,
            4:2,
            3:4,
            2:5
        }
    }
]


@pytest.mark.parametrize(
    ('capacity', 'instance', 'demands'), [
        (test_case['capacity'], test_case['instance'], test_case['demands'])
        for test_case in TEST_CASES
    ]
    
)
def test_cg_fit(capacity: int, instance: Sequence[int], demands: Dict[int, int]):
    model = CGFit(capacity, instance, demands)
    model.build()
    assert model.plan is not None
    solution, _ = model.solve()
    assert solution.total_items == len(instance)





