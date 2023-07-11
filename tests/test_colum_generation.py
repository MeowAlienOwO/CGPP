import pytest
from bpp1d.structure import BinPattern
from bpp1d.models.mip import ColumnGeneration, SolutionStatus

TEST_CASES = [
    
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

@pytest.mark.parametrize(
    ('instance', 'capacity', 'expected_plan'), [
        (
            [5, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2],
            10,
            {
                BinPattern((5, 3, 2)): 1,
                BinPattern((4, 3, 3)): 2,
                BinPattern((2, 2, 2, 2, 2)): 1
            }
        )

    ]
)
def test_column_generation(instance, capacity, expected_plan):

    items = sorted(set(instance))

    demands = {i: instance.count(i) for i in items}

    cg = ColumnGeneration(capacity, demands)
    plan = cg.solve()
    assert plan is not None
    assert cg.status == SolutionStatus.FINISHED
    # patterns = [key for key in expected_plan.keys()]

    plan_demand = {}
    for pattern, p_count in plan.items():
        for item in items:
            pattern_dict = pattern.as_dict()
            plan_demand[item] = plan_demand.get(item, 0) + pattern_dict.get(item, 0) * p_count

    for i in items:
        assert plan_demand[i] >= demands[i]

    # for key in result:
    #     assert key in patterns
    #     assert result[key] == expected_plan[key]

