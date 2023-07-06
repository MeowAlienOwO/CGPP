import json
from bpp1d.structure import BinPattern, BppBin, BinWithPattern
import pytest

TEST_CASES = [
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

TEST_CAPACITY = 10



def test_patterns_creation():
    for case in TEST_CASES[:2]:
        patterns = [BinPattern(p) for p in case["patterns"]]
        assert patterns[0] == BinPattern((5, 3, 2))
        assert patterns[0] == BinPattern((5, 2, 3))


def test_bin_creation():
    bppbin = BppBin(TEST_CAPACITY)

    bppbin.pack(1)
    assert bppbin.filled_space == 1

    bppbin.append(3)
    assert bppbin.filled_space == 4

    bppbin.pack(5)
    assert bppbin.items == [1, 3, 5]

    with pytest.raises(ValueError):
        bppbin.pack(3)

    assert bppbin.to_json() == json.dumps(bppbin.items)

@pytest.mark.parametrize(
    ('pattern', 'items', 'expected'), [
        (BinPattern((4, 3, 2, 1)), [4], ([4], [1, 2, 3], [])),
        (BinPattern((5, 3, 2)), [5], ([5], [2, 3], [])),
        (BinPattern((4, 3, 2, 1)), [4, 5], ([4], [1, 2, 3], [5])),
    ]
)
def test_bin_pattern(pattern, items, expected):
    bppbin = BinWithPattern(TEST_CAPACITY, pattern)
    for i in items:
        bppbin.pack(i)

    # both bin and items are sorted so order matters
    assert bppbin.check() == expected




