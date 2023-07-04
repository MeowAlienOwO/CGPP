from bpp1d.solutions import BinPattern

# see the paper
TEST_CASES = [
    {
        "capacity": 10,
        "instance": [5, 4, 4, 3, 2, 2],
        "patterns": [(5, 3, 2), (4, 4, 2)]

    }, 
    {
        "capacity": 10,
        "instance": [5, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2],
        "patterns": [(5, 3, 2), (4, 3, 3), (2, 2, 2, 2, 2)]
    }
    
    
]


def test_patterns():
    for case in TEST_CASES:
        patterns = [BinPattern(p) for p in case["patterns"]]
        assert patterns[0] == BinPattern((5, 3,2))


