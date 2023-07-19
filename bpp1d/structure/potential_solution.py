import json

from typing import List
from .solution import Solution


class PotentialSolution(Solution):
    def __init__(self, capacity: int, potential: List[int], filled_bins:int) -> None:
        assert len(potential) == capacity
        self.potential = potential
        self.filled_bins = filled_bins
        self.capacity = capacity

    def __len__(self):
        """return total bins
        """
        return sum(self.potential)

    def __getitem__(self, key: object):
        if isinstance(key, int) and key >= 0 and key <= self.capacity:
            return self.potential[key]
        else:
            return NotImplemented

    @property
    def waste(self):
        return sum([(self.capacity - idx) * num for idx, num in enumerate(self.potential[:-1])])
    @property
    def num_bins(self):
        return sum(self.potential) + self.filled_bins

    @property
    def metrics(self):
        return {
            'capacity': int(self.capacity),
            'bins': self.num_bins,
            'waste': int(self.waste),
            'filled_rate': self.filled_bins / self.num_bins
        }

    def __repr__(self):
        return self.potential.__repr__()


    def to_json(self) -> str:
        return json.dumps({
            "metrics": self.metrics,
            "potential": self.potential
        })

