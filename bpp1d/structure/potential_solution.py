import json

from typing import List
from .solution import Solution


class PotentialSolution(Solution):
    def __init__(self, capacity: int, potential: List[int]) -> None:
        assert len(potential) == capacity + 1
        self.potential = potential
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
    def metrics(self):
        return {
            'capacity': int(self.capacity),
            'bins': self.__len__(),
            'waste': int(self.waste)
        }

    def __repr__(self):
        return self.potential.__repr__()


    def to_json(self) -> str:
        return json.dumps({
            "metrics": self.metrics(),
            "potential": self.potential
        })

