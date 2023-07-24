from typing import Sequence
from .solution import Solution
import json

from bpp1d.structure.bpp_bin import BppBin




class BinSolution(Solution):
    def __init__(self,capacity: int, bins:Sequence[BppBin] | None = None) -> None:
        self.bins = [b for b in bins if not b.empty] if bins is not None else []
        self.capacity = capacity

    def __getitem__(self, key):
        if isinstance(key, slice):
            return BinSolution(self.bins[key.start:key.stop], self.capacity)
        else:
            return self.bins[key]

    def __len__(self):
        return len(self.bins)
        
    def __repr__(self):
        repr_str = "\n".join(["{}: {}".format(k, v) for k, v in self.metrics.items()])
        return repr_str

    def show_exact(self):
        all_str = self.__repr__() + "\n" \
                        + "\n".join([str(b) + f"({b.filled_space}, {b.empty_space})" 
                                        for b in sorted(self.bins, 
                                                        key=lambda b: b.empty_space, reverse=True)])
        return all_str
    
    @property
    def metrics(self):
        return {
            'capacity': int(self.capacity),
            'item_count': int(self.total_items),
            'bins': self.num_bins,
            'waste': int(self.waste),
            'filled_rate': self.num_filled_bins / self.num_bins,
        }
    @property
    def waste(self):
        return sum([b.empty_space for b in self.bins])

    @property
    def num_filled_bins(self) -> int:
        return len([b for b in self.bins if b.filled_space == b.capacity])


    @property
    def num_bins(self) -> int:
        return len(self.bins)

    @property 
    def total_items(self):
        return sum([len(b) for b in self.bins])


    def write(self, file_path):
        lines = []
        lines.append(str(self.capacity))
        for bin in self.bins:
            lines.append(str(bin.items))
        
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')
    
    @property
    def min_filled_bin(self):
        return min(self.bins, key=lambda b: b.filled_space)

    @property
    def max_filled_bin(self):
        return max(self.bins, key=lambda b: b.filled_space)

    def to_json(self) -> str:
        return json.dumps({
            "metrics": self.metrics,
            "bins": self.bins
        })
