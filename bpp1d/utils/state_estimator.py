
from typing import Any, Callable, Sequence
from bpp3d_dataset.utils.distributions import Discrete
import numpy as np

def estimate_distribution(sequence: Sequence[int], smooth=True) -> Discrete:
    items = sorted(list(set(sequence)))
    if smooth:
        probs = [max(sequence.count(i), 1) / len(sequence) for i in items]
    else:
        probs = [sequence.count(i) / len(sequence) for i in items]
    return Discrete(probs, items)


def kl_divergence(d1: Discrete, d2: Discrete) -> float:
    items = sorted(list(set().union(d1.items, d2.items)))
    return sum([d1.p(t) * np.log((d1.p(t) + 1e-4) / (d2.p(t) + 1e-4))  for t in items])


class StateEstimator:

    def __init__(self, priori: Discrete):
        self.priori = priori

    @property
    def distribution(self):
        return self.priori

    def fit_estimation(self, sequence: Sequence[int]) -> bool:
        raise NotImplementedError

        
    def estimate(self, sequence: Sequence[int]) -> Discrete:
        raise NotImplementedError

    def update_priori(self, dist: Discrete):
        self.priori = dist

        
    def __call__(self, *args: Any, **kwargs: Any) -> Discrete:
        return self.estimate(*args, **kwargs)


class SimpleStateEstimator(StateEstimator):
    def __init__(self, priori: Discrete, kl_theshold:float = 0.7):
        super().__init__(priori)
        self.kl_theshold = kl_theshold

    def fit_estimation(self, sequence: Sequence[int]) -> bool:
        estimated = estimate_distribution(sequence)
        return kl_divergence(self.priori, estimated) < self.kl_theshold

    def estimate(self, sequence: Sequence[int]) -> Discrete:
        return estimate_distribution(sequence)
