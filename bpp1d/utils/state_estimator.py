
from typing import Any, Callable, Sequence
from bpp3d_dataset.utils.distributions import Discrete
from sklearn.neighbors import KernelDensity
from scipy.special import softmax
import numpy as np


def kl_divergence(d1: Discrete, d2: Discrete) -> float:
    items = sorted(list(set().union(d1.items, d2.items)))
    return sum([d1.p(t) * np.log((d1.p(t) + 1e-4) / (d2.p(t) + 1e-4))  for t in items])


class StateEstimator:

    def __init__(self, priori: Discrete):
        self.priori = priori

    @property
    def distribution(self):
        return self.priori

    def is_fit(self, sequence: Sequence[int]) -> bool:
        raise NotImplementedError

        
    def estimate(self, sequence: Sequence[int]) -> Discrete:
        weights = [sequence.count(i) for i in self.priori.items]
        probs = [w / sum(weights) for w in weights]
        return Discrete(probs, self.priori.items)

    def update_priori(self, sequence: Sequence[int]):
        self.priori = self.estimate(sequence)

    def __call__(self, *args: Any, **kwargs: Any) -> Discrete:
        return self.estimate(*args, **kwargs)


class SimpleStateEstimator(StateEstimator):
    def __init__(self, priori: Discrete, kl_theshold:float = 0.7, smooth: bool=True):
        super().__init__(priori)
        self.kl_theshold = kl_theshold
        self.smooth = smooth

    def is_fit(self, sequence: Sequence[int]) -> bool:
        estimated = self.estimate(sequence)
        return kl_divergence(self.priori, estimated) < self.kl_theshold

    def estimate(self, sequence: Sequence[int]) -> Discrete:

        if self.smooth:
            weights = [max(sequence.count(i), 1)  for i in self.priori.items]
        else:
            weights = [sequence.count(i) for i in self.priori.items]

        probs = [w / sum(weights) for w in weights]
        return Discrete(probs, self.priori.items)

    def update_priori(self, sequence: Sequence[int]):
        """Smoothed update"""
        if self.smooth:
            # weights = [dist.p(i) * len(self.priori.items) for i in self.priori.items]
            # probs = [(dist.p(i) + 0.01) / (len(self.priori.items)*0.01 +1) 
                        # for i in self.priori.items]
            weights = [max(sequence.count(i), 1) for i in self.priori.items]

            probs = [w / sum(weights) for w in weights]
            self.priori = Discrete(probs , self.priori.items)
        else:
            return super().update_priori(sequence)

class KernelDensityEstimator(StateEstimator):
    def __init__(self, priori: Discrete, kl_theshold:float = 0.7, memorize_all=True):
        super().__init__(priori)
        # self.bandwidth = (max(self.priori.items) - min(self.priori.items)) / (len(self.priori.items) * 2)
        self.bandwidth = (max(self.priori.items) - min(self.priori.items)) / (4)
        self.memory = []
        # print(self.bandwidth, max(self.priori.items), min(self.priori.items))
        self.memorize_all = memorize_all
        self.kl_theshold = kl_theshold
        self.model = KernelDensity(bandwidth=self.bandwidth,kernel='gaussian') # assume gaussian
        

    def is_fit(self, sequence: Sequence[int]) -> bool:
        estimated = self.estimate(sequence)
        # print(kl_divergence(self.priori, estimated))
        return kl_divergence(self.priori, estimated) < self.kl_theshold
        
    def estimate(self, sequence: Sequence[int]) -> Discrete:
        # self.model.fit(np.asarray(sequence).reshape((-1, 1)))
        # logits = self.model.score_samples(np.asarray(self.priori.items).reshape((-1, 1)))
        # # probs = softmax(logits)
        if self.memorize_all:

            return super().estimate(self.memory + list(sequence)) # 
        else:
            return super().estimate(list(sequence)) # 
        
        # return Discrete(probs, self.priori.items)

    def update_priori(self, sequence: Sequence[int]):
        self.model.fit(np.asarray(sequence).reshape((-1, 1)))
        if self.memorize_all:
            self.memory += list(sequence)
        logits = self.model.score_samples(np.asarray(self.priori.items).reshape((-1, 1)))
        # probs = np.exp(probs).tolist()
        probs = softmax(logits)
        self.priori = Discrete(probs, self.priori.items)







# class BayesianStateEstimator(StateEstimator):
#     def __init__(self, priori: Discrete, kl_theshold:float = 0.7, smooth: bool=False):
#         super().__init__(priori)
#         self.kl_theshold = kl_theshold
#         self.smooth = smooth

#     def fit(self, sequence: Sequence[int]) -> bool:
#         estimated = self.estimate(sequence)
#         return kl_divergence(self.priori, estimated) < self.kl_theshold



#     def update_priori(self, dist: Sequence[int]):
#         # return super().update_priori(dist)
    