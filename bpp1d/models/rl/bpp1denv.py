from typing import Callable, List, Sequence
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from bpp3d_dataset.utils.distributions import Discrete


BIG_NEG_REWARD = -100
BIG_POS_REWARD = 10

class Bpp1DPotentialEnv(gym.Env):
    """A 1d bin packing environment wrapper
    follows or-gym formulation, define the action space to be a potential table
    """
    def __init__(self, instance: Sequence[int], capacity: int, items: List[int],
                    reset_distribution: Discrete | None = None):  
        self.instance = instance
        self.capacity = capacity
        self.reset_distribution = reset_distribution

        self.action_space = spaces.Discrete(self.capacity)
        # masked
        self.observation_space = spaces.Dict({
            "mask": spaces.Box(0, 1, shape = (self.capacity,), dtype=np.uint8), 
            "state": spaces.Box(
                low=np.array([0] * (self.capacity + 1)),
                high=np.array([len(instance)] + [max(items)]),
                dtype=np.int32
            )
        })

    @property
    def is_random_reset(self) -> bool:
        return self.reset_distribution is not None

    def reset(self):
        self.step_count = 0
        self.filled_bins = 0
        self.total_waste = 0
        self.total_reward = 0
        self.decision_sequence = []
        self.bin_levels:np.ndarray = np.zeros(self.capacity)

        if self.reset_distribution is not None:
            self.instance = self.reset_distribution.sample(len(self.instance))

        self.item = self.instance[self.step_count]

        self._update_state()
        return self.state
        


    def _update_state(self):
        self.state = {
            "mask": self.mask_potentials(self.item),
            "state": self.bin_levels + np.array([self.item])
        }

    @property
    def used_bins(self) -> int:
        return int(sum(self.bin_levels)) + self.filled_bins

    def mask_potentials(self, item: int):
        assert item <= self.capacity
        mask = (self.bin_levels > 0).astype(float)
        overflow = self.capacity - item
        mask[overflow+1:] = 0
        mask[0] = 1
        return mask


    def step(self, action: int):
        assert action >= 0 and action < self.capacity, f"Action should between 0 and {self.capacity}"
        self.decision_sequence.append(action)
        terminated = False
        truncated = False
        reward = 0
        waste = 0
        if action > self.capacity - self.item or self.bin_levels[action] == 0:
            reward = BIG_NEG_REWARD
            truncated = True
        elif action == 0:
            # new bin
            self.bin_levels[self.item] += 1
            waste = self.capacity -self.item
            reward = -1 * waste / self.capacity
        else:
            if action + self.item == self.capacity:
                self.filled_bins += 1
            else:
                self.bin_levels[action + self.item] += 1
            waste = -self.item
            reward = -1 * waste / self.capacity

            self.bin_levels[action] -= 1
        
        self.total_reward += reward
        self.total_waste += waste
        self.step_count += 1

        if self.step_count >= len(self.instance):
            terminated = True

        self._update_state()
        info = {
            "used_bins": self.used_bins,
            "waste": self.waste
        }

        return self.state, reward, terminated, truncated, info, terminated or truncated




