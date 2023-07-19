from pathlib import Path

from typing import Any, Dict, Sequence, Tuple
from bpp1d.models.model import Model, ModelStatus
from bpp1d.models.rl.bpp1denv import Bpp1DPotentialEnv
from bpp1d.models.rl.net import Actor
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from bpp1d.structure import Solution, PotentialSolution


class RLModel(Model):

    def __init__(self, capacity: int, instance: Sequence[int], 
                    checkpoint_path: Path, name: str = 'rl_model'):
        """A implementation of 

        Args:
            capacity (int): _description_
            instance (Sequence[int]): _description_
            checkpoint_path (Path): _description_
            name (str, optional): _description_. Defaults to 'rl_model'.
        """
        super().__init__(capacity, instance, name)
        self.checkpoint_path = checkpoint_path
        self.actor = Actor(capacity + 1, capacity)

        self.env = Bpp1DPotentialEnv(self.instance, self.capacity)
        self.step_count = 0

    def build(self) -> Any:
        # try:
        actor_dict = self.actor.state_dict()
        actor_keys = ['actor.' + k for k in actor_dict]
        checkpoint = torch.load(self.checkpoint_path)
        state_dict = {k.split('.', 1)[1]:v for k,v in checkpoint.items() if k in actor_keys}

        actor_dict.update(state_dict)
        self.actor.load_state_dict(actor_dict)

        self.actor.eval()
        return super().build()
        # except Exception:
            # self.status = ModelStatus.ERROR
        
    def solve(self) -> Tuple[Solution, Dict | None]:
        self.status = ModelStatus.SOLVING
        obs, info = self.env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        while not (terminated or truncated):
            logits, _ = self.actor(obs)

            dist = Categorical(F.softmax(logits, dim=-1))
            action = dist.sample().item()
            obs, rew, terminated, truncated, info = self.env.step(action)
            total_reward += rew
            self.step_count += 1

        self.status = ModelStatus.FINISHED if not truncated else ModelStatus.ERROR
        return PotentialSolution(self.capacity, self.env.bin_levels.tolist(), self.env.filled_bins), info

        
