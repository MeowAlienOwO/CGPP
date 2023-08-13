from pathlib import Path
# from typing import NamedTuple
from .net import Actor, Critic
# import numpy as np
import torch
from .bpp1denv import Bpp1DPotentialEnv
from bpp3d_dataset.utils.distributions import Discrete
from bpp3d_dataset.problems import Problem
from bpp3d_dataset.problems import BppInstance
from torch.optim import Adam
import tianshou as ts
from tianshou.policy import PPOPolicy
from tianshou.data.collector import Collector
import torch.nn.functional as F
from dataclasses import dataclass
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

@dataclass
class RLHyperparam:
    lr:float = 1e-3
    epoch:int = 10
    batch_size:int = 128
    buffer_size:int = 20000
    step_per_epoch:int = 10000
    step_per_collect:int = 1000
    item_per_instance:int = 1000
    train_num:int = 100
    eps_clip:float = 0.2
    discount_factor:float = 0.995


def train_ppo(distribution: Discrete, capacity:int, target_problem: Problem, 
                model_path: Path, filename: str | None = None, param: RLHyperparam | None = None):

    logger= ts.utils.TensorboardLogger(SummaryWriter(model_path / 'summary'))

    param = RLHyperparam() if param is None else param
    def _train_env_create():
        return Bpp1DPotentialEnv(distribution.sample(param.item_per_instance), 
                            capacity, reset_distribution=distribution)
    
    def _test_env_create(instance: BppInstance):
        # print(instance.configuration['capacity'])
        return Bpp1DPotentialEnv(instance, instance.configuration['capacity'], reset_distribution=distribution)

    def _dist_fn(*logits):
        return Categorical(F.softmax(*logits, dim=-1))

    # train_envs = ts.env.DummyVectorEnv([ _train_env_create for _ in range(param.train_num)])
    # test_envs = ts.env.DummyVectorEnv([lambda: _test_env_create(instance) for instance in target_problem])

    train_envs = ts.env.SubprocVectorEnv([_train_env_create for _ in range(param.train_num)])
    test_envs = ts.env.SubprocVectorEnv([lambda: _test_env_create(instance) for instance in target_problem])

    actor = Actor(capacity + 1, capacity)
    critic = Critic(capacity + 1)

    optim = Adam(set(actor.parameters()).union(critic.parameters()), lr=param.lr)
    policy = PPOPolicy(actor, critic, optim, _dist_fn, 
                        eps_clip=param.eps_clip, discount_factor=param.discount_factor)
    rep_buffer = ts.data.VectorReplayBuffer(param.buffer_size, param.train_num)

    train_collector = Collector(policy, train_envs, rep_buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    filename = "policy.pth" if filename is None else filename

    def _save_best_fn(policy):
        torch.save(policy.state_dict(), model_path / filename)
    

    result = ts.trainer.onpolicy_trainer(policy, train_collector, test_collector, 
                                            max_epoch=param.epoch, step_per_epoch=param.step_per_epoch,
                                            step_per_collect=param.step_per_collect,
                                            repeat_per_collect=len(test_envs),
                                            episode_per_test=len(test_envs), batch_size=param.batch_size,
                                            update_per_step=1 / param.step_per_collect,
                                            save_best_fn = _save_best_fn,
                                            logger=logger)
    return result
