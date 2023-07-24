from pathlib import Path
from typing import  Sequence
import pytest
from bpp1d.models.model import ModelStatus
import random
from bpp1d.models.rl.bpp1denv import Bpp1DPotentialEnv
from bpp1d.models.rl_model import RLModel 


ENV_TEST_CASES = [
    {
        "instance": [5, 5, 3, 3, 2, 2],
        "capacity": 10,
        "actions": [0, 5, 0, 3, 6, 8],
        "filled": [0, 1, 1, 1, 1, 2],
        "waste": [5, 0, 7, 4, 2, 0],
    }
]

RL_TEST_CASES = [
    {
        "instance": [random.randint(10, 60) for _ in range(1000)],
        "capacity": 100,
        "checkpoint_path": Path("./models/policy.pth")
    }
]


@pytest.mark.parametrize(
    ('instance', 'capacity', 'actions', 'filled', 'waste'), [
        (test_case['instance'], test_case['capacity'], 
            test_case['actions'], test_case['filled'], test_case['waste'])
        for test_case in ENV_TEST_CASES
    ]
)

def test_bpp1d_env(instance: Sequence[int], capacity: int, 
                    actions:Sequence[int], filled: Sequence[int], waste: Sequence[int]):

    env = Bpp1DPotentialEnv(instance, capacity)
    obs, _ = env.reset()

    for i, action in enumerate(actions):
        obs, rew, terminated, truncated, _ = env.step(action)
        print(env.total_waste, env.filled_bins, env.state)
        assert env.filled_bins == filled[i]
        assert env.total_waste == waste[i]
        if truncated:
            break


    assert (terminated and not truncated) or (not terminated and truncated)
    assert env.decision_sequence == actions


@pytest.mark.parametrize(
    ('instance', 'capacity', 'checkpoint_path'), [
        (test_case['instance'], test_case['capacity'], test_case['checkpoint_path'])
        for test_case in RL_TEST_CASES
    ]
)
def test_rl_model(instance: Sequence[int], capacity: int, checkpoint_path: Path):

    model = RLModel(capacity, instance, checkpoint_path)
    model.build()
    # if model.status == ModelStatus.BUILD:
    assert model.status == ModelStatus.BUILD
    solution, _ = model.solve()
    # print(solution.metrics)
    # assert solution.waste == 0
    assert model.status == ModelStatus.FINISHED
    assert model.step_count == len(instance)