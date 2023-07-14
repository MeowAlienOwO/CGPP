from torch import nn
from .helpers import masked_log_softmax, obs2tensor


class Actor(nn.Module):
    def __init__(self, state_size, action_size, inner_dim=256):

        super().__init__()

        self.state_size = state_size
        self.action_shape = action_size

        self.model = nn.Sequential(
            nn.Linear(state_size, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, action_size)
        )

    def forward(self, obs, state=None):
        obs_state, mask = obs2tensor(obs)
        log_prob = self.model(obs_state)

        return masked_log_softmax(log_prob, mask), None


class Critic(nn.Module):
    def __init__(self, state_size, inner_dim=256):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.model = nn.Sequential(
            nn.Linear(state_size, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, 1)
        )

    def forward(self, obs):
        obs_state, _ = obs2tensor(obs)
        return self.model(obs_state)

