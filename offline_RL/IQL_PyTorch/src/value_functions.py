import torch
import torch.nn as nn
from .util import mlp


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))
    
    def log_reward(self, s, a_arctanh, alpha=1.0):
        a = torch.tanh(a_arctanh)
        q_sa = self(s, a)
        r = q_sa + alpha*torch.log((1 - (a)**2) + 1e-7).sum(1)
        return r

    def score(self, s, a, alpha=1.0):
        a = a.detach()
        a.requires_grad_(True)
        r = self.log_reward(s, a, alpha=alpha)
        # get gradient wrt r_sa
        score = torch.clamp(torch.autograd.grad(r.sum(), a)[0], -100, 100)
        return score.detach()


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)