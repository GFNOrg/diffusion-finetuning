import torch
import torch.nn as nn


class LangevinModel(nn.Module):
    def __init__(self, problem_dim: int, t_dim: int, hidden_dim: int = 256, out_dim: int = 1, num_layers: int = 3,
                 zero_init: bool = False):
        super(LangevinModel, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=t_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(t_dim)[None])

        self.lgv_model = nn.Sequential(
            nn.Linear(problem_dim + (2 * t_dim), hidden_dim),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.register_buffer('pe', pe)

        if zero_init:
            self.lgv_model[-1].weight.data.fill_(0.0)
            self.lgv_model[-1].bias.data.fill_(0.01)

    def forward(self, x, t):
        bs, _, _, _ = x.shape
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        t_emb = t_emb.repeat(bs, 1)
        x = torch.flatten(x, start_dim=1)
        scaling_factor = self.lgv_model(torch.cat([x, t_emb], dim=-1)).reshape(bs, 1, 1, 1)
        return scaling_factor