import math

import torch
from torch import nn

from enum import Enum, auto


class GaussianFourierProjection(nn.Module):
    """
    Random Fourier features for time embedding.
    """

    def __init__(self, embed_dim: int, scale: float = 16.0) -> None:
        super().__init__()
        assert embed_dim % 2 == 0
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [B, 1]
        # returns: [B, embed_dim]
        x = 2.0 * math.pi * t * self.W[None, :]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class PredictorType(Enum):
    X_START = auto()
    NOISE = auto()
    SCORE = auto()


class Predictor(nn.Module):
    def __init__(
        self,
        predictor_type: PredictorType = PredictorType.X_START,
        x_dim: int = 2,
        time_embed_dim: int = 64,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.predictor_type = predictor_type
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(time_embed_dim),
            nn.Linear(time_embed_dim, hidden),
            nn.SiLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(x_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, 2], t: [B, 1]
        te = self.time_embed(t)
        h = torch.cat([x, te], dim=-1)
        return self.net(h)
