from enum import StrEnum

import torch
from torch import nn


class PredictorType(StrEnum):
    X_START = "x_start"
    NOISE = "noise"
    SCORE = "score"


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
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_data = torch.hstack([x, t])
        return self.net(input_data)
