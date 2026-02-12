from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from diffusion_model_2d.sde.sde import SDE
from diffusion_model_2d.model import Predictor


@dataclass(frozen=True)
class SolverState:
    """Internal state for multistep solvers."""

    t_prev: list[torch.Tensor]  # list of shape [1] tensors (times)
    model_prev: list[torch.Tensor]  # list of model outputs at corresponding times


class Solver(ABC):
    """
    Base class for samplers / solvers.

    - Works with continuous-time SDE wrappers that provide alpha(t) and sigma(t).
    - Assumes the solver integrates from t1 -> t0 (decreasing times).
    """

    def __init__(
        self,
        sde: SDE,
        predictor: Predictor,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        self.sde = sde
        self.predictor = predictor
        self.device = device

    def to(self, device: torch.device) -> "Solver":
        self.device = device
        return self

    @abstractmethod
    def denoise_to_x0(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return x0 prediction from x_t at time t. Used when denoise_to_x0=True in sample()."""
        raise NotImplementedError

    @abstractmethod
    def init_state(self) -> SolverState:
        raise NotImplementedError

    @abstractmethod
    def step(
        self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, state: SolverState
    ) -> tuple[torch.Tensor, SolverState]:
        raise NotImplementedError
