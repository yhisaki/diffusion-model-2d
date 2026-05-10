from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from diffusion_model_2d.model import Predictor
from diffusion_model_2d.sde.sde import SDE


@dataclass(frozen=True)
class SolverState:
    """Internal state for multistep solvers."""

    t_prev: list[torch.Tensor]  # list of shape [1] tensors (times)
    model_prev: list[torch.Tensor]  # list of model outputs at corresponding times


class Solver(ABC):
    """
    Base class for samplers / solvers.
    """

    def __init__(
        self,
        sde: SDE,
        predictor: Predictor,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.sde = sde
        self.predictor = predictor
        self.device = device

    def to(self, device: torch.device) -> Solver:
        self.device = device
        return self

    @abstractmethod
    def denoise_to_x0(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return x0 prediction from x_t at time t."""
        raise NotImplementedError

    @abstractmethod
    def init_state(self) -> SolverState:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        state: SolverState,
        guidance: object | None = None,  # <--- Added
    ) -> tuple[torch.Tensor, SolverState]:
        """
        Perform one solver step from time s to time t.
        Returns (x_t, new_state).
        """
        raise NotImplementedError

    def get_time_schedule(
        self, steps: int, t_start: float, t_end: float, device: torch.device
    ) -> torch.Tensor:
        """
        Generate the time schedule for sampling.
        """
        return torch.linspace(t_start, t_end, steps + 1, device=device)

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        steps: int = 20,
        t_start: float = 1.0,
        t_end: float = 1e-4,
        timesteps: torch.Tensor | None = None,
        guidance: object | None = None,  # <--- Added
    ) -> list[torch.Tensor]:
        """
        Run the sampling loop from t_start to t_end.

        Args:
            x: Initial noise tensor [B, D].
            steps: Number of sampling steps.
            t_start: Starting time.
            t_end: Ending time.
            timesteps: Optional explicit time schedule.
            guidance: Optional Guidance object to control generation.

        Returns:
            List of sample tensors at each step.
        """
        device = x.device if self.device is None else self.device
        x = x.to(device)

        # 1. Determine the schedule
        if timesteps is None:
            timesteps = self.get_time_schedule(steps, t_start, t_end, device)
        else:
            timesteps = timesteps.to(device)
            steps = len(timesteps) - 1

        state = self.init_state()
        history = [x.clone()]

        # 2. Sampling loop
        for i in range(steps):
            s = timesteps[i]  # Current time
            t = timesteps[i + 1]  # Next time

            x, state = self.step(x, s, t, state, guidance=guidance)
            history.append(x.clone())

        return history
