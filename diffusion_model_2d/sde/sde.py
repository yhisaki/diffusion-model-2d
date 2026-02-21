from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class SDE(ABC):
    """
    Abstract base class for diffusion SDEs used in score-based models.

    We consider an Itô SDE on x_t in R^d:
        dx = f(x, t) dt + g(t) dW,   t in [0, 1]
    where:
      - f(x,t): drift (same shape as x)
      - g(t): diffusion coefficient (broadcastable to x), often shape [B, 1] or [B]
      - W: standard Brownian motion

    Conventions:
      - t is a torch.Tensor of shape [B, 1] (recommended) or [B]
      - x is a torch.Tensor of shape [B, d] (or [B, ...])
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return drift f(x,t), same shape as x."""
        raise NotImplementedError

    @abstractmethod
    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Return signal-to-noise ratio (SNR) of the SDE."""
        raise NotImplementedError

    @abstractmethod
    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Return log signal-to-noise ratio (log SNR) of the SDE."""
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return diffusion coefficient g(t).
        Must be broadcastable to x in drift/diffusion usage.
        Typical shapes: [B, 1] or [B]
        """
        raise NotImplementedError

    @abstractmethod
    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (mean, std) of p(x_t | x_0) for this SDE.
          x_t = mean(x0,t) + std(t) * z,  z ~ N(0, I)
        std must be broadcastable to x0.
        """
        raise NotImplementedError

    @abstractmethod
    def prior_sampling(self, shape: torch.Size) -> torch.Tensor:
        """Sample from p(x_{t1}) (the prior / terminal distribution)."""
        raise NotImplementedError

    @abstractmethod
    def prior_logp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x_{t1}) per sample (shape [B]).
        Useful for likelihood estimation / debugging.
        """
        raise NotImplementedError

    def reverse_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        probability_flow: bool = False,
    ) -> torch.Tensor:
        """
        Return the drift of the reverse-time dynamics.

        Reverse-time SDE (stochastic):
            dx = [ f(x,t) - g(t)^2 * score(x,t) ] dt + g(t) dW_bar

        Probability flow ODE (deterministic):
            dx = [ f(x,t) - 0.5 * g(t)^2 * score(x,t) ] dt

        score is ∇_x log p_t(x) (unconditional score).
        """
        f = self.drift(x, t)
        g = self.diffusion(t)
        gg = g * g
        coef = 0.5 if probability_flow else 1.0
        return f - coef * gg * score
