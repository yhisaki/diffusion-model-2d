from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch


class Guidance(ABC):
    """
    Abstract base class for sampling guidance.
    Guidance modifies the diffusion process by injecting gradients from a
    potential/loss function.
    """

    @abstractmethod
    def compute_correction(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: torch.Tensor,
        alpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the correction term to be applied to the predicted x0.

        Args:
            x_t: Current noisy sample [B, D].
            x0_pred: Predicted clean sample [B, D].
            t: Current time [B, 1].
            alpha_t: Alpha value at t [B, 1].
            sigma_t: Sigma value at t [B, 1].

        Returns:
            delta_x0: The correction vector to add to x0_pred.
        """
        raise NotImplementedError


class LossGuidance(Guidance):
    """
    Guidance based on minimizing a loss function L(x0).

    The score modification is equivalent to:
        nabla_x log p(y|x) approx - nabla_x Loss(x0(x))

    For x0-prediction models, the update rule derived from the score correction is:
        x0_new = x0_old - scale * (sigma_t^2 / alpha_t) * nabla_xt Loss(x0)

    Approximating nabla_xt Loss(x0) approx (1/alpha_t) * nabla_x0 Loss(x0), we get:
        x0_new = x0_old - scale * (sigma_t / alpha_t)^2 * nabla_x0 Loss(x0)
    """

    def __init__(
        self, loss_fn: Callable[[torch.Tensor], torch.Tensor], scale: float = 1.0
    ) -> None:
        """
        Args:
            loss_fn: A function that takes x0 [B, D] and returns a scalar loss
                [B] or [1]. Example: lambda x: torch.relu(-x).sum()
            scale: Guidance scale (lambda). Positive values minimize the loss.
        """
        self.loss_fn = loss_fn
        self.scale = scale

    def compute_correction(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: torch.Tensor,
        alpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        # Enable gradient calculation for x0_pred
        with torch.enable_grad():
            x0_in = x0_pred.detach().requires_grad_(True)
            loss = self.loss_fn(x0_in)

            # Compute gradient w.r.t x0
            grad = torch.autograd.grad(outputs=loss.sum(), inputs=x0_in)[0]

        # Formula: delta_x0 = - scale * (sigma_t^2 / alpha_t^2) * grad
        # This assumes the approximation nabla_xt L = (1/alpha) * nabla_x0 L
        # which is computationally cheaper than backpropping through the U-Net.

        coef = (sigma_t**2) / (alpha_t**2 + 1e-12)
        delta_x0 = -self.scale * coef * grad

        return delta_x0
