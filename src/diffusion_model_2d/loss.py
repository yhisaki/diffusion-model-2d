from collections.abc import Callable

import torch

from diffusion_model_2d.model import Predictor, PredictorType
from diffusion_model_2d.sde.sde import SDE


def loss_fn(
    sde: SDE,
    predictor: Predictor,
    x0: torch.Tensor,
    eps: float,
    weight_fn: Callable[[torch.Tensor], torch.Tensor] = lambda lam: 1.0,
) -> torch.Tensor:
    """
    Loss function for score-based models.
    """
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}.")
    if eps >= 1.0:
        raise ValueError(f"eps must be in [0, 1), got {eps}.")
    B = x0.shape[0]
    device = x0.device
    span = 1.0 - eps
    t = torch.rand(B, 1, device=device) * span + eps

    mean, std = sde.marginal_prob(x0, t)
    z = torch.randn_like(x0)
    xt = mean + std * z

    pred = predictor(xt, t)

    if predictor.predictor_type == PredictorType.X_START:
        target = x0
        snr = sde.snr(t).detach()
        log_snr = sde.log_snr(t).detach()
        weight = snr * weight_fn(log_snr).detach()
    elif predictor.predictor_type == PredictorType.NOISE:
        target = z
        log_snr = sde.log_snr(t).detach()
        weight = weight_fn(log_snr).detach()
    elif predictor.predictor_type == PredictorType.SCORE:
        target = -z / std
        log_snr = sde.log_snr(t).detach()
        weight = (std**2).detach() * weight_fn(log_snr).detach()
    else:
        raise ValueError(f"Unknown predictor type: {predictor.predictor_type}")

    loss = (pred - target).pow(2) * weight

    return loss.mean()
