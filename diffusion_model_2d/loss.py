import torch
from torch import nn
from diffusion_model_2d.sde.sde import SDE
from diffusion_model_2d.model import Predictor, PredictorType


def loss_fn(
    sde: SDE,
    predictor: Predictor,
    x0: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Loss function for score-based models.
    """
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}.")
    B = x0.shape[0]
    device = x0.device
    span = (sde.t1 - sde.t0) - eps
    if span <= 0.0:
        raise ValueError(
            f"Require t1 - t0 > eps, got t0={sde.t0}, t1={sde.t1}, eps={eps}."
        )
    t = torch.rand(B, 1, device=device) * span + sde.t0 + eps

    mean, std = sde.marginal_prob(x0, t)
    z = torch.randn_like(x0)
    xt = mean + std * z

    pred = predictor(xt, t)

    if predictor.predictor_type == PredictorType.X_START:
        target = x0
    elif predictor.predictor_type == PredictorType.NOISE:
        target = z
    elif predictor.predictor_type == PredictorType.SCORE:
        target = -z / std
    else:
        raise ValueError(f"Unknown predictor type: {predictor.predictor_type}")

    loss = (pred - target).pow(2)

    return loss.mean()
