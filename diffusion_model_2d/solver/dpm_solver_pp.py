# DPM-Solver++ implementation

from __future__ import annotations

import torch

from diffusion_model_2d.model import Predictor, PredictorType
from diffusion_model_2d.sde.vp_sde import VPSDE

from .solver import Solver, SolverState


class DPMSolverPP2M(Solver):
    """
    DPM-Solver++(2M) for VP-type schedules, implemented in x0-prediction form.

    This is the second-order multistep variant (2M):
      - first step uses the 1st-order update (DDIM-equivalent in x0-form),
      - subsequent steps use the 2nd-order multistep update reusing the previous x0 predictions.

    Notes:
      - Requires PredictorType.X_START and VPSDE; raises otherwise.
      - Uses time-uniform spacing by default (handled by SolverBase.sample()).
    """

    def __init__(
        self,
        sde: VPSDE,
        predictor: Predictor,
        *,
        device: torch.device | None = None,
    ) -> None:
        if predictor.predictor_type != PredictorType.X_START:
            raise ValueError(
                f"DPMSolverPP2M requires PredictorType.X_START, "
                f"got {predictor.predictor_type}"
            )
        if not isinstance(sde, VPSDE):
            raise TypeError(f"DPMSolverPP2M requires VPSDE, got {type(sde).__name__}")
        super().__init__(sde, predictor, device=device)

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        a = self.sde.alpha(t)
        return a.clamp_min(1e-12)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        s = self.sde.sigma(t)
        return s.clamp_min(1e-12)

    def _lambda(self, t: torch.Tensor) -> torch.Tensor:
        """Half-log-SNR: lambda(t) = log(alpha(t)) - log(sigma(t)). Shape follows input t."""
        a = self._alpha(t)
        s = self._sigma(t)
        return torch.log(a) - torch.log(s)

    def _predict_x0(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert network output to x0. This solver uses X_START only, so out is already x0."""
        B = x.shape[0]
        if t.dim() == 1 and t.shape[0] == 1:
            t_in = t.expand(B).reshape(B, 1)
        else:
            t_in = t
        return self.predictor(x, t_in)

    def denoise_to_x0(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._predict_x0(x, t)

    def init_state(self) -> SolverState:
        return SolverState(t_prev=[], model_prev=[])

    @torch.no_grad()
    def step(
        self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, state: SolverState
    ) -> tuple[torch.Tensor, SolverState]:
        # Evaluate model at current time s (x0 prediction)
        B = x.shape[0]
        s_in = s.expand(B).reshape(B, 1)
        x0_s = self._predict_x0(x, s_in)

        # Update buffers
        t_prev = state.t_prev + [s.clone()]
        model_prev = state.model_prev + [x0_s]

        # Keep only the last 2 entries (2M)
        if len(t_prev) > 2:
            t_prev = t_prev[-2:]
            model_prev = model_prev[-2:]

        # If we don't yet have two previous model values, do 1st-order update
        if len(t_prev) < 2:
            x_next = self._first_order_update(x, s, t, x0_s)
            return x_next, SolverState(t_prev=t_prev, model_prev=model_prev)

        # Otherwise do multistep 2nd order update
        x_next = self._multistep_second_update(
            x=x,
            model_prev_1=model_prev[-2],
            model_prev_0=model_prev[-1],
            t_prev_1=t_prev[-2],
            t_prev_0=t_prev[-1],
            t=t,
        )
        return x_next, SolverState(t_prev=t_prev, model_prev=model_prev)

    # -------------------------
    # DPM-Solver++(2M) updates
    # -------------------------
    def _first_order_update(
        self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_s: torch.Tensor
    ) -> torch.Tensor:
        """
        DPM-Solver++ 1st-order update (DDIM-equivalent in x0 form):
          x_t = (sigma_t / sigma_s) * x_s - alpha_t * expm1(-(lambda_t - lambda_s)) * x0_s
        """
        B = x.shape[0]
        s_in = s.expand(B).reshape(B, 1)
        t_in = t.expand(B).reshape(B, 1)

        lam_s = self._lambda(s_in)
        lam_t = self._lambda(t_in)
        h = lam_t - lam_s

        sigma_s = self._sigma(s_in)
        sigma_t = self._sigma(t_in)
        alpha_t = self._alpha(t_in)

        phi_1 = torch.expm1(-h)
        return (sigma_t / sigma_s) * x - alpha_t * phi_1 * x0_s

    def _multistep_second_update(
        self,
        *,
        x: torch.Tensor,
        model_prev_1: torch.Tensor,
        model_prev_0: torch.Tensor,
        t_prev_1: torch.Tensor,
        t_prev_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPM-Solver++ 2nd-order multistep update (2M):

        Let:
          h0 = lambda(t_prev_0) - lambda(t_prev_1)
          h  = lambda(t)        - lambda(t_prev_0)
          r0 = h0 / h
          D1_0 = (1/r0) * (model_prev_0 - model_prev_1)
          phi_1 = expm1(-h)

        Then:
          x_t = (sigma_t / sigma_prev_0) * x
                - alpha_t * phi_1 * model_prev_0
                - 0.5 * alpha_t * phi_1 * D1_0
        """
        B = x.shape[0]
        t0_in = t_prev_0.expand(B).reshape(B, 1)
        t1_in = t_prev_1.expand(B).reshape(B, 1)
        t_in = t.expand(B).reshape(B, 1)

        lam_1 = self._lambda(t1_in)
        lam_0 = self._lambda(t0_in)
        lam_t = self._lambda(t_in)

        h0 = lam_0 - lam_1
        h = lam_t - lam_0

        # Avoid division by zero if times collapse numerically
        h_safe = h.clamp_min(1e-12)
        r0 = h0 / h_safe

        D1_0 = (model_prev_0 - model_prev_1) / r0.clamp_min(1e-12)
        phi_1 = torch.expm1(-h)

        sigma_0 = self._sigma(t0_in)
        sigma_t = self._sigma(t_in)
        alpha_t = self._alpha(t_in)

        return (
            (sigma_t / sigma_0) * x
            - (alpha_t * phi_1) * model_prev_0
            - 0.5 * (alpha_t * phi_1) * D1_0
        )
