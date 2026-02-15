# DPM-Solver++ implementation

from __future__ import annotations

import torch

from diffusion_model_2d.guidance import Guidance
from diffusion_model_2d.model import Predictor, PredictorType
from diffusion_model_2d.sde.sde import SDE
from diffusion_model_2d.sde.vp_sde import VPSDE

from .solver import Solver, SolverState


class DPMSolverPP2M(Solver):
    """
    DPM-Solver++(2M) for VP-type schedules, implemented in x0-prediction form.

    This is the second-order multistep variant (2M):
      - first step uses the 1st-order update (DDIM-equivalent in x0-form),
      - subsequent steps use the 2nd-order multistep update reusing the
        previous x0 predictions.

    Notes:
      - Requires PredictorType.X_START and VPSDE.
      - Implements Log-SNR uniform scheduling in get_time_schedule.
    """

    def __init__(
        self,
        sde: SDE,
        predictor: Predictor,
        *,
        device: torch.device | None = None,
    ) -> None:
        if predictor.predictor_type not in (PredictorType.X_START, PredictorType.NOISE):
            raise ValueError(
                "DPMSolverPP2M requires PredictorType.X_START or PredictorType.NOISE, "
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
        """Half-log-SNR: lambda(t) = log(alpha(t)) - log(sigma(t))."""
        a = self._alpha(t)
        s = self._sigma(t)
        return torch.log(a) - torch.log(s)

    def _lambda_to_t(
        self, lambda_val: torch.Tensor, t_min: float = 1e-5, t_max: float = 1.0
    ) -> torch.Tensor:
        """
        Invert lambda(t) -> t for linear VPSDE analytically (more stable than Newton).

        Matches the reference implementation used in DPM-Solver / DPM-Solver++.
        """
        beta0 = float(self.sde.beta_min)
        beta1 = float(self.sde.beta_max)
        bdiff = beta1 - beta0
        if bdiff <= 0.0:
            raise ValueError("Require beta_max > beta_min for VPSDE.")

        # tmp = 2*(beta1-beta0)*log(1 + exp(-2*lambda))
        zeros = torch.zeros_like(lambda_val)
        tmp = 2.0 * bdiff * torch.logaddexp(-2.0 * lambda_val, zeros)
        delta = beta0 * beta0 + tmp
        t = tmp / (torch.sqrt(delta) + beta0) / bdiff
        return t.clamp(t_min, t_max)

    def get_time_schedule(
        self, steps: int, t_start: float, t_end: float, device: torch.device
    ) -> torch.Tensor:
        """
        Generate Log-SNR uniform schedule.
        Calculates lambda(t_start) and lambda(t_end), interpolates linearly
        in lambda space, and then inverts back to time t.
        """
        t_start_tensor = torch.tensor(t_start, device=device).reshape(1)
        t_end_tensor = torch.tensor(t_end, device=device).reshape(1)

        lambda_start = self._lambda(t_start_tensor)
        lambda_end = self._lambda(t_end_tensor)

        # Uniform steps in lambda space
        lambdas = torch.linspace(
            lambda_start.item(), lambda_end.item(), steps + 1, device=device
        )

        # Invert back to time t
        # We process all steps in parallel
        timesteps = self._lambda_to_t(
            lambdas, t_min=min(t_end, t_start), t_max=max(t_end, t_start)
        )

        # Ensure endpoints are exact
        timesteps[0] = t_start
        timesteps[-1] = t_end

        return timesteps

    def _predict_x0(
        self, x: torch.Tensor, t: torch.Tensor, guidance: Guidance | None = None
    ) -> torch.Tensor:
        """
        Convert network output to x0 and apply guidance if provided.
        """
        B = x.shape[0]
        if t.dim() == 0 or t.dim() == 1 and t.shape[0] == 1:
            t_in = t.expand(B).reshape(B, 1)
        else:
            t_in = t

        # 1. Base model prediction
        alpha_t = self._alpha(t_in)
        sigma_t = self._sigma(t_in)

        if self.predictor.predictor_type == PredictorType.X_START:
            x0_pred = self.predictor(x, t_in)
        elif self.predictor.predictor_type == PredictorType.NOISE:
            eps_pred = self.predictor(x, t_in)
            x0_pred = (x - sigma_t * eps_pred) / alpha_t
        else:
            raise ValueError(f"Unknown predictor type: {self.predictor.predictor_type}")

        # 2. Apply Guidance
        if guidance is not None:
            # Compute correction: delta = -scale * (sigma^2/alpha^2) * grad_Loss
            correction = guidance.compute_correction(x, x0_pred, t_in, alpha_t, sigma_t)
            x0_pred = x0_pred + correction

        return x0_pred

    def denoise_to_x0(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._predict_x0(x, t, guidance=None)

    def init_state(self) -> SolverState:
        return SolverState(t_prev=[], model_prev=[])

    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        state: SolverState,
        guidance: Guidance | None = None,
    ) -> tuple[torch.Tensor, SolverState]:
        # Evaluate model at current time s
        B = x.shape[0]
        s_in = s.expand(B).reshape(B, 1) if s.dim() == 0 else s

        # Apply guidance inside _predict_x0
        x0_s = self._predict_x0(x, s_in, guidance=guidance)

        # Update buffers
        t_prev = state.t_prev + [s.clone().detach()]
        model_prev = state.model_prev + [x0_s]

        if len(t_prev) > 2:
            t_prev = t_prev[-2:]
            model_prev = model_prev[-2:]

        # 1st-order update
        if len(t_prev) < 2:
            x_next = self._first_order_update(x, s, t, x0_s)
            return x_next, SolverState(t_prev=t_prev, model_prev=model_prev)

        # 2nd-order update
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
    # Updates (Identical to previous implementation)
    # -------------------------
    def _first_order_update(
        self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_s: torch.Tensor
    ) -> torch.Tensor:
        B = x.shape[0]
        s_in = s.expand(B).reshape(B, 1) if s.dim() == 0 else s
        t_in = t.expand(B).reshape(B, 1) if t.dim() == 0 else t

        lam_s = self._lambda(s_in)
        lam_t = self._lambda(t_in)
        h = lam_t - lam_s

        sigma_s = self._sigma(s_in)
        sigma_t = self._sigma(t_in)
        alpha_t = self._alpha(t_in)

        # DPM-Solver++ (data/x0-prediction) uses expm1(-h).
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
        B = x.shape[0]
        t0_in = t_prev_0.expand(B).reshape(B, 1) if t_prev_0.dim() == 0 else t_prev_0
        t1_in = t_prev_1.expand(B).reshape(B, 1) if t_prev_1.dim() == 0 else t_prev_1
        t_in = t.expand(B).reshape(B, 1) if t.dim() == 0 else t

        lam_1 = self._lambda(t1_in)
        lam_0 = self._lambda(t0_in)
        lam_t = self._lambda(t_in)

        h0 = lam_0 - lam_1
        h = lam_t - lam_0

        # Multistep DPM-Solver++(2M) uses expm1(-h).
        r0 = h0 / (h + 1e-12)
        D1_0 = (model_prev_0 - model_prev_1) / (r0 + 1e-12)
        phi_1 = torch.expm1(-h)

        sigma_0 = self._sigma(t0_in)
        sigma_t = self._sigma(t_in)
        alpha_t = self._alpha(t_in)

        return (
            (sigma_t / sigma_0) * x
            - (alpha_t * phi_1) * model_prev_0
            - 0.5 * (alpha_t * phi_1) * D1_0
        )
