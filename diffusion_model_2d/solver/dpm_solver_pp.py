# DPM-Solver++ implementation

from __future__ import annotations

import torch

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
        """Half-log-SNR: lambda(t) = log(alpha(t)) - log(sigma(t))."""
        a = self._alpha(t)
        s = self._sigma(t)
        return torch.log(a) - torch.log(s)

    def _lambda_to_t(
        self, lambda_val: torch.Tensor, t_min: float = 1e-5, t_max: float = 1.0
    ) -> torch.Tensor:
        """
        Numerically invert lambda(t) to find t using Newton's method.
        f(t) = lambda(t) - lambda_val = 0
        """
        # Initial guess: linear interpolation is usually close enough for convergence
        # But simply starting from t_max or t_min is safer. Let's use midpoint.
        t = torch.full_like(lambda_val, 0.5 * (t_min + t_max))

        # Newton iterations
        for _ in range(10):
            # Calculate f(t) and f'(t)
            # lambda(t) = log(alpha) - log(sigma)
            # d(lambda)/dt = - beta(t) / (2 * sigma(t)^2)
            # This derivative is derived from VP-SDE definitions.

            # Clamp t to stay in valid range during iteration
            t = t.clamp(t_min, t_max)

            lam = self._lambda(t)
            f = lam - lambda_val

            beta_t = self.sde.beta(t)
            sigma_t = self._sigma(t)
            # Derivative: dlambda/dt
            # sigma^2 can be small, add epsilon
            d_lam = -0.5 * beta_t / (sigma_t.pow(2) + 1e-12)

            # Update: t = t - f / f'
            delta = f / (d_lam - 1e-12)  # Avoid div by zero
            t = t - delta

            if torch.abs(delta).max() < 1e-6:
                break

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

    def _predict_x0(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if t.dim() == 0 or t.dim() == 1 and t.shape[0] == 1:
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
        # Evaluate model at current time s
        B = x.shape[0]
        s_in = s.expand(B).reshape(B, 1) if s.dim() == 0 else s

        x0_s = self._predict_x0(x, s_in)

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

        r0 = h0 / (h + 1e-12)
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
