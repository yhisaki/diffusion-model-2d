import torch

from diffusion_model_2d.sde.sde import SDE


class VESDE(SDE):
    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
    ) -> None:
        super().__init__()
        if not (sigma_min > 0.0 and sigma_max > 0.0):
            raise ValueError("sigma_min and sigma_max must be positive.")
        if not (sigma_min < sigma_max):
            raise ValueError(
                f"Require sigma_min < sigma_max, got sigma_min={sigma_min}, "
                f"sigma_max={sigma_max}."
            )
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        ratio = self.sigma_max / self.sigma_min
        self._log_sigma_ratio = float(torch.log(torch.tensor(ratio)))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        log_ratio = t.new_tensor(self._log_sigma_ratio)
        return self.sigma_min * torch.exp(t * log_ratio)

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        return 1 / (self.sigma(t) ** 2)

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        return -2.0 * torch.log(self.sigma(t))

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        log_ratio = t.new_tensor(self._log_sigma_ratio)
        return self.sigma(t) * torch.sqrt(2.0 * log_ratio).clamp_min(1e-12)

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x0
        std = self.sigma(t)
        return mean, std

    def prior_sampling(self, shape: torch.Size) -> torch.Tensor:
        return torch.randn(shape) * self.sigma_max

    def prior_logp(self, x: torch.Tensor) -> torch.Tensor:
        # Normal log-density with variance sigma_max^2 per sample
        x_flat = x.reshape(x.shape[0], -1)
        d = x_flat.shape[1]
        quad = (x_flat * x_flat).sum(dim=1)
        sigma2 = x.new_tensor(self.sigma_max * self.sigma_max)
        log_2pi = torch.log(2.0 * torch.pi).to(x)
        log_det = d * (log_2pi + torch.log(sigma2))
        return -0.5 * (quad / sigma2 + log_det)
