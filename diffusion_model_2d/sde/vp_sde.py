import torch
from diffusion_model_2d.sde.sde import SDE


class VPSDE(SDE):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def int_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t * t

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.int_beta(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        exp_neg = torch.exp(-self.int_beta(t))
        return torch.sqrt((1.0 - exp_neg).clamp_min(1e-12))

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.beta(t) * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta(t)).clamp_min(1e-12)

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.alpha(t) * x0
        std = self.sigma(t)
        return mean, std

    def prior_sampling(self, shape: torch.Size) -> torch.Tensor:
        return torch.randn(shape)

    def prior_logp(self, x: torch.Tensor) -> torch.Tensor:
        # Standard normal log-density per sample
        # log N(x;0,I) = -0.5*(||x||^2 + d*log(2π))
        x_flat = x.reshape(x.shape[0], -1)
        d = x_flat.shape[1]
        quad = (x_flat * x_flat).sum(dim=1)
        log_2pi = torch.log(2.0 * torch.pi).to(x)
        return -0.5 * (quad + d * log_2pi)