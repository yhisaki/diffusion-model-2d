from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from diffusion_model_2d.data import make_dataset
from diffusion_model_2d.loss import loss_fn
from diffusion_model_2d.model import Predictor, PredictorType
from diffusion_model_2d.plot import (
    plot_generated_data,
    plot_loss_curve,
    plot_sampling_gif,
    plot_training_data,
)
from diffusion_model_2d.sde.ve_sde import VESDE
from diffusion_model_2d.sde.vp_sde import VPSDE
from diffusion_model_2d.seed import set_seed
from diffusion_model_2d.solver import DPMSolverPP2M


def build_sde(config: DictConfig) -> VPSDE | VESDE:
    sde_cfg = config.model.sde
    sde_name = str(sde_cfg.type).upper()

    if sde_name == "VP":
        return VPSDE(
            beta_min=float(sde_cfg.beta_min),
            beta_max=float(sde_cfg.beta_max),
        )
    if sde_name == "VE":
        return VESDE(
            sigma_min=float(sde_cfg.sigma_min),
            sigma_max=float(sde_cfg.sigma_max),
        )

    raise ValueError(f"Unknown SDE: {sde_name}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    # --- Setup ---
    set_seed(int(config.seed))
    device = torch.device(config.device)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    x_train = make_dataset(
        str(config.data.type),
        int(config.data.n_train_samples),
        float(config.data.noise),
        int(config.seed),
    )

    # Standardize for stable training and save stats for inverse transform.
    data_mean = x_train.mean(dim=0, keepdim=True)
    data_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - data_mean) / data_std

    predictor = Predictor(
        predictor_type=PredictorType(config.model.predictor),
        x_dim=2,
        hidden=int(config.model.hidden),
    ).to(device)

    predictor = torch.compile(predictor)

    sde = build_sde(config)

    optimizer = Adam(
        predictor.parameters(),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
    )

    steps = int(config.training.steps)
    batch_size = int(config.training.batch_size)
    eps = float(config.training.eps)
    log_every = int(config.training.log_every)

    print(f"Training on device={device} for {steps} steps")
    print(
        f"dataset={config.data.type} "
        f"n_train_samples={config.data.n_train_samples} "
        f"noise={config.data.noise}"
    )

    x_train = x_train.to(device)
    n_data = x_train.shape[0]

    loss_steps: list[int] = []
    loss_values: list[float] = []

    for step in range(1, steps + 1):
        indices = torch.randint(0, n_data, (batch_size,), device=device)
        x0 = x_train[indices]

        optimizer.zero_grad(set_to_none=True)

        def weight_fn(lam: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(-lam + 5.0)

        loss = loss_fn(
            sde=sde,
            predictor=predictor,
            x0=x0,
            eps=eps,
            weight_fn=weight_fn,
        )
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_steps.append(step)
        loss_values.append(loss_val)
        writer.add_scalar("train/loss", float(loss_val), step)

        if step % log_every == 0 or step == 1:
            print(f"step={step:6d} loss={loss_val:.6f}")

    # Plot loss convergence
    loss_png = out_dir / "loss_curve.png"
    plot_loss_curve(loss_steps, loss_values, loss_png)
    print(f"Saved loss curve to {loss_png}")

    ckpt_path = out_dir / "predictor.pt"
    torch.save(
        {
            "state_dict": predictor.state_dict(),
            "config": OmegaConf.to_container(config, resolve=True),
            "data_mean": data_mean,
            "data_std": data_std,
        },
        ckpt_path,
    )
    print(f"Saved model checkpoint to {ckpt_path}")

    solver = DPMSolverPP2M(sde=sde, predictor=predictor, device=device)

    with torch.no_grad():
        x_t = sde.prior_sampling(torch.Size([int(config.sampling.num_samples), 2])).to(
            device
        )
        history = solver.sample(x_t, steps=int(config.sampling.steps))

        # Denormalize all steps
        mean_dev = data_mean.to(device)
        std_dev = data_std.to(device)
        history = [h * std_dev + mean_dev for h in history]
        x_gen = history[-1]

    samples_pt = out_dir / "samples.pt"
    samples_csv = out_dir / "samples.csv"
    x_gen_cpu = x_gen.cpu().numpy()
    torch.save(x_gen.cpu(), samples_pt)
    np.savetxt(samples_csv, x_gen_cpu, delimiter=",")
    print(f"Saved generated samples to {samples_pt} and {samples_csv}")

    x_train_orig = x_train.cpu() * data_std + data_mean
    train_np = x_train_orig.numpy()
    pad = 0.1
    x_range = [float(train_np[:, 0].min()) - pad, float(train_np[:, 0].max()) + pad]
    y_range = [float(train_np[:, 1].min()) - pad, float(train_np[:, 1].max()) + pad]

    out_dir.mkdir(parents=True, exist_ok=True)

    training_png = out_dir / "training_data.png"
    plot_training_data(x_train_orig, training_png, x_range, y_range)
    print(f"Saved training data plot to {training_png}")

    generated_png = out_dir / "generated_samples.png"
    plot_generated_data(history[-1], generated_png, x_range, y_range)
    print(f"Saved generated samples plot to {generated_png}")

    sampling_gif = out_dir / "sampling_process.gif"
    plot_sampling_gif(history, sampling_gif, x_range, y_range)
    print(f"Saved sampling animation to {sampling_gif}")

    writer.close()


if __name__ == "__main__":
    main()
