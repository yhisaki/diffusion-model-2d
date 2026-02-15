from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from diffusion_model_2d.data import make_dataset
from diffusion_model_2d.loss import loss_fn
from diffusion_model_2d.model import Predictor, PredictorType
from diffusion_model_2d.plot import plot_sampling_process, plot_training_data
from diffusion_model_2d.sde.ve_sde import VESDE
from diffusion_model_2d.sde.vp_sde import VPSDE
from diffusion_model_2d.seed import set_seed
from diffusion_model_2d.solver import DPMSolverPP2M


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2D diffusion model on two moons"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config yaml",
    )
    return parser.parse_args()


def build_sde(config: dict[str, Any]) -> VPSDE | VESDE:
    sde_cfg = config["model"]["sde"]
    sde_name = str(sde_cfg["type"]).upper()

    if sde_name == "VP":
        vp_cfg = sde_cfg["VP"]
        return VPSDE(
            beta_min=float(vp_cfg["beta_min"]),
            beta_max=float(vp_cfg["beta_max"]),
        )
    if sde_name == "VE":
        ve_cfg = sde_cfg["VE"]
        return VESDE(
            sigma_min=float(ve_cfg["sigma_min"]),
            sigma_max=float(ve_cfg["sigma_max"]),
        )

    raise ValueError(f"Unknown SDE: {sde_name}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # --- Setup ---
    set_seed(int(config["seed"]))
    device = torch.device(config["device"])
    out_dir = Path(config["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    x_train = make_dataset(
        str(config["data"]["type"]),
        int(config["data"]["n_train_samples"]),
        float(config["data"]["noise"]),
        int(config["seed"]),
    )

    # Standardize for stable training and save stats for inverse transform.
    data_mean = x_train.mean(dim=0, keepdim=True)
    data_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - data_mean) / data_std

    predictor = Predictor(
        predictor_type=PredictorType(config["model"]["predictor"]),
        x_dim=2,
        hidden=int(config["model"]["hidden"]),
    ).to(device)

    sde = build_sde(config)

    optimizer = Adam(
        predictor.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    steps = int(config["training"]["steps"])
    batch_size = int(config["training"]["batch_size"])
    eps = float(config["training"]["eps"])
    log_every = int(config["training"]["log_every"])

    print(f"Training on device={device} for {steps} steps")
    print(
        f"dataset={config['data']['type']} "
        f"n_train_samples={config['data']['n_train_samples']} "
        f"noise={config['data']['noise']}"
    )

    x_train = x_train.to(device)
    n_data = x_train.shape[0]

    for step in range(1, steps + 1):
        indices = torch.randint(0, n_data, (batch_size,), device=device)
        x0 = x_train[indices]

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(sde=sde, predictor=predictor, x0=x0, eps=eps)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        writer.add_scalar("train/loss", float(loss_val), step)

        if step % log_every == 0 or step == 1:
            print(f"step={step:6d} loss={loss_val:.6f}")

    ckpt_path = out_dir / "predictor.pt"
    torch.save(
        {
            "state_dict": predictor.state_dict(),
            "config": config,
            "data_mean": data_mean,
            "data_std": data_std,
        },
        ckpt_path,
    )
    print(f"Saved model checkpoint to {ckpt_path}")

    solver = DPMSolverPP2M(sde=sde, predictor=predictor, device=device)

    with torch.no_grad():
        x_t = sde.prior_sampling(
            torch.Size([int(config["sampling"]["num_samples"]), 2])
        ).to(device)
        history = solver.sample(x_t, steps=int(config["sampling"]["steps"]))

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

    sampling_fig = plot_sampling_process(history, x_range, y_range)
    sampling_html = out_dir / "samples.html"
    sampling_fig.write_html(str(sampling_html))
    print(f"Saved sampling plot to {sampling_html}")

    training_fig = plot_training_data(x_train_orig)
    training_html = out_dir / "training_data.html"
    training_fig.write_html(str(training_html))
    print(f"Saved training data plot to {training_html}")

    writer.close()


if __name__ == "__main__":
    main()
