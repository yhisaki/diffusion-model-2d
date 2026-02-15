from __future__ import annotations

import numba as nb
import numpy as np
import torch
from sklearn.datasets import (
    make_circles,
    make_moons,
    make_s_curve,
    make_swiss_roll,
)

DATASETS = [
    "two_moons",
    "swiss_roll",
    "circles",
    "s_curve",
    "spiral",
    "pinwheel",
    "checkerboard",
]


def _make_spiral(n_samples: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.sqrt(rng.uniform(0, 1, n_samples)) * 3 * np.pi
    r = theta
    x = r * np.cos(theta) + rng.normal(0, noise, n_samples)
    y = r * np.sin(theta) + rng.normal(0, noise, n_samples)
    return np.column_stack([x, y])


def _make_pinwheel(
    n_samples: int, noise: float, seed: int, n_arms: int = 5
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_arm = n_samples // n_arms
    remainder = n_samples - per_arm * n_arms
    points = []
    for i in range(n_arms):
        n = per_arm + (1 if i < remainder else 0)
        r = rng.uniform(0.1, 1.0, n)
        angle_offset = 2 * np.pi * i / n_arms
        theta = r * 3.0 + angle_offset
        x = r * np.cos(theta) + rng.normal(0, noise, n)
        y = r * np.sin(theta) + rng.normal(0, noise, n)
        points.append(np.column_stack([x, y]))
    return np.concatenate(points, axis=0)


@nb.njit
def _checkerboard_reject(n_samples: int, grid: int, seed: int) -> np.ndarray:  # type: ignore[type-arg]
    """Rejection-sample points on even-parity cells (numba-accelerated)."""
    np.random.seed(seed)
    out = np.empty((n_samples, 2), dtype=np.float64)
    filled = 0
    while filled < n_samples:
        x = np.random.random() * grid
        y = np.random.random() * grid
        if (int(np.floor(x)) + int(np.floor(y))) % 2 == 0:
            out[filled, 0] = x
            out[filled, 1] = y
            filled += 1
    return out


def _make_checkerboard(
    n_samples: int, noise: float, seed: int, grid: int = 4
) -> np.ndarray:
    arr = _checkerboard_reject(n_samples, grid, seed)
    rng = np.random.default_rng(seed + 1)
    arr += rng.normal(0.0, noise, (n_samples, 2))
    return arr


def make_dataset(name: str, n_samples: int, noise: float, seed: int) -> torch.Tensor:
    """Create a 2D dataset by name.

    Args:
        name: One of ``DATASETS``.
        n_samples: Number of points.
        noise: Standard deviation of Gaussian noise added.
        seed: Random seed.

    Returns:
        Tensor of shape ``[n_samples, 2]``.
    """
    if name == "two_moons":
        x_np, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif name == "swiss_roll":
        x_3d, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
        x_np = x_3d[:, [0, 2]]
    elif name == "circles":
        x_np, _ = make_circles(
            n_samples=n_samples, noise=noise, random_state=seed, factor=0.5
        )
    elif name == "s_curve":
        x_3d, _ = make_s_curve(n_samples=n_samples, noise=noise, random_state=seed)
        x_np = x_3d[:, [0, 2]]
    elif name == "spiral":
        x_np = _make_spiral(n_samples, noise, seed)
    elif name == "pinwheel":
        x_np = _make_pinwheel(n_samples, noise, seed)
    elif name == "checkerboard":
        x_np = _make_checkerboard(n_samples, noise, seed)
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Valid: {DATASETS}")
    return torch.from_numpy(x_np).to(torch.float32)
