from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.artist as martist
import matplotlib.pyplot as plt
import torch


def plot_training_data(
    x_train: torch.Tensor,
    save_path: Path,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
) -> None:
    """Save a scatter plot of the training data distribution.

    Args:
        x_train: Training data [M, 2] (original scale).
        save_path: Output path for the PNG file.
        x_range: Optional [min, max] for x-axis.
        y_range: Optional [min, max] for y-axis.
    """
    train_np = x_train.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(train_np[:, 0], train_np[:, 1], s=2, alpha=0.4, c="gray")
    ax.set_title("Training Data")
    ax.set_aspect("equal")
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_generated_data(
    samples: torch.Tensor,
    save_path: Path,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
) -> None:
    """Save a scatter plot of the generated samples.

    Args:
        samples: Generated data [N, 2] (original scale).
        save_path: Output path for the PNG file.
        x_range: Optional [min, max] for x-axis.
        y_range: Optional [min, max] for y-axis.
    """
    pts = samples.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.4, c="blue")
    ax.set_title("Generated Samples")
    ax.set_aspect("equal")
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sampling_gif(
    history: list[torch.Tensor],
    save_path: Path,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    interval: int = 100,
) -> None:
    """Save a GIF animation of the sampling (denoising) process.

    Args:
        history: List of [N, 2] tensors from solver.sample() (already denormalized).
        save_path: Output path for the GIF file.
        x_range: Optional [min, max] for x-axis.
        y_range: Optional [min, max] for y-axis.
        interval: Delay between frames in milliseconds.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter([], [], s=2, alpha=0.4, c="blue")
    ax.set_title("Sampling Process")
    ax.set_aspect("equal")
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)

    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")

    def update(frame: int) -> tuple[martist.Artist, martist.Artist]:
        pts = history[frame].cpu().numpy()
        scatter.set_offsets(pts)
        step_text.set_text(f"Step {frame}/{len(history) - 1}")
        return scatter, step_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(history), interval=interval, blit=True
    )
    anim.save(str(save_path), writer="pillow")
    plt.close(fig)
