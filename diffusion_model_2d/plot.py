from __future__ import annotations

import plotly.graph_objects as go
import torch


def plot_sampling_process(
    history: list[torch.Tensor],
    x_range: list[float],
    y_range: list[float],
) -> go.Figure:
    """Build an interactive plotly figure showing the denoising process with a slider.

    Args:
        history: List of [N, 2] tensors from solver.sample() (already denormalized).
        x_range: [min, max] for x-axis.
        y_range: [min, max] for y-axis.

    Returns:
        A plotly Figure with slider animation.
    """
    n_steps = len(history)

    # Initial data (step 0)
    init_pts = history[0].cpu().numpy()
    fig = go.Figure(
        data=[
            go.Scatter(
                x=init_pts[:, 0],
                y=init_pts[:, 1],
                mode="markers",
                marker=dict(size=2, opacity=0.4, color="blue"),
            ),
        ],
    )

    # Build one frame per step
    frames = []
    for i, h in enumerate(history):
        pts = h.cpu().numpy()
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode="markers",
                        marker=dict(size=2, opacity=0.4, color="blue"),
                    ),
                ],
                name=str(i),
            )
        )

    fig.frames = frames

    # Slider
    sliders = [
        dict(
            active=0,
            currentvalue=dict(prefix="Step: "),
            pad=dict(t=50),
            steps=[
                dict(
                    args=[[str(i)], dict(mode="immediate", frame=dict(duration=0))],
                    method="animate",
                    label=str(i),
                )
                for i in range(n_steps)
            ],
        )
    ]

    fig.update_layout(
        title="Sampling Process",
        xaxis=dict(range=x_range, scaleanchor="y"),
        yaxis=dict(range=y_range),
        sliders=sliders,
        width=800,
        height=700,
        showlegend=False,
    )

    return fig


def plot_training_data(
    x_train: torch.Tensor,
) -> go.Figure:
    """Build a plotly figure showing the training data distribution.

    Args:
        x_train: Training data [M, 2] (original scale).

    Returns:
        A plotly Figure.
    """
    train_np = x_train.cpu().numpy()

    fig = go.Figure(
        data=[
            go.Scatter(
                x=train_np[:, 0],
                y=train_np[:, 1],
                mode="markers",
                marker=dict(size=2, opacity=0.4, color="gray"),
            ),
        ],
    )

    fig.update_layout(
        title="Training Data",
        xaxis=dict(scaleanchor="y"),
        width=800,
        height=700,
        showlegend=False,
    )

    return fig
