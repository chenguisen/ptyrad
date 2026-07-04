"""vis — Visualization utilities for orbital ptychography results.

Follows ptyrad's matplotlib conventions: returns plt.Figure when save_path=None.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_orbital_positions(
    positions,
    grid_shape=None,
    pixel_size=None,
    title="Orbital Positions",
    save_path=None,
):
    """Scatter plot of orbital positions.

    Parameters
    ----------
    positions : np.ndarray or torch.Tensor
        Shape (K, 2) in pm.
    grid_shape : tuple, optional
        (Ny, Nx) for axis limits.
    pixel_size : float, optional
        Pixel size in pm, for axis limits.
    title : str
    save_path : str, optional

    Returns
    -------
    plt.Figure or None
    """
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(positions[:, 0], positions[:, 1], s=15, alpha=0.7, c="C0")
    ax.set_aspect("equal")
    ax.set_xlabel("x (pm)")
    ax.set_ylabel("y (pm)")

    if grid_shape is not None and pixel_size is not None:
        half = max(grid_shape) * pixel_size / 2
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_rendered_object(
    obj,
    title="Reconstructed Object",
    cmap_amp="gray",
    cmap_phase="RdBu_r",
    save_path=None,
):
    """Plot rendered object amplitude and phase side by side.

    Parameters
    ----------
    obj : np.ndarray or torch.Tensor
        Complex-valued object wave, shape (Ny, Nx).
    title : str
    cmap_amp, cmap_phase : str
    save_path : str, optional

    Returns
    -------
    plt.Figure or None
    """
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(abs(obj), cmap=cmap_amp, origin="lower")
    ax1.set_title(f"{title} — Amplitude")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(np.angle(obj), cmap=cmap_phase, origin="lower")
    ax2.set_title(f"{title} — Phase")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    for ax in (ax1, ax2):
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_loss_curve(
    loss_iters,
    title="Loss Curve",
    save_path=None,
):
    """Plot reconstruction loss history.

    Parameters
    ----------
    loss_iters : list of (int, float)
        [(iter, loss), ...]
    title : str
    save_path : str, optional

    Returns
    -------
    plt.Figure or None
    """
    iters = [x[0] for x in loss_iters]
    losses = [x[1] for x in loss_iters]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(iters, losses, "b-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_diffraction_comparison(
    measured,
    predicted,
    idx=0,
    title="Diffraction Comparison",
    save_path=None,
):
    """Side-by-side comparison of measured vs predicted diffraction.

    Parameters
    ----------
    measured : np.ndarray or torch.Tensor
        Shape (N, Ny, Nx) or (Ny, Nx).
    predicted : np.ndarray or torch.Tensor
        Same shape.
    idx : int
        Index into batch if 3D.
    title : str
    save_path : str, optional

    Returns
    -------
    plt.Figure or None
    """
    if isinstance(measured, torch.Tensor):
        measured = measured.detach().cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()

    if measured.ndim == 3 and predicted.ndim == 3:
        measured = measured[idx]
        predicted = predicted[idx]

    vmax = max(measured.max(), predicted.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Measured", "Predicted", "Residual"]
    data = [measured, predicted, measured - predicted]
    cmaps = ["gray", "gray", "RdBu_r"]

    for ax, d, t, cm in zip(axes, data, titles, cmaps):
        im = ax.imshow(d, cmap=cm, origin="lower", vmin=0, vmax=vmax if cm != "RdBu_r" else None)
        ax.set_title(t)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_probe(probe, title="Reconstructed Probe", save_path=None):
    """Plot probe amplitude and phase.

    Parameters
    ----------
    probe : np.ndarray or torch.Tensor
        Complex probe, shape (Ny, Nx).
    title : str
    save_path : str, optional

    Returns
    -------
    plt.Figure or None
    """
    if isinstance(probe, torch.Tensor):
        probe = probe.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(abs(probe), cmap="gray", origin="lower")
    ax1.set_title("Probe Amplitude")
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    im2 = ax2.imshow(np.angle(probe), cmap="RdBu_r", origin="lower")
    ax2.set_title("Probe Phase")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
