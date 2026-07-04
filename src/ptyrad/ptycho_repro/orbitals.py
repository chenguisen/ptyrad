"""Gaussian orbital representation for local-orbital ptychography.

LOP represents the object as a sum of Gaussian orbitals:

    O(r) = Σ_i A_i · exp(-|r - r_i|² / 2σ_i²)

Each orbital i has center position r_i = (x_i, y_i), complex amplitude
A_i, and width σ_i.
"""

import torch


def render_orbital(
    x: torch.Tensor,
    y: torch.Tensor,
    amplitude: torch.Tensor,
    sigma: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
) -> torch.Tensor:
    """Render a single Gaussian orbital onto a 2D grid.

    Parameters
    ----------
    x, y : torch.Tensor
        Orbital center position, scalar tensors (pm).
    amplitude : torch.Tensor
        Complex amplitude, scalar tensor.
    sigma : torch.Tensor
        Gaussian width, scalar tensor (pm).
    grid_x, grid_y : torch.Tensor
        2D meshgrid coordinates, shape (N, N) (pm).

    Returns
    -------
    torch.Tensor
        Orbital value on grid, shape (N, N), complex64.
    """
    dx = grid_x - x
    dy = grid_y - y
    r2 = dx**2 + dy**2
    return amplitude * torch.exp(-r2 / (2 * sigma**2))


def render_orbitals(
    positions: torch.Tensor,
    amplitudes: torch.Tensor,
    sigmas: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
) -> torch.Tensor:
    """Render multiple Gaussian orbitals onto a 2D grid as superposition.

    Parameters
    ----------
    positions : torch.Tensor
        Orbital center positions, shape (K, 2) (pm).
    amplitudes : torch.Tensor
        Complex amplitudes, shape (K,).
    sigmas : torch.Tensor
        Gaussian widths, shape (K,) (pm).
    grid_x, grid_y : torch.Tensor
        2D meshgrid coordinates, shape (N, N) (pm).

    Returns
    -------
    torch.Tensor
        Superposition of all orbitals on grid, shape (N, N), complex64.
    """
    K = positions.shape[0]
    obj = torch.zeros_like(grid_x, dtype=torch.complex64)
    for i in range(K):
        dx = grid_x - positions[i, 0]
        dy = grid_y - positions[i, 1]
        r2 = dx**2 + dy**2
        obj = obj + amplitudes[i] * torch.exp(-r2 / (2 * sigmas[i] ** 2))
    return obj


def render_orbitals_vectorized(
    positions: torch.Tensor,
    amplitudes: torch.Tensor,
    sigmas: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
) -> torch.Tensor:
    """Vectorized version: render all orbitals at once.

    Uses broadcasting for better GPU performance.

    Parameters
    ----------
    positions : torch.Tensor
        Orbital center positions, shape (K, 2) (pm).
    amplitudes : torch.Tensor
        Complex amplitudes, shape (K,).
    sigmas : torch.Tensor
        Gaussian widths, shape (K,) (pm).
    grid_x, grid_y : torch.Tensor
        2D meshgrid coordinates, shape (N, N) (pm).

    Returns
    -------
    torch.Tensor
        Superposition of all orbitals on grid, shape (N, N), complex64.
    """
    K = positions.shape[0]
    N = grid_x.shape[-1]

    # (K, 1, 1)
    px = positions[:, 0].view(K, 1, 1)
    py = positions[:, 1].view(K, 1, 1)
    amp = amplitudes.view(K, 1, 1)
    sig = sigmas.view(K, 1, 1)

    # (K, N, N) each
    dx = grid_x.unsqueeze(0) - px
    dy = grid_y.unsqueeze(0) - py
    r2 = dx**2 + dy**2
    gauss = amp * torch.exp(-r2 / (2 * sig**2))

    return gauss.sum(dim=0)  # (N, N)


# ─── 3D orbital functions for nLOT ────────────────────────────────────────────


def render_orbitals_3d(
    positions_3d: torch.Tensor,
    amplitudes: torch.Tensor,
    sigmas: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    grid_z: torch.Tensor,
) -> torch.Tensor:
    """Render 3D Gaussian orbitals on a 3D grid.

    Parameters
    ----------
    positions_3d : torch.Tensor
        3D positions, shape (K, 3) (pm).
    amplitudes : torch.Tensor
        Complex amplitudes, shape (K,).
    sigmas : torch.Tensor
        Isotropic widths, shape (K,) (pm).
    grid_x, grid_y, grid_z : torch.Tensor
        3D meshgrid coordinates (pm).

    Returns
    -------
    torch.Tensor
        3D orbital volume, complex64.
    """
    K = positions_3d.shape[0]
    obj = torch.zeros_like(grid_x, dtype=torch.complex64)
    for i in range(K):
        dx = grid_x - positions_3d[i, 0]
        dy = grid_y - positions_3d[i, 1]
        dz = grid_z - positions_3d[i, 2]
        r2 = dx**2 + dy**2 + dz**2
        obj = obj + amplitudes[i] * torch.exp(-r2 / (2 * sigmas[i] ** 2))
    return obj


def tilt_transform(
    positions_3d: torch.Tensor,
    tilt_angle: torch.Tensor,
    axis: str = "y",
) -> torch.Tensor:
    """Rotate 3D positions around a specified axis.

    Parameters
    ----------
    positions_3d : torch.Tensor
        3D positions, shape (K, 3) (pm).
    tilt_angle : torch.Tensor
        Rotation angle in radians.
    axis : str
        Rotation axis: 'x' or 'y'. Default 'y'.

    Returns
    -------
    torch.Tensor
        Rotated positions, shape (K, 3).
    """
    cos_a = torch.cos(tilt_angle)
    sin_a = torch.sin(tilt_angle)

    if axis == "y":
        rotated = torch.stack(
            [
                positions_3d[:, 0] * cos_a + positions_3d[:, 2] * sin_a,
                positions_3d[:, 1],
                -positions_3d[:, 0] * sin_a + positions_3d[:, 2] * cos_a,
            ],
            dim=1,
        )
    elif axis == "x":
        rotated = torch.stack(
            [
                positions_3d[:, 0],
                positions_3d[:, 1] * cos_a - positions_3d[:, 2] * sin_a,
                positions_3d[:, 1] * sin_a + positions_3d[:, 2] * cos_a,
            ],
            dim=1,
        )
    else:
        raise ValueError(f"Unknown axis: {axis}")

    return rotated


def project_to_slice(
    positions_3d: torch.Tensor,
    amplitudes: torch.Tensor,
    sigmas: torch.Tensor,
    tilt_angle: torch.Tensor,
    z_slice: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
) -> torch.Tensor:
    """Project 3D orbitals to a 2D slice for a given tilt angle and depth.

    Parameters
    ----------
    positions_3d : torch.Tensor
        3D orbital positions, shape (K, 3) (pm).
    amplitudes : torch.Tensor
        Complex amplitudes, shape (K,).
    sigmas : torch.Tensor
        Gaussian widths: shape (K,) isotropic (pm).
    tilt_angle : torch.Tensor
        Sample tilt angle in radians.
    z_slice : torch.Tensor
        Depth of target slice (pm).
    grid_x, grid_y : torch.Tensor
        2D meshgrid (pm).

    Returns
    -------
    torch.Tensor
        2D slice projection, shape (N, N), complex64.
    """
    rotated = tilt_transform(positions_3d, tilt_angle, axis="y")

    obj = torch.zeros_like(grid_x, dtype=torch.complex64)
    for i in range(positions_3d.shape[0]):
        sigma_2d = sigmas[i]
        sigma_z = sigmas[i]

        z_dist = (rotated[i, 2] - z_slice).abs()
        if z_dist > 3 * sigma_z:
            continue

        px, py = rotated[i, 0], rotated[i, 1]
        dx = grid_x - px
        dy = grid_y - py
        r2 = dx**2 + dy**2
        obj = obj + amplitudes[i] * torch.exp(-r2 / (2 * sigma_2d**2))

    return obj
