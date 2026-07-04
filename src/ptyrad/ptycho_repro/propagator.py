"""Adaptive Fresnel propagator for APP (Adaptive Propagation Factor) ptychography.

Uses the exact Angular Spectrum Method (ASM) for wave propagation:

    H(Kx, Ky) = exp(i * dz * sqrt(k^2 - Kx^2 - Ky^2))

With tilt correction (APP's innovation):
    H_adj = exp(i * dz * (Kz + Ky*tan(theta_y) + Kx*tan(theta_x)))

References
----------
Sha, H., Cui, J., & Yu, R. (2022). Science Advances, 8(19), eabn2275.
"""

import torch
import torch.nn as nn


class AdaptivePropagator(nn.Module):
    """Differentiable ASM propagator with adaptive dz and tilt angles.

    Parameters
    ----------
    grid_shape : tuple of int
        (ny, nx) size of the 2D grid.
    pixel_size : float
        Real-space pixel size (same unit as wavelength).
    wavelength : float
        Electron wavelength (same unit as pixel_size).
    """

    def __init__(self, grid_shape: tuple, pixel_size: float, wavelength: float):
        super().__init__()
        ny, nx = grid_shape
        self.grid_shape = grid_shape
        self.pixel_size = pixel_size
        self.wavelength = wavelength

        fx = torch.fft.fftfreq(nx, d=pixel_size)
        fy = torch.fft.fftfreq(ny, d=pixel_size)
        Kx_grid, Ky_grid = torch.meshgrid(2 * torch.pi * fx, 2 * torch.pi * fy, indexing="ij")
        self.register_buffer("Kx", Kx_grid)
        self.register_buffer("Ky", Ky_grid)
        k = 2 * torch.pi / wavelength
        self.register_buffer("k", torch.tensor(k, dtype=torch.float32))

    def forward(
        self,
        dz: torch.Tensor,
        theta_x: torch.Tensor | None = None,
        theta_y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ASM propagation transfer function H.

        Parameters
        ----------
        dz : torch.Tensor
            Propagation distance(s). Scalar or 1-D (n_layers,).
        theta_x, theta_y : torch.Tensor, optional
            Tilt angle(s) in radians.

        Returns
        -------
        torch.Tensor
            Complex propagator H: shape (ny, nx) or (n, ny, nx).
        """
        inputs_are_all_scalar = dz.ndim == 0
        if theta_x is not None:
            inputs_are_all_scalar = inputs_are_all_scalar and theta_x.ndim == 0
        if theta_y is not None:
            inputs_are_all_scalar = inputs_are_all_scalar and theta_y.ndim == 0

        if dz.ndim == 0:
            dz = dz.unsqueeze(0)

        device = dz.device
        dtype = dz.dtype
        Kx = self.Kx.to(device=device, dtype=dtype)
        Ky = self.Ky.to(device=device, dtype=dtype)
        k = self.k.to(device=device, dtype=dtype)

        dz_batch = dz[:, None, None]
        kz_sq = torch.clamp(k**2 - Kx**2 - Ky**2, min=0.0)
        Kz = torch.sqrt(kz_sq)
        phase = dz_batch * Kz[None, :, :]

        if theta_x is not None:
            tx = theta_x
            if tx.dtype != dtype:
                tx = tx.to(dtype)
            if tx.ndim == 0:
                tx = tx.unsqueeze(0)
            phase = phase + dz_batch * Kx[None, :, :] * torch.tan(tx)[:, None, None]

        if theta_y is not None:
            ty = theta_y
            if ty.dtype != dtype:
                ty = ty.to(dtype)
            if ty.ndim == 0:
                ty = ty.unsqueeze(0)
            phase = phase + dz_batch * Ky[None, :, :] * torch.tan(ty)[:, None, None]

        H = torch.exp(1j * phase)
        if inputs_are_all_scalar:
            return H[0]
        return H
