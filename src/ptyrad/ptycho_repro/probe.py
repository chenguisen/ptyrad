"""Position-dependent probe generation for eLOP.

The core innovation of eLOP is a position-dependent probe: aberration
coefficients vary spatially across the scan, enabling thick-specimen imaging.

References
----------
Cui, J., Sha, H., Yang, W., & Yu, R. (2025). arXiv:2502.18294.
"""

import torch
import torch.nn as nn
from torch.nn.functional import grid_sample


def make_aberration_basis(
    grid_shape: tuple[int, int],
    n_modes: int = 6,
) -> torch.Tensor:
    """Compute Zernike-like aberration basis functions on a frequency grid.

    Parameters
    ----------
    grid_shape : tuple of int
        (ny, nx) size of the 2D grid.
    n_modes : int, optional
        Number of aberration modes. Default 6.
        Modes: 0=defocus, 1=astig0, 2=astig45, 3=spherical, 4=comax, 5=comay

    Returns
    -------
    torch.Tensor
        Basis functions, shape (n_modes, ny, nx), float32.
    """
    ny, nx = grid_shape
    cy, cx = ny // 2, nx // 2

    fy = (torch.arange(ny, dtype=torch.float32) - cy) / (ny // 2)
    fx = (torch.arange(nx, dtype=torch.float32) - cx) / (nx // 2)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")

    r2 = fxx**2 + fyy**2
    r = torch.sqrt(r2 + 1e-10)

    basis = []
    if n_modes >= 1:
        basis.append(r2)
    if n_modes >= 2:
        basis.append(fxx**2 - fyy**2)
    if n_modes >= 3:
        basis.append(2.0 * fxx * fyy)
    if n_modes >= 4:
        basis.append(r2**2)
    if n_modes >= 5:
        basis.append(fxx * r2)
    if n_modes >= 6:
        basis.append(fyy * r2)
    if n_modes > 6:
        for m in range(6, n_modes):
            order = m - 3
            basis.append(r**order * torch.cos(order * torch.atan2(fyy, fxx)))

    return torch.stack(basis).float()


class PositionDependentProbe(nn.Module):
    """Position-dependent electron probe generator.

    Aberration coefficients vary with scan position via a coarse coefficient
    grid with bilinear interpolation.

    Parameters
    ----------
    grid_shape : tuple of int
        (ny, nx) reconstruction grid size.
    aperture_radius_pixels : float
        Aperture radius in pixels.
    n_aberration_modes : int, optional
        Number of aberration modes. Default 6.
    scan_grid_shape : tuple of int, optional
        (gh, gw) coarse grid for spatial coefficient variation. Default (8, 8).
    aperture_smoothing : float, optional
        Smoothing width of aperture edge in pixels. Default 1.0.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        aperture_radius_pixels: float,
        n_aberration_modes: int = 6,
        scan_grid_shape: tuple[int, int] = (8, 8),
        aperture_smoothing: float = 1.0,
    ):
        super().__init__()
        ny, nx = grid_shape
        self.grid_shape = grid_shape
        self.aperture_radius = aperture_radius_pixels
        self.n_aberration_modes = n_aberration_modes
        self.scan_grid_shape = scan_grid_shape

        basis = make_aberration_basis(grid_shape, n_modes=n_aberration_modes)
        self.register_buffer("basis", basis)

        aperture = self._make_aperture(ny, nx, aperture_radius_pixels, aperture_smoothing)
        self.register_buffer("aperture", aperture)

        self.base_coefficients = nn.Parameter(
            torch.zeros(n_aberration_modes, dtype=torch.float32)
        )

        gh, gw = scan_grid_shape
        self.coeff_grid = nn.Parameter(
            torch.zeros(n_aberration_modes, gh, gw, dtype=torch.float32)
        )

    @staticmethod
    def _make_aperture(ny: int, nx: int, radius: float, smoothing: float = 1.0) -> torch.Tensor:
        cy, cx = ny // 2, nx // 2
        yy, xx = torch.meshgrid(
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        return torch.sigmoid((radius - r) / smoothing)

    def _interpolate_coefficients(self, positions: torch.Tensor) -> torch.Tensor:
        n_scan = positions.shape[0]
        gh, gw = self.scan_grid_shape
        half_range = max(gh, gw) * 10.0
        pos_norm = positions / half_range

        coeff_grid = self.coeff_grid.unsqueeze(0)
        grid = pos_norm.view(1, n_scan, 1, 2).to(device=coeff_grid.device, dtype=coeff_grid.dtype)

        interpolated = grid_sample(
            coeff_grid, grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        return interpolated.squeeze(0).squeeze(-1).T

    def forward(self, scan_positions: torch.Tensor) -> torch.Tensor:
        """Generate position-dependent probes.

        Parameters
        ----------
        scan_positions : torch.Tensor
            Shape (n_scan, 2), in pm.

        Returns
        -------
        torch.Tensor
            Complex probes, shape (n_scan, ny, nx), complex64.
        """
        device = scan_positions.device
        ny, nx = self.grid_shape

        delta_coeffs = self._interpolate_coefficients(scan_positions)
        total_coeffs = self.base_coefficients[None, :] + delta_coeffs

        basis = self.basis.to(device=device)
        chi = torch.einsum("sm,myn->syn", total_coeffs, basis)

        aperture = self.aperture.to(device=device)
        probe_k = aperture[None, :, :] * torch.exp(-1j * chi)

        probe_k = torch.fft.ifftshift(probe_k, dim=(-2, -1))
        probe_r = torch.fft.ifft2(probe_k)
        probe_r = torch.fft.fftshift(probe_r, dim=(-2, -1))

        return probe_r.to(torch.complex64)
