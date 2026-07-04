"""LOPOrbitalModel — Local-Orbital Ptychography model compatible with PtyRAD's recon_loop.

Forward model:
    O(r) = Σ_i A_i · exp(-|r - r_i|² / 2σ_i²)
    ψ_j(r) = P(r - r_j) · O(r)
    I_j(k) = |F[ψ_j(r)]|²

Attributes compatible with ptyrad's recon_loop:
    - forward(batch) -> diffractions
    - get_measurements(indices) -> measured data
    - get_complex_probe_view() -> complex probe
    - optimizable_tensors, lr_params, start_iter, end_iter
    - _current_object_patches, omode_occu
    - loss_iters, iter_times, lr_iters (logging)
    - clear_cache()
"""

from collections import defaultdict
import logging
import torch
import torch.nn as nn
from torch.fft import fft2

from ptyrad.io.dataloader import MeasDataLoader

from ptyrad.ptycho_repro.orbitals import render_orbitals, render_orbitals_vectorized

logger = logging.getLogger(__name__)


class LOPOrbitalModel(nn.Module):
    """LOP orbital ptychography model.

    Parameters
    ----------
    init_variables : dict
        Must contain:
            - 'measurements': (N, Ny, Nx) diffraction data
            - 'scan_positions': (N, 2) scan positions in pm
            - 'probe': (Ny, Nx) complex probe
            - 'n_orbitals': int
            - 'grid_shape': (Ny, Nx)
            - 'pixel_size': float (pm)
            - 'init_position_spread': float (pm)
            - 'init_sigma': float (pm)
            - 'dx': float (real-space pixel size in Å)
            - 'dk': float (reciprocal-space pixel size)
    model_params : dict
        PtyRAD-style model parameters. Must have:
            - 'update_params': dict of {param_name: {'lr': float, 'start_iter': int}}
            - 'optimizer_params': dict
    device : str
        Device string, default 'cuda'.
    """

    def __init__(self, init_variables, model_params, device='cuda'):
        super().__init__()
        self.device = device

        # ── Unpack init variables ──────────────────────────────────────────
        n_orb   = init_variables['n_orbitals']
        grid_shape = init_variables['grid_shape']
        ny, nx    = grid_shape
        self.pixel_size = init_variables['pixel_size']
        self.n_orbitals = n_orb

        # Coordinate grid for rendering (pm, centered at 0)
        half_x = (nx // 2) * self.pixel_size
        half_y = (ny // 2) * self.pixel_size
        xv = torch.linspace(-half_x, half_x, nx, device=device)
        yv = torch.linspace(-half_y, half_y, ny, device=device)
        self.register_buffer("grid_x", xv[None, :].expand(ny, nx))
        self.register_buffer("grid_y", yv[:, None].expand(ny, nx))
        self.center_px = nx // 2
        self.center_py = ny // 2

        # ── Data loader ────────────────────────────────────────────────────
        preload = model_params.get('preload_data', True)
        self.meas_loader = MeasDataLoader(
            init_variables['measurements'],
            preload_data=preload,
            device=self.device,
        )

        # ── Orbital parameters (optimizable) ───────────────────────────────
        init_spread = init_variables.get('init_position_spread', 30.0)
        init_sigma  = init_variables.get('init_sigma', 30.0)
        init_amp    = init_variables.get('init_amplitude', 1.0)

        self.orbital_positions = nn.Parameter(
            torch.randn(n_orb, 2, device=device) * init_spread
        )
        self.orbital_amplitudes = nn.Parameter(
            torch.full((n_orb,), init_amp, dtype=torch.complex64, device=device)
        )
        self.orbital_sigmas = nn.Parameter(
            torch.full((n_orb,), init_sigma, dtype=torch.float32, device=device)
        )

        # ── Probe (like PtychoModel: (pmode, Ny, Nx) as real/imag pair) ────
        probe = init_variables['probe']  # complex (Ny, Nx)
        probe_t = torch.tensor(probe, dtype=torch.complex64, device=device)
        self.opt_probe = nn.Parameter(
            torch.view_as_real(probe_t[None, ...])
        )  # (1, Ny, Nx, 2)

        # ── Scan positions with sub-pixel refinements ──────────────────────
        scan_pos = init_variables['scan_positions']  # (N, 2) in pm
        self.register_buffer(
            "scan_positions",
            torch.tensor(scan_pos, dtype=torch.float32, device=device),
        )
        self.opt_probe_pos_shifts = nn.Parameter(
            torch.zeros(scan_pos.shape[0], 2, dtype=torch.float32, device=device)
        )

        # ── Learning-rate schedule & grad control ──────────────────────────
        update_params = model_params['update_params']
        self.lr_params = {}
        self.start_iter = {}
        self.end_iter = {}
        for key, cfg in update_params.items():
            self.lr_params[key]    = cfg.get('lr', 0.0)
            self.start_iter[key]   = cfg.get('start_iter')
            self.end_iter[key]     = cfg.get('end_iter')

        # ── Optimizable tensors dict (for grad toggling in recon_loop) ────
        self.optimizable_tensors = {
            'orbital_positions':  self.orbital_positions,
            'orbital_amplitudes': self.orbital_amplitudes,
            'orbital_sigmas':     self.orbital_sigmas,
            'probe':              self.opt_probe,
            'probe_pos_shifts':   self.opt_probe_pos_shifts,
        }

        # ── Optimizer params & optimizable params list ───────────────────────
        opt_params = model_params['optimizer_params']
        self.optimizer_params = {'name': opt_params.get('name', 'Adam')}
        if 'configs' in opt_params and opt_params['configs']:
            self.optimizer_params['configs'] = opt_params['configs']

        # Build optimizable_params list (for create_optimizer)
        self.optimizable_params = []
        for name, tensor in self.optimizable_tensors.items():
            lr = self.lr_params.get(name, 0.0)
            tensor.requires_grad = (lr != 0) and (self.start_iter.get(name, 0) == 1)
            if lr != 0:
                self.optimizable_params.append({'params': [tensor], 'lr': lr})

        # ── LR scheduler ───────────────────────────────────────────────────
        self.scheduler_params = model_params.get('scheduler_params', None)

        # ── Compatibility buffers (for loss & logging in recon_loop) ───────
        self.register_buffer(
            "omode_occu",
            torch.tensor([1.0], dtype=torch.float32, device=device),
        )
        # Dummy attributes for ptyrad recon_loop compatibility
        self.opt_obj_tilts = nn.Parameter(
            torch.zeros(1, 2, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.opt_slice_thickness = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device),
            requires_grad=False,
        )

        # ── Logging (recon_loop expects these) ─────────────────────────────
        self.compilation_iters  = {1}
        self.loss_iters         = []
        self.iter_times         = []
        self.dz_iters           = []
        self.avg_tilt_iters     = defaultdict(list)
        self.lr_iters           = defaultdict(list)
        self.recon_provenance   = init_variables.get('recon_provenance', [])
        self._current_object_patches = None

        self._log_summary()

    # ── Public API for ptyrad's save/compat ────────────────────────────────

    def get_complex_probe_view(self):
        """Return complex probe tensor (same API as PtychoModel)."""
        return torch.view_as_complex(self.opt_probe)

    def get_measurements(self, indices):
        """Load measured diffraction data for the given indices."""
        return self.meas_loader[indices]

    def clear_cache(self):
        self._current_object_patches = None

    def render_object(self):
        """Render the orbital representation to a pixel image.

        Returns
        -------
        torch.Tensor
            Complex object wave, shape (Ny, Nx), complex64.
        """
        return render_orbitals_vectorized(
            positions=self.orbital_positions,
            amplitudes=self.orbital_amplitudes,
            sigmas=self.orbital_sigmas,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
        )

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(self, batch_indices):
        """Forward pass: compute diffraction patterns for a batch of positions.

        Parameters
        ----------
        batch_indices : torch.Tensor
            1D tensor of scan-position indices for this batch.

        Returns
        -------
        torch.Tensor
            Diffraction patterns, shape (B, Ny, Nx), float32.
        """
        ny, nx = self.grid_x.shape

        # Render the full object once
        obj = self.render_object()  # (Ny, Nx) complex

        batch_size = batch_indices.shape[0]
        diffractions = torch.zeros(batch_size, ny, nx, device=self.device)

        for i, idx in enumerate(batch_indices):
            # Combined scan position (initial + learned shift)
            pos = self.scan_positions[idx] + self.opt_probe_pos_shifts[idx]

            # Convert to pixel shift from center
            sx = pos[0] / self.pixel_size + self.center_px
            sy = pos[1] / self.pixel_size + self.center_py
            shift_x = int(torch.round(sx).item())
            shift_y = int(torch.round(sy).item())

            if 0 <= shift_x < nx and 0 <= shift_y < ny:
                # Roll object so the scan position aligns with probe center
                shifted_obj = torch.roll(
                    obj,
                    shifts=(shift_y - self.center_py, shift_x - self.center_px),
                    dims=(0, 1),
                )
                exit_wave = self.get_complex_probe_view().squeeze(0) * shifted_obj

                # |FFT(exit_wave)|²  (avoid .abs() on complex — buggy on some CUDA)
                dp = fft2(exit_wave)
                r = torch.view_as_real(dp)
                diffractions[i] = r[..., 0]**2 + r[..., 1]**2

        # Store for loss computation (dummy patches — LOP doesn't use pixel patches)
        self._current_object_patches = (
            torch.tensor([], device=self.device),
            torch.tensor([], device=self.device),
        )

        return diffractions

    # ── Internals ──────────────────────────────────────────────────────────

    def _log_summary(self):
        """Log model configuration (mirrors PtychoModel.print_model_summary)."""
        logger.info("### LOPOrbitalModel optimizable variables ###")
        for name, tensor in self.optimizable_tensors.items():
            lr = self.lr_params.get(name, 0)
            logger.info(
                f"{name.ljust(20)}: {str(tensor.shape).ljust(24)}  "
                f"lr={lr:.0e}  requires_grad={tensor.requires_grad}"
            )
        n_meas = self.meas_loader.meas_arr.size if hasattr(self.meas_loader, 'meas_arr') else 0
        n_var  = sum(t.numel() for t in self.optimizable_tensors.values() if t.requires_grad)
        logger.info(f"Total measurements: {n_meas:,d}")
        logger.info(f"Total variables:    {n_var:,d}")
        if n_var > 0:
            logger.info(f"Overdetermined ratio: {n_meas / n_var:.2f}")
