"""ELOPOrbitalModel — Extended Local-orbital Ptychography model.

Combines position-dependent probe (spatially-varying aberration coefficients)
with Gaussian orbital object representation. Compatible with ptyrad's recon_loop.

References
----------
Cui, J., Sha, H., Yang, W., & Yu, R. (2025). arXiv:2502.18294.
"""

from collections import defaultdict
import logging
import torch
import torch.nn as nn
from torch.fft import fft2

from ptyrad.io.dataloader import MeasDataLoader
from ptyrad.ptycho_repro.orbitals import render_orbitals_vectorized
from ptyrad.ptycho_repro.probe import PositionDependentProbe

logger = logging.getLogger(__name__)


class ELOPOrbitalModel(nn.Module):
    """eLOP model with position-dependent probe + orbital object.

    Parameters
    ----------
    init_variables : dict
        Required: measurements, scan_positions, probe (unused with pos-dep probe),
        n_orbitals, grid_shape, pixel_size, wavelength, n_aberration_modes,
        scan_grid_shape, aperture_radius_fraction.
    model_params : dict
        update_params with learning rates.
    device : str
    """

    def __init__(self, init_variables, model_params, device='cuda'):
        super().__init__()
        self.device = device

        n_orb = init_variables['n_orbitals']
        grid_shape = tuple(init_variables['grid_shape'])
        ny, nx = grid_shape
        self.pixel_size = init_variables['pixel_size']
        self.wavelength = init_variables.get('wavelength', 2.51)
        self.grid_shape = grid_shape
        self.center_y, self.center_x = ny // 2, nx // 2

        # Data loader
        self.meas_loader = MeasDataLoader(
            init_variables['measurements'],
            preload_data=model_params.get('preload_data', True),
            device=self.device,
        )

        # Coordinate grids for orbital rendering
        half_x = (nx // 2) * self.pixel_size
        half_y = (ny // 2) * self.pixel_size
        xv = torch.linspace(-half_x, half_x, nx, device=device)
        yv = torch.linspace(-half_y, half_y, ny, device=device)
        gx, gy = torch.meshgrid(xv, yv, indexing="ij")
        self.register_buffer("grid_x", gx)
        self.register_buffer("grid_y", gy)

        # 2D orbital parameters
        spread = init_variables.get('init_position_spread', 20.0)
        isigma = init_variables.get('init_sigma', 30.0)
        iamp = init_variables.get('init_amplitude', 1.0)
        self.orbital_positions = nn.Parameter(
            (torch.rand(n_orb, 2, device=device) - 0.5) * spread
        )
        self.orbital_amplitudes = nn.Parameter(
            torch.full((n_orb,), iamp, dtype=torch.complex64, device=device)
        )
        self.orbital_sigmas = nn.Parameter(
            torch.full((n_orb,), isigma, dtype=torch.float32, device=device)
        )

        # Position-dependent probe
        n_modes = init_variables.get('n_aberration_modes', 6)
        sg_shape = init_variables.get('scan_grid_shape', (8, 8))
        aperture_frac = init_variables.get('aperture_radius_fraction', 0.4)
        ap_radius_px = min(ny, nx) // 2 * aperture_frac
        self.probe_gen = PositionDependentProbe(
            grid_shape=grid_shape, aperture_radius_pixels=ap_radius_px,
            n_aberration_modes=n_modes, scan_grid_shape=sg_shape,
        )

        # Scan positions
        sp = init_variables['scan_positions']
        self.register_buffer("scan_positions", torch.tensor(sp, dtype=torch.float32, device=device))

        # LR schedule
        up = model_params['update_params']
        self.lr_params = {}
        self.start_iter = {}
        self.end_iter = {}
        for key, cfg in up.items():
            self.lr_params[key] = cfg.get('lr', 0.0)
            self.start_iter[key] = cfg.get('start_iter')
            self.end_iter[key] = cfg.get('end_iter')

        self.optimizable_tensors = {
            'orbital_positions': self.orbital_positions,
            'orbital_amplitudes': self.orbital_amplitudes,
            'orbital_sigmas': self.orbital_sigmas,
            'base_coefficients': self.probe_gen.base_coefficients,
            'coeff_grid': self.probe_gen.coeff_grid,
        }
        opt_p = model_params['optimizer_params']
        self.optimizer_params = {'name': opt_p.get('name', 'Adam')}
        if 'configs' in opt_p and opt_p['configs']:
            self.optimizer_params['configs'] = opt_p['configs']
        self.scheduler_params = model_params.get('scheduler_params', None)

        self.optimizable_params = []
        for name, tensor in self.optimizable_tensors.items():
            lr = self.lr_params.get(name, 0.0)
            tensor.requires_grad = (lr != 0) and (self.start_iter.get(name, 0) == 1)
            if lr != 0:
                self.optimizable_params.append({'params': [tensor], 'lr': lr})

        self.register_buffer("omode_occu", torch.tensor([1.0], device=device))
        self.opt_obj_tilts = nn.Parameter(torch.zeros(1, 2, device=device), requires_grad=False)
        self.opt_slice_thickness = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
        self.compilation_iters = {1}
        self.loss_iters = []
        self.iter_times = []
        self.dz_iters = []
        self.avg_tilt_iters = defaultdict(list)
        self.lr_iters = defaultdict(list)
        self.recon_provenance = init_variables.get('recon_provenance', [])
        self._current_object_patches = None

    def get_complex_probe_view(self):
        return torch.view_as_complex(self.opt_probe) if hasattr(self, 'opt_probe') else self.probe_gen(torch.zeros(1, 2, device=self.device))[0:1]

    def get_measurements(self, indices):
        return self.meas_loader[indices]

    def clear_cache(self):
        self._current_object_patches = None

    def forward(self, batch_indices):
        ny, nx = self.grid_shape
        cy, cx = self.center_y, self.center_x

        # Render orbital object
        obj = render_orbitals_vectorized(
            self.orbital_positions, self.orbital_amplitudes, self.orbital_sigmas,
            self.grid_x, self.grid_y,
        )

        batch_size = batch_indices.shape[0]
        diffractions = torch.zeros(batch_size, ny, nx, device=self.device)

        for i, idx in enumerate(batch_indices):
            pos = self.scan_positions[idx]
            sx = pos[0] / self.pixel_size + cx
            sy = pos[1] / self.pixel_size + cy
            sx_i = int(torch.round(sx).item())
            sy_i = int(torch.round(sy).item())

            # Generate position-dependent probe for this one position
            probe_i = self.probe_gen(pos.unsqueeze(0))[0]
            shifted = torch.roll(probe_i, shifts=(sy_i - cy, sx_i - cx), dims=(0, 1))
            ew = shifted * obj
            dp = fft2(ew)
            r = torch.view_as_real(dp)
            diffractions[i] = r[..., 0]**2 + r[..., 1]**2

        self._current_object_patches = (torch.tensor([], device=self.device), torch.tensor([], device=self.device))
        return diffractions
