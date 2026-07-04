"""APPOrbitalModel — Adaptive Propagation Factor Ptychography model.

Multi-slice forward model with adaptive propagators and learnable dz/tilt.
Compatible with ptyrad's recon_loop interface.

References
----------
Sha, H., Cui, J., & Yu, R. (2022). Science Advances, 8(19), eabn2275.
"""

from collections import defaultdict
import logging
import torch
import torch.nn as nn
from torch.fft import fft2, ifft2

from ptyrad.io.dataloader import MeasDataLoader
from ptyrad.ptycho_repro.propagator import AdaptivePropagator

logger = logging.getLogger(__name__)


class APPOrbitalModel(nn.Module):
    """APP multi-slice ptychography model with adaptive propagators.

    Parameters
    ----------
    init_variables : dict
        Must contain: measurements, scan_positions, probe, n_layers,
        grid_shape, pixel_size, wavelength.
    model_params : dict
        update_params with learning rates for obj_amp, obj_phase, dz, tilt, etc.
    device : str
    """

    def __init__(self, init_variables, model_params, device='cuda'):
        super().__init__()
        self.device = device

        n_layers = init_variables['n_layers']
        grid_shape = tuple(init_variables['grid_shape'])
        ny, nx = grid_shape
        self.pixel_size = init_variables['pixel_size']
        self.wavelength = init_variables.get('wavelength', 2.51)
        self.n_layers = n_layers
        self.grid_shape = grid_shape

        # Data loader
        self.meas_loader = MeasDataLoader(
            init_variables['measurements'],
            preload_data=model_params.get('preload_data', True),
            device=self.device,
        )

        # Object layers: amplitude + phase
        self.obj_amp = nn.Parameter(
            torch.ones(n_layers, ny, nx, device=device) * 0.9
        )
        self.obj_phase = nn.Parameter(
            torch.zeros(n_layers, ny, nx, device=device)
        )

        # Adaptive propagator
        self.propagator = AdaptivePropagator(
            grid_shape=grid_shape, pixel_size=self.pixel_size,
            wavelength=self.wavelength,
        )

        # Learnable dz and tilt
        self.dz = nn.Parameter(
            torch.full((max(1, n_layers - 1),), init_variables.get('init_dz', 2.0), device=device)
        )
        self.tilt = nn.Parameter(
            torch.zeros(max(1, n_layers - 1), 2, device=device)
        )

        # Probe
        probe_t = torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device)
        self.opt_probe = nn.Parameter(torch.view_as_real(probe_t[None, ...]))

        # Scan positions
        scan_pos = init_variables['scan_positions']
        self.register_buffer("scan_positions", torch.tensor(scan_pos, dtype=torch.float32, device=device))
        self.opt_probe_pos_shifts = nn.Parameter(
            torch.zeros(scan_pos.shape[0], 2, device=device)
        )

        # LR schedule
        update_p = model_params['update_params']
        self.lr_params = {}
        self.start_iter = {}
        self.end_iter = {}
        for key, cfg in update_p.items():
            self.lr_params[key] = cfg.get('lr', 0.0)
            self.start_iter[key] = cfg.get('start_iter')
            self.end_iter[key] = cfg.get('end_iter')

        # Optimizable tensors
        self.optimizable_tensors = {
            'obj_amp': self.obj_amp,
            'obj_phase': self.obj_phase,
            'dz': self.dz,
            'tilt': self.tilt,
            'probe': self.opt_probe,
            'probe_pos_shifts': self.opt_probe_pos_shifts,
        }

        # Optimizer & scheduler params
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

        # Compatibility with recon_loop
        self.register_buffer("omode_occu", torch.tensor([1.0], device=device))
        self.opt_obj_tilts = self.tilt  # reuse
        self.opt_slice_thickness = nn.Parameter(
            torch.tensor(1.0, device=device), requires_grad=False
        )
        self.compilation_iters = {1}
        self.loss_iters = []
        self.iter_times = []
        self.dz_iters = []
        self.avg_tilt_iters = defaultdict(list)
        self.lr_iters = defaultdict(list)
        self.recon_provenance = init_variables.get('recon_provenance', [])
        self._current_object_patches = None

    def get_complex_probe_view(self):
        return torch.view_as_complex(self.opt_probe)

    def get_measurements(self, indices):
        return self.meas_loader[indices]

    def clear_cache(self):
        self._current_object_patches = None

    def forward(self, batch_indices):
        ny, nx = self.grid_shape
        center_y, center_x = ny // 2, nx // 2

        # Pre-compute propagators
        if self.n_layers > 1:
            H = self.propagator(dz=self.dz, theta_x=self.tilt[:, 0], theta_y=self.tilt[:, 1])
        else:
            H = torch.ones(ny, nx, dtype=torch.complex64, device=self.device)

        batch_size = batch_indices.shape[0]
        diffractions = torch.zeros(batch_size, ny, nx, device=self.device)
        probe = self.get_complex_probe_view().squeeze(0)

        for i, idx in enumerate(batch_indices):
            pos = self.scan_positions[idx] + self.opt_probe_pos_shifts[idx]
            sx = pos[0] / self.pixel_size + center_x
            sy = pos[1] / self.pixel_size + center_y
            shift_x = int(torch.round(sx).item())
            shift_y = int(torch.round(sy).item())

            psi = probe.clone()
            for n in range(self.n_layers):
                obj_n = torch.polar(self.obj_amp[n], self.obj_phase[n])
                shifted_obj = torch.roll(obj_n, shifts=(shift_y - center_y, shift_x - center_x), dims=(0, 1))
                psi = psi * shifted_obj
                if n < self.n_layers - 1:
                    H_n = H if H.ndim == 2 else H[n]
                    pk = fft2(psi) * H_n
                    psi = ifft2(pk)

            dp = fft2(psi)
            r = torch.view_as_real(dp)
            diffractions[i] = r[..., 0]**2 + r[..., 1]**2

        self._current_object_patches = (torch.tensor([], device=self.device), torch.tensor([], device=self.device))
        return diffractions
