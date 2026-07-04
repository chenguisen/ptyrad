"""NLOTOrbitalModel — Multiple-section Local-orbital Tomography model.

3D Gaussian orbitals projected to 2D slices for multi-tilt, multi-defocus
ptychography. Compatible with ptyrad's recon_loop interface.

References
----------
Mao, L., Cui, J., & Yu, R. (2025). Science Bulletin, 70(1), 64-69.
"""

from collections import defaultdict
import logging
import torch
import torch.nn as nn
from torch.fft import fft2

from ptyrad.io.dataloader import MeasDataLoader
from ptyrad.ptycho_repro.orbitals import project_to_slice

logger = logging.getLogger(__name__)


class NLOTOrbitalModel(nn.Module):
    """3D orbital tomography model with multi-tilt and multi-defocus.

    Parameters
    ----------
    init_variables : dict
        Required: measurements, scan_positions, probe, n_orbitals, grid_shape,
        pixel_size, wavelength, tilt_angles, defocus_values.
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

        # Data loader
        self.meas_loader = MeasDataLoader(
            init_variables['measurements'],
            preload_data=model_params.get('preload_data', True),
            device=self.device,
        )

        # Coordinate grids
        half_x = (nx // 2) * self.pixel_size
        half_y = (ny // 2) * self.pixel_size
        xv = torch.linspace(-half_x, half_x, nx, device=device)
        yv = torch.linspace(-half_y, half_y, ny, device=device)
        gx, gy = torch.meshgrid(xv, yv, indexing="ij")
        self.register_buffer("grid_x", gx)
        self.register_buffer("grid_y", gy)

        # FFT frequency grid for defocus propagation
        fx = torch.fft.fftfreq(nx, d=self.pixel_size, device=device)
        fy = torch.fft.fftfreq(ny, d=self.pixel_size, device=device)
        fxx, fyy = torch.meshgrid(fx, fy, indexing="ij")
        self.register_buffer("k2", fxx**2 + fyy**2)

        # 3D orbital parameters
        spread = init_variables.get('init_position_spread', 20.0)
        isigma = init_variables.get('init_sigma', 30.0)
        iamp = init_variables.get('init_amplitude', 1.0)
        self.orbital_positions = nn.Parameter(
            (torch.rand(n_orb, 3, device=device) - 0.5) * spread
        )
        self.orbital_amplitudes = nn.Parameter(
            torch.full((n_orb,), iamp, dtype=torch.complex64, device=device)
        )
        self.orbital_sigmas = nn.Parameter(
            torch.full((n_orb,), isigma, dtype=torch.float32, device=device)
        )

        # Probe
        probe_t = torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device)
        self.opt_probe = nn.Parameter(torch.view_as_real(probe_t[None, ...]))

        # Scan positions
        sp = init_variables['scan_positions']
        self.register_buffer("scan_positions", torch.tensor(sp, dtype=torch.float32, device=device))
        self.opt_probe_pos_shifts = nn.Parameter(torch.zeros(sp.shape[0], 2, device=device))

        # Tilt and defocus values
        self.register_buffer("tilt_angles", torch.tensor(
            init_variables.get('tilt_angles', [0.0]), device=device))
        self.register_buffer("defocus_values", torch.tensor(
            init_variables.get('defocus_values', [0.0]), device=device))

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
            'probe': self.opt_probe,
            'probe_pos_shifts': self.opt_probe_pos_shifts,
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
        return torch.view_as_complex(self.opt_probe)

    def get_measurements(self, indices):
        return self.meas_loader[indices]

    def clear_cache(self):
        self._current_object_patches = None

    def _apply_defocus(self, probe, defocus):
        phase = -torch.pi * self.wavelength * defocus * self.k2.to(probe.device, probe.real.dtype)
        H = torch.exp(1j * phase.to(dtype=torch.complex64))
        pk = fft2(probe) * H
        return torch.fft.ifft2(pk)

    def forward(self, batch_indices):
        ny, nx = self.grid_shape
        cy, cx = ny // 2, nx // 2
        batch_size = batch_indices.shape[0]
        n_tilt = self.tilt_angles.shape[0]
        n_defocus = self.defocus_values.shape[0]

        diffractions = torch.zeros(batch_size, n_tilt, n_defocus, ny, nx, device=self.device)
        probe_center = self.get_complex_probe_view().squeeze(0)

        for t in range(n_tilt):
            obj_2d = project_to_slice(
                self.orbital_positions, self.orbital_amplitudes, self.orbital_sigmas,
                self.tilt_angles[t], torch.tensor(0.0, device=self.device),
                self.grid_x, self.grid_y,
            )
            for d in range(n_defocus):
                probe_def = self._apply_defocus(probe_center, self.defocus_values[d])
                for i, idx in enumerate(batch_indices):
                    pos = self.scan_positions[idx] + self.opt_probe_pos_shifts[idx]
                    sx = pos[0] / self.pixel_size + cx
                    sy = pos[1] / self.pixel_size + cy
                    sx_i = int(torch.round(sx).item())
                    sy_i = int(torch.round(sy).item())

                    sp = torch.roll(probe_def, shifts=(sy_i - cy, sx_i - cx), dims=(0, 1))
                    ew = sp * obj_2d
                    dp = fft2(ew)
                    r = torch.view_as_real(dp)
                    diffractions[i, t, d] = r[..., 0]**2 + r[..., 1]**2

        self._current_object_patches = (torch.tensor([], device=self.device), torch.tensor([], device=self.device))
        return diffractions.view(batch_size, ny, nx)  # flatten tilt/defocus dims for loss
