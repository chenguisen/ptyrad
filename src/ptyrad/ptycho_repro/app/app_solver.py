"""APPOrbitalSolver — Solver for APP multi-slice reconstruction.

Integrates APP model with ptyrad's recon_loop infrastructure.
"""

import logging
from copy import deepcopy
import numpy as np
import torch

from ptyrad.core.losses import CombinedLoss
from ptyrad.core.constraints import CombinedConstraint
from ptyrad.solver.reconstruction import create_optimizer, create_scheduler, recon_loop
from ptyrad.io.save import safe_filename
from ptyrad.utils.time import get_time
from ptyrad.ptycho_repro.app.app_model import APPOrbitalModel

logger = logging.getLogger(__name__)


class _InitWrapper:
    def __init__(self, init_variables):
        self.init_variables = init_variables


class APPOrbitalSolver:
    """APP multi-slice solver, compatible with ptyrad's recon_loop."""

    _ALL_CONSTRAINTS = [
        'probe_mask_k', 'probe_mask_r', 'ortho_pmode', 'fix_probe_int',
        'obj_rblur', 'obj_zblur', 'kr_filter', 'kz_filter', 'kr_thresh',
        'complex_ratio', 'mirrored_amp', 'obj_z_recenter', 'obja_thresh',
        'objp_postiv', 'pos_recenter', 'tilt_smooth',
    ]
    _ALL_LOSSES = ['loss_single', 'loss_poissn', 'loss_pacbed', 'loss_sparse', 'loss_simlar']

    def __init__(self, params, device=None, seed=None):
        self.params = deepcopy(params)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed or params.get('init_params', {}).get('random_seed', 42)
        self._fill_defaults()
        self.init_variables = {}
        self._init_components()
        self.loss_fn = CombinedLoss(self.params.get('loss_params', {}), device=self.device)
        self.constraint_fn = CombinedConstraint(self.params.get('constraint_params', {}), device=self.device)

    def _fill_defaults(self):
        cp = self.params.setdefault('constraint_params', {})
        for c in self._ALL_CONSTRAINTS:
            cp.setdefault(c, {'start_iter': None, 'step': 1, 'end_iter': None})
        lp = self.params.setdefault('loss_params', {})
        for l in self._ALL_LOSSES:
            lp.setdefault(l, {'state': False, 'weight': 0})
        rp = self.params.setdefault('recon_params', {})
        rp.setdefault('INDICES_MODE', {'mode': 'full'})
        rp.setdefault('GROUP_MODE', 'random')
        rp.setdefault('selected_figs', 'default')
        rp.setdefault('compiler_configs', {'enable': False})
        rp.setdefault('recon_dir_affixes', ['minimal'])
        rp.setdefault('prefix_time', False)
        rp.setdefault('prefix', '')
        rp.setdefault('postfix', '')
        rp.setdefault('copy_params', False)

    def _init_components(self):
        p = self.params
        ip = p.get('init_params', {})
        D = self.init_variables
        D['n_layers'] = ip.get('n_layers', 3)
        D['grid_shape'] = tuple(ip.get('grid_shape', [64, 64]))
        D['pixel_size'] = ip.get('pixel_size', 10.0)
        D['wavelength'] = ip.get('wavelength', 2.51)
        D['init_dz'] = ip.get('init_dz', 2.0)
        D['dx'] = ip.get('dx', 0.1)
        D['measurements'] = p.get('data', {}).get('measurements')
        D['scan_positions'] = p.get('data', {}).get('scan_positions')
        D['probe'] = p.get('data', {}).get('probe')
        if D['probe'] is None and D['measurements'] is not None:
            D['probe'] = np.sqrt(np.abs(D['measurements'].mean(0).astype(np.complex64)) + 1e-12)
        D['N_scan_slow'] = p.get('data', {}).get('N_scan_slow', 1)
        D['N_scan_fast'] = p.get('data', {}).get('N_scan_fast', 1)

    def reconstruct(self):
        p = self.params
        rp = p.get('recon_params', {})
        model = APPOrbitalModel(self.init_variables, p.get('model_params', {}), device=self.device)
        optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
        scheduler = create_scheduler(model.scheduler_params, optimizer)

        N = self.init_variables['scan_positions'].shape[0]
        indices = np.arange(N)
        bs = rp.get('BATCH_SIZE', {}).get('size', 32)
        nb = max(1, N // bs)
        shuf = np.random.default_rng(seed=self.seed).permutation(indices)
        batches = [torch.from_numpy(b).to(self.device) for b in np.array_split(shuf, nb)]

        SAVE = rp.get('SAVE_ITERS')
        output_path = self._make_output_path(rp) if SAVE else None
        recon_loop(model, _InitWrapper(self.init_variables), self.params,
                   optimizer, scheduler, self.loss_fn, self.constraint_fn,
                   indices, batches, output_path)
        self.model = model

    def _make_output_path(self, rp):
        import os
        op = rp.get('output_dir', 'output/app')
        p = safe_filename(os.path.join(op, f"APP_{get_time('datetime')}"))
        os.makedirs(p, exist_ok=True)
        return p
