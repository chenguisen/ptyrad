"""NLOTOrbitalSolver — Solver for nLOT 3D orbital tomography."""

import logging, os
from copy import deepcopy
import numpy as np, torch
from ptyrad.core.losses import CombinedLoss
from ptyrad.core.constraints import CombinedConstraint
from ptyrad.solver.reconstruction import create_optimizer, create_scheduler, recon_loop
from ptyrad.io.save import safe_filename
from ptyrad.utils.time import get_time
from ptyrad.ptycho_repro.nlot.nlot_model import NLOTOrbitalModel

logger = logging.getLogger(__name__)


class _IW:
    def __init__(self, d): self.init_variables = d

_ALL_C = ['probe_mask_k','probe_mask_r','ortho_pmode','fix_probe_int',
          'obj_rblur','obj_zblur','kr_filter','kz_filter','kr_thresh',
          'complex_ratio','mirrored_amp','obj_z_recenter','obja_thresh',
          'objp_postiv','pos_recenter','tilt_smooth']
_ALL_L = ['loss_single','loss_poissn','loss_pacbed','loss_sparse','loss_simlar']


class NLOTOrbitalSolver:
    def __init__(self, params, device=None, seed=None):
        self.params = deepcopy(params)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed or params.get('init_params', {}).get('random_seed', 42)
        cp = self.params.setdefault('constraint_params', {})
        for c in _ALL_C: cp.setdefault(c, {'start_iter': None, 'step': 1, 'end_iter': None})
        lp = self.params.setdefault('loss_params', {})
        for l in _ALL_L: lp.setdefault(l, {'state': False, 'weight': 0})
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

        D = params.get('init_params', {})
        self.init_variables = {
            'n_orbitals': D.get('n_orbitals', 100),
            'grid_shape': tuple(D.get('grid_shape', [64, 64])),
            'pixel_size': D.get('pixel_size', 10.0),
            'wavelength': D.get('wavelength', 2.51),
            'init_position_spread': D.get('init_position_spread', 20.0),
            'init_sigma': D.get('init_sigma', 30.0),
            'tilt_angles': D.get('tilt_angles', [0.0]),
            'defocus_values': D.get('defocus_values', [0.0]),
            'measurements': params.get('data', {}).get('measurements'),
            'scan_positions': params.get('data', {}).get('scan_positions'),
            'probe': params.get('data', {}).get('probe'),
        }
        iv = self.init_variables
        if iv['probe'] is None and iv['measurements'] is not None:
            iv['probe'] = np.sqrt(np.abs(iv['measurements'].mean(0).astype(np.complex64)) + 1e-12)
        self.loss_fn = CombinedLoss(self.params.get('loss_params', {}), device=self.device)
        self.constraint_fn = CombinedConstraint(self.params.get('constraint_params', {}), device=self.device)

    def reconstruct(self):
        rp = self.params.get('recon_params', {})
        model = NLOTOrbitalModel(self.init_variables, self.params.get('model_params', {}), device=self.device)
        optim = create_optimizer(model.optimizer_params, model.optimizable_params)
        sched = create_scheduler(model.scheduler_params, optim)
        N = self.init_variables['scan_positions'].shape[0]
        idx = np.arange(N)
        bs = rp.get('BATCH_SIZE', {}).get('size', 32)
        nb = max(1, N // bs)
        sh = np.random.default_rng(seed=self.seed).permutation(idx)
        batches = [torch.from_numpy(b).to(self.device) for b in np.array_split(sh, nb)]
        SAVE = rp.get('SAVE_ITERS')
        op = rp.get('output_dir', 'output/nlot')
        out = safe_filename(os.path.join(op, f"NLOT_{get_time('datetime')}")) if SAVE else None
        if out: os.makedirs(out, exist_ok=True)
        recon_loop(model, _IW(self.init_variables), self.params, optim, sched,
                   self.loss_fn, self.constraint_fn, idx, batches, out)
        self.model = model
