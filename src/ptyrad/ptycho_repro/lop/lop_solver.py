"""LOPOrbitalSolver — High-level solver for LOP reconstruction.

Integrates the LOP orbital model with PtyRAD's reconstruction infrastructure:
  - Uses ptyrad's CombinedLoss, CombinedConstraint, recon_loop
  - Uses ptyrad's create_optimizer, create_scheduler
  - Compatible with ptyrad's YAML parameter conventions
"""

import logging
from copy import deepcopy

import numpy as np
import torch

from ptyrad.core.losses import CombinedLoss
from ptyrad.core.constraints import CombinedConstraint
from ptyrad.solver.reconstruction import (
    create_optimizer,
    create_scheduler,
    recon_loop,
)
from ptyrad.io.save import safe_filename
from ptyrad.utils.time import get_time

from ptyrad.ptycho_repro.lop.lop_model import LOPOrbitalModel

logger = logging.getLogger(__name__)


class _InitWrapper:
    """Wraps init_variables dict to match recon_loop's expected interface."""
    def __init__(self, init_variables):
        self.init_variables = init_variables


class LOPOrbitalSolver:
    """Orbital LOP solver, compatible with ptyrad's workflow conventions.

    Usage::

        params = load_params('params/sto.yaml')
        solver = LOPOrbitalSolver(params, device='cuda')
        solver.reconstruct()
    """

    def __init__(self, params, device=None, seed=None):
        self.params = deepcopy(params)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed or params.get('init_params', {}).get('random_seed', 42)

        self._fill_defaults()

        logger.info("### Initializing LOPOrbitalSolver ###")

        self.init_variables = {}
        self._init_components()
        self.loss_fn = self._build_loss()
        self.constraint_fn = self._build_constraint()

        logger.info("### LOPOrbitalSolver ready ###\n")

    def _fill_defaults(self):
        """Fill missing ptyrad params with defaults so recon_loop works."""
        # --- constraint_params: disable all by default ---
        ALL_CONSTRAINTS = [
            'probe_mask_k', 'probe_mask_r', 'ortho_pmode', 'fix_probe_int',
            'obj_rblur', 'obj_zblur', 'kr_filter', 'kz_filter', 'kr_thresh',
            'complex_ratio', 'mirrored_amp', 'obj_z_recenter', 'obja_thresh',
            'objp_postiv', 'pos_recenter', 'tilt_smooth',
        ]
        cp = self.params.setdefault('constraint_params', {})
        for cname in ALL_CONSTRAINTS:
            cp.setdefault(cname, {'start_iter': None, 'step': 1, 'end_iter': None})

        # --- loss_params: disable all except what user specified ---
        ALL_LOSSES = ['loss_single', 'loss_poissn', 'loss_pacbed',
                      'loss_sparse', 'loss_simlar']
        lp = self.params.setdefault('loss_params', {})
        for lname in ALL_LOSSES:
            lp.setdefault(lname, {'state': False, 'weight': 0})

        # --- recon_params ---
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

    # ── Initialization ─────────────────────────────────────────────────────

    def _init_components(self):
        """Extract all init variables from params dict."""
        p = self.params
        init_p = p.get('init_params', {})

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        D = self.init_variables

        D['n_orbitals']   = init_p.get('n_orbitals', 100)
        D['grid_shape']   = tuple(init_p.get('grid_shape', [64, 64]))
        D['pixel_size']   = init_p.get('pixel_size', 10.0)
        D['init_position_spread'] = init_p.get('init_position_spread', 30.0)
        D['init_sigma']   = init_p.get('init_sigma', 30.0)
        D['init_amplitude'] = init_p.get('init_amplitude', 1.0)
        D['dx']           = init_p.get('dx', 0.1)
        D['dk']           = init_p.get('dk', 0.1)

        D['measurements'] = p.get('data', {}).get('measurements')
        if D['measurements'] is None:
            raise ValueError("'data.measurements' is required in params")

        D['scan_positions'] = p.get('data', {}).get('scan_positions')
        if D['scan_positions'] is None:
            raise ValueError("'data.scan_positions' is required in params")

        D['probe'] = p.get('data', {}).get('probe')
        if D['probe'] is None:
            logger.warning("No probe provided; estimating from mean diffraction")
            D['probe'] = self._estimate_probe()

        D['N_scan_slow'] = p.get('data', {}).get('N_scan_slow', 1)
        D['N_scan_fast'] = p.get('data', {}).get('N_scan_fast', 1)

        logger.info(
            f"  Scan: {D['N_scan_slow']}x{D['N_scan_fast']}  "
            f"Grid: {D['grid_shape']}  "
            f"Orbitals: {D['n_orbitals']}"
        )

    def _estimate_probe(self):
        meas = self.init_variables['measurements']
        avg = meas.mean(0).astype(np.complex64)
        return np.sqrt(np.abs(avg) + 1e-12)

    def _build_loss(self):
        return CombinedLoss(self.params.get('loss_params', {}), device=self.device)

    def _build_constraint(self):
        return CombinedConstraint(self.params.get('constraint_params', {}), device=self.device)

    # ── Reconstruction ─────────────────────────────────────────────────────

    def reconstruct(self):
        """Run LOP reconstruction using ptyrad's recon_loop."""
        p = self.params
        device = self.device
        recon_p = p.get('recon_params', {})
        model_p = p.get('model_params', {})

        SAVE_ITERS = recon_p.get('SAVE_ITERS', None)
        batch_size = recon_p.get('BATCH_SIZE', {}).get('size', 64)

        model = LOPOrbitalModel(self.init_variables, model_p, device=device)

        optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
        scheduler = create_scheduler(model.scheduler_params, optimizer)

        N = self.init_variables['scan_positions'].shape[0]
        indices = np.arange(N)
        rng = np.random.default_rng(seed=self.seed)
        num_batches = max(1, N // batch_size)
        shuffled = rng.permutation(indices)
        batches = np.array_split(shuffled, num_batches)
        batches = [torch.from_numpy(b).to(device) for b in batches]
        logger.info(f"Batches: {num_batches} x ~{batch_size}")

        output_path = self._make_output_path(recon_p) if SAVE_ITERS else None

        init_wrapper = _InitWrapper(self.init_variables)
        recon_loop(
            model, init_wrapper, self.params,
            optimizer, scheduler,
            self.loss_fn, self.constraint_fn,
            indices, batches, output_path,
        )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        logger.info("### LOP reconstruction complete ###")

    def _make_output_path(self, recon_p):
        output_dir = recon_p.get('output_dir', 'output/lop')
        time_str = get_time('datetime')
        dir_name = f"LOP_{time_str}"
        import os
        output_path = safe_filename(os.path.join(output_dir, dir_name))
        os.makedirs(output_path, exist_ok=True)
        return output_path
