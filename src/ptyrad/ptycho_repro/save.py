"""save — Save utilities for orbital ptychography results.

Follows ptyrad's naming conventions where possible:
  - HDF5 save with model state_dict
  - TIFF export of rendered pixel images
  - Loss curve data export
"""

import logging
import os
import h5py
import numpy as np
import torch
from tifffile import imwrite

from ptyrad.io.save import safe_filename
from ptyrad.utils.image_proc import normalize_by_bit_depth

logger = logging.getLogger(__name__)


def save_orbital_results(output_path, model, niter):
    """Save orbital model state and rendered images.

    Parameters
    ----------
    output_path : str
        Output directory path.
    model : LOPOrbitalModel
        The trained model.
    niter : int
        Current iteration number.
    """
    iter_str = f"_iter{str(niter).zfill(4)}"
    os.makedirs(output_path, exist_ok=True)

    # ── HDF5: full state_dict ──────────────────────────────────────────────
    h5_path = safe_filename(os.path.join(output_path, f"model{iter_str}.hdf5"))
    with h5py.File(h5_path, "w") as f:
        f.attrs["niter"] = niter
        f.attrs["n_orbitals"] = model.n_orbitals
        f.attrs["pixel_size"] = model.pixel_size

        # Orbital params
        grp = f.create_group("orbitals")
        grp.create_dataset("positions", data=model.orbital_positions.detach().cpu().numpy())
        grp.create_dataset("amplitudes_real", data=model.orbital_amplitudes.real.detach().cpu().numpy())
        grp.create_dataset("amplitudes_imag", data=model.orbital_amplitudes.imag.detach().cpu().numpy())
        grp.create_dataset("sigmas", data=model.orbital_sigmas.detach().cpu().numpy())

        # Probe
        probe = model.get_complex_probe_view().detach().cpu().numpy()
        f.create_dataset("probe_real", data=probe.real)
        f.create_dataset("probe_imag", data=probe.imag)

        # Loss history
        if model.loss_iters:
            loss_arr = np.array(model.loss_iters, dtype=object)
            f.create_dataset("loss_iters", data=loss_arr.astype(np.float32))

    logger.info(f"Saved {h5_path}")

    # ── TIFF: rendered object ──────────────────────────────────────────────
    with torch.no_grad():
        obj = model.render_object()  # (Ny, Nx) complex

    obj_amp = obj.abs().detach().cpu().numpy()
    obj_phase = obj.angle().detach().cpu().numpy()

    tiff_amp = safe_filename(os.path.join(output_path, f"obj_amp{iter_str}.tif"))
    tiff_phase = safe_filename(os.path.join(output_path, f"obj_phase{iter_str}.tif"))
    imwrite(tiff_amp, normalize_by_bit_depth(obj_amp, "32"))
    imwrite(tiff_phase, normalize_by_bit_depth(obj_phase, "32"))

    # ── TIFF: probe ────────────────────────────────────────────────────────
    probe_amp = abs(probe).squeeze()
    tiff_probe = safe_filename(os.path.join(output_path, f"probe_amp{iter_str}.tif"))
    imwrite(tiff_probe, normalize_by_bit_depth(probe_amp, "32"))


def save_loss_curve(output_path, model):
    """Save loss curve plot data as CSV."""
    if not model.loss_iters:
        return
    path = safe_filename(os.path.join(output_path, "loss_curve.csv"))
    arr = np.array([(n, float(v)) for n, v in model.loss_iters])
    np.savetxt(path, arr, header="iter,loss", delimiter=",", comments="")
    logger.info(f"Saved {path}")
