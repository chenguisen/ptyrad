"""ptycho_repro — Reproduction of local-orbital ptychography algorithms.

This package provides orbital-based ptychography models (LOP, APP, nLOT, eLOP)
that integrate with PtyRAD's reconstruction infrastructure (recon_loop,
CombinedLoss, CombinedConstraint, etc.) without modifying PtyRAD source code.

Each algorithm lives in its own sub-package:
  lop/   — Local-Orbital Ptychography (Nature Nanotechnology 2024)
  app/   — Adaptive Propagation Factor (Science Advances 2022)
  nlot/  — Multi-slice Local Orbital Tomography (Science Bulletin 2024)
  elop/  — Extended Local-Orbital Ptychography (arXiv 2025)
"""

from . import orbitals
