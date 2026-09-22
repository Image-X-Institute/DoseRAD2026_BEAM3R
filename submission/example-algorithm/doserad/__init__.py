"""Self-contained CNN-xLSTM segment dose prediction.

Everything needed to turn a CT ``.mha`` plus beam-level metadata into per
control-point dose lives under this package -- there are no imports from
outside this repository.

Layout::

    geometry.py    beam basis + MLC aperture, straight from the plan JSON
    bev_grid.py    BEV cuboid geometry config
    bev_build.py   CT -> BEV resampling and the inverse BEV-to-CT affine (CuPy)
    ct_sample.py   Triton bicubic samplers, incl. the packed-coefficient path
    fast_preprocess.py  cached affine CT/aperture preprocessing (Triton)
    nets/          the CNN-xLSTM model
    predict.py     the DosePredictor entry point
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys



def _preload_nvrtc() -> None:
    """Make ``libnvrtc`` resolvable before CuPy's extension modules load.

    CuPy compiles its ElementwiseKernels through NVRTC, but ``cupy-cuda12x``
    declares no dependency on it and its extension resolves ``libnvrtc.so.12``
    through the normal dynamic loader. In wheel-based installs (including the
    ``pytorch/pytorch`` images) NVRTC lives under ``nvidia/cuda_nvrtc/lib`` in
    site-packages, which is not on the loader path. Loading it RTLD_GLOBAL here
    -- the same trick torch uses for its own CUDA deps -- fixes that without
    baking a Python version into ``LD_LIBRARY_PATH``.

    A no-op when the loader can already find it, or when nothing matches.
    """
    try:
        ctypes.CDLL("libnvrtc.so.12", mode=ctypes.RTLD_GLOBAL)
        return
    except OSError:
        pass
    for site_dir in sys.path:
        if not site_dir:
            continue
        pattern = os.path.join(site_dir, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*")
        for candidate in sorted(glob.glob(pattern)):
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


_preload_nvrtc()

__all__ = ["geometry", "bev_grid", "bev_build", "ct_sample", "predict"]
