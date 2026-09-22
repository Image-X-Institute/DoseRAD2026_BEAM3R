# Mirrors the training environment in DL-segment-dose-calculation/docker/Dockerfile:
# same CUDA 12.8.0, same Python 3.11 venv at /opt/venv, same torch 2.8.0+cu128,
# same numpy/cupy/SimpleITK pins. Only the packages this checkpoint actually needs
# are installed -- see the notes at each pip step for what was dropped and why.
#
# -runtime rather than training's -devel: nothing here compiles against the CUDA
# toolkit. Triton ships its own ptxas/nvdisasm/cuobjdump under
# site-packages/triton/backends/nvidia/bin, and no .so in the venv resolves to
# /usr/local/cuda. Two things the -devel base provided implicitly have to be
# supplied explicitly instead -- see the gcc note below and _preload_nvrtc() in
# doserad/__init__.py. Re-adding mamba-ssm or causal-conv1d would need nvcc and
# so would mean going back to -devel or adding a builder stage.
FROM --platform=linux/amd64 nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS example_algorithm_amd64

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Training's apt list minus the build/vision/HDF5 extras that only the training
# and preprocessing stacks needed (cmake, ninja, ffmpeg, libgl, libhdf5, git...).
# The venv is built as root so the runtime user gets it read-only.
#
# gcc is not a build-time convenience: Triton compiles a small CPython extension
# (its CUDA driver utils) with the *host* compiler on first use, via
# triton/runtime/build.py, which searches PATH for gcc then clang and raises
# "Failed to find C compiler" if neither is there. The -devel base happened to
# supply one; -runtime does not, so the first kernel launch died here. The
# headers it also needs come from python3.11-dev, already in this list.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv gcc \
 && python3.11 -m venv /opt/venv \
 && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel \
 && rm -rf /var/lib/apt/lists/*

# Same wheel and index as training. Pulls pytorch-triton, which is what the
# CT-space packed-coefficient sampler and the mLSTM limit-chunk kernel compile
# against -- so it must not be resolved from the default PyPI index.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0

# Everything else, pinned to the training versions. Installed after torch so the
# numpy pin wins over whatever the torch wheel resolved.
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --requirement /opt/app/requirements.txt

# CNN-Mamba stack. Three things make this work without nvcc on the -runtime base:
#
#  * triton 3.5.0 (torch 2.8 ships 3.4.0). mamba-ssm 2.3.2's mamba3 SISO kernel
#    does a tl.dot(q*k, ones[K,1]) N=1 reduction that 3.4's stricter min-tile
#    check rejects. Training pins the same upgrade. Only torch.compile would
#    care about the version skew, and nothing here compiles.
#  * MAMBA_SKIP_CUDA_BUILD / CAUSAL_CONV1D_SKIP_CUDA_BUILD. mamba3's scan is
#    Triton, not the Mamba-1 selective_scan CUDA extension, so the source build
#    that needs nvcc can be skipped. causal-conv1d then falls back to F.conv1d;
#    measured at the shapes this model runs, that costs nothing.
#  * huggingface_hub and transformers are imported by mamba_ssm/__init__.py
#    (via MambaLMHeadModel) even though no language model is ever built.
#
# Despite the training image's note, mamba3 is NOT Hopper-only: verified running
# on Ampere sm_86, which is what Grand Challenge's A10G is.
RUN pip install --no-deps "triton==3.5.0" einops \
 && pip install huggingface_hub transformers==4.57.1 \
 && CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE MAMBA_SKIP_CUDA_BUILD=TRUE \
    pip install --no-build-isolation --no-deps \
      causal-conv1d==1.6.1 \
      mamba-ssm==2.3.2.post1

# The sCT stage. MONAI supplies SwinUNETR and sliding_window_inference, which
# are the generator's architecture and its inference tiling -- both are part of
# what the checkpoint was trained as, not implementation choices, so the version
# is pinned to the one the generator was validated against.
#
# --no-deps: MONAI's core needs only numpy and torch, both already pinned above,
# and its full dependency set would pull in the training-only stack this image
# spent effort removing (see the note below).
RUN pip install --no-deps "monai==1.6.0"

# Deliberately NOT installed, though training's image has them:
#   xlstm==1.0.3           - doserad/vendor ships the modified fork this checkpoint
#                            needs (install_mlstm_backend / triton_limit_chunk are
#                            not in the PyPI release). Installing it would put a
#                            second, wrong xlstm on the path.
#   mamba-ssm, causal-conv1d - CNN-Mamba only; needs nvcc at install time.
#   torchmetrics, tensorboard, pytest, pre-commit - training/eval only.
#   scipy, pydicom, h5py, itk, pymedphys, uproot, numba, matplotlib, notebook,
#   transformers, einops   - preprocessing, analysis or unused. doserad.geometry
#                            reimplements the one scipy call (zoom order=1) it
#                            needed; tests/ still use scipy, but tests do not ship.

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

# Triton and CuPy both JIT-compile kernels on first use and need a writable cache;
# without one the first /invoke either pays the full compile cost or fails.
# On this -runtime base there is no /usr/local/cuda/lib64/libnvrtc.so.12, so
# doserad/__init__.py's _preload_nvrtc() -- which pulls NVRTC out of torch's
# nvidia-cuda-nvrtc-cu12 wheel dep -- is the only thing putting it on CuPy's
# loader path. It is load-bearing here, not the no-op it is under -devel.
ENV HOME=/home/user \
    TRITON_CACHE_DIR=/home/user/.cache/triton \
    CUPY_CACHE_DIR=/home/user/.cache/cupy

WORKDIR /opt/app

# Fail the build rather than the submission if the CUDA wiring is wrong. This only
# checks that the extension modules load; kernel compilation needs a GPU and is
# therefore first exercised on invoke.
# RUN python -c "import torch, triton, cupy, SimpleITK, numpy; \
# print('deps ok', torch.__version__, triton.__version__, cupy.__version__, numpy.__version__)"

# The dose-prediction library, including the vendored xlstm + mlstm_kernels.
# Copied before the entrypoints so edits there do not invalidate this layer.
COPY --chown=user:user doserad /opt/app/doserad

COPY --chown=user:user app.py inference.py /opt/app/

# Which of the four challenge interfaces this image is built for. Grand
# Challenge cannot set environment variables, so the task has to be baked in:
# it decides which input socket is read (source-mri-image, not source-ct-image)
# and, through inference.NEEDS_SCT, whether the MR-to-CT generators are loaded
# at all. An image built for photon-mri will not serve photon-ct and vice versa.
ARG DOSERAD_TASK=photon-mri
ENV TASK=${DOSERAD_TASK}

# The BEV grid and arch JSON are derived from TASK in app.py -- the photon runs
# record bev_grid_384_x_zalign.json (shape_dhw [384, 200, 200]) and the proton
# runs bev_grid_proton_zalign.json ([512, 128, 48]). They are deliberately NOT
# baked here: a fixed value would silently serve one modality on the other's
# grid, which is a train/test mismatch rather than a narrower field of view.
# Both remain overridable at run time for local A/B work.

# The shipped checkpoint's architecture, likewise baked in. Both must match
# config_hyper.json for thorax_1.5x_..._upscale2 or torch.load_state_dict fails
# at startup: --otf-input-upscale-factor 2 packs 2x2 phases into the encoder's
# input channels (8, not 2), and --model-capacity-scale 1.5 widens every conv
# stage (16/32/64 -> 24/48/96).
ENV DOSERAD_INPUT_UPSCALE_FACTOR=2

# Which architecture the shipped weights are. "mamba-packed" serves the 2x
# capacity, upscale-2 CNN-Mamba3 checkpoint.  DOSERAD_MAMBA_VARIANT chooses its
# directional checkpoint and matching arch JSON in app.py.  Keeping this as a
# build argument gives the unidirectional and bidirectional submissions distinct
# immutable images while preserving the established unidirectional default.
# The architecture is fixed in app.py; nothing selects between families here.

# Inference-only Mamba3 kernels are selected before startup warmup.  The warmup
# therefore JIT-compiles the exact recurrence baked into the image before
# /health becomes ready.  Both production variants use their validated
# chunk-16 scratch-free recurrence by default.  Fused-angle and fused merge
# remain disabled until separately validated on the complete checkpoint.
ARG DOSERAD_MAMBA3_SCRATCH_FREE=1
ARG DOSERAD_MAMBA3_FUSED_ANGLE=0
ARG DOSERAD_MAMBA_BIDIRECTIONAL_MERGE=0
ENV DOSERAD_MAMBA3_SCRATCH_FREE=${DOSERAD_MAMBA3_SCRATCH_FREE} \
    DOSERAD_MAMBA3_FUSED_ANGLE=${DOSERAD_MAMBA3_FUSED_ANGLE} \
    DOSERAD_MAMBA_BIDIRECTIONAL_MERGE=${DOSERAD_MAMBA_BIDIRECTIONAL_MERGE}

# Which of the two packed CNN-Mamba3 checkpoints to serve. "ct" is the run
# trained on real CTs; "sct" is that run fine-tuned with --sct-dir, on the
# synthetic CTs the photon-mri task feeds it at inference. Same architecture,
# BEV grid and dose_scale either way -- only the weights differ, so both are
# served by the same arch JSON above and both ship in every model tarball.
#
# There is no automatic coupling to DOSERAD_TASK: which checkpoint suits the
# MRI task is an empirical question, so it is chosen explicitly per build.

# Compile the Triton/CuPy kernels at startup instead of inside the first
# /invoke. The health endpoint stays unready until it completes, which is the
# behaviour Grand Challenge's start-up probe expects.
ENV DOSERAD_STARTUP_WARMUP=1

# Baked-in runtime tunables. Grand Challenge sets no environment, so these have
# to be baked in to take effect on a real submission.
#
# Precision and sliding-window batch of the SwinUNETR sCT stage (MRI tasks
# only). bf16 is what DoseRAD2026_sCT's validation epoch used and is ~2x
# faster; sw_batch_size is bit-identical output, trading VRAM for utilisation.
# The three stack/compression workers below are output-writing throughput.
ARG DOSERAD_SCT_PRECISION=fp32
ARG DOSERAD_SCT_SW_BATCH_SIZE=1
ENV DOSERAD_SCT_PRECISION=${DOSERAD_SCT_PRECISION} \
    DOSERAD_SCT_SW_BATCH_SIZE=${DOSERAD_SCT_SW_BATCH_SIZE} \
    DOSERAD_PARALLEL_STACK_WRITERS=1 \
    DOSERAD_MAX_PARALLEL_STACK_WRITERS=4 \
    DOSERAD_MHA_COMPRESSION_WORKERS=4

# Proton inference fusions: combined RSP/WET conditioning + channels-last
# encoder input, fused CT->material sampling, and the evaluator's minimum
# cutoff applied on device before the output copy. Fused backprojection stays
# off: it measured 0.6% slower than the reference sequence on RTX A6000.
ENV DOSERAD_PROTON_FUSED_PREPROCESS=1 \
    DOSERAD_PROTON_FUSED_CT_MATERIAL=1 \
    DOSERAD_PROTON_FUSED_BACKPROJECTION=0 \
    DOSERAD_PROTON_GPU_OUTPUT_CUTOFF=1

LABEL org.grand-challenge.api-method="invoke"

ENTRYPOINT ["python", "app.py"]
