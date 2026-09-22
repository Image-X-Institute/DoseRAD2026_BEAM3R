# Training environment for the CNN-Mamba3 dose model and the MR -> sCT
# generators. This is the image ../../../submission/example-algorithm/Dockerfile
# mirrors: same CUDA 12.8.0, same Python 3.11 venv at /opt/venv, same
# torch 2.8.0+cu128, same numpy/cupy/SimpleITK pins, same triton 3.5.0.
#
# Reconstructed for this repository from the pins the inference image records
# plus the imports the training code actually makes. It installs what
# `train_bidi.sh` and `train_sct.sh` need and nothing else -- the wider research
# stack (pydicom, itk, pymedphys, uproot, numba, notebook, torchmetrics) belonged
# to preprocessing and analysis code that is not part of this extract.
#
# -devel rather than inference's -runtime: mamba-ssm and causal-conv1d compile
# CUDA extensions at install time and need nvcc. That is the one structural
# difference between the two images.
#
#   docker build -t doserad-train external/DL-segment-dose-calculation/docker
#   docker run --rm -it --gpus device=0 \
#     -v "$PWD:/workspace" -v /path/to/doserad-data:/data \
#     doserad-train
#
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# python3.11 from deadsnakes: 22.04 ships 3.10, and the venv has to match the
# interpreter the inference image builds against so a checkpoint pickled here
# loads there.
#
# git, ninja-build and g++ are for the mamba-ssm / causal-conv1d source builds
# below; python3.11-dev supplies the headers both they and Triton's host-compiler
# step need. The CUDA toolkit itself comes from the -devel base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    gcc g++ git ninja-build \
 && python3.11 -m venv /opt/venv \
 && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel \
 && rm -rf /var/lib/apt/lists/*

# torch must come from the cu128 index, not PyPI. It also pulls pytorch-triton,
# which the triton pin below deliberately replaces.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0

# Installed after torch so the numpy pin wins over whatever the torch wheel
# resolved. numpy stays on 1.26.4 rather than floating to 2.x: beam geometry and
# aperture construction are numpy, and NEP 50 changed scalar type promotion.
#
# cupy runs the BEV resampling and the aperture projection ElementwiseKernel;
# SimpleITK is the .mha reader/writer.
RUN pip install \
    numpy==1.26.4 \
    cupy-cuda12x==13.6.0 \
    simpleitk==2.3.1

# triton 3.5.0, replacing the 3.4.0 that torch 2.8 ships.
#
# This is a correctness pin, not an upgrade for its own sake: mamba-ssm 2.3.2's
# mamba3 SISO kernel does a tl.dot(q*k, ones[K,1]) N=1 reduction that 3.4's
# stricter min-tile check rejects. The inference image pins the same version for
# the same reason, and the CT-space packed-coefficient sampler compiles against
# it too.
#
# --no-deps so pip does not walk back up to torch and re-resolve it.
RUN pip install --no-deps "triton==3.5.0"

# The CNN-Mamba stack.
#
# Versions are the ones the submission image loads the checkpoint with; they must
# not drift apart, since the state dict is written by this side and read by that
# one. mamba3 requires >=2.3.2 (see model/CNN_Mamba/model.py).
#
# Unlike the inference image, the CUDA extensions are *built* here rather than
# skipped -- nvcc is present, and training wants the fused causal-conv1d path
# rather than the F.conv1d fallback. That build is the slow step in this
# Dockerfile; MAX_JOBS caps its parallelism, which is what keeps it from
# exhausting RAM on a many-core host. Raise it if you have the memory.
#
# einops, transformers and huggingface_hub are imported by mamba_ssm/__init__.py
# (via MambaLMHeadModel) even though no language model is ever built. They are
# installed explicitly because the mamba-ssm install below uses --no-deps, which
# is what stops it from pulling its own triton and undoing the pin above.
RUN pip install einops huggingface_hub transformers==4.57.1 \
 && MAX_JOBS=4 pip install --no-build-isolation --no-deps \
      causal-conv1d==1.6.1 \
      mamba-ssm==2.3.2.post1

# The sCT stage. MONAI supplies SwinUNETR and sliding_window_inference, which are
# the generator's architecture and its inference tiling -- both are part of what
# the checkpoint was trained as, not implementation choices, so the version is
# pinned to the one the generator was validated against.
#
# --no-deps: MONAI's core needs only numpy and torch, both already pinned, and
# its full dependency set would re-resolve them.
RUN pip install --no-deps "monai==1.6.0"

# Training-only, and the reason this image exists separately from the inference
# one. None of these ship in the submission container.
#
#   h5py         train/utils.py, cached dataset I/O
#   scipy        sCT body masks (ndimage); the inference path reimplements the
#                one zoom(order=1) call it needed so it can drop scipy entirely
#   matplotlib   validation figures written during a run
#   tqdm         progress bars in both trainers
#   tensorboard  --tensorboard in train_bidi.sh; imported via
#                torch.utils.tensorboard, which needs the package present
#   schedulefree the schedule-free optimiser, train/utils.py
RUN pip install \
    h5py \
    scipy \
    matplotlib \
    tqdm \
    tensorboard \
    schedulefree

# Triton and CuPy both JIT-compile kernels on first use and need a writable
# cache. Pointing them inside the image keeps a bind-mounted workspace clean; set
# them elsewhere if you run with --user and cannot write here.
ENV TRITON_CACHE_DIR=/opt/cache/triton \
    CUPY_CACHE_DIR=/opt/cache/cupy
RUN mkdir -p /opt/cache/triton /opt/cache/cupy && chmod -R 777 /opt/cache

# Pin check: fails the build rather than the first epoch if a resolution step
# above silently moved a version -- most importantly triton, which torch would
# otherwise hold at 3.4.0, and mamba-ssm, which must stay at the release the
# submission image loads the checkpoint with.
#
# Deliberately metadata-only, no imports: the build host has no GPU, and
# importing cupy or mamba_ssm's compiled extensions there is exactly why the
# inference image's equivalent check ended up commented out. Whether the kernels
# actually compile is first exercised on a real run.
RUN python - <<'EOF'
from importlib.metadata import version
expected = {
    "torch": "2.8.0",
    "triton": "3.5.0",
    "mamba_ssm": "2.3.2.post1",
    "causal_conv1d": "1.6.1",
    "numpy": "1.26.4",
    "cupy-cuda12x": "13.6.0",
    "SimpleITK": "2.3.1",
    "monai": "1.6.0",
}
bad = {p: (v, got) for p, v in expected.items()
       if not (got := version(p)).startswith(v)}
if bad:
    raise SystemExit("version drift: %r" % bad)
for p in ("h5py", "scipy", "matplotlib", "tqdm", "tensorboard", "schedulefree"):
    version(p)
print("pins ok:", ", ".join(f"{p}=={version(p)}" for p in expected))
EOF

WORKDIR /workspace

CMD ["bash"]
