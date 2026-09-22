# BEAM3R

**Beam's-eye-view architecture with Mamba-3 for implicit dose reconstruction**

BEAM3R is a beam's-eye-view (BEV) deep-learning framework for rapid photon control-point and proton beamlet dose prediction, developed for the **DoseRAD2026 Grand Challenge**. The BEAM3R submission won both photon tasks, for dose calculation on CT and MRI.

BEAM3R combines a 2D CNN encoder-decoder with Mamba-3 depth-sequence modelling to capture long-range radiation transport without expensive 3D convolutions. Implicit super-resolution enables high-resolution dose prediction while keeping the main sequence computation at lower spatial resolution, with custom Triton inference kernels to reduce memory traffic and latency.

Photon models use bidirectional Mamba-3. Proton models use forward Mamba-3 with physics-based range conditioning, learned energy-prefix tokens, and Bragg-peak residual refinement.

- **Paper:** [arXiv:2609.04747](https://arxiv.org/abs/2609.04747)
- **Dataset:** [DoseRAD2026](https://doserad2026.grand-challenge.org/data/)

## Pipelines

| Task | Pipeline |
|---|---|
| `photon-ct` | CT → bidirectional Mamba-3 → dose |
| `photon-mri` | MR → sCT → bidirectional Mamba-3 → dose |
| `proton-ct` | CT → physics-conditioned forward Mamba-3 → dose |
| `proton-mri` | MR → sCT → physics-conditioned forward Mamba-3 → dose |

MRI tasks use anatomy-specific SwinUNETR models to generate synthetic CTs before dose prediction.

## Repository structure

```text
external/
├── DL-segment-dose-calculation/   # Dose models and training
└── DoseRAD2026_sCT/               # MR → sCT training

configs/                            # BEV grids and normalisation
submission/                         # Inference and challenge packaging
train_bidi.sh                       # Photon training
train_proton.sh                     # Proton training
train_sct.sh                        # sCT training
validation.py                       # Validation metrics
```

The directory structure under `external/` should be preserved because several modules rely on these paths.

## Environment

Training uses Docker with CUDA 12.8, Python 3.11 and PyTorch 2.8.0.

```bash
docker build -t doserad-train \
  external/DL-segment-dose-calculation/docker

docker run --rm -it --gpus device=0 \
  -v "$PWD:/workspace" \
  -v /path/to/doserad-data:/data \
  doserad-train
```

## Training

### Photon

```bash
DATA_ROOT=/data ./train_bidi.sh
```

The photon model uses a bidirectional Mamba-3 core and implicit BEV super-resolution.

### Proton

```bash
DATA_ROOT=/data/proton/training \
SPLIT=/path/to/proton_split.json \
./train_proton.sh
```

The proton model uses a forward Mamba-3 core with energy-prefix tokens, material-density conditioning, 2× implicit super-resolution, and Bragg-peak residual refinement. `BEAM_PARAMETERS`, `OUT_ROOT`, `INIT_FROM`, `EPOCHS`, `BATCH_SIZE` and `LR` can be overridden through environment variables.

The challenge checkpoint was warm-started from an earlier energy-token model using `INIT_FROM`; leaving this unset trains the same architecture from scratch.

For both modalities, training and inference must use the same normalisation and `dose_scale`, and deployment-critical architecture options must remain consistent with the inference implementation.

### Synthetic CT

Train one generator for each anatomy, for whichever modality you need:

```bash
ANATOMY=thorax  DATA_ROOT=/data ./train_sct.sh
ANATOMY=abdomen DATA_ROOT=/data ./train_sct.sh

MODALITY=proton ANATOMY=thorax  DATA_ROOT=/data ./train_sct.sh
MODALITY=proton ANATOMY=abdomen DATA_ROOT=/data ./train_sct.sh
```

`MODALITY` defaults to `photon` and selects both the image tree (`<modality>/training/`) and the loss weighting: the released proton generators were trained with a heavier reconstruction term than the photon ones, so each modality has its own config.

The SwinUNETR generators use body-masked reconstruction and adversarial losses. Low-density HU bias and water-equivalent-path error are also recorded as diagnostics relevant to downstream dose calculation.

## Inference optimisation

BEAM3R includes several optimisations for low-latency inference:

- **Implicit super-resolution** produces high-resolution dose while performing the main latent and Mamba-3 computation at lower spatial resolution.
- **Custom Triton kernels** implement scratch-free Mamba-3 recurrence, fused projection preparation, fused LayerNorm, and fused decoder operations.
- **Startup warm-up** JIT-compiles inference kernels before timed execution.
- **Parallel output I/O** overlaps GPU inference with MetaImage compression and writing.

These optimisations are applied without changing the trained model state dictionary.

## Submission

Challenge inference and packaging scripts are contained in `submission/`. Stage the relevant dose model and, for MRI, the thorax and abdomen sCT generators before running `do_save.sh`. Use `do_test_run.sh` to validate the complete packaged pipeline before submission.

## Reproducibility

- Keep training and inference normalisation and dose scaling identical.
- Keep deployment architecture settings consistent with the packaged model.
- Use `--force` when replacing an already staged checkpoint.
- Keep anatomy-specific sCT models paired with their corresponding CT statistics.
- Use the fixed patient split when reproducing the proton model.
- Validate the complete packaged pipeline before submission.

## Citation

```bibtex
@article{cheng2026beam3r,
  title   = {{BEAM3R}: Beam's-eye-view architecture with Mamba-3 for implicit dose reconstruction},
  author  = {Cheng, Chen and Ferraro, Michael and Grover, James and Waddington, David E. J. and Hewson, Emily},
  journal = {arXiv preprint arXiv:2609.04747},
  year    = {2026},
  eprint  = {2609.04747},
  archivePrefix = {arXiv},
  primaryClass  = {physics.med-ph},
  doi     = {10.48550/arXiv.2609.04747}
}
```