# `doserad` — inference package

Turns a CT and its beam-level metadata into dose. For the MRI tasks it first
turns the MR into a synthetic CT, then runs the same dose pipeline on that.

This package has no dependencies outside the container image, so the algorithm
container runs it unchanged.

## Tasks

| Task | Pipeline | Scored unit |
|---|---|---|
| `photon-ct` | CT → bidirectional Mamba-3 → dose | control point |
| `photon-mri` | MR → sCT → bidirectional Mamba-3 → dose | control point |
| `proton-ct` | CT → physics-conditioned forward Mamba-3 → dose | beamlet |
| `proton-mri` | MR → sCT → physics-conditioned forward Mamba-3 → dose | beamlet |

`TASK` selects the pipeline. Photon and proton use different checkpoints, BEV
grids and normalisation statistics; the defaults are chosen from the task, so
nothing needs setting by hand.

## Model files

These are not in git. They are uploaded as the Grand Challenge model tarball and
extracted to `/opt/ml/model`, which `do_test_run.sh` mounts from `./model/`.

**Photon dose**

| File | What it is |
|---|---|
| `best_model_thorax_mamba3_2x_upscale2_bidi.pth` | the trained bidirectional Mamba-3 weights |
| `mamba3_2x_upscale2_bidi_arch.json` | the architecture the weights were trained with |
| `dl_segment_stats_thorax.json` | HU normalisation and `dose_scale` |
| `bev_grid_384_x_zalign.json` | BEV cuboid geometry |

**Proton dose**

| File | What it is |
|---|---|
| `best_model_proton_thorax_mamba3_energy_bragg.pth` | the trained forward Mamba-3 weights |
| `proton_mamba3_energy_bragg_arch.json` | the architecture the weights were trained with |
| `dl_segment_stats_proton_thorax.json` | HU normalisation and `dose_scale` |
| `bev_grid_proton_zalign.json` | BEV cuboid geometry |
| `beam_parameters.json` | the energy table used to tokenise beamlet energies |

**MRI tasks only**

| File | What it is |
|---|---|
| `sct_{photon,proton}_{thorax,abdomen}.pth` | one SwinUNETR generator per modality and anatomy |
| `sct_ct_population_statistics.json` | the HU mean and standard deviation each generator was normalised against |

The statistics file is not optional. `ct_min` and `ct_max` are baked into the
model constructor and `dose_scale` converts the network output back to physical
dose, so a checkpoint served against different statistics produces silently
wrong dose rather than an error.

The staging scripts in `submission/` copy these out of their training runs.

## Pipeline

```
MLC leaves ─► aperture (400²) ─┐
                               ├─► BEV cuboid 384×200×200 ─► CNN-Mamba
gantry + iso ─► beam basis U ──┤                                  │
CT .mha ─► spline coeffs ──────┘                                  ▼
                                                    packed spline coefficients
                                                       (B, 384, 4, 200, 200)
                                                                  │
                            inverse BEV→CT affine ─► Triton packed sampler
                                                                  ▼
                                                    dose in the CT ROI × dose_scale
```

The model emits **packed bicubic spline coefficients, not a dose map**, because
it was trained with `--ct-space-direct-packed-coefficients`. Dose exists only
after `bicubic_iir_sample_affine_from_packed_coeff_triton` evaluates those
coefficients on the CT region of interest. That evaluation is also what places
the result directly in CT space, so there is no separate back-projection step.

## Synthetic CT

The MRI tasks are the CT tasks with one stage in front. The dose model reads a
CT — it normalises HU against the statistics file and samples the volume as
bicubic spline coefficients — so `sct.py` turns the MR into a synthetic CT
first, and that is what the dose model sees. The sCT is produced on the MR's own
voxel grid and is never written to disk.

```
MR .mha ─► z-score ─► SwinUNETR (sliding window, 64×160×160, overlap 0.5)
                              │
                              ├─► × population std + mean  (per anatomy)
                              ├─► outside the body mask := -1024 HU
                              └─► clip to [-1024, 3000] ─► sCT ─► the pipeline above
```

The generators were trained with the body mask taken from the real CT beside the
MR. A container has no real CT, so `sct.body_mask_from_mr` derives the mask from
the MR instead. Measured against the CT-derived mask on 12 held-out patients,
the finished volume differs by 2.2 HU MAE on average (worst patient 9.0 HU), at
Dice 0.990.

Generators are per-anatomy and are paired with their own anatomy's entry in the
statistics file. Crossing them shifts every voxel of the sCT by roughly 90 HU.
The dose checkpoints are not per-anatomy: one thorax-trained model serves both
regions, while each region still gets its own generator.

## Modules

| Module | Role |
|---|---|
| `geometry.py` | beam geometry — aperture from MLC leaves, beam basis, proton ray vectors |
| `bev_grid.py` | BEV cuboid geometry and its configuration |
| `bev_build.py` | builds the BEV input volumes on the GPU |
| `ct_sample.py` | Triton samplers that evaluate packed coefficients on the CT grid |
| `_spline_torch.py` | the torch reference implementation of that sampling |
| `fast_preprocess.py` | fixed-shape Triton preprocessing for CT, aperture and the packed encoder input |
| `predict.py` | the shared predictor — CT loading, plan caching, batching, output assembly |
| `mamba_predict.py` | the photon CNN-Mamba predictors |
| `proton_predict.py`, `proton_mamba_predict.py` | the proton predictors — beamlet geometry, energy tokens, Bragg residual head |
| `proton_preprocess_fusions.py` | fused proton preprocessing kernels |
| `mha_stream.py` | streaming MetaImage writer, overlapping compression with GPU work |
| `sct.py` | the MR-to-CT stage |
| `nets/` | the model definitions, plus the physics conditioning and pixel-shuffle layers |

`nets/` matches the trained architecture apart from import rewiring. Editing it
risks changing `state_dict` keys and breaking checkpoint loading; the predictors
load with `strict=True`, so a mismatch fails loudly rather than producing
garbage.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `DOSERAD_INPUT_UPSCALE_FACTOR` | `2` | samples CT and aperture on the `(2D, 2H, W)` grid and packs the four depth/height phases into model channels. A checkpoint whose first encoder layer does not match is rejected at startup. |
| `DOSERAD_MAMBA_ARCH` | per task | architecture JSON to load |
| `DOSERAD_BEV_GRID` | per task | BEV grid configuration to load |
| `DOSERAD_SCT_PRECISION` | `fp32` | sCT generator forward precision |
| `DOSERAD_SCT_SW_BATCH_SIZE` | `1` | sliding-window batch size for the sCT generator |
