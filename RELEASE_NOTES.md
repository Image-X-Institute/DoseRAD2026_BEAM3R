# BEAM3R v1.0.0 — trained models

Trained weights for the DoseRAD2026 Grand Challenge submission: two dose models
(photon and proton) and four MR→sCT generators, one per modality and anatomy.

Every configuration file these weights need — architecture JSONs, BEV grids,
normalisation statistics, the proton energy table — is already in the
repository. Only the `.pth` weights are distributed here, because they are too
large for git.

## Contents

| Asset | Size | Used by |
|---|---|---|
| `best_model_thorax_mamba3_2x_upscale2_bidi.pth` | 6.8 MB | `photon-ct`, `photon-mri` |
| `best_model_proton_thorax_mamba3_energy_bragg.pth` | 2.8 MB | `proton-ct`, `proton-mri` |
| `sct_photon_thorax.pth` | 284.8 MB | `photon-mri` |
| `sct_photon_abdomen.pth` | 284.8 MB | `photon-mri` |
| `sct_proton_thorax.pth` | 284.8 MB | `proton-mri` |
| `sct_proton_abdomen.pth` | 284.8 MB | `proton-mri` |
| `SHA256SUMS.txt` | — | checksum manifest |

Both dose models are thorax-trained and serve both anatomies. The sCT
generators are genuinely per-anatomy — each was trained on its own anatomy's
split — so they must not be crossed.

Download only what your task needs. The CT tasks need one small file each; only
the MRI tasks pull the large generators.

| Task | Weights needed | Total |
|---|---|---|
| `photon-ct` | photon dose model | 6.8 MB |
| `proton-ct` | proton dose model | 2.8 MB |
| `photon-mri` | photon dose model + both photon generators | 577 MB |
| `proton-mri` | proton dose model + both proton generators | 572 MB |

## Download

The weights go in `submission/example-algorithm/model/`, alongside the
configuration files already in the clone. Do not rename them — each is found by
filename.

```bash
git clone https://github.com/Image-X-Institute/DoseRAD2026_BEAM3R.git
cd DoseRAD2026_BEAM3R

# everything (1.2 GB)
gh release download v1.0.0 --dir submission/example-algorithm/model/

# or one task, e.g. photon-ct (6.8 MB)
gh release download v1.0.0 \
  --pattern 'best_model_thorax_mamba3_2x_upscale2_bidi.pth' \
  --pattern 'SHA256SUMS.txt' \
  --dir submission/example-algorithm/model/
```

Without the `gh` CLI, download the assets from the release page into that same
directory.

## Verify

```bash
cd submission/example-algorithm/model
sha256sum --check --ignore-missing SHA256SUMS.txt
```

`--ignore-missing` lets the check pass when you fetched only one task's weights.
Every file present must report `OK`.

## Where the files go

Configuration from the clone, weights from this release:

```
submission/example-algorithm/model/
├── best_model_thorax_mamba3_2x_upscale2_bidi.pth      ← release
├── mamba3_2x_upscale2_bidi_arch.json                  ← in the repo
├── dl_segment_stats_thorax.json                       ← in the repo
├── bev_grid_384_x_zalign.json                         ← in the repo
├── sct_photon_{thorax,abdomen}.pth                    ← release   (MRI only)
└── sct_ct_population_statistics.json                  ← in the repo (MRI only)
```

Proton tasks use `best_model_proton_thorax_mamba3_energy_bragg.pth` with
`proton_mamba3_energy_bragg_arch.json`, `dl_segment_stats_proton_thorax.json`,
`bev_grid_proton_zalign.json` and `beam_parameters.json`.

Statistics files are not optional and must stay paired with their weights:
`ct_min` and `ct_max` are baked into the model constructor and `dose_scale`
converts the network output back to physical dose. A checkpoint served against
different statistics produces silently wrong dose rather than an error.

## Build an inference image

Requires Docker with the NVIDIA container runtime and a CUDA GPU — the BEV
builder uses CuPy kernels and the samplers are Triton, so there is no CPU path.

```bash
cd submission

# photon — the shipped photon checkpoint is bidirectional
TASK=photon-ct  MAMBA_VARIANT=bidirectional ./do_build.sh
TASK=photon-mri MAMBA_VARIANT=bidirectional ./do_build.sh

# proton — bidirectional is photon-only and will be rejected
TASK=proton-ct  ./do_build.sh
TASK=proton-mri ./do_build.sh
```

`do_build.sh` bakes the task into the image and checks that the weights that
task needs are present, so a missing file fails at build time rather than at
container startup. Each task gets its own image tag, so building one never
overwrites another.

## Run inference

The image implements Grand Challenge's `invoke` API. It does not process a case
and exit: the entrypoint starts an HTTP server on port 4743 and waits, so run it
detached, publish the port, and drive it over HTTP.

```bash
cd submission

CASE=/path/to/case          # one case, laid out as below
OUT=/path/to/results        # must be writable by the container's non-root user

mkdir -p "$OUT" && chmod 777 "$OUT"

docker run --rm -d --name beam3r --gpus device=0 -p 4743:4743 \
  --volume "$PWD/example-algorithm/model":/opt/ml/model:ro \
  --volume "$CASE":/input:ro \
  --volume "$OUT":/output \
  doserad2026_photon_ct_bidi

# startup JITs the Triton kernels -- allow ~60-95 s before it reports healthy
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:4743/health)" = "200" ]; do
  sleep 3
done

curl -X POST http://localhost:4743/invoke     # 201 Created on success

docker stop beam3r
```

Dose lands in `$OUT/images/stacked-radiation-dose-map-<n>/output.mha`, one
4-D stack per output socket.

`/input` holds one case in the challenge's layout: the CT (or MR) volumes under
`images/`, plus the run's beam-level metadata JSON at the top level.

```
/input/
├── images/
│   └── radiation-dose-calculation-source-ct-image-<n>/…
└── stacked-photon-beam-level-metadata.json
```

`submission/do_test_run.sh` automates the whole loop (build, boot, health-check,
invoke, collect), but it expects packaged cases under `submission/test-data/`,
which are not distributed with this repository.

To package for upload, `./do_save.sh` writes the image and model tarballs.

## Notes

- Photon and proton use different architectures, BEV grids and normalisation,
  selected automatically from `TASK`.
- Training these models from scratch is documented in the repository README.
