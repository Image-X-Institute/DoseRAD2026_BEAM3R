#!/usr/bin/env bash
# Train the proton CNN-Mamba3 dose model.
#
# Every flag below was used for the shipped checkpoint. Changing any of the
# architecture ones (--mamba-*, --spatial-mid, --temporal-conv,
# --model-capacity-scale, --proton-input-upscale-factor, --proton-energy-token,
# --proton-output-refiner, --ct-space-*) produces a model whose state dict will
# not load into the submission image.
#
#   DATA_ROOT   proton training tree:
#                 <DATA_ROOT>/<patient>/<patient>.json   beam/ray/energy plan
#                 <DATA_ROOT>/<patient>/image/ct.mha     planning CT
#                 <DATA_ROOT>/<patient>/dose/*.mha       one file per beamlet
#                 <DATA_ROOT>/beam_parameters.json       proton.energy_table
#   SPLIT       JSON of {"train": [...], "val": [...], "test": [...]} holding
#               out patients by ID. Without it, --validation-fraction splits
#               patients at random, which is not what the shipped run did.
#   OUT_ROOT    where the run directory is created
#
# The shipped checkpoint was warm-started from an earlier energy-token run via
# --init-from. Set INIT_FROM to reproduce that lineage; leaving it empty trains
# the same architecture from scratch, which is a different (longer) trajectory.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the proton training tree}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/runs}"
RUN_NAME="${RUN_NAME:-CNN-Mamba3_proton_x2_thorax_energy_braggresid}"
SPLIT="${SPLIT:?set SPLIT to the patient split JSON}"
BEAM_PARAMETERS="${BEAM_PARAMETERS:-$DATA_ROOT/beam_parameters.json}"
# Committed alongside the code: dose_scale must match what the checkpoint was
# trained with. A mismatch does not error -- it predicts at the wrong scale.
STATS="${STATS:-$REPO/configs/dl_segment_stats_proton_thorax.json}"
INIT_FROM="${INIT_FROM:-}"

mkdir -p "$OUT_ROOT/training_logs"
cd "$REPO"

extra=()
[ -n "$INIT_FROM" ] && extra+=(--init-from "$INIT_FROM")

nohup python -u external/DL-segment-dose-calculation/train/CNN-Mamba/train.py \
  --data-format otf_gpu --otf-gpu-full --otf-modality proton \
  --data-dir "$DATA_ROOT" \
  --proton-beam-parameters "$BEAM_PARAMETERS" \
  --patient-split-json "$SPLIT" \
  --out-dir "$OUT_ROOT/$RUN_NAME" \
  --bev-grid-config configs/bev_grid_proton_zalign.json \
  --stats "$STATS" \
  --proton-energy-token \
  --proton-input-upscale-factor 2 --proton-density-calibration g4dcm_rsp \
  --normalise --otf-bev-mode cubic --otf-stream-dose --otf-gpu-ct-cache-patients 0 \
  --ct-space-loss --ct-space-bev-pixel-shuffle --ct-space-bev-upscale-factor 2 \
  --ct-space-direct-spline-coefficients --ct-space-direct-packed-coefficients \
  --ct-space-valid-voxel-mse --ct-space-negative-dose-weight 0.01 \
  --mamba-core mamba3 --mamba-d 64 --mamba3-headdim 64 --mamba3-chunk-size 64 \
  --spatial-mid --temporal-conv --model-capacity-scale 1.5 \
  --n-prefix 4 --prefix-mode constant --use-scaler --scaler-pool max --scaler-layers 3 \
  --decoder-final-relu --channels-last --fast-cudnn --amp --amp-dtype bfloat16 \
  --batch-size "${BATCH_SIZE:-4}" --num-workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-1e-4}" --epochs "${EPOCHS:-200}" --grad-clip 1.0 \
  --tensorboard --val-metrics-freq 1 --val-beam-idd --masked-mae-weight 1e-4 \
  --proton-fast-preprocess --proton-fast-cache-patients "${FAST_CACHE_PATIENTS:-39}" \
  --proton-ray-aware-batches \
  --proton-output-refiner bragg_residual \
  "${extra[@]}" \
  "$@" \
  > "$OUT_ROOT/training_logs/$RUN_NAME.log" 2>&1 &
echo "started: $OUT_ROOT/training_logs/$RUN_NAME.log"
