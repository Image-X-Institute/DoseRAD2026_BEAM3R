#!/usr/bin/env bash
# Train the bidirectional CNN-Mamba3 dose model.
#
# Every flag below was used for the trained checkpoint; changing any of the
# architecture ones (--mamba-*, --spatial-mid, --temporal-conv,
# --bidirectional-mamba, --model-capacity-scale, --otf-input-upscale-factor,
# --ct-space-*) produces a model whose state dict will not load into the
# submission image.
#
#   DATA_ROOT   contains photon/<split>/<patient>/{<patient>.json,image,dose}
#               and segment_mac/{mac,segments}
#   OUT_ROOT    where the run directory is created
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the Doserad data root}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/runs}"
RUN_NAME="${RUN_NAME:-merged_2x_CNN-Mamba3_bidi}"
SPLIT="${SPLIT:-$DATA_ROOT/configs/bev_patient_split_merged_wed_seed333.json}"
# Committed alongside the code: dose_scale must match what the checkpoint
# was trained with, and this is the same 73-byte file the submission image
# serves. A mismatch does not error -- it predicts at the wrong scale.
STATS="${STATS:-$REPO/configs/dl_segment_stats_thorax.json}"

mkdir -p "$OUT_ROOT/training_logs"
cd "$REPO"

nohup python -u external/DL-segment-dose-calculation/train/CNN-Mamba/train.py \
  --data-format otf_gpu --otf-gpu-full --otf-bev-mode cubic \
  --baseline-pb-dir "$DATA_ROOT" --baseline-split training \
  --segment-mac-dir "$DATA_ROOT/segment_mac" \
  --patient-split-json "$SPLIT" \
  --out-dir "$OUT_ROOT/$RUN_NAME" \
  --stats "$STATS" \
  --bev-grid-config configs/bev_grid_384_x_zalign.json \
  --normalise --tensorboard --val-metrics-freq 1 --val-beam-idd \
  --batch-size "${BATCH_SIZE:-4}" --num-workers "${NUM_WORKERS:-8}" \
  --lr "${LR:-1e-4}" --lr-patience 10 --max-grad-norm 1.0 --epochs "${EPOCHS:-200}" \
  --amp --amp-dtype bfloat16 \
  --masked-mae-weight 2e-4 --masked-mae-threshold 0.10 \
  --spatial-mid --temporal-conv --mamba-core mamba3 --mamba-d 64 \
  --bidirectional-mamba \
  --ct-space-loss --ct-space-bev-pixel-shuffle \
  --ct-space-direct-spline-coefficients --ct-space-direct-packed-coefficients \
  --otf-input-upscale-factor 2 \
  --otf-batched-affine-preprocess --otf-batched-affine-cache-patients 66 \
  --otf-gpu-ct-cache-patients 2 --otf-collapsed-ct-dtype float16 \
  --model-capacity-scale 2 --fast-cudnn --resume \
  "$@" \
  > "$OUT_ROOT/training_logs/$RUN_NAME.log" 2>&1 &
echo "started: $OUT_ROOT/training_logs/$RUN_NAME.log"
