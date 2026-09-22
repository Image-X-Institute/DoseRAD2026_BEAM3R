#!/usr/bin/env bash
# Train one MR -> sCT generator. Run once per anatomy, per modality.
#
#   ANATOMY=thorax  DATA_ROOT=.../Doserad ./train_sct.sh
#   ANATOMY=abdomen DATA_ROOT=.../Doserad ./train_sct.sh
#
#   MODALITY=proton ANATOMY=thorax  DATA_ROOT=.../Doserad ./train_sct.sh
#   MODALITY=proton ANATOMY=abdomen DATA_ROOT=.../Doserad ./train_sct.sh
#
# MODALITY selects both the image tree and the loss weighting, and defaults to
# photon. The proton generators were trained with a heavier reconstruction term
# (lambda 100 rather than 50), so each modality has its own config; see CONFIG
# below.
#
# DATA_ROOT must contain <modality>/training/<patient>/image/{ct.mha,mr.mha} --
# the sCT stage reads the paired images only, not the plans or dose the dose
# model needs. The shipped splits hold out the same patients the released
# checkpoints held out (thorax 3, abdomen 6); see configs/<anatomy>_split.json.
#
# The generator architecture is fixed in train_sct.py
# (SwinUNETR feature_size=48, use_v2=True) because doserad.sct.SCTGenerator
# constructs exactly that and loads the state dict strictly. There is no flag
# for it here on purpose.
#
# Checkpoint selection is best validation L1, which is what produced both
# shipped generators (thorax epoch 972, abdomen epoch 914). It is a weak proxy
# for dose quality -- read validation_wep_error.npy and
# validation_low_density_bias.npy alongside it rather than the L1 curve alone.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCT_SRC="$REPO/external/DoseRAD2026_sCT/src"
CONFIGS="$SCT_SRC/doserad2026_sct/configs"

ANATOMY="${ANATOMY:?set ANATOMY to thorax or abdomen}"
case "$ANATOMY" in
  thorax|abdomen) ;;
  *) echo "ERROR: ANATOMY must be thorax or abdomen, got '$ANATOMY'" >&2; exit 1 ;;
esac
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the Doserad data root}"
MODALITY="${MODALITY:-photon}"
case "$MODALITY" in
  photon|proton) ;;
  *) echo "ERROR: MODALITY must be photon or proton, got '$MODALITY'" >&2; exit 1 ;;
esac
SPLIT_DIR="${SPLIT_DIR:-$CONFIGS}"
# Per modality, because the reconstruction weight differs: the released photon
# generators were trained at lambda 50 and the proton ones at 100. Serving one
# config to both would silently train the wrong arm.
if [ "$MODALITY" = "proton" ]; then
  CONFIG="${CONFIG:-$CONFIGS/train_sCT_proton.json}"
else
  CONFIG="${CONFIG:-$CONFIGS/train_sCT.json}"
fi
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/sCT_runs}"
DEVICE="${DEVICE:-cuda:0}"
RUN_LOG="${RUN_LOG:-$OUT_ROOT/training_logs/sct_${MODALITY}_${ANATOMY}.log}"

mkdir -p "$OUT_ROOT/training_logs"
# generate_unique_log_directory() calls os.path.relpath, so the run directory is
# created relative to the CWD at launch -- do not cd during a run.
cd "$REPO"

# -u because Python block-buffers stdout to a file: without it a redirected run
# shows nothing for hours.
PYTHONPATH="$SCT_SRC" nohup python -u -m doserad2026_sct.train_sct \
  --path_to_data "$DATA_ROOT/$MODALITY/training" \
  --modality "$MODALITY" \
  --anatomy "$ANATOMY" \
  --config_file "$CONFIG" \
  --path_to_splits "$SPLIT_DIR" \
  --path_to_logs "$OUT_ROOT" \
  --device "$DEVICE" \
  "$@" \
  > "$RUN_LOG" 2>&1 &
echo "started: $RUN_LOG"
echo "run dir  : newest directory under $OUT_ROOT (printed at the top of the log)"
