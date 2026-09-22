#!/usr/bin/env bash
#
# Build the algorithm image for one challenge task.
#
#   ./do_build.sh                                    # photon-ct, CT-trained dose model
#   TASK=photon-mri ./do_build.sh
#   MAMBA_VARIANT=bidirectional ./do_build.sh          # bidirectional, chunk 16 + scratch-free
#
# Grand Challenge cannot set environment variables at runtime, so the task is
# baked into the image via the Dockerfile's ARG DOSERAD_TASK -> ENV TASK. The
# tag carries the task too: one shared tag meant building either task silently
# overwrote the other.

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

TASK="${TASK:-photon-ct}"
case "$TASK" in
  photon-ct|proton-ct) ;;
  photon-mri|proton-mri)
    # The MRI tasks run MR -> sCT before the dose model, so the two generators
    # and the population statistics they denormalise with have to be staged.
    # Checked here rather than left to runtime: app.py loads them during
    # startup, so a missing one is a container that never reports healthy.
    for required in \
      sct_photon_thorax.pth \
      sct_photon_abdomen.pth \
      sct_ct_population_statistics.json; do
      if [ ! -f "$SCRIPT_DIR/example-algorithm/model/$required" ]; then
        echo "ERROR: TASK=${TASK} needs a staged sCT file that is missing: $required" >&2
        echo "Run: $SCRIPT_DIR/stage_sct_generators.py --thorax-snapshot <...> --abdomen-snapshot <...>" >&2
        exit 1
      fi
    done
    ;;
  *)
    echo "ERROR: unknown TASK '${TASK}'; expected one of photon-ct, photon-mri, proton-ct, proton-mri" >&2
    exit 1
    ;;
esac

# Which dose checkpoint the image serves: "ct" (trained on real CTs) or "sct"
# (that run fine-tuned on synthetic CTs). Part of the tag, so the two builds of
# a task do not overwrite each other. "ct" adds no suffix, keeping the names of
# everything built before this option existed.
DOSE_VARIANT="${DOSE_VARIANT:-ct}"
case "$DOSE_VARIANT" in
  ct)  DOSE_SUFFIX="" ;;
  sct) DOSE_SUFFIX="_sct" ;;
  *)
    echo "ERROR: unknown DOSE_VARIANT '${DOSE_VARIANT}'; expected ct or sct" >&2
    exit 1
    ;;
esac

MAMBA_VARIANT="${MAMBA_VARIANT:-unidirectional}"
case "$MAMBA_VARIANT" in
  unidirectional)
    MAMBA_SUFFIX=""
    DEFAULT_SCRATCH_FREE=1
    ;;
  bidirectional)
    MAMBA_SUFFIX="_bidi"
    DEFAULT_SCRATCH_FREE=1
    if [ "$DOSE_VARIANT" != "ct" ]; then
      echo "ERROR: MAMBA_VARIANT=bidirectional has no sCT-tuned checkpoint; use DOSE_VARIANT=ct" >&2
      exit 1
    fi
    case "$TASK" in
      photon-ct|photon-mri) ;;
      *)
        echo "ERROR: MAMBA_VARIANT=bidirectional is only available for photon-ct or photon-mri" >&2
        exit 1
        ;;
    esac
    for required in \
      best_model_thorax_mamba3_2x_upscale2_bidi.pth \
      mamba3_2x_upscale2_bidi_arch.json; do
      if [ ! -f "$SCRIPT_DIR/example-algorithm/model/$required" ]; then
        echo "ERROR: missing staged bidirectional model file: $required" >&2
        echo "Run: $SCRIPT_DIR/stage_bidirectional_mamba.py" >&2
        exit 1
      fi
    done
    ;;
  *)
    echo "ERROR: unknown MAMBA_VARIANT '${MAMBA_VARIANT}'; expected unidirectional or bidirectional" >&2
    exit 1
    ;;
esac

MAMBA3_SCRATCH_FREE="${MAMBA3_SCRATCH_FREE:-$DEFAULT_SCRATCH_FREE}"
case "$MAMBA3_SCRATCH_FREE" in
  0|1) ;;
  *)
    echo "ERROR: MAMBA3_SCRATCH_FREE must be 0 or 1" >&2
    exit 1
    ;;
esac

DOCKER_IMAGE_TAG="doserad2026_${TASK//-/_}${DOSE_SUFFIX}${MAMBA_SUFFIX}"
printf 'Building %s (dose=%s, mamba=%s, scratch_free=%s)\n' \
  "$DOCKER_IMAGE_TAG" "$DOSE_VARIANT" "$MAMBA_VARIANT" "$MAMBA3_SCRATCH_FREE"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  --build-arg DOSERAD_TASK="$TASK" \
  --build-arg DOSERAD_SCT_PRECISION="${DOSERAD_SCT_PRECISION:-fp32}" \
  --build-arg DOSERAD_SCT_SW_BATCH_SIZE="${DOSERAD_SCT_SW_BATCH_SIZE:-1}" \
  --build-arg DOSERAD_MAMBA3_SCRATCH_FREE="$MAMBA3_SCRATCH_FREE" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$SCRIPT_DIR/example-algorithm" 2>&1
