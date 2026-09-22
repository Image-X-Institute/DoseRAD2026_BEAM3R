#!/usr/bin/env bash
#
# do_test_run.sh
#
# Usage: ./do_test_run.sh <TASK>
#   <TASK> matches a test/input/<TASK>/ directory, 
#   e.g. proton-mri, proton-ct, photon-mri, photon-ct.
#
# Builds the algorithm's Docker image, boots it as an HTTP server
# implementing Grand Challenge's "invoke" API, then runs it against every
# packaged run under test/input/<TASK>/run_*/. For each run: stage inputs,
# call /invoke, collect outputs. Also writes test/output/<TASK>/
# predictions.json, an evaluate.py-compatible predictions file covering
# all runs, so evaluate.py can be pointed at this output directly.
#
# Run this after changing the algorithm to confirm the container still
# behaves correctly before uploading it (see ./do_save.sh).

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

INPUT_DIR="${SCRIPT_DIR}/test-data/input"
OUTPUT_DIR="${SCRIPT_DIR}/test-data/output"

HELPER_IMAGE="curlimages/curl:latest"

# Startup warmup compiles the model's Triton kernels before readiness, so this
# budget has to cover the slowest one: ~11s for CNN-xLSTM but ~95s for
# CNN-Mamba3. Override with DOSERAD_HEALTH_ATTEMPTS.
HEALTH_CHECK_MAX_ATTEMPTS="${DOSERAD_HEALTH_ATTEMPTS:-90}"
HEALTH_CHECK_DELAY_SECONDS=3
HEALTH_CHECK_TIMEOUT_SECONDS=10

# A full run is one /invoke covering every control point of every plan, so the
# ceiling scales with the run, not with the algorithm. 600s suits the smoke run
# (60 control points); run_001_full is 5400 and needs far longer. Override with
# DOSERAD_INVOKE_TIMEOUT rather than editing this.
INVOKE_TIMEOUT_SECONDS="${DOSERAD_INVOKE_TIMEOUT:-600}"

DOCKER_VOLUME_TAG=""
DOCKER_NETWORK_TAG=""
TESTER_NAME=""
RUN_DIRS=()
SELECTED_RUNS=()
PREDICTIONS_ENTRIES=()
LOG_FOLLOW_PID=""

if [ "$#" -lt 1 ]; then
  echo "Usage: $(basename "$0") <TASK> [--max-runs N] [--runs NAME ...]"
  echo "  <TASK> matches a test/input/<TASK>/ directory, e.g. proton-mri, proton-ct, photon-mri, photon-ct."
  echo "  --runs picks specific run_* directories; --max-runs caps how many run."
  exit 1
fi
TASK="$1"
shift

# Same naming do_build.sh derives, so this tests the image that was built for
# this task rather than whichever one was built last.
DOSE_VARIANT="${DOSE_VARIANT:-ct}"
[ "$DOSE_VARIANT" = "ct" ] && DOSE_SUFFIX="" || DOSE_SUFFIX="_${DOSE_VARIANT}"
MAMBA_VARIANT="${MAMBA_VARIANT:-unidirectional}"
[ "$MAMBA_VARIANT" = "unidirectional" ] && MAMBA_SUFFIX="" || MAMBA_SUFFIX="_bidi"
VARIANT_SUFFIX="${DOSE_SUFFIX}${MAMBA_SUFFIX}"
DOCKER_IMAGE_TAG="doserad2026_${TASK//-/_}${VARIANT_SUFFIX}"
# Input is per task; output is per task *and* variant, so running the two dose
# checkpoints back to back leaves both sets of predictions to compare instead of
# the second overwriting the first.
OUTPUT_TASK_DIR="${TASK}${VARIANT_SUFFIX}"
CONTAINER_NAME="${DOCKER_IMAGE_TAG}_container"
STAGING_INPUT_VOLUME="${DOCKER_IMAGE_TAG}-staging-input"
STAGING_OUTPUT_VOLUME="${DOCKER_IMAGE_TAG}-staging-output"

MAX_RUNS=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --max-runs)
      MAX_RUNS="$2"
      if ! [[ "$MAX_RUNS" =~ ^[0-9]+$ ]] || [ "$MAX_RUNS" -lt 1 ]; then
        echo "Error: --max-runs requires a positive integer"
        exit 1
      fi
      shift 2
      ;;
    --runs)
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        SELECTED_RUNS+=("$1")
        shift
      done
      if [ "${#SELECTED_RUNS[@]}" -eq 0 ]; then
        echo "Error: --runs requires at least one run directory name"
        exit 1
      fi
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done
printf "Running test for task: %s\n" "$TASK"

main() {
  require_jq

  trap cleanup EXIT
  build_container

  discover_runs
  local run_dirs=("${RUN_DIRS[@]}")
  local total_found="${#run_dirs[@]}"
  if [ -n "$MAX_RUNS" ] && [ "$MAX_RUNS" -lt "$total_found" ]; then
    run_dirs=("${run_dirs[@]:0:$MAX_RUNS}")
    log "Found ${total_found} run(s); limiting to first ${MAX_RUNS} (--max-runs)"
  else
    log "Found ${total_found} run(s): ${run_dirs[*]}"
  fi

  local run_name
  for run_name in "${run_dirs[@]}"; do
    setup
    start_container
    check_health
    process_run "$run_name"
    cleanup
  done

  write_predictions_json

  log "Wrote results to ${OUTPUT_DIR}/${OUTPUT_TASK_DIR}/ (one subdirectory per run, plus predictions.json)"
  log "To run the evaluator against this output locally:"
  log "  DOSERAD_INPUT_DIR=${OUTPUT_DIR}/${OUTPUT_TASK_DIR} DOSERAD_OUTPUT_DIR=<metrics output dir> DOSERAD_GT_DIR=<ground truth dir> python3 evaluate.py"

  log "Save this image for uploading via ./do_save.sh"
}


require_jq() {
  if ! command -v jq &> /dev/null; then
    log "ERROR: jq is required (used to build predictions.json) but was not found on PATH."
    exit 1
  fi
}


# Populates the global RUN_DIRS array (bash 3.2 has no namerefs).
discover_runs() {
  RUN_DIRS=()

  local d
  for d in "${INPUT_DIR}/${TASK}"/run_*; do
    [ -d "$d" ] && RUN_DIRS+=("$(basename "$d")")
  done

  if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
    log "ERROR: no run_* directories found under ${INPUT_DIR}/${TASK}/"
    exit 1
  fi

  IFS=$'\n' RUN_DIRS=($(sort <<<"${RUN_DIRS[*]}")); unset IFS

  # --runs names a subset explicitly. Unknown names are an error rather than a
  # silent no-op: a typo would otherwise look like a run that produced nothing.
  if [ "${#SELECTED_RUNS[@]}" -gt 0 ]; then
    local wanted found kept=()
    for wanted in "${SELECTED_RUNS[@]}"; do
      found=""
      for d in "${RUN_DIRS[@]}"; do
        [ "$d" = "$wanted" ] && found="$d" && break
      done
      if [ -z "$found" ]; then
        log "ERROR: --runs ${wanted}: no such directory under ${INPUT_DIR}/${TASK}/"
        log "Available: ${RUN_DIRS[*]}"
        exit 1
      fi
      kept+=("$found")
    done
    RUN_DIRS=("${kept[@]}")
  fi
}


setup() {
  log "Setup ..."
  chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/example-algorithm/model"

  export DOCKER_CLI_HINTS=false

  docker volume rm -f "$STAGING_INPUT_VOLUME" "$STAGING_OUTPUT_VOLUME" > /dev/null 2>&1 || true
  docker volume create "$STAGING_INPUT_VOLUME" > /dev/null
  docker volume create "$STAGING_OUTPUT_VOLUME" > /dev/null

  # New volumes are root-owned and not writable by the non-root users in
  # the helper and algorithm images.
  docker run --rm --user root \
      --volume "$STAGING_INPUT_VOLUME":/vol_in \
      --volume "$STAGING_OUTPUT_VOLUME":/vol_out \
      "$HELPER_IMAGE" sh -c "chmod 777 /vol_in /vol_out"

  DOCKER_VOLUME_TAG="${DOCKER_IMAGE_TAG}-scratch"
  docker volume create "$DOCKER_VOLUME_TAG" > /dev/null

  CONTAINER_PORT=4743
  BASE_URL="http://${CONTAINER_NAME}:${CONTAINER_PORT}"

  # Isolated network mimicking Grand Challenge's network restrictions.
  DOCKER_NETWORK_TAG="${DOCKER_IMAGE_TAG}-isolated"
  docker network create --internal "$DOCKER_NETWORK_TAG" > /dev/null

  # Sidecar for issuing health/invoke requests from inside the isolated network.
  TESTER_NAME="${DOCKER_IMAGE_TAG}-tester"
  docker run --detach --name "$TESTER_NAME" \
      --network "$DOCKER_NETWORK_TAG" \
      curlimages/curl:latest sleep infinity > /dev/null
}


cleanup() {
  log "Cleanup ..."

  stop_log_follow

  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm -f "$TESTER_NAME" >/dev/null 2>&1 || true
  log "Container stopped"

  docker volume rm -f "$STAGING_INPUT_VOLUME" "$STAGING_OUTPUT_VOLUME" > /dev/null 2>&1 || true
  docker volume rm "$DOCKER_VOLUME_TAG" > /dev/null 2>&1 || true
  docker network rm "$DOCKER_NETWORK_TAG" > /dev/null 2>&1 || true
}


build_container() {
  log "(Re)build the container"
  source "${SCRIPT_DIR}/do_build.sh"

  log "Verifying container labels"
  local api_method
  api_method=$(docker inspect \
      --format='{{index .Config.Labels "org.grand-challenge.api-method"}}' \
      "$DOCKER_IMAGE_TAG" 2>/dev/null || echo "")

  if [ "$api_method" != "invoke" ]; then
    log "ERROR: The container image is missing the required label:"
    log "  LABEL org.grand-challenge.api-method=\"invoke\""
    log ""
    log "Without this label, Grand Challenge will not recognize that your"
    log "container implements the invoke API and will default to exec mode."
    log "Please add this label to your Dockerfile."
    exit 1
  fi
}


start_container() {
  log "Starting container"
  # Which GPUs the container may use, in `docker run --gpus` syntax. Grand
  # Challenge gives the algorithm a dedicated GPU; this box does not, and the
  # dose model plus the sCT generators peak around 8.5 GB, so a run scheduled
  # onto a card another job has nearly filled dies with no HTTP response at all
  # (the tester reports status 000000) rather than a clean error. Pin to an idle
  # card when the default one is busy:
  #
  #   DOSERAD_GPUS='"device=1"' ./do_test_run.sh photon-mri
  #
  # The selected card is always cuda:0 inside the container, so nothing else
  # needs changing.
  local gpus="${DOSERAD_GPUS:-all}"
  log "Requesting GPUs: ${gpus}"
  local docker_run_args=(
    --detach
    --name "$CONTAINER_NAME"
    --gpus "$gpus"
    --platform=linux/amd64
    --volume "${SCRIPT_DIR}/example-algorithm/model":/opt/ml/model:ro
    --volume "$STAGING_INPUT_VOLUME":/input:ro
    --volume "$STAGING_OUTPUT_VOLUME":/output
    --volume "$DOCKER_VOLUME_TAG":/tmp
    --network "$DOCKER_NETWORK_TAG"
    -e TASK="$TASK" # This is NOT available on the Grand Challenge platform unless set in your Dockerfile, only added here for flexibility changing tasks.
  )

  # Any DOSERAD_* knob exported on the host (DOSERAD_VERBOSE, DOSERAD_BEV_GRID,
  # DOSERAD_MHA_BACKEND, ...) is passed through for local experiments. Grand
  # Challenge sets no environment, so anything that must hold for a real
  # submission belongs in the Dockerfile's ENV, not here.
  local var
  for var in $(compgen -v | grep '^DOSERAD_' || true); do
    docker_run_args+=(-e "${var}=${!var}")
    log "Forwarding ${var}=${!var} into the container"
  done

  docker run "${docker_run_args[@]}" "$DOCKER_IMAGE_TAG" >/dev/null

  start_log_follow

  log "Container started; reachable from the tester sidecar at ${BASE_URL}"
}


# Streams container logs to the terminal continuously in the background,
# instead of polling snapshots at a few checkpoints. `docker logs -f`
# first replays anything already emitted, then follows new lines live,
# so nothing written between container start and this call is lost.
start_log_follow() {
  docker logs --follow --timestamps "$CONTAINER_NAME" 2>&1 &
  LOG_FOLLOW_PID=$!
  disown "$LOG_FOLLOW_PID" 2>/dev/null || true
}


stop_log_follow() {
  if [ -n "$LOG_FOLLOW_PID" ] && kill -0 "$LOG_FOLLOW_PID" 2>/dev/null; then
    kill "$LOG_FOLLOW_PID" 2>/dev/null || true
    wait "$LOG_FOLLOW_PID" 2>/dev/null || true
  fi
  LOG_FOLLOW_PID=""
}


http_status() {
  local method="$1"
  local timeout_seconds="$2"
  local url="$3"

  docker exec "$TESTER_NAME" \
      curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout_seconds" \
      -X "$method" "$url" \
    || echo "000"
}


check_health() {
  log "Waiting for health endpoint..."

  local status
  for ((i = 1; i <= HEALTH_CHECK_MAX_ATTEMPTS; i++)); do
    status=$(http_status "GET" "$HEALTH_CHECK_TIMEOUT_SECONDS" "${BASE_URL}/health")

    log "Health check attempt $i/${HEALTH_CHECK_MAX_ATTEMPTS} returned $status"

    if [[ "$status" == "200" ]]; then
      log "API healthy!"
      return 0
    fi

    if [[ "$status" == "302" ]]; then
      log "Health endpoint returned 302 — failing"
      return 1
    fi

    log "Retrying in ${HEALTH_CHECK_DELAY_SECONDS}s"
    sleep "$HEALTH_CHECK_DELAY_SECONDS"
  done

  log "Health endpoint never returned 200"
  return 1
}


provision() {
  local interface_dir="$1"

  log "Provisioning input for ${interface_dir}"

  # Docker silently creates a missing bind-mount source, and creates it as
  # root -- which leaves an undeletable-looking directory inside test-data and
  # makes setup()'s `chmod -R` fail on every subsequent run, long after the
  # typo that caused it. Fail here instead, while the cause is still obvious.
  if [ ! -d "${INPUT_DIR}/${interface_dir}" ]; then
    log "ERROR: no input directory at ${INPUT_DIR}/${interface_dir}"
    exit 1
  fi

  docker exec --user root "$CONTAINER_NAME" /bin/sh -c "rm -rf /output/* /output/.[!.]* 2>/dev/null; true"

  # --user root: needed for cp -a to preserve source ownership and for
  # chmod to touch the volume's own root entry, not just world-writable
  # access to it.
  docker run --rm --user root \
      --volume "${INPUT_DIR}/${interface_dir}":/src:ro \
      --volume "$STAGING_INPUT_VOLUME":/dst \
      "$HELPER_IMAGE" \
      sh -c "rm -rf /dst/* /dst/.[!.]* 2>/dev/null; cp -a /src/. /dst/ && chmod -R a+rX /dst" \
      > /dev/null
}


invoke() {
  log "Calling invoke endpoint..."

  local status
  status=$(http_status "POST" "$INVOKE_TIMEOUT_SECONDS" "${BASE_URL}/invoke")

  if [ "$status" != "201" ]; then
    log "Invoke failed with status $status"
    exit 1
  fi

  log "...invoke completed"
}


collect_output() {
  local interface_dir="$1"
  local destination="${OUTPUT_DIR}/${interface_dir}"

  log "Collecting output for ${interface_dir}"

  if [ -d "$destination" ]; then
    log "Cleaning up any earlier collected output in $destination"
    rm -rf "${destination:?}"/*
  else
    mkdir -p -m o+rwX "$destination"
  fi

  log "Copying outputs to output directory"
  docker exec --user root "$CONTAINER_NAME" /bin/sh -c "chmod -R a+rX /output"

  # docker cp runs on the host, not inside a container, so it isn't
  # subject to rootless Docker's user-namespace UID remapping -- files
  # land owned by whoever is running this script.
  docker cp "${CONTAINER_NAME}:/output/." "$destination/output"
}


build_socket_entries_for_run() {
  # One JSON object per image-N socket in the run's input directory, plus
  # one for the metadata socket. The metadata value is null, matching how
  # the real platform behaves (see evaluate.py's job-identification logic).
  local run_input_dir="$1"

  local slot_dir slug filename mha_path
  for slot_dir in "$run_input_dir"/images/radiation-dose-calculation-source-*-image-*; do
    [ -d "$slot_dir" ] || continue
    slug=$(basename "$slot_dir")
    mha_path=$(find "$slot_dir" -maxdepth 1 -type f | head -n1)
    filename=$(basename "$mha_path")
    jq -n --arg slug "$slug" --arg name "$filename" \
        '{socket: {slug: $slug}, image: {pk: "local-test", name: $name}, value: null}'
  done

  local metadata_path metadata_slug
  metadata_path=$(find "$run_input_dir" -maxdepth 1 -name 'stacked-*-beam-level-metadata.json')
  metadata_slug=$(basename "$metadata_path" .json)
  jq -n --arg slug "$metadata_slug" '{socket: {slug: $slug}, image: null, value: null}'
}


process_run() {
  local run_name="$1"
  # Input is shared by both dose variants and lives under the task; output is
  # kept apart per variant so a second run does not overwrite the first.
  local input_interface_dir="${TASK}/${run_name}"
  local output_interface_dir="${OUTPUT_TASK_DIR}/${run_name}"

  provision "$input_interface_dir"

  SECONDS=0
  invoke
  local duration_iso="PT${SECONDS}S"

  collect_output "$output_interface_dir"

  local inputs_json
  inputs_json=$(build_socket_entries_for_run "${INPUT_DIR}/${input_interface_dir}" | jq -s '.')

  local job_json
  job_json=$(jq -n --arg pk "$run_name" --arg duration "$duration_iso" --argjson inputs "$inputs_json" \
      '{pk: $pk, invoke_duration: $duration, inputs: $inputs}')

  PREDICTIONS_ENTRIES+=("$job_json")

  log "Wrote results to ${OUTPUT_DIR}/${output_interface_dir} (invoke took ${duration_iso})"
}


write_predictions_json() {
  local destination="${OUTPUT_DIR}/${OUTPUT_TASK_DIR}/predictions.json"
  printf '%s\n' "${PREDICTIONS_ENTRIES[@]}" | jq -s '.' > "$destination"
  log "Wrote ${destination} (${#PREDICTIONS_ENTRIES[@]} job(s))"
}


log() {
  local message="$1"
  if [[ -t 1 ]]; then
    printf "\e[38;2;36;150;237m> %s\e[0m\n" "$message"
  else
    printf "%s\n" "$message"
  fi
}


main
