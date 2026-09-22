#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Which challenge task to package; do_build.sh bakes it into the image and
# derives the tag from it, so image, image tarball and model tarball are all
# named per task and building one never overwrites the other.
#
#   ./do_save.sh                     # photon-ct
#   TASK=photon-mri ./do_save.sh
#   TASK=photon-mri DOSE_VARIANT=sct ./do_save.sh
#   MAMBA_VARIANT=bidirectional ./do_save.sh
#
# The model tarball is deliberately NOT per-variant: both dose checkpoints are a
# few MB and ship in every tarball, so the ~570 MB photon-mri model only has to
# be uploaded once and both images select from it by filename.
TASK="${TASK:-photon-ct}"
# Same variant suffix do_build.sh derives; see the comment there.
DOSE_VARIANT="${DOSE_VARIANT:-ct}"
[ "$DOSE_VARIANT" = "ct" ] && DOSE_SUFFIX="" || DOSE_SUFFIX="_${DOSE_VARIANT}"
MAMBA_VARIANT="${MAMBA_VARIANT:-unidirectional}"
[ "$MAMBA_VARIANT" = "unidirectional" ] && MAMBA_SUFFIX="" || MAMBA_SUFFIX="_bidi"
DOCKER_IMAGE_TAG="doserad2026_${TASK//-/_}${DOSE_SUFFIX}${MAMBA_SUFFIX}"

echo ""
echo "= STEP 1 = (Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"
echo "==== Done"
echo ""

# Get the build information from the Docker image tag
build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $DOCKER_IMAGE_TAG"
    exit 1
fi

# Format the build information to remove special characters
formatted_build_info=$(echo $build_timestamp | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

# Set the output filename with timestamp and build information
output_filename="${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
output_path="${SCRIPT_DIR}/$output_filename"

# Save the Docker-container image and gzip it
echo "= STEP 2 = Saving the image"
echo "This can take a while."

# pigz is gzip-format compatible and parallelises across cores, which dominates
# the wall time for an 11 GB image. Fall back to gzip where it is not installed;
# the output is a valid .tar.gz either way, so uploads are unaffected.
if command -v pigz >/dev/null; then
  docker save "$DOCKER_IMAGE_TAG" | pigz -c > "$output_path"
else
  docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_path"
fi
printf "Saved as: \e[32m${output_filename}\e[0m\n"

echo "==== Done"
echo ""


# Create the tarbal
echo "= STEP 3 = Packing the model"
echo "This can take a while."
output_tarball_name="${SCRIPT_DIR}/model_${TASK//-/_}.tar.gz"

# The whole model directory goes in every task's tarball, including the
# checkpoints that task does not serve: they are a few MB each, and app.py picks
# by name. The sCT generators are the exception at ~300 MB apiece, which is why
# photon-mri's tarball is far larger than photon-ct's.
tar -czf $output_tarball_name -C "${SCRIPT_DIR}/example-algorithm/model" .
printf "Saved as: \e[32mmodel_${TASK//-/_}.tar.gz\e[0m\n"

echo "==== Done"
echo ""

printf "\e[31mIMPORTANT: Please upload the model.tar.gz as seperate Model to your Algorithm!\e[0m\n"
