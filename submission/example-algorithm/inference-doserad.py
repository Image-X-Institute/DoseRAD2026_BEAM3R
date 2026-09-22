"""CNN-xLSTM dose-calculation algorithm for Grand-Challenge.org.

Runs inside a container. Locally:
    ./do_test_run.sh   # reads ./test/input, writes ./test/output
    ./do_save.sh       # packages the container for upload

Runtime docs: https://grand-challenge.org/documentation/runtime-environment/

The TASK environment variable selects one of four interfaces:

    TASK         input images                   beam-level metadata
    ----------   ---------------------------    ------------------------------------
    photon-ct    ...-source-ct-image-{1..10}    stacked-photon-beam-level-metadata
    proton-ct    ...-source-ct-image-{1..10}    stacked-proton-beam-level-metadata
    photon-mri   ...-source-mri-image-{1..10}   stacked-photon-beam-level-metadata
    proton-mri   ...-source-mri-image-{1..10}   stacked-proton-beam-level-metadata

Only ``photon-ct`` is implemented: the shipped checkpoint is a thorax photon
model trained on CT. The other three raise rather than emit plausible-looking
dose from a model that was never trained for them.

Dose is predicted one control point at a time and written as a 4D stack, one
3D CT-grid volume per control point, thresholded at that control point's
``minimum_cutoff``.
"""

import glob
import json
import os
from pathlib import Path

import SimpleITK as sitk

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

DEFAULT_TASK = "photon-ct"
NUM_OUTPUT_FILES = 10

CT_DIR_BASE = "radiation-dose-calculation-source-ct-image"
MR_DIR_BASE = "radiation-dose-calculation-source-mri-image"
PHOTON_JSON_NAME = "stacked-photon-beam-level-metadata"
PROTON_JSON_NAME = "stacked-proton-beam-level-metadata"

# Each TASK maps to (input image directory base, beam-level metadata file).
TASK_CONFIG = {
    "photon-ct":  (CT_DIR_BASE, PHOTON_JSON_NAME),
    "proton-ct":  (CT_DIR_BASE, PROTON_JSON_NAME),
    "photon-mri": (MR_DIR_BASE, PHOTON_JSON_NAME),
    "proton-mri": (MR_DIR_BASE, PROTON_JSON_NAME),
}
SUPPORTED_TASKS = ("photon-ct",)

# This is NOT available on Grand Challenge unless set in the Dockerfile; it is
# only here for flexibility when changing tasks locally.
TASK = os.environ.get("TASK", DEFAULT_TASK)
if TASK not in TASK_CONFIG:
    raise ValueError(f"Unknown TASK {TASK!r}; expected one of {sorted(TASK_CONFIG)}")

INPUT_DIR_BASE, INPUT_JSON_NAME = TASK_CONFIG[TASK]

# Batching barely improves throughput (~87 -> 80 ms/segment) but scales GPU
# memory linearly, so predict one control point at a time by default.
BATCH_SIZE = int(os.environ.get("DOSERAD_BATCH_SIZE", "1"))


def run(model):
    print(
        f"Running TASK {TASK!r} "
        f"(input_dir_base={INPUT_DIR_BASE!r}, input_json_name={INPUT_JSON_NAME!r})"
    )
    if TASK not in SUPPORTED_TASKS:
        raise NotImplementedError(
            f"TASK {TASK!r} is not supported by this algorithm. The shipped "
            f"checkpoint is a thorax photon model trained on CT; supported: "
            f"{sorted(SUPPORTED_TASKS)}."
        )

    from doserad import geometry

    print("Loading json metadata:")
    metadata = load_json_file(INPUT_PATH / f"{INPUT_JSON_NAME}.json")
    segments = list(geometry.iter_photon_segments(metadata))
    print(f"Parsed {len(segments)} control points")

    # Group control points per output file, keyed by their slice position.
    per_output = [dict() for _ in range(NUM_OUTPUT_FILES)]
    for segment in segments:
        if segment.output_file_idx >= NUM_OUTPUT_FILES:
            raise ValueError(
                f"{segment.name}: output_file_idx={segment.output_file_idx} "
                f"exceeds the {NUM_OUTPUT_FILES} declared outputs"
            )
        per_output[segment.output_file_idx][segment.idx_in_output] = segment

    stack_sizes = [max(slot) + 1 if slot else 0 for slot in per_output]
    print(f"Stack sizes: {stack_sizes}")

    for output_index in range(NUM_OUTPUT_FILES):
        slot = per_output[output_index]
        stack_size = stack_sizes[output_index]

        output_dir = OUTPUT_PATH / f"images/stacked-radiation-dose-map-{output_index + 1}"
        os.makedirs(output_dir, exist_ok=True)

        if stack_size == 0:
            # Empty stack: write a placeholder to honor the output contract.
            sitk.WriteImage(sitk.Image(1, 1, sitk.sitkFloat32), output_dir / "output.mha")
            continue

        missing = [i for i in range(stack_size) if i not in slot]
        if missing:
            raise ValueError(
                f"output {output_index + 1}: metadata has no control point for "
                f"idx_in_output {missing}"
            )
        ordered = [slot[i] for i in range(stack_size)]

        # Every slice in a stack shares the same source image.
        image_indices = {segment.image_file_idx for segment in ordered}
        if len(image_indices) != 1:
            raise ValueError(
                f"output {output_index + 1}: control points span multiple source "
                f"images {sorted(image_indices)}"
            )
        patient_ct = load_ct_by_index(model, image_indices.pop())

        print(
            f"Predicting dose stack for output file index {output_index + 1} "
            f"with {stack_size} slices"
        )
        dose_slices = [None] * stack_size
        by_name = {segment.name: i for i, segment in enumerate(ordered)}
        for segment, dose_np in model.predict_segments(
            ordered, patient_ct, batch_size=BATCH_SIZE
        ):
            # Threshold below the cutoff to keep the written file small.
            dose_np[dose_np < segment.minimum_cutoff] = 0.0
            dose_slice = sitk.GetImageFromArray(dose_np)
            dose_slice.CopyInformation(patient_ct.image)
            dose_slices[by_name[segment.name]] = dose_slice

        stacked = sitk.JoinSeries(dose_slices)
        print(stacked.GetSize())
        sitk.WriteImage(stacked, output_dir / "output.mha", useCompression=False)

    return 0


def load_json_file(location):
    with open(location) as f:
        return json.load(f)


def load_sitk_path(location):
    """Return the first .mha file found in a directory."""
    mha_files = glob.glob(str(location / "*.mha"))
    print(f"Searching for input images in {location}")
    if not mha_files:
        raise FileNotFoundError(f"No .mha file under {location}")
    return mha_files[0]


# One-entry CT cache. Not functools.lru_cache: the predictor is a dataclass and
# so unhashable, and holding two spline-filtered volumes on the GPU at once is
# wasteful. Outputs are processed in order and every control point in a stack
# shares its source image, so consecutive lookups hit and a switch evicts.
_CT_CACHE: dict[int, object] = {}


def load_ct_by_index(model, input_file_idx):
    """Load and spline-prefilter one source CT, reusing the previous one."""
    cached = _CT_CACHE.get(input_file_idx)
    if cached is not None:
        return cached

    location = INPUT_PATH / f"images/{INPUT_DIR_BASE}-{input_file_idx + 1}"
    patient_ct = model.load_ct(load_sitk_path(location))
    print(
        f"Loaded input image {input_file_idx + 1} "
        f"with shape {patient_ct.shape} and spacing {patient_ct.spacing}"
    )
    _CT_CACHE.clear()
    _CT_CACHE[input_file_idx] = patient_ct
    return patient_ct


if __name__ == "__main__":
    from app import init_model
    raise SystemExit(run(model=init_model()))
