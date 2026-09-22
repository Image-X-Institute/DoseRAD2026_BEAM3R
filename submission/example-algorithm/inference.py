"""CNN-xLSTM / CNN-Mamba dose-calculation algorithm for Grand-Challenge.org.

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

``photon-ct`` and ``photon-mri`` are implemented. The two proton tasks raise
rather than emit plausible-looking dose from a photon model.

``photon-mri`` is the CT task with one stage in front: the dose network reads a
CT, so each source MRI is first turned into a synthetic CT by the region's
SwinUNETR generator (``doserad/sct.py``) and that is what the dose model sees.
The sCT lands on the MRI's own voxel grid, so the output dose stacks keep the
input image's geometry exactly as they do on the CT tasks.

Each source image carries an ``anatomical_region`` in the metadata, which
selects the region's checkpoint. One job can mix regions, so the choice is made
per output stack rather than once per invoke. This submission maps both
supported regions onto the same checkpoint (see app.py's REGION_CHECKPOINTS),
but the per-stack lookup stays -- an unrecognized region must still fail rather
than be served by whichever model happens to be loaded.

Dose is predicted one control point at a time and written as a 4D stack, one
3D CT-grid volume per control point, thresholded at that control point's
``minimum_cutoff``.
"""

import glob
import json
import os
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import numpy as np
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
SUPPORTED_TASKS = ("photon-ct", "photon-mri", "proton-ct", "proton-mri")

# Maps the metadata's anatomical_region onto app.py's REGION_CHECKPOINTS keys.
# Applying the wrong region's checkpoint rescales every voxel by that region's
# dose_scale, which produces plausible-looking but wrong dose, so an
# unrecognized region raises instead of falling back to a default.
REGION_TO_MODEL = {
    "thoracic": "thorax",
    "abdominal": "abdomen",
}

# This is NOT available on Grand Challenge unless set in the Dockerfile; it is
# only here for flexibility when changing tasks locally.
TASK = os.environ.get("TASK", DEFAULT_TASK)
if TASK not in TASK_CONFIG:
    raise ValueError(f"Unknown TASK {TASK!r}; expected one of {sorted(TASK_CONFIG)}")

INPUT_DIR_BASE, INPUT_JSON_NAME = TASK_CONFIG[TASK]


def is_proton_task() -> bool:
    """Whether this interface carries proton beamlets rather than control points."""
    return TASK in {"proton-ct", "proton-mri"}


def is_mri_task() -> bool:
    """Whether the source image is an MRI and needs the sCT stage in front."""
    return INPUT_DIR_BASE == MR_DIR_BASE


# Set by _run from app.py's bundle. Both MRI tasks route through the same
# manager, which owns the per-region generators and the per-invoke sCT cache.
_SCT_MANAGER = None
# MRI tasks need the sCT stage in front of the dose model; CT tasks feed the
# source image to it directly.
NEEDS_SCT = INPUT_DIR_BASE == MR_DIR_BASE

# Batching offers little extra throughput after cached affine preprocessing but
# scales model activation memory linearly, so predict one control point at a
# time by default.
BATCH_SIZE = int(os.environ.get("DOSERAD_BATCH_SIZE", "1"))
# Per-control-point logging. Off by default: each line costs a max/count reduce
# over the predicted volume, which is wasted work on a scored submission. There
# is no ground truth inside the container, so these are distribution sanity
# checks (is the dose non-zero, is its peak plausible), not accuracy metrics --
# for those, score the collected output with ./run_beam_eval.py afterwards.
VERBOSE = os.environ.get("DOSERAD_VERBOSE", "0").strip().lower() in {"1", "true", "yes"}
MHA_BACKEND = os.environ.get("DOSERAD_MHA_BACKEND", "stream").strip().lower()
if MHA_BACKEND not in {"sparse", "stream", "sitk"}:
    raise ValueError(
        "DOSERAD_MHA_BACKEND must be 'sparse', 'stream', or 'sitk'"
    )
# The evaluator counts every uncompressed output stack as an implementation
# error (one per beam in the stack), so compression is on by default. Level 1
# is the deliberate choice: on this data it reaches ~2.1x for a few seconds per
# stack, while level 6 buys another ~3% for roughly triple the time, and invoke
# duration is itself regressed and reported. Set 0 to write raw.
COMPRESS_LEVEL = int(os.environ.get("DOSERAD_MHA_COMPRESS_LEVEL", "1"))
if not 0 <= COMPRESS_LEVEL <= 9:
    raise ValueError("DOSERAD_MHA_COMPRESS_LEVEL must be between 0 and 9")
# Sparse writing positions ROIs by file offset, which a sequential deflate
# stream cannot do. Asking for both is a configuration error rather than
# something to silently resolve in one direction.
if MHA_BACKEND == "sparse" and COMPRESS_LEVEL:
    raise ValueError(
        "DOSERAD_MHA_BACKEND='sparse' cannot be combined with "
        "DOSERAD_MHA_COMPRESS_LEVEL>0; use the 'stream' backend for "
        "compressed output, or set the level to 0 to keep sparse writes"
    )
# ISA-L deflates several times faster than stdlib zlib at the same level and is
# picked automatically when importable; "zlib" forces the fallback, which is
# what runs if the optional wheel is missing.
MHA_COMPRESSOR = os.environ.get(
    "DOSERAD_MHA_COMPRESSOR", "auto"
).strip().lower()
if MHA_COMPRESSOR not in {"auto", "isal", "zlib"}:
    raise ValueError("DOSERAD_MHA_COMPRESSOR must be auto, isal, or zlib")
# Interleave several output stacks so one stack's compression overlaps the
# next's GPU work instead of serialising behind it. Only meaningful for the
# stream backend; sparse writes are already offset-positioned and uncompressed.
PARALLEL_STACK_WRITERS = os.environ.get(
    "DOSERAD_PARALLEL_STACK_WRITERS", "1"
).strip().lower() in {"1", "true", "yes"}
MHA_QUEUE_DEPTH = int(os.environ.get("DOSERAD_MHA_QUEUE_DEPTH", "1"))
if MHA_QUEUE_DEPTH < 1:
    raise ValueError("DOSERAD_MHA_QUEUE_DEPTH must be positive")
# Each open writer holds a full stack's worth of buffers, so the group size
# bounds peak host memory, not just concurrency.
MAX_PARALLEL_STACK_WRITERS = int(
    os.environ.get("DOSERAD_MAX_PARALLEL_STACK_WRITERS", "4")
)
if MAX_PARALLEL_STACK_WRITERS < 1:
    raise ValueError("DOSERAD_MAX_PARALLEL_STACK_WRITERS must be positive")
# Total compression threads, split across the stacks currently open rather than
# applied per stack, so the CPU budget does not scale with the group size.
MHA_COMPRESSION_WORKERS = int(
    os.environ.get("DOSERAD_MHA_COMPRESSION_WORKERS", "1")
)
if MHA_COMPRESSION_WORKERS < 1:
    raise ValueError("DOSERAD_MHA_COMPRESSION_WORKERS must be positive")
MHA_COMPRESSION_SLAB_MIB = int(
    os.environ.get("DOSERAD_MHA_COMPRESSION_SLAB_MIB", "32")
)
if MHA_COMPRESSION_SLAB_MIB < 1:
    raise ValueError("DOSERAD_MHA_COMPRESSION_SLAB_MIB must be positive")
MHA_COMPRESSION_SLAB_BYTES = MHA_COMPRESSION_SLAB_MIB * 1024 * 1024


def compression_worker_allocations(stack_count: int) -> list[int]:
    """Distribute the fixed CPU compression budget across active stacks."""
    count = int(stack_count)
    if count <= 0:
        return []
    base, extra = divmod(MHA_COMPRESSION_WORKERS, count)
    return [max(1, base + (index < extra)) for index in range(count)]




def _run(bundle):
    """Predict every control point in the job.

    ``bundle`` is app.py's ``{"dose": {region: predictor}, "sct": manager}``.
    The sCT entry is ``None`` for the CT tasks, which need no generator at all.
    """
    global _SCT_MANAGER
    print(
        f"Running TASK {TASK!r} "
        f"(input_dir_base={INPUT_DIR_BASE!r}, input_json_name={INPUT_JSON_NAME!r}, "
        f"needs_sct={NEEDS_SCT})"
    )
    if TASK not in SUPPORTED_TASKS:
        raise NotImplementedError(
            f"TASK {TASK!r} is not supported by this algorithm; "
            f"supported: {sorted(SUPPORTED_TASKS)}."
        )

    from doserad import geometry

    models = bundle["dose"]
    _SCT_MANAGER = bundle["sct"]
    if NEEDS_SCT and _SCT_MANAGER is None:
        raise RuntimeError(
            f"TASK {TASK!r} reads MRI and needs the sCT stage, "
            "but no generator manager was loaded"
        )
    if _SCT_MANAGER is not None:
        _SCT_MANAGER.begin_invoke()

    print("Loading json metadata:")
    metadata = load_json_file(INPUT_PATH / f"{INPUT_JSON_NAME}.json")
    model_by_image = build_model_index(metadata, models)
    segments = list(
        geometry.iter_proton_segments(metadata)
        if is_proton_task()
        else geometry.iter_photon_segments(metadata)
    )
    print(
        f"Parsed {len(segments)} "
        + ("beamlets" if is_proton_task() else "control points")
    )

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

    if MHA_BACKEND == "stream" and PARALLEL_STACK_WRITERS:
        return run_parallel_output_stacks(
            per_output, stack_sizes, model_by_image, models
        )

    for output_index in range(NUM_OUTPUT_FILES):
        slot = per_output[output_index]
        stack_size = stack_sizes[output_index]

        output_dir = OUTPUT_PATH / f"images/stacked-radiation-dose-map-{output_index + 1}"
        os.makedirs(output_dir, exist_ok=True)

        if stack_size == 0:
            # Empty stack: write a placeholder to honor the output contract.
            # Compressed like every other stack -- the evaluator's check is on
            # the file, and it has no exemption for placeholders.
            placeholder_type = (
                sitk.sitkFloat64 if is_proton_task() else sitk.sitkFloat32
            )
            sitk.WriteImage(
                sitk.Image(1, 1, placeholder_type),
                output_dir / "output.mha",
                useCompression=True,
            )
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
        image_idx = image_indices.pop()

        # The stack's single source image fixes the region, so the checkpoint is
        # chosen once per stack rather than per control point.
        model_key = model_by_image[image_idx]
        model = models[model_key]
        patient_ct = load_ct_by_index(model, model_key, image_idx)

        print(
            f"Predicting dose stack for output file index {output_index + 1} "
            f"with {stack_size} slices using the {model_key} model"
        )
        output_path = output_dir / "output.mha"
        t_stack = time.perf_counter()
        if MHA_BACKEND in {"sparse", "stream"}:
            from doserad.mha_stream import StreamingMHAWriter

            with StreamingMHAWriter(
                output_path,
                size_xyz=patient_ct.shape,
                stack_size=stack_size,
                spacing_xyz=patient_ct.spacing,
                origin_xyz=patient_ct.origin,
                direction_xyz=patient_ct.image.GetDirection(),
                queue_depth=MHA_QUEUE_DEPTH,
                sparse=MHA_BACKEND == "sparse",
                compress_level=COMPRESS_LEVEL,
                compressor_backend=MHA_COMPRESSOR,
                compression_workers=MHA_COMPRESSION_WORKERS,
                compression_slab_bytes=MHA_COMPRESSION_SLAB_BYTES,
            ) as writer:
                if MHA_BACKEND == "sparse":
                    for segment, dose_roi, roi_box in predict_rois_for_writer(
                        model, ordered, patient_ct
                    ):
                        ensure_minimum_cutoff(model, dose_roi, segment.minimum_cutoff)
                        log_segment(segment, dose_roi, roi_box=roi_box)
                        writer.submit_roi(dose_roi, roi_box)
                else:
                    for segment, dose_np in model.predict_segments(
                        ordered, patient_ct, batch_size=BATCH_SIZE
                    ):
                        dose_np[dose_np < cutoff_threshold(segment.minimum_cutoff)] = 0.0
                        log_segment(segment, dose_np)
                        writer.submit(dose_np)
        else:
            dose_slices = [None] * stack_size
            by_name = {segment.name: i for i, segment in enumerate(ordered)}
            for segment, dose_np in model.predict_segments(
                ordered, patient_ct, batch_size=BATCH_SIZE
            ):
                dose_np[dose_np < cutoff_threshold(segment.minimum_cutoff)] = 0.0
                log_segment(segment, dose_np)
                dose_slice = sitk.GetImageFromArray(dose_np)
                dose_slice.CopyInformation(patient_ct.image)
                dose_slices[by_name[segment.name]] = dose_slice
            stacked = sitk.JoinSeries(dose_slices)
            print(stacked.GetSize())
            sitk.WriteImage(
                stacked, output_path, useCompression=bool(COMPRESS_LEVEL)
            )

        elapsed = time.perf_counter() - t_stack
        print(
            f"Finished output file index {output_index + 1} "
            f"({model_key}, {stack_size} slices) in {elapsed:.1f}s "
            f"({elapsed / stack_size:.2f}s/control point)",
            flush=True,
        )

    return 0


# The optional profiling wrapper is not part of this repository: it only ever
# wrapped the production callable when DOSERAD_PROFILE was set, which no
# submission build does, so `run` is the plain function.
run = _run


def run_parallel_output_stacks(
    per_output, stack_sizes, model_by_image, models
):
    """Feed bounded groups of stacks while their compressors run in parallel.

    The sequential path finishes one stack before starting the next, so every
    stack's deflate runs with the GPU idle. Here a group of writers stays open
    at once and the predictors are advanced round-robin, one control point at a
    time, which keeps compression threads busy against the next stack's
    prediction. Output is byte-identical either way -- each writer still
    receives its own stack's slices in index order.
    """
    from doserad.mha_stream import StreamingMHAWriter

    states = []
    for output_index, (slot, stack_size) in enumerate(
        zip(per_output, stack_sizes)
    ):
        output_dir = OUTPUT_PATH / (
            f"images/stacked-radiation-dose-map-{output_index + 1}"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "output.mha"
        if stack_size == 0:
            placeholder_type = (
                sitk.sitkFloat64 if is_proton_task() else sitk.sitkFloat32
            )
            sitk.WriteImage(
                sitk.Image(1, 1, placeholder_type),
                output_path,
                useCompression=True,
            )
            continue

        missing = [index for index in range(stack_size) if index not in slot]
        if missing:
            raise ValueError(
                f"output {output_index + 1}: metadata has no control point for "
                f"idx_in_output {missing}"
            )
        ordered = [slot[index] for index in range(stack_size)]
        image_indices = {segment.image_file_idx for segment in ordered}
        if len(image_indices) != 1:
            raise ValueError(
                f"output {output_index + 1}: control points span multiple "
                f"source images {sorted(image_indices)}"
            )
        image_idx = image_indices.pop()
        model_key = model_by_image[image_idx]
        states.append({
            "output_index": output_index,
            "output_path": output_path,
            "ordered": ordered,
            "stack_size": stack_size,
            "model_key": model_key,
            "model": models[model_key],
            "image_idx": image_idx,
            "written": 0,
        })

    if not states:
        return 0

    total_items = sum(state["stack_size"] for state in states)
    started = time.perf_counter()
    print(
        f"Starting {len(states)} output stacks in groups of at most "
        f"{MAX_PARALLEL_STACK_WRITERS} writers for "
        f"{total_items} control points (compressor={MHA_COMPRESSOR}, "
        f"queue_depth={MHA_QUEUE_DEPTH})",
        flush=True,
    )
    for group_start in range(0, len(states), MAX_PARALLEL_STACK_WRITERS):
        group = states[
            group_start : group_start + MAX_PARALLEL_STACK_WRITERS
        ]
        compression_workers = compression_worker_allocations(len(group))
        # The module-level one-entry CT cache is sized for the sequential path;
        # a group holds several patients open at once, so it is cached here for
        # the group's lifetime instead. Stacks sharing a source image share the
        # prepared volume rather than spline-filtering it twice.
        patient_cache = {}
        with ExitStack() as writers:
            for state, worker_count in zip(group, compression_workers):
                cache_key = (state["model_key"], state["image_idx"])
                patient_ct = patient_cache.get(cache_key)
                if patient_ct is None:
                    patient_ct = load_ct_by_index(state["model"], *cache_key)
                    patient_cache[cache_key] = patient_ct
                state["patient_ct"] = patient_ct
                state["writer"] = writers.enter_context(StreamingMHAWriter(
                    state["output_path"],
                    size_xyz=patient_ct.shape,
                    stack_size=state["stack_size"],
                    spacing_xyz=patient_ct.spacing,
                    origin_xyz=patient_ct.origin,
                    direction_xyz=patient_ct.image.GetDirection(),
                    queue_depth=MHA_QUEUE_DEPTH,
                    compress_level=COMPRESS_LEVEL,
                    compressor_backend=MHA_COMPRESSOR,
                    compression_workers=worker_count,
                    compression_slab_bytes=MHA_COMPRESSION_SLAB_BYTES,
                ))
                state["generator"] = iter(predict_rois_for_writer(
                    state["model"], state["ordered"], patient_ct
                ))

            resolved_backends = sorted({
                state["writer"].compressor_backend for state in group
            })
            print(
                f"Resolved MHA compressor backend(s): {resolved_backends}; "
                f"workers per stack: {compression_workers}",
                flush=True,
            )

            active = deque(group)
            while active:
                state = active.popleft()
                try:
                    segment, dose_roi, roi_box = next(state["generator"])
                except StopIteration as error:
                    raise RuntimeError(
                        f"output {state['output_index'] + 1}: predictor stopped "
                        f"after {state['written']} of {state['stack_size']} "
                        f"control points"
                    ) from error
                # predict_segment_rois may hand back a view of a reusable
                # pinned buffer, so ordering matters: the writer copies on
                # submit, and nothing may advance this generator before then.
                expected = state["ordered"][state["written"]]
                if segment.name != expected.name:
                    raise RuntimeError(
                        f"output {state['output_index'] + 1}: predictor "
                        f"returned {segment.name}, expected {expected.name}"
                    )
                ensure_minimum_cutoff(
                    state["model"], dose_roi, segment.minimum_cutoff
                )
                log_segment(segment, dose_roi, roi_box=roi_box)
                state["writer"].submit_roi(dose_roi, roi_box)
                state["written"] += 1
                if state["written"] < state["stack_size"]:
                    active.append(state)

        for state in group:
            state.pop("generator", None)
            state.pop("writer", None)
            state.pop("patient_ct", None)

    elapsed = time.perf_counter() - started
    print(
        f"Finished {len(states)} output stacks and {total_items} control "
        f"points in {elapsed:.1f}s ({elapsed / total_items:.3f}s/control point)",
        flush=True,
    )
    return 0


def apply_minimum_cutoff(dose, minimum_cutoff):
    """Apply the evaluator cutoff in the array's current numerical precision."""
    threshold = (
        float(minimum_cutoff)
        if np.asarray(dose).dtype == np.float64
        else cutoff_threshold(minimum_cutoff)
    )
    dose[dose < threshold] = 0.0

def ensure_minimum_cutoff(model, dose, minimum_cutoff):
    """Apply the cutoff unless the predictor already did so on its device."""
    if getattr(model, "applies_minimum_cutoff_on_device", False):
        return
    apply_minimum_cutoff(dose, minimum_cutoff)

def predict_rois_for_writer(model, ordered, patient_ct):
    """Request transferable pinned ROIs when the predictor supports them."""
    kwargs = {"batch_size": BATCH_SIZE}
    if getattr(model, "writer_owned_roi_outputs", False):
        kwargs["writer_owned_outputs"] = True
    return model.predict_segment_rois(ordered, patient_ct, **kwargs)

def load_source_image_path(input_file_idx):
    location = INPUT_PATH / f"images/{INPUT_DIR_BASE}-{input_file_idx + 1}"
    return load_sitk_path(location)

def cutoff_threshold(minimum_cutoff):
    """Smallest float32 threshold whose ``<`` test matches the evaluator's.

    Dose is written as float32; the evaluator casts it back to float64 before
    flagging any voxel with ``0 < v < minimum_cutoff``. numpy 1.x compares a
    float32 array against a Python float in float32, so when a cutoff rounds
    *down* under float32 the single value float32(cutoff) passes the filter
    here and fails there -- one voxel, one implementation error. Stepping the
    threshold up one ULP pulls that value into the zeroed set.

    Cutoffs that round *up* are already conservative; bumping those would
    discard dose the evaluator considers legitimate, hence the branch.
    """
    c32 = np.float32(minimum_cutoff)
    if float(c32) < minimum_cutoff:
        return np.nextafter(c32, np.float32(np.inf))
    return c32


def log_segment(segment, dose, roi_box=None):
    """Log one control point's predicted dose distribution.

    No ground truth is available inside the container, so this reports shape
    and dose statistics only -- enough to spot an all-zero or wildly scaled
    prediction. Gated on DOSERAD_VERBOSE because the reductions are not free.
    """
    if not VERBOSE:
        return

    nonzero = int((dose > 0).sum())
    total = int(dose.size)
    extent = "" if roi_box is None else f" roi={tuple(roi_box)}"
    print(
        f"  {segment.name}: max={float(dose.max()):.4g} "
        f"mean={float(dose.mean()):.4g} "
        f"nonzero={nonzero}/{total} ({100.0 * nonzero / total:.1f}%) "
        f"cutoff={segment.minimum_cutoff}{extent}",
        flush=True,
    )


def load_json_file(location):
    with open(location) as f:
        return json.load(f)


def build_model_index(metadata, models):
    """Map each image_file_idx onto the model key its region calls for.

    Resolved up front, before any prediction runs, so a job carrying a region
    with no matching checkpoint fails immediately rather than part-way through
    writing output stacks.
    """
    index = {}
    for image in metadata:
        image_idx = int(image["image_file_idx"])
        region = image.get("anatomical_region")

        model_key = REGION_TO_MODEL.get(region)
        if model_key is None:
            raise ValueError(
                f"image {image_idx + 1}: unknown anatomical_region {region!r}; "
                f"expected one of {sorted(REGION_TO_MODEL)}"
            )
        if model_key not in models:
            raise ValueError(
                f"image {image_idx + 1}: region {region!r} needs the "
                f"{model_key!r} checkpoint, which was not loaded; available: "
                f"{sorted(models)}"
            )
        index[image_idx] = model_key

    print(
        "Region assignment per image: "
        + ", ".join(f"{i + 1}={index[i]}" for i in sorted(index))
    )
    return index


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
#
# Keyed by (model key, image index), not image index alone: each predictor
# normalizes the CT with its own stats, so a volume prepared by the thorax model
# is not reusable by the abdomen one. Today both stats files share a CT window
# and the arrays would be identical, but the cache should not depend on that.
_CT_CACHE: dict[tuple[str, int], object] = {}


def load_ct_by_index(model, generator, model_key, input_file_idx):
    """Prepare one source image for the dose model, reusing the previous one.

    On the CT tasks that is the source CT, spline-prefiltered. On the MRI tasks
    the source image is an MRI, so ``generator`` synthesises a CT from it first;
    the result is prepared identically from there on, and the sCT is never
    written to disk.
    """
def load_ct_by_index(model, model_key, input_file_idx):
    """Load and spline-prefilter one source CT, reusing the previous one."""
    cache_key = (model_key, input_file_idx)
    cached = _CT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    source_path = load_source_image_path(input_file_idx)
    if is_mri_task():
        if _SCT_MANAGER is None:
            raise RuntimeError("proton-mri sCT manager was not initialized")
        source_path = _SCT_MANAGER.materialize(
            source_path, model_key, input_file_idx
        )
    patient_ct = model.load_ct(source_path)
    print(
        f"Loaded input image {input_file_idx + 1} "
        f"with shape {patient_ct.shape} and spacing {patient_ct.spacing}"
    )
    _CT_CACHE.clear()
    _CT_CACHE[cache_key] = patient_ct
    return patient_ct
