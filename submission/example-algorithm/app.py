"""
The following is an example algorithm inference server.

Any implementation will do as long as it:

1. Starts the inference server and loads the algorithm
2. On the health endpoint indicates if the server is healthy (i.e. returns HTTP 200 OK)
3. On the invoke endpoint invokes the algorithm for inference and returns HTTP 201 CREATED

"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response, status
import torch
import uvicorn
import os

# This server only ever serves the real model. The upstream dummy-dose module
# (a Gaussian blob per control point, no weights) and the ALGORITHM switch that
# selected it are not carried into this repository, so there is no configuration
# under which a build can ship anything but real predictions.
import inference


from uvicorn.config import LOGGING_CONFIG


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


# One checkpoint per anatomical region, as (weights, stats) under /opt/ml/model.
# Each checkpoint is paired with the stats it was trained against: the stats
# files differ in dose_scale, which rescales every predicted voxel.
#
# This submission serves one checkpoint for both regions -- the 1.5x-capacity,
# upscale-2 thorax model. It is therefore paired with the *thorax* stats even
# for abdominal patients: dose_scale belongs to the training run, not to the
# patient, so abdominal stats here would rescale the output by the wrong
# constant. Regions sharing an entry share a single predictor instance.
#
# This repository ships one architecture: the packed CNN-Mamba3 checkpoint that
# the submitted photon-ct and photon-mri images serve. The xLSTM and unpacked
# Mamba families the upstream server could also serve are not included, so the
# kind is fixed rather than selected.
MODEL_KIND = "mamba-packed"

# The bidirectional checkpoint has extra backward stacks and its own arch JSON;
# it is the only dose model in this repository. The build scripts still take
# MAMBA_VARIANT=bidirectional, but only to name the image -- the server does not
# select between architectures.

# The bidirectional checkpoint was trained on real CTs and has no sCT-tuned
# counterpart -- photon-mri feeds it synthetic CTs from the sCT stage, using
# these same weights. The upstream DOSERAD_DOSE_VARIANT switch therefore has
# only one reachable value here and is not carried over.
# One entry per modality. Both are thorax-trained runs serving both anatomies,
# so each is paired with its own run's stats for every region -- dose_scale
# belongs to the training run, not to the patient.
#
# Mapping both regions onto one checkpoint (rather than collapsing the region
# keys) keeps each image's anatomy visible to inference.py, which is what the
# MRI tasks use to pick that anatomy's sCT generator.
PHOTON_CHECKPOINT = (
    "best_model_thorax_mamba3_2x_upscale2_bidi.pth",
    "dl_segment_stats_thorax.json",
)
PROTON_CHECKPOINT = (
    "best_model_proton_thorax_mamba3_energy_bragg.pth",
    "dl_segment_stats_proton_thorax.json",
)
SHARED_CHECKPOINT = (
    PROTON_CHECKPOINT if inference.is_proton_task() else PHOTON_CHECKPOINT
)
REGION_CHECKPOINTS = {
    "thorax": SHARED_CHECKPOINT,
    "abdomen": SHARED_CHECKPOINT,
}

# The MR-to-CT generators, one per region -- these are genuinely per-region
# (each was trained on its own anatomy's split), unlike the dose checkpoints
# above. Only loaded for the MRI tasks; inference.NEEDS_SCT decides.
#
# Both entries read their denormalisation constants out of the same
# ct_statistics.json under their own anatomy key. Crossing them would shift
# every voxel of the sCT by ~90 HU, so the anatomy is named rather than
# defaulted.
SCT_MODALITY = "proton" if inference.is_proton_task() else "photon"
REGION_SCT_GENERATORS = {
    "thorax": f"sct_{SCT_MODALITY}_thorax.pth",
    "abdomen": f"sct_{SCT_MODALITY}_abdomen.pth",
}
# Sliding windows per SwinUNETR forward. Output is bit-identical to 1 -- MONAI
# batches and blends the same windows -- so this only trades VRAM for GPU
# utilisation. Measured peak reserved on a 152x520x547 volume: ~4.9 GiB at 1,
# ~11.9 GiB at 4 (bf16), against the A10G's 22.06 GiB shared with the dose model.
SCT_SW_BATCH_SIZE = int(os.environ.get("DOSERAD_SCT_SW_BATCH_SIZE", "1"))
if SCT_SW_BATCH_SIZE < 1:
    raise ValueError(
        f"DOSERAD_SCT_SW_BATCH_SIZE must be >= 1, got {SCT_SW_BATCH_SIZE}"
    )
# fp32 is the default; bf16 is what DoseRAD2026_sCT's own validation epoch used
# and is roughly 2x faster. Denormalisation stays fp32 either way.
SCT_PRECISION = os.environ.get("DOSERAD_SCT_PRECISION", "fp32").strip().lower()
if SCT_PRECISION not in {"fp32", "bf16"}:
    raise ValueError(
        f"Unknown DOSERAD_SCT_PRECISION {SCT_PRECISION!r}; expected fp32 or bf16"
    )
SCT_POPULATION_STATS = os.environ.get(
    "DOSERAD_SCT_STATS", "sct_ct_population_statistics.json"
)
# The CNN-Mamba run's config_hyper.json, shipped beside the weights so the
# architecture travels with the checkpoint instead of being restated here.
# An explicit environment value remains available for local experiments; a
# production build normally chooses through DOSERAD_MAMBA_VARIANT alone.
MAMBA_ARCH_CONFIG = os.environ.get(
    "DOSERAD_MAMBA_ARCH",
    "proton_mamba3_energy_bragg_arch.json"
    if inference.is_proton_task()
    else "mamba3_2x_upscale2_bidi_arch.json",
)
# Must match the grid the checkpoints were trained on -- both config_hyper.json
# files for the shipped x384 checkpoints record bev_grid_384_x_zalign.json,
# shape_dhw [384, 200, 200]. The depth axis is the xLSTM's sequence dimension,
# so running a different depth is a train/test mismatch, not just a narrower
# field of view. The Dockerfile pins the same value (Grand Challenge cannot set
# environment variables); this default only matters when running app.py
# directly. Overridable for A/B testing against the earlier 256-deep
# checkpoints with bev_grid_default_zalign.json.
BEV_GRID_CONFIG = os.environ.get(
    "DOSERAD_BEV_GRID",
    "bev_grid_proton_zalign.json"
    if inference.is_proton_task()
    else "bev_grid_384_x_zalign.json",
)
# Opt-in fine CT/aperture sampling followed by inverse pixel shuffle. Earlier
# submitted checkpoints use 1; the shipped checkpoint was trained with
# --otf-input-upscale-factor 2, which packs 2x2 phases into the encoder's input
# channels, so a mismatch fails at load (encoder.0.weight is 8 channels, not 2).
INPUT_UPSCALE_FACTOR = int(os.environ.get("DOSERAD_INPUT_UPSCALE_FACTOR", "2"))
# --model-capacity-scale from training. Widens every conv stage (16/32/64 ->
# 24/48/96 at 1.5), so it must match the checkpoint exactly or the state dict
# will not load. The shipped checkpoint was trained at 1.5.
# Compile the fixed-shape Triton/CuPy kernels during startup rather than inside
# the first /invoke. init_model() runs in the lifespan handler, before uvicorn
# serves anything, so /health cannot return 200 until this finishes.
STARTUP_WARMUP = os.environ.get(
    "DOSERAD_STARTUP_WARMUP", "1"
).strip().lower() in {"1", "true", "yes"}


def init_model():
    # Loading happens once at server startup: each /invoke has a timeout, and
    # building the model plus JIT-compiling the Triton kernels is far too slow
    # to do inside a request.
    _show_torch_cuda_info()

    if inference.is_proton_task():
        from doserad.proton_mamba_predict import (
            ProtonMambaDosePredictor as DosePredictor,
        )
    else:
        from doserad.mamba_predict import PackedMambaDosePredictor as DosePredictor

    # The model tarball is extracted to `model_dir` at runtime on Grand
    # Challenge. When testing locally, the local `./model` directory is mounted
    # here. Upload it under Algorithm -> Models.
    model_dir = Path("/opt/ml/model")

    # Every checkpoint stays resident: they are a couple of MB each, and a single
    # job can mix thoracic and abdominal patients, so loading on demand would
    # mean rebuilding a predictor mid-invoke. inference.py maps each image's
    # anatomical_region onto these keys.
    #
    # Regions pointing at the same (weights, stats) pair get the same predictor
    # rather than two identical ones: building a second would duplicate the
    # weights, the BEV index grid and the Triton kernel warm-up for nothing.
    predictors = {}
    by_checkpoint = {}
    for region, checkpoint in REGION_CHECKPOINTS.items():
        predictor = by_checkpoint.get(checkpoint)
        if predictor is None:
            weights, stats = checkpoint
            kwargs = dict(
                weights=model_dir / weights,
                stats=model_dir / stats,
                bev_grid_config=model_dir / BEV_GRID_CONFIG,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )
            # Capacity and phase channels come from mamba_arch; the upscale
            # factor also sizes the input grid outside the model, so it is
            # passed too and the predictor checks the two against each other.
            # This run trained --amp --amp-dtype bfloat16 and is served so.
            kwargs.update(
                arch=model_dir / MAMBA_ARCH_CONFIG,
                input_upscale_factor=INPUT_UPSCALE_FACTOR,
            )
            if inference.is_proton_task():
                # Spot widths and the energy-token table both come from here.
                kwargs["beam_parameters"] = model_dir / "beam_parameters.json"
            predictor = DosePredictor(**kwargs)
            # Fails loudly on a mis-staged checkpoint. Each modality ships one
            # architecture: photon the bidirectional run, proton the
            # energy-token + Bragg-residual run.
            if inference.is_proton_task():
                if int(getattr(predictor._arch, "energy_token_levels", 0)) <= 0:
                    raise RuntimeError(
                        f"{MAMBA_ARCH_CONFIG} has energy_token_levels=0; the "
                        "staged checkpoint is not the energy-token proton model"
                    )
            elif not bool(getattr(predictor._arch, "bidirectional_mamba", False)):
                raise RuntimeError(
                    f"{MAMBA_ARCH_CONFIG} has bidirectional_mamba=False; "
                    "the staged checkpoint is not the bidirectional model"
                )
            by_checkpoint[checkpoint] = predictor
            print(
                f"Loaded CNN-{MODEL_KIND} dose predictor from {model_dir / weights} "
                f"(stats={stats}, grid={BEV_GRID_CONFIG}, "
                f"arch={MAMBA_ARCH_CONFIG}, upscale={INPUT_UPSCALE_FACTOR})"
            )
        predictors[region] = predictor
        print(f"Region {region!r} serves from {checkpoint[0]}")

    # The MR-to-CT stage, only for the MRI tasks. inference.py owns the
    # task-to-interface mapping, so it decides whether generators are needed
    # rather than this being restated here.
    sct = None
    if inference.NEEDS_SCT:
        from doserad.sct import SwinSCTManager

        sct = SwinSCTManager(
            model_dir,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            generators=REGION_SCT_GENERATORS,
            population_stats=SCT_POPULATION_STATS,
            modality=SCT_MODALITY,
            precision=SCT_PRECISION,
            sw_batch_size=SCT_SW_BATCH_SIZE,
        )
        print(
            f"Loaded the {SCT_MODALITY} SwinUNETR sCT generators "
            f"({SCT_PRECISION}, sw_batch_size={SCT_SW_BATCH_SIZE}) from "
            f"{model_dir} (population stats: {SCT_POPULATION_STATS})"
        )

    if STARTUP_WARMUP:
        # Distinct predictor objects only -- regions sharing a checkpoint share
        # the object, and warming it twice would compile nothing new.
        for index, predictor in enumerate(by_checkpoint.values()):
            elapsed = predictor.warmup(include_preprocessing=index == 0)
            print(
                f"Warmed predictor {index + 1}/{len(by_checkpoint)} before "
                f"readiness in {elapsed:.2f}s",
                flush=True,
            )
        if sct is not None:
            sct.warmup()

    return {"dose": predictors, "sct": sct}


MODELS = {}


# During the lifespan of your inference server, your model should be ready
# for invocations.It is important to load your model here, and not just
# before running inference, to allow the inference time to be as short as
# possible. Each invocation will have a timeout, so if your model still
# needs to be loaded when the /invoke endpoint is called, there may not be
# enough time for processing.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    MODELS["answer_to_everything"] = init_model()
    yield
    # Clean up the models and release the resources
    MODELS.clear()


app = FastAPI(lifespan=lifespan)


# After starting your inference server, the health endpoint will
# be called repeatedly until it returns a 200 response.
# Redirect responses will not be followed and will raise an exception.
# Any other response will be ignored.
@app.get("/health")
async def health():
    try:
        # check if the model is initialized
        _ = MODELS["answer_to_everything"]
        return Response(status_code=status.HTTP_200_OK)
    except KeyError:
        return Response(status_code=status.HTTP_404_NOT_FOUND)


# After the health endpoint returns a 200 response,
# the invoke endpoint will be called (one or more times)
# to invoke inference on the inputs in the input folder.
# When inference is done, this endpoint should return a 201 response.
# Any other response will raise an exception and fail.
@app.post("/invoke")
async def invoke():
    # First print a tree of /input
    for root, dirs, files in os.walk("/input"):
        level = root.count(os.sep)
        indent = "    " * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = "    " * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

    model = MODELS["answer_to_everything"]
    inference.run(model)
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    log_config = LOGGING_CONFIG.copy()
    log_config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    uvicorn.run(app, host="0.0.0.0", port=4743, log_config=log_config)
