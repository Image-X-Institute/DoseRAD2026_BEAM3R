#!/usr/bin/env python3
"""Stage the packed bidirectional CNN-Mamba3 deployment files.

The training checkpoint and its full ``config_hyper.json`` remain in the run
directory.  This script copies the checkpoint into the separately uploaded
Grand Challenge model bundle and writes a deployment config which:

* retains the checkpoint's bidirectional architecture;
* removes inactive training-only fields unknown to the inference model; and
* selects the validated inference chunk size (16 by default).

The resulting files are selected at image build time with::

    MAMBA_VARIANT=bidirectional ./do_build.sh
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "example-algorithm" / "model"
# No default: the training run lives wherever --out-dir put it, which differs
# per machine. Passing --run-dir explicitly is required rather than inheriting
# one developer's layout.
DEFAULT_RUN_DIR = None
WEIGHTS_NAME = "best_model_thorax_mamba3_2x_upscale2_bidi.pth"
ARCH_NAME = "mamba3_2x_upscale2_bidi_arch.json"

# These fields were recorded by a newer training model but were disabled in
# this run.  The compact submission model intentionally omits those optional
# modules, so carrying their inactive settings into MambaDoseArchConfig would
# make startup reject an otherwise compatible checkpoint.
INACTIVE_TRAINING_FIELDS = {
    "depth_film",
    "film_depth_spacing_mm",
    "film_hidden",
    "film_n_freq",
    "film_sad_mm",
    "unet_skips",
}


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise SystemExit(f"{path}: checkpoint is not a state dict")
    return state


def _deployment_config(path: Path, chunk_size: int) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if "mamba_arch" not in config:
        raise SystemExit(f"{path}: no mamba_arch block")
    arch = dict(config["mamba_arch"])

    expected = {
        "bidirectional_mamba": True,
        "return_packed_dh_coefficients": True,
        "input_phase_channels": 4,
        "model_capacity_scale": 2.0,
        "mamba_core": "mamba3",
    }
    wrong = {key: arch.get(key) for key, value in expected.items()
             if arch.get(key) != value}
    if wrong:
        raise SystemExit(f"bidirectional run has incompatible architecture: {wrong}")
    if arch.get("depth_film") or arch.get("unet_skips"):
        raise SystemExit(
            "the bidirectional run enables an inference module absent from the "
            "submission model (depth_film or unet_skips)"
        )

    for key in INACTIVE_TRAINING_FIELDS:
        arch.pop(key, None)
    arch["mamba3_chunk_size"] = int(chunk_size)
    # Ship only what inference reads. The rest of config_hyper.json records the
    # argv, out_dir, stats and split paths of the machine that trained the run:
    # unused here (both predictors read nothing but ``mamba_arch``) and it would
    # publish local filesystem layout in a public repo.
    return {
        "mamba_arch": arch,
        # Keep the resolved top-level echo consistent for provenance readers.
        "mamba3_chunk_size": int(chunk_size),
        "deployment_mamba3_chunk_size": int(chunk_size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Training run directory: contains model/ and script/config_hyper.json")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--weights", type=str, default="best_model.pth")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    source_weights = args.run_dir / "model" / args.weights
    source_config = args.run_dir / "script" / "config_hyper.json"
    for path in (source_weights, source_config):
        if not path.is_file():
            raise SystemExit(f"missing source: {path}")

    state = _state_dict(source_weights)
    if not any(key.startswith("blocks_bwd.") for key in state):
        raise SystemExit(f"{source_weights}: no blocks_bwd parameters")
    if not any(key.startswith("blocks2_bwd.") for key in state):
        raise SystemExit(f"{source_weights}: no blocks2_bwd parameters")
    config = _deployment_config(source_config, args.chunk_size)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target_weights = MODEL_DIR / WEIGHTS_NAME
    target_config = MODEL_DIR / ARCH_NAME
    if args.force or not target_weights.exists():
        shutil.copyfile(source_weights, target_weights)
        print(f"{WEIGHTS_NAME} <- {source_weights}")
    else:
        print(f"skip {WEIGHTS_NAME} (already staged)")
    if args.force or not target_config.exists():
        target_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"{ARCH_NAME} <- {source_config} (chunk={args.chunk_size})")
    else:
        print(f"skip {ARCH_NAME} (already staged)")

    print(
        f"Staged bidirectional Mamba3: {len(state):,} tensors, "
        f"chunk={args.chunk_size}. Do not commit the raw checkpoint without LFS."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
