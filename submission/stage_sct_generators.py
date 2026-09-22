#!/usr/bin/env python3
"""Stage the MR->sCT generators into the model bundle for the photon-mri task.

A training snapshot is ~900 MB because it carries the discriminator and both
optimizer states for resuming. Only the generator is served, so this extracts
``GENERATOR_MODEL_STATE`` into a ~300 MB flat state dict -- the shape
``doserad.sct.SCTGenerator`` loads with ``weights_only=True`` and ``strict=True``.

    ./stage_sct_generators.py \
        --thorax-snapshot  /path/to/<thorax run>/snapshot.pt \
        --abdomen-snapshot /path/to/<abdomen run>/snapshot.pt \
        --force

Both are required: ``app.py``'s ``REGION_SCT_GENERATORS`` maps each region to its
own generator, and crossing them shifts every voxel by ~90 HU because the two
denormalise with different population statistics.

Like ``stage_bidirectional_mamba.py``, ``--force`` matters: without it an
already-staged file is *skipped* with a printed notice, so a stale generator
would be packaged with no error.

The generators are checked against the architecture the submission constructs
(SwinUNETR feature_size=48, use_v2=True -> 167 tensors) rather than trusted by
filename, so a mis-pointed run fails here instead of at container startup.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "example-algorithm" / "model"
REPO_ROOT = SCRIPT_DIR.parent
# The population mean/std the generators denormalise with. Byte-identical to the
# training package's configs/ct_statistics.json -- the same relationship
# dl_segment_stats_thorax.json has to the dose model's --stats.
STATS_SOURCE = (
    REPO_ROOT
    / "external"
    / "DoseRAD2026_sCT"
    / "src"
    / "doserad2026_sct"
    / "configs"
    / "ct_statistics.json"
)
STATS_NAME = "sct_ct_population_statistics.json"

# SwinUNETR(in_channels=1, out_channels=1, feature_size=48, use_v2=True)
EXPECTED_TENSORS = 167


def extract_generator(snapshot_path: Path, target: Path) -> tuple[int, object]:
    """Pull GENERATOR_MODEL_STATE out of a training snapshot, or pass one through.

    Accepts either a full resumable ``snapshot.pt`` or a generator-only
    ``snapshot_epoch_XXXX.pt`` -- the trainer writes both in the same shape.
    """
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    if not isinstance(snapshot, dict):
        raise SystemExit(f"{snapshot_path}: checkpoint is not a state dict")
    if "GENERATOR_MODEL_STATE" in snapshot:
        epoch = snapshot.get("EPOCH", "?")
        state = snapshot["GENERATOR_MODEL_STATE"]
    else:
        # An already-extracted flat generator state.
        epoch = "?"
        state = snapshot
    if len(state) != EXPECTED_TENSORS:
        raise SystemExit(
            f"{snapshot_path}: {len(state)} generator tensors, expected "
            f"{EXPECTED_TENSORS} for SwinUNETR(feature_size=48, use_v2=True). "
            f"This is not the architecture the submission constructs."
        )
    state = {k: v.cpu() for k, v in state.items()}
    torch.save(state, target)
    return sum(v.numel() for v in state.values()), epoch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thorax-snapshot", type=Path, required=True,
                        help="snapshot.pt from the thorax sCT run")
    parser.add_argument("--abdomen-snapshot", type=Path, required=True,
                        help="snapshot.pt from the abdomen sCT run")
    parser.add_argument("--force", action="store_true",
                        help="overwrite files already staged")
    args = parser.parse_args()

    generators = [
        (args.thorax_snapshot, "sct_photon_thorax.pth", "thorax"),
        (args.abdomen_snapshot, "sct_photon_abdomen.pth", "abdomen"),
    ]
    for source, _, _ in generators:
        if not source.is_file():
            raise SystemExit(f"missing source: {source}")
    if not STATS_SOURCE.is_file():
        raise SystemExit(f"missing source: {STATS_SOURCE}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for source, name, region in generators:
        target = MODEL_DIR / name
        if target.exists() and not args.force:
            print(f"skip {name} (already staged)")
            continue
        parameters, epoch = extract_generator(source, target)
        print(f"{name} <- {source}")
        print(f"  {region}: epoch {epoch}, {parameters / 1e6:.1f}M parameters")

    target_stats = MODEL_DIR / STATS_NAME
    if target_stats.exists() and not args.force:
        print(f"skip {STATS_NAME} (already staged)")
    else:
        shutil.copyfile(STATS_SOURCE, target_stats)
        print(f"{STATS_NAME} <- {STATS_SOURCE}")

    print("\nStaged for photon-mri:")
    for name in ("sct_photon_thorax.pth", "sct_photon_abdomen.pth", STATS_NAME):
        path = MODEL_DIR / name
        print(f"  {name:38s} {path.stat().st_size / 1e6:9.1f} MB")
    print("\nDo not commit these; *.pth is gitignored for this reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
