"""Train one MR -> sCT generator (one anatomy, one modality).

    PYTHONPATH=external/DoseRAD2026_sCT/src python -m doserad2026_sct.train_sct \
      --path_to_data <DATA_ROOT>/photon/training \
      --modality photon --anatomy thorax \
      --config_file <configs>/train_sCT.json \
      --path_to_splits <configs> \
      --path_to_logs <OUT_ROOT> \
      --device cuda:0

``--path_to_splits`` is a directory containing ``<anatomy>_split.json``.
``--resume_from`` continues an existing run directory in place; without it a new
timestamped directory is created under ``--path_to_logs`` and the resolved config
and split are written into it.
"""

import argparse
import json
import os
from pathlib import Path

from monai.networks.nets.swin_unetr import SwinUNETR
import torch

from doserad2026_sct.datasets import paired_mr_ct
from doserad2026_sct.trainers import sct
from doserad2026_sct.utils import save, patchgan


def _parse_args() -> dict[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", type=str, required=True)
    parser.add_argument("--modality", type=str, required=True)
    parser.add_argument("--anatomy", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument(
        "--path_to_logs",
        type=str,
        default=os.path.join(Path(__file__).resolve().parent, "runs"),
    )
    parser.add_argument(
        "--path_to_splits",
        type=str,
        default=os.path.join(Path(__file__).resolve().parent, "configs"),
    )
    parser.add_argument("--resume_from", type=str)
    args = parser.parse_args()
    return vars(args)


def load_split(path_to_configs: str, anatomy: str) -> dict[str, dict]:
    with open(os.path.join(path_to_configs, f"{anatomy}_split.json"), "r") as f:
        split = json.load(f)
    return split


def load_config(path_to_config: str) -> dict[str, str | int | float]:
    with open(os.path.abspath(path_to_config), "r") as f:
        config = json.load(f)
    return config


def main() -> None:
    args = _parse_args()
    split = load_split(args["path_to_splits"], args["anatomy"])
    config = load_config(args["config_file"])
    torch.random.manual_seed(config["seed"])
    config["anatomy"] = args["anatomy"]
    config["modality"] = args["modality"]
    train_dataset = paired_mr_ct.PairedMRCTDataset(
        path_to_data=args["path_to_data"],
        patient_ids=split["train"],
        train=True,
        patch_size=config["patch_size"],
        number_of_patches=config["batch_size"],
    )
    validation_dataset = paired_mr_ct.PairedMRCTDataset(
        path_to_data=args["path_to_data"],
        patient_ids=split["test"],
        train=False,
    )
    # Loader workers. At num_workers=0 each item costs ~0.45-0.59 s of read +
    # preprocess on the critical path, against ~2.5 s of GPU work; 4-8 workers
    # take it to ~0.12-0.17 s and hide it entirely behind compute. The dataset is
    # built once in the parent and inherited copy-on-write, so the per-patient
    # body-mask precompute is NOT paid per worker under the default fork start
    # method. If a cluster forces "spawn", set num_workers=0 or expect that cost
    # per worker.
    num_workers = int(config.get("num_workers", 4))
    loader_kwargs = {"batch_size": 1}  # batch size is handled by MONAI's transforms
    if num_workers > 0:
        loader_kwargs.update(
            num_workers=num_workers,
            persistent_workers=True,  # otherwise workers are re-forked every epoch
            prefetch_factor=int(config.get("prefetch_factor", 2)),
            pin_memory=True,
        )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, **loader_kwargs
    )
    # Validation is a handful of sliding-window inferences; loading is not the
    # bottleneck there and extra workers just hold memory.
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=1
    )
    # feature_size and use_v2 are the trained architecture, not tunable here:
    # doserad.sct.SCTGenerator constructs exactly this and loads the state dict
    # strictly, so changing either produces a generator the submission rejects.
    generator = SwinUNETR(
        in_channels=1, out_channels=1, feature_size=48, use_v2=True
    ).to(args["device"])

    discriminator = patchgan.Discriminator(in_channels=1).to(args["device"])

    if args["resume_from"] is not None:
        out_path = args["resume_from"]
        resume_training = True
    else:
        out_path = save.generate_unique_log_directory(
            output_directory=args["path_to_logs"]
        )
        with open(os.path.join(out_path, "config.json"), "w") as f:
            json.dump(config, f)
        with open(os.path.join(out_path, "split.json"), "w") as f:
            json.dump(split, f)
        resume_training = False

    # The trainer is otherwise silent until the first tqdm bar of epoch 0, which
    # is several minutes in, and the run directory is never reported anywhere.
    print(
        f"run dir     : {os.path.abspath(out_path)}\n"
        f"anatomy     : {args['modality']} {args['anatomy']}\n"
        f"split       : {len(split['train'])} train / {len(split['test'])} test"
        f"  held out: {sorted(split['test'])}\n"
        f"device      : {args['device']}\n"
        f"patch/batch : {config['patch_size']} x {config['batch_size']}"
        f"   num_workers {num_workers}\n"
        f"lambda      : recon {config['lambda_generator_discriminator']}\n"
        f"losses      : image {config['generator_image_criterion']}"
        f" | disc {config['discriminator_criterion']} (LSGAN)\n"
        f"resuming    : {bool(args['resume_from'])}",
        flush=True,
    )

    trainer = sct.sCTTrainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        device=args["device"],
        out_path=out_path,
        resume_training=resume_training,
    )
    trainer.train()


if __name__ == "__main__":
    main()
