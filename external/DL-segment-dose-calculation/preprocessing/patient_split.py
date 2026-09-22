"""Stratified patient-level train/val splits for DoseRAD (1ABB / 1THB)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Literal

_SAMPLE_KEY_RE = re.compile(r"^(.+)_(\d+)_CP(\d{3})$")

SplitRole = Literal["train", "val", "test", "all", "unused"]


def patient_cohort(patient_id: str) -> str:
    if patient_id.startswith("1ABB"):
        return "abdomen"
    if patient_id.startswith("1THB"):
        return "thorax"
    return "other"


def list_doserad_patients(
    baseline_pb_dir: str | Path,
    split: str,
    *,
    modality: str = "photon",
) -> list[str]:
    """Patients with a plan JSON under ``{baseline}/{modality}/{split}/``."""
    root = Path(baseline_pb_dir) / modality / split
    if not root.is_dir():
        raise FileNotFoundError(f"Split directory not found: {root}")
    patients: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        plan = entry / f"{entry.name}.json"
        if plan.is_file():
            patients.append(entry.name)
    return patients


def _take_from_shuffled(
    rng: random.Random,
    pool: list[str],
    n_train: int,
    n_val: int,
) -> tuple[list[str], list[str], list[str]]:
    ids = list(pool)
    rng.shuffle(ids)
    need = n_train + n_val
    if len(ids) < need:
        raise ValueError(
            f"Not enough patients in cohort (have {len(ids)}, need {need}: "
            f"{n_train} train + {n_val} val)"
        )
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    unused = ids[n_train + n_val :]
    return train, val, unused


def stratified_fixed_split(
    patient_ids: list[str],
    *,
    seed: int = 333,
    n_train_abdomen: int = 5,
    n_train_thorax: int = 5,
    n_val_abdomen: int = 2,
    n_val_thorax: int = 2,
) -> dict:
    """Shuffle within 1ABB / 1THB and assign fixed train/val counts per cohort."""
    rng = random.Random(seed)
    abb = sorted(p for p in patient_ids if p.startswith("1ABB"))
    thb = sorted(p for p in patient_ids if p.startswith("1THB"))
    other = sorted(
        p for p in patient_ids
        if not p.startswith("1ABB") and not p.startswith("1THB")
    )

    tr_a, va_a, un_a = _take_from_shuffled(rng, abb, n_train_abdomen, n_val_abdomen)
    tr_t, va_t, un_t = _take_from_shuffled(rng, thb, n_train_thorax, n_val_thorax)

    train = sorted(tr_a + tr_t)
    val = sorted(va_a + va_t)
    unused = sorted(un_a + un_t + other)

    return {
        "seed": seed,
        "counts": {
            "train": {"abdomen": n_train_abdomen, "thorax": n_train_thorax},
            "val": {"abdomen": n_val_abdomen, "thorax": n_val_thorax},
        },
        "train": train,
        "val": val,
        "unused": unused,
        "by_cohort": {
            "abdomen": {"train": tr_a, "val": va_a, "unused": un_a},
            "thorax": {"train": tr_t, "val": va_t, "unused": un_t},
            "other": {"train": [], "val": [], "unused": other},
        },
    }


def patients_for_role(split: dict, role: SplitRole) -> list[str]:
    if role == "train":
        return list(split["train"])
    if role == "val":
        return list(split["val"])
    if role == "test":
        return list(split.get("test", []))
    if role == "unused":
        return list(split.get("unused", []))
    if role == "all":
        return sorted(set(split["train"]) | set(split["val"]) | set(split.get("test", [])))
    raise ValueError(f"Unknown split role: {role!r}")


def save_patient_split(path: str | Path, split: dict, *, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(split)
    if meta:
        payload["meta"] = meta
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_patient_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def patient_id_from_sample_key(sample_id: str) -> str:
    m = _SAMPLE_KEY_RE.match(sample_id)
    if m:
        return m.group(1)
    return sample_id.split("_", 1)[0]


def h5_group_for_sample(split: dict, sample_id: str) -> str | None:
    """Return HDF5 group name ``train`` / ``validation`` / ``test``, or ``None`` to skip."""
    patient_id = patient_id_from_sample_key(sample_id)
    if patient_id in split["train"]:
        return "train"
    if patient_id in split["val"]:
        return "validation"
    if patient_id in split.get("test", []):
        return "test"
    return None
