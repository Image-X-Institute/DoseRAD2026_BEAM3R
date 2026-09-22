"""Optional patient-plan CT validation for OTF dose training."""

from __future__ import annotations

import argparse
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


PLAN_GAMMA_SPECS: tuple[tuple[str, float, float], ...] = (
    ("plan_gamma_2pct_3mm", 2.0, 3.0),
    ("plan_gamma_2pct_0mm", 2.0, 0.0),
    ("plan_gamma_1pct_1mm", 1.0, 1.0),
    ("plan_gamma_1pct_0mm", 1.0, 0.0),
)


def add_plan_validation_args(parser: argparse.ArgumentParser) -> None:
    """Add opt-in plan validation options without changing legacy defaults."""
    parser.add_argument(
        "--ct-roundtrip-idd-freq",
        type=int,
        default=0,
        help=(
            "Log raw and positive-clipped CT-to-BEV round-trip IDD every N "
            "epochs. 0 disables round-trip IDD (default: 0)."
        ),
    )
    parser.add_argument(
        "--plan-validation-metrics-freq",
        type=int,
        default=0,
        help=(
            "Accumulate patient-level CT plans and log 10-30%%, 30-80%%, and >=80%% "
            "Dmax dose-band metrics every N epochs. 0 disables plan validation "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--plan-validation-gamma-freq",
        type=int,
        default=0,
        help=(
            "Also calculate exact local 2%%/3mm, 2%%/0mm, 1%%/1mm, and 1%%/0mm "
            "plan gamma above 10%% of reference Dmax every N epochs. Gamma implies "
            "plan accumulation for that epoch. 0 disables gamma (default: 0)."
        ),
    )
    parser.add_argument(
        "--plan-validation-gamma-max-patients",
        type=int,
        default=0,
        help=(
            "Maximum sorted validation patients used for gamma on a gamma epoch. "
            "0 evaluates all accumulated patients (default: 0)."
        ),
    )
    parser.add_argument(
        "--plan-validation-gamma-random-subset",
        type=int,
        default=None,
        help=(
            "Optional pymedphys random reference-voxel subset per patient and gamma "
            "criterion. Omit for exact gamma."
        ),
    )
    parser.add_argument(
        "--plan-validation-gamma-interp-fraction",
        type=int,
        default=10,
        help="pymedphys gamma interpolation fraction (default: 10).",
    )


def validate_plan_validation_args(args: Any) -> None:
    """Validate plan metric options while leaving their default path disabled."""
    frequencies = (
        int(getattr(args, "ct_roundtrip_idd_freq", 0)),
        int(getattr(args, "plan_validation_metrics_freq", 0)),
        int(getattr(args, "plan_validation_gamma_freq", 0)),
    )
    if any(value < 0 for value in frequencies):
        raise ValueError("validation metric frequencies must be non-negative")
    if int(getattr(args, "plan_validation_gamma_max_patients", 0)) < 0:
        raise ValueError("--plan-validation-gamma-max-patients must be non-negative")
    random_subset = getattr(args, "plan_validation_gamma_random_subset", None)
    if random_subset is not None and int(random_subset) <= 0:
        raise ValueError("--plan-validation-gamma-random-subset must be positive")
    if int(getattr(args, "plan_validation_gamma_interp_fraction", 10)) <= 0:
        raise ValueError("--plan-validation-gamma-interp-fraction must be positive")
    if any(frequencies):
        if not bool(getattr(args, "tensorboard", False)):
            raise ValueError("plan validation metrics require --tensorboard")
        if not bool(getattr(args, "ct_space_loss", False)):
            raise ValueError("plan validation metrics require --ct-space-loss")
        if getattr(args, "data_format", None) != "otf_gpu":
            raise ValueError("plan validation metrics require --data-format otf_gpu")
        if not bool(getattr(args, "otf_gpu_full", False)):
            raise ValueError("plan validation metrics require --otf-gpu-full")


def plan_metrics_due(epoch: int, frequency: int) -> bool:
    return int(frequency) > 0 and int(epoch) % int(frequency) == 0


def save_plan_validation_rows(
    experiment_dir: str | Path,
    epoch: int,
    rows: list[dict[str, Any]],
    *,
    gamma_computed: bool,
) -> Path:
    """Persist per-patient metrics while TensorBoard receives macro means."""
    output_dir = Path(experiment_dir) / "plan_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"epoch_{int(epoch):04d}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": 1,
                "epoch": int(epoch),
                "prescription_source": "summed_reference_dmax",
                "gamma_computed": bool(gamma_computed),
                "patients": rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return path


@functools.lru_cache(maxsize=4)
def build_plan_gantry_lookup(
    baseline_pb_dir: str | Path, split: str = "training"
) -> dict[tuple[str, int, int], float]:
    """Map ``(patient_id, beam_idx, cp_idx)`` to gantry angle from the plan JSONs.

    Deliberately a local reader rather than ``eval_catalog.inference``: that lives
    outside this subtree and is not importable from a training job, and it rounds
    the angle to whole degrees. Rounding happens to be lossless for the current
    plans (all angles are integers at a 2 deg step) but a 1 deg error would move
    the resampled IDD plane by several millimetres, so full precision is kept.

    Cached because the accumulator is rebuilt every validation epoch and this
    walks every patient directory.
    """
    root = Path(baseline_pb_dir) / "photon" / split
    if not root.is_dir():
        raise FileNotFoundError(
            f"no plan directory at {root}; --val-beam-idd needs "
            "--baseline-pb-dir pointing at the tree holding photon/<split>/"
        )
    lookup: dict[tuple[str, int, int], float] = {}
    for plan_path in sorted(root.glob("*/*.json")):
        if plan_path.stem != plan_path.parent.name:
            continue                    # provenance.json and friends
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        patient_id = plan_path.parent.name
        for beam in plan.get("beams", ()):
            beam_idx = int(beam["beam_idx"])
            for cp in beam.get("control_points", ()):
                lookup[(patient_id, beam_idx, int(cp["cp_idx"]))] = float(
                    cp["gantry_angle"]
                )
    if not lookup:
        raise ValueError(f"no gantry angles found under {root}")
    return lookup


def _zero_distance_gamma(
    prediction: np.ndarray,
    target: np.ndarray,
    prescription: float,
    *,
    dose_percent: float,
) -> float:
    """Local-dose gamma with zero distance tolerance above 10% prescription."""
    mask = target >= 0.10 * prescription
    if prescription <= 0 or not np.any(mask):
        return float("nan")
    tolerance = (dose_percent / 100.0) * np.abs(target[mask])
    return float(np.mean(np.abs(prediction[mask] - target[mask]) <= tolerance))


def _dose_band_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    prescription: float,
    lower: float,
    upper: float | None,
) -> tuple[float, int]:
    if prescription <= 0:
        return float("nan"), 0
    mask = target >= lower * prescription
    if upper is not None:
        mask &= target < upper * prescription
    count = int(mask.sum())
    if count == 0:
        return float("nan"), 0
    value = float(np.mean(np.abs(prediction[mask] - target[mask])) / prescription)
    return value, count


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


@dataclass
class _PlanEntry:
    prediction_xyz: torch.Tensor
    target_xyz: torch.Tensor
    spacing_xyz_mm: tuple[float, float, float]
    control_points: int = 0


class PatientPlanCtAccumulator:
    """Accumulate detached control-point CT ROIs into one plan per patient.

    Per-control-point metrics do not belong here: see
    :class:`beam_idd_validation.CtBeamIddAccumulator`, which scores the same
    ROIs on the ``--val-metrics-freq`` pass.
    """

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog
        self._plans: dict[str, _PlanEntry] = {}

    def update(
        self,
        sample: dict[str, Any],
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        sample_id = str(sample["sample_id"])
        patient_id = str(self.catalog.records[sample_id].patient_id)
        shape = tuple(int(value) for value in sample["ct_shape"])
        spacing = tuple(float(value) for value in sample["spacing_xyz_mm"])
        entry = self._plans.get(patient_id)
        if entry is None:
            entry = _PlanEntry(
                prediction_xyz=torch.zeros(shape, dtype=torch.float32),
                target_xyz=torch.zeros(shape, dtype=torch.float32),
                spacing_xyz_mm=spacing,
            )
            self._plans[patient_id] = entry
        elif (
            tuple(entry.prediction_xyz.shape) != shape
            or entry.spacing_xyz_mm != spacing
        ):
            raise ValueError(
                f"inconsistent CT geometry for patient {patient_id}: "
                f"{tuple(entry.prediction_xyz.shape)}/{entry.spacing_xyz_mm} "
                f"versus {shape}/{spacing}"
            )

        xs, xe, ys, ye, zs, ze = (int(value) for value in sample["roi_box"])
        valid_prediction = prediction[0].detach().float().masked_fill(
            ~valid[0], 0.0
        )
        valid_target = target[0].detach().float().masked_fill(~valid[0], 0.0)

        entry.prediction_xyz[xs:xe, ys:ye, zs:ze].add_(
            valid_prediction.cpu()
        )
        entry.target_xyz[xs:xe, ys:ye, zs:ze].add_(valid_target.cpu())
        entry.control_points += 1

    def summaries(
        self,
        *,
        compute_gamma: bool,
        gamma_max_patients: int = 0,
        gamma_random_subset: int | None = None,
        gamma_interp_fraction: int = 10,
        gamma_function: Callable[..., float] | None = None,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Return patient-macro means and detailed per-patient rows."""
        if gamma_function is None and compute_gamma:
            from validation import ct_gamma_pass_rate

            gamma_function = ct_gamma_pass_rate

        gamma_patients = sorted(self._plans)
        if gamma_max_patients > 0:
            gamma_patients = gamma_patients[:gamma_max_patients]
        gamma_patient_set = set(gamma_patients)
        rows: list[dict[str, Any]] = []

        for patient_id, entry in sorted(self._plans.items()):
            # The materializer uses XYZ; validation gamma expects ZYX.
            prediction = entry.prediction_xyz.numpy().transpose(2, 1, 0)
            target = entry.target_xyz.numpy().transpose(2, 1, 0)
            sx, sy, sz = entry.spacing_xyz_mm
            spacing_zyx = (sz, sy, sx)
            prescription = float(target.max())

            low_mae, low_count = _dose_band_mae(
                prediction, target, prescription, 0.10, 0.30
            )
            mid_mae, mid_count = _dose_band_mae(
                prediction, target, prescription, 0.30, 0.80
            )
            high_mae, high_count = _dose_band_mae(
                prediction, target, prescription, 0.80, None
            )
            evaluation_mask = target >= 0.10 * prescription
            if prescription > 0 and np.any(evaluation_mask):
                normalized_error = (
                    prediction[evaluation_mask] - target[evaluation_mask]
                ) / prescription
                plan_mse = float(np.mean(normalized_error * normalized_error))
                plan_mae = float(np.mean(np.abs(normalized_error)))
            else:
                plan_mse = float("nan")
                plan_mae = float("nan")

            row: dict[str, Any] = {
                "patient_id": patient_id,
                "control_points": entry.control_points,
                "prescription_estimate_normalized": prescription,
                "prescription_source": "summed_reference_dmax",
                "plan_mse_10pct_rx": plan_mse,
                "plan_mae_10pct_rx": plan_mae,
                "plan_low_dose_mae": low_mae,
                "plan_low_dose_voxels": low_count,
                "plan_mid_dose_mae": mid_mae,
                "plan_mid_dose_voxels": mid_count,
                "plan_high_dose_mae": high_mae,
                "plan_high_dose_voxels": high_count,
            }

            do_gamma = compute_gamma and patient_id in gamma_patient_set
            if do_gamma:
                row["plan_gamma_2pct_0mm"] = _zero_distance_gamma(
                    prediction, target, prescription, dose_percent=2.0
                )
                row["plan_gamma_1pct_0mm"] = _zero_distance_gamma(
                    prediction, target, prescription, dose_percent=1.0
                )
                gamma_mask = target >= 0.10 * prescription
                if np.any(gamma_mask):
                    zz, yy, xx = np.where(gamma_mask)
                    margin = 2
                    z0, z1 = max(0, int(zz.min()) - margin), min(
                        target.shape[0], int(zz.max()) + 1 + margin
                    )
                    y0, y1 = max(0, int(yy.min()) - margin), min(
                        target.shape[1], int(yy.max()) + 1 + margin
                    )
                    x0, x1 = max(0, int(xx.min()) - margin), min(
                        target.shape[2], int(xx.max()) + 1 + margin
                    )
                    crop = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
                    pred_gamma, target_gamma = prediction[crop], target[crop]
                else:
                    pred_gamma, target_gamma = prediction, target
                gamma_dmax = float(target_gamma.max())
                cutoff_percent = (
                    100.0 * 0.10 * prescription / gamma_dmax
                    if gamma_dmax > 0
                    else 100.0
                )
                common_options = {
                    "lower_percent_dose_cutoff": cutoff_percent,
                    "interp_fraction": int(gamma_interp_fraction),
                    "max_gamma": 2,
                    "random_subset": gamma_random_subset,
                    "local_gamma": True,
                }
                assert gamma_function is not None
                row["plan_gamma_2pct_3mm"] = gamma_function(
                    pred_gamma,
                    target_gamma,
                    spacing_zyx,
                    gamma_options={
                        **common_options,
                        "dose_percent_threshold": 2,
                        "distance_mm_threshold": 3,
                    },
                )
                row["plan_gamma_1pct_1mm"] = gamma_function(
                    pred_gamma,
                    target_gamma,
                    spacing_zyx,
                    gamma_options={
                        **common_options,
                        "dose_percent_threshold": 1,
                        "distance_mm_threshold": 1,
                    },
                )
            else:
                for name, _dose, _distance in PLAN_GAMMA_SPECS:
                    row[name] = float("nan")
            rows.append(row)

        metric_names = (
            "plan_mse_10pct_rx",
            "plan_mae_10pct_rx",
            "plan_low_dose_mae",
            "plan_mid_dose_mae",
            "plan_high_dose_mae",
            *(name for name, _dose, _distance in PLAN_GAMMA_SPECS),
        )
        means = {
            name: _finite_mean([float(row[name]) for row in rows])
            for name in metric_names
        }
        means["plan_patients"] = float(len(rows))
        means["plan_gamma_patients"] = float(
            len(gamma_patient_set) if compute_gamma else 0
        )
        return means, rows
