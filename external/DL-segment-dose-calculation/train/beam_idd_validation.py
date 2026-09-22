"""Evaluator beam-direction IDD, scored per control point during validation.

IDD is a per-beam metric and cannot be recovered from a summed plan, so it is
scored as the CT-space ROIs arrive. That is the only thing it ever needed from
:mod:`plan_validation`, so it rides on the ordinary validation metrics pass
(``--val-metrics-freq``) instead of the patient-plan accumulation, and is fed
the same per-sample tensors as the CT-space metrics.

Kept free of ``utils`` imports so the wiring stays unit-testable without the
training stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from idd_ct import direction_from_gantry, idd_curve_distance_loss_ct


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


class CtBeamIddAccumulator:
    """Score each control point's CT-space ROI with the challenge's beam IDD.

    Aggregation is a patient macro-mean -- mean over each patient's control
    points, then mean across patients -- which is how the challenge aggregates
    and what the plan-validation rows reported before this moved.
    """

    def __init__(
        self,
        catalog: Any,
        gantry_lookup: dict[tuple[str, int, int], float],
        idd_floor_frac: float = 0.005,
    ) -> None:
        self.catalog = catalog
        self.gantry_lookup = gantry_lookup
        self.idd_floor_frac = float(idd_floor_frac)
        self._per_patient: dict[str, list[float]] = {}
        self._per_patient_floored: dict[str, list[float]] = {}
        self._missing_gantry: set[str] = set()

    def update(
        self,
        sample: dict[str, Any],
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Score one control point from the tensors the CT metrics already get."""
        sample_id = str(sample["sample_id"])
        patient_id = str(self.catalog.records[sample_id].patient_id)
        shape = tuple(int(value) for value in sample["ct_shape"])
        spacing = tuple(float(value) for value in sample["spacing_xyz_mm"])
        xs, _xe, ys, _ye, _zs, _ze = (int(value) for value in sample["roi_box"])

        invalid = ~valid[0].detach().bool()
        valid_prediction = prediction[0].detach().float().masked_fill(invalid, 0.0)
        valid_target = target[0].detach().float().masked_fill(invalid, 0.0)

        kw = dict(roi_origin_xy=(xs, ys), full_shape_xy=(shape[0], shape[1]),
                  spacing=spacing)
        self._per_patient.setdefault(patient_id, []).append(
            self._beam_idd(sample_id, patient_id, valid_prediction,
                           valid_target, **kw)
        )

        # Same metric on floored dose: both prediction and target zeroed below
        # a fraction of the target's own max, matching the submission-scoring
        # convention (postprocess_eval.py) for beams with no real
        # minimum_cutoff to read. Symmetric, so it cannot flatter the model.
        if self.idd_floor_frac > 0.0:
            thr = self.idd_floor_frac * float(valid_target.max())
            floored_prediction = torch.where(
                valid_prediction < thr,
                torch.zeros_like(valid_prediction),
                valid_prediction,
            )
            floored_target = torch.where(
                valid_target < thr, torch.zeros_like(valid_target), valid_target
            )
            self._per_patient_floored.setdefault(patient_id, []).append(
                self._beam_idd(sample_id, patient_id, floored_prediction,
                               floored_target, **kw)
            )

    def _beam_idd(
        self,
        sample_id: str,
        patient_id: str,
        prediction_roi: torch.Tensor,
        target_roi: torch.Tensor,
        *,
        roi_origin_xy: tuple[int, int],
        full_shape_xy: tuple[int, int],
        spacing: tuple[float, float, float],
    ) -> float:
        """Evaluator IDD for one control point, on the ROI, without gradients."""
        record = self.catalog.records[sample_id]
        key = (patient_id, int(record.beam_id), int(record.cp_id))
        gantry = self.gantry_lookup.get(key)
        if gantry is None:
            # a warning per patient, not per control point
            if patient_id not in self._missing_gantry:
                self._missing_gantry.add(patient_id)
                print(
                    f"  [val metrics] no gantry angle for {key}; beam IDD "
                    f"will be nan for {patient_id}"
                )
            return float("nan")
        if float(target_roi.max()) <= 0.0:
            return float("nan")
        with torch.no_grad():
            value = idd_curve_distance_loss_ct(
                prediction_roi,
                target_roi,
                direction_from_gantry(gantry),
                spacing,
                roi_origin_xy=roi_origin_xy,
                full_shape_xy=full_shape_xy,
            )
        return float(value)

    def summary_means(self) -> dict[str, float]:
        """Patient macro-mean, the per-patient worst control point, and a count.

        ``beam_idd_true_cps`` is the number of control points that actually
        scored, so a silently broken gantry join reads as 0 rather than as a
        plausible curve.
        """
        if not self._per_patient:
            return {}
        per_patient_mean: list[float] = []
        per_patient_max: list[float] = []
        scored = 0
        for values in self._per_patient.values():
            finite = [value for value in values if np.isfinite(value)]
            scored += len(finite)
            per_patient_mean.append(_finite_mean(values))
            per_patient_max.append(
                float(np.max(finite)) if finite else float("nan")
            )
        means = {
            "beam_idd_true": _finite_mean(per_patient_mean),
            "beam_idd_true_max": _finite_mean(per_patient_max),
            "beam_idd_true_cps": float(scored),
        }
        if self._per_patient_floored:
            means["beam_idd_true_floored"] = _finite_mean(
                [_finite_mean(v) for v in self._per_patient_floored.values()]
            )
        return {
            name: value
            for name, value in means.items()
            if np.isfinite(value)
        }
