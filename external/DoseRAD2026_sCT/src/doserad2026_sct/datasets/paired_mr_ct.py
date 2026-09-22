import os

from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    EnsureTyped,
)
import numpy as np
import SimpleITK as sitk
import torch
import tqdm

from doserad2026_sct.utils import load, masks, preprocess


def get_train_transforms(
    patch_size: list[int, int, int], number_of_patches: int
) -> Compose:
    # Spatial transforms carry the body mask alongside image/label so it stays
    # registered after cropping and flipping; the intensity transforms below are
    # image-only, since the CT label and the mask must not be perturbed.
    spatial_keys = ["image", "label", "mask"]
    return Compose(
        [
            RandCropByPosNegLabeld(
                keys=spatial_keys,
                label_key="mask",
                spatial_size=patch_size,
                pos=3,
                neg=1,
                num_samples=number_of_patches,
            ),
            RandFlipd(
                keys=spatial_keys,
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=spatial_keys,
                prob=0.5,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=spatial_keys,
                prob=0.5,
                spatial_axis=2,
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=0.1,
                mean=0.0,
                std=0.1,
            ),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.15,
                gamma=(0.7, 1.5),
            ),
            RandScaleIntensityd(
                keys=["image"],
                factors=0.25,
                prob=0.15,
            ),
            EnsureTyped(keys=spatial_keys),
        ]
    )


class PairedMRCTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_data: str,
        patient_ids: list[str],
        train: bool,
        patch_size: list[int, int, int] | None = None,
        number_of_patches: int | None = None,
    ) -> None:
        self.path_to_data = path_to_data
        self.patient_ids = patient_ids
        self.train = train
        self.patch_size = patch_size
        self.number_of_patches = number_of_patches
        self._get_population_zscore()

    def _get_population_zscore(self) -> None:
        """Body masks and the population CT mean/std, computed once up front.

        The z-score the generator is trained against uses the mean of the
        per-patient means over *this split*, so the statistics depend on the
        patient list. That is why inference does not recompute them: the
        submission denormalises with the fixed constants in
        ``configs/ct_statistics.json`` instead.
        """
        self.ct_masks = {}
        self.ct_population_statistics = {}
        self.ct_population_statistics["mean"] = []
        self.ct_population_statistics["std"] = []
        # ~1 s/patient of binary_closing + connected components. Reported
        # because it is otherwise a silent wait before training starts.
        for patient_id in tqdm.tqdm(
            self.patient_ids,
            desc=f"  masks ({'train' if self.train else 'val'})",
            leave=False,
        ):
            ct = load.load_mha_as_np(
                os.path.join(self.path_to_data, patient_id, "image", "ct.mha")
            )
            mask = masks.get_body_mask(ct)
            self.ct_masks[patient_id] = mask
            ct[mask == 0] = preprocess.CT_LOWEST_VALUE
            ct = np.clip(ct, preprocess.CT_LOWEST_VALUE, preprocess.CT_HIGHEST_VALUE)
            self.ct_population_statistics["mean"].append(np.mean(ct))
            self.ct_population_statistics["std"].append(np.std(ct))

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, dict[str, float]] | dict[str, np.ndarray],
    ]:
        # returns the preprocessed mr, ct, mask, and a dict containing normalisation information for postprocessing
        stats = {}
        patient_id = self.patient_ids[idx]

        # load CT
        ct_sitk_image = load.load_mha_as_sitk_image(
            os.path.join(self.path_to_data, patient_id, "image", "ct.mha")
        )
        ct = sitk.GetArrayFromImage(ct_sitk_image)
        if self.train:
            ct, ct_stats = preprocess.preprocess_ct(
                ct,
                ct_mean=np.mean(self.ct_population_statistics["mean"]),
                ct_std=np.mean(self.ct_population_statistics["std"]),
                mask=self.ct_masks[patient_id],
            )
            stats["ct"] = ct_stats
        else:
            ct, _ = preprocess.preprocess_ct(
                ct, mask=self.ct_masks[patient_id], method="mask_clip"
            )
        stats["ct_population"] = {
            "mean": self.ct_population_statistics["mean"],
            "std": self.ct_population_statistics["std"],
        }

        # load MR
        mr_sitk_image = load.load_mha_as_sitk_image(
            os.path.join(self.path_to_data, patient_id, "image", "mr.mha")
        )
        mr = sitk.GetArrayFromImage(mr_sitk_image)
        mr, mr_stats = preprocess.preprocess_mr(mr)
        stats["mr"] = mr_stats
        stats["patient_id"] = patient_id
        mask = self.ct_masks[patient_id]
        if not self.train:
            return mr, ct, mask, stats
        else:
            data = {
                "image": mr[None].astype(np.float32),
                "label": ct[None].astype(np.float32),
                "mask": mask[None].astype(np.uint8),
            }
            train_transforms = get_train_transforms(
                self.patch_size, self.number_of_patches
            )
            data = train_transforms(data)
            return data, stats
