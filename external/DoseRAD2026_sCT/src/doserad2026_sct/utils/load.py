"""Volume loading. Everything is .mha, read through SimpleITK."""

import numpy as np
import SimpleITK as sitk


def load_mha_as_sitk_image(path: str) -> sitk.Image:
    return sitk.ReadImage(path)


def load_mha_as_np(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(load_mha_as_sitk_image(path))
