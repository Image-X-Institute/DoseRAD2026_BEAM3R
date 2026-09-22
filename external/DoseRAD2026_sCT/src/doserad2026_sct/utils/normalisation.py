import numpy as np
import torch


def z_score_normalise(
    img: np.ndarray, mean: float | None = None, std: float | None = None
) -> tuple[np.ndarray, dict[str, float]]:
    if not np.issubdtype(img.dtype, np.floating):
        raise TypeError(f"Expected floating-point input image, got {img.dtype}")
    if mean is None:
        mean = np.mean(img)
    if std is None:
        std = np.std(img)

    if std == 0:
        raise ValueError("Standard deviation zero, will incur div. by zero.")

    normalised = (img - mean) / std

    stats = {
        "mean": mean,
        "std": std,
        "min": np.min(img),
        "max": np.max(img),
    }
    return normalised, stats


def z_score_denormalise(
    img: np.ndarray | torch.Tensor, mean: float, std: float
) -> np.ndarray | torch.Tensor:
    return (img * std) + mean
