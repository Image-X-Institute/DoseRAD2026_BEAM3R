import numpy as np

from doserad2026_sct.utils import masks, normalisation

CT_LOWEST_VALUE = -1024.0
CT_HIGHEST_VALUE = 3000.0


def preprocess_ct(
    ct: np.ndarray,
    method: str = "mask_clip_z-score_population",
    ct_mean: float | None = None,
    ct_std: float | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    match method:
        case "mask_clip_z-score_population":
            if mask is None:
                mask = masks.get_body_mask(ct)
            ct[mask == 0] = CT_LOWEST_VALUE
            ct = np.clip(ct, CT_LOWEST_VALUE, CT_HIGHEST_VALUE)
            ct, ct_stats = normalisation.z_score_normalise(ct, mean=ct_mean, std=ct_std)
            return ct, ct_stats
        case "mask_clip":
            if mask is None:
                mask = masks.get_body_mask(ct)
            ct[mask == 0] = CT_LOWEST_VALUE
            ct = np.clip(ct, CT_LOWEST_VALUE, CT_HIGHEST_VALUE)
            ct_stats = {}
            return ct, ct_stats
        case _:
            raise NotImplementedError(f"Preprocessing method: {method} not recognised")


def preprocess_mr(
    mr: np.ndarray,
    method: str = "z-score",
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    match method:
        case "z-score":
            mr, mr_stats = normalisation.z_score_normalise(
                mr,
            )
            return mr, mr_stats
        case _:
            raise NotImplementedError(f"Preprocessing method: {method} not recognised")
