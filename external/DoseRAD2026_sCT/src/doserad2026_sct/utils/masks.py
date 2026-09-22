import numpy as np
from scipy import ndimage


def get_body_mask(ct: np.ndarray, threshold: int = -850) -> np.ndarray:
    mask = ct > threshold
    # Remove small gaps
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5, 5)))
    labels, num = ndimage.label(mask)

    if num == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = ndimage.sum(mask, labels, range(1, num + 1))
    mask = labels == (np.argmax(sizes) + 1)
    mask = ndimage.binary_fill_holes(mask)
    return mask


def get_enclosed_low_density_mask(
    ct: np.ndarray,
    threshold: int = -900,
    dilate: int = 1,
    minimum_size: int = 50,
) -> np.ndarray:
    """Enclosed regions below ``threshold`` HU -- aerated lung, airways, gas.

    Named for what it measures rather than what it was first assumed to be. In
    the thorax the mask is dominated by *aerated lung parenchyma*, not airways:
    on 1THB121 the two largest components are 2241 mL and 956 mL, right and
    left, each surrounded ~87-90 % by lung-band tissue. Trachea and bronchi are
    the 44 mL and 25 mL components below them. In the abdomen it is genuinely
    just bowel gas, and tiny (~17 mL), which is why it carries none of that
    cohort's dose error.

    Together with the [-900, -500) band this covers lung density either side of
    the threshold; the two are disjoint in HU and roughly additive in effect.

    ``get_body_mask`` fills holes, so these regions sit *inside* the body and
    are neither clamped to air at inference nor given any weight of their own by
    a voxel-count-weighted reconstruction loss. They are ~1.7 % of a thorax
    volume but carry roughly half of the beam-level dose error, because every
    chest field passes through them.

    Enclosed is defined by connectivity rather than by the body mask: any
    ``ct < threshold`` component that does not touch the volume border. That
    excludes the external air the inference clamp already handles, and needs no
    body mask of its own.

    ``dilate`` grows the mask by one voxel so the boundary wall -- where the
    air/tissue transition actually is, and where partial volume makes the
    generator hedge -- is included. ``minimum_size`` drops isolated speckle that
    is usually noise rather than anatomy.
    """
    air = ct < threshold
    labels, num = ndimage.label(air)
    if num == 0:
        return np.zeros_like(air, dtype=bool)

    border = np.unique(
        np.concatenate(
            [
                labels[0].ravel(),
                labels[-1].ravel(),
                labels[:, 0].ravel(),
                labels[:, -1].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            ]
        )
    )
    enclosed = air & ~np.isin(labels, border[border != 0])

    if minimum_size > 0 and enclosed.any():
        labels, num = ndimage.label(enclosed)
        sizes = ndimage.sum(enclosed, labels, range(1, num + 1))
        keep = [i + 1 for i, size in enumerate(sizes) if size >= minimum_size]
        enclosed = np.isin(labels, keep)

    if dilate > 0 and enclosed.any():
        enclosed = ndimage.binary_dilation(enclosed, iterations=dilate)
    return enclosed
