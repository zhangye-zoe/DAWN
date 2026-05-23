from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology


def point_gaussian_map(points: np.ndarray, shape: tuple[int, int], r1: int, r2: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Create DAWN extended Gaussian target and valid mask from a binary point map.

    Returns target values in [0,1] and a valid mask. Pixels outside r2 are ignored.
    """
    points = (points > 0).astype(np.uint8)
    if points.sum() == 0:
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    dist = ndi.distance_transform_edt(1 - points)
    target = np.zeros(shape, np.float32)
    fg = dist <= r1
    bg = (dist > r1) & (dist <= r2)
    target[fg] = np.exp(-(dist[fg] ** 2) / (2.0 * sigma ** 2))
    target[bg] = 0.0
    valid = (fg | bg).astype(np.float32)
    return target, valid


def combined_pseudo_label(seg_prob: np.ndarray, det_prob: np.ndarray, point_map: np.ndarray, theta: float, distance: int, min_area: int = 8) -> np.ndarray:
    """Combined Pseudo-Label optimization from DAWN.

    1. threshold detector output, 2. union with segmentation foreground,
    3. retain connected components supported by nearby point annotations.
    """
    seg_mask = seg_prob > 0.5
    det_mask = det_prob > theta
    fused = np.logical_or(seg_mask, det_mask)
    if min_area > 0:
        fused = morphology.remove_small_objects(fused, min_size=min_area)
    labels = measure.label(fused)
    if labels.max() == 0 or point_map.max() == 0:
        return fused.astype(np.uint8)
    dist = ndi.distance_transform_edt(point_map <= 0)
    keep_region = dist <= distance
    keep_ids = np.unique(labels[keep_region])
    keep_ids = keep_ids[keep_ids != 0]
    out = np.isin(labels, keep_ids)
    return out.astype(np.uint8)


def filter_mask_by_points(mask: np.ndarray, point_map: np.ndarray, distance: int = 25, min_area: int = 8, keep_if_no_point: bool = False) -> np.ndarray:
    """Retain predicted connected components supported by point annotations.

    This is used for source-HoVerNet bootstrap pseudo labels. It keeps the
    segmentation-mask morphology from the pre-trained segmentor, but removes
    components that are not close to a target-domain point annotation.
    """
    mask = mask > 0
    if min_area > 0:
        mask = morphology.remove_small_objects(mask, min_size=min_area)
    labels = measure.label(mask)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    point_map = point_map > 0
    if point_map.max() == 0:
        return mask.astype(np.uint8) if keep_if_no_point else np.zeros_like(mask, dtype=np.uint8)
    dist = ndi.distance_transform_edt(~point_map)
    keep_region = dist <= distance
    keep_ids = np.unique(labels[keep_region])
    keep_ids = keep_ids[keep_ids != 0]
    out = np.isin(labels, keep_ids)
    if min_area > 0:
        out = morphology.remove_small_objects(out, min_size=min_area)
    return out.astype(np.uint8)


def remove_bad_regions(mask: np.ndarray, min_area: int = 12, max_area: int = 2000) -> np.ndarray:
    lab = measure.label(mask > 0)
    if lab.max() == 0:
        return mask.astype(np.uint8)
    small_removed = morphology.remove_small_objects(lab, min_size=min_area) > 0
    if max_area > 0:
        large = morphology.remove_small_objects(lab, min_size=max_area) > 0
        small_removed = small_removed & (~large)
    return small_removed.astype(np.uint8)
