from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage import measure, morphology


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_label(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def remap_label(label: np.ndarray, by_size: bool = False) -> np.ndarray:
    """Remap instance ids to consecutive ids: 0, 1, 2, ... ."""
    label = label.astype(np.int32, copy=False)
    ids = [i for i in np.unique(label) if i != 0]
    if by_size:
        ids = sorted(ids, key=lambda x: int((label == x).sum()), reverse=True)
    out = np.zeros_like(label, dtype=np.int32)
    for new_id, old_id in enumerate(ids, start=1):
        out[label == old_id] = new_id
    return out


def binary_to_instances(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    mask = mask > 0
    if min_area > 0:
        mask = morphology.remove_small_objects(mask, min_size=min_area)
    return remap_label(measure.label(mask).astype(np.int32))


def fast_aji(true: np.ndarray, pred: np.ndarray) -> float:
    """Aggregated Jaccard Index following common HoVer-Net evaluation code."""
    true = remap_label(true)
    pred = remap_label(pred)
    true_ids = list(np.unique(true)); pred_ids = list(np.unique(pred))
    true_ids = [i for i in true_ids if i != 0]
    pred_ids = [i for i in pred_ids if i != 0]
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    true_masks = {i: true == i for i in true_ids}
    pred_masks = {i: pred == i for i in pred_ids}
    pair_inter = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)
    pair_union = np.zeros_like(pair_inter)

    for ti, t_id in enumerate(true_ids):
        t_mask = true_masks[t_id]
        overlapping = np.unique(pred[t_mask])
        for p_id in overlapping:
            if p_id == 0:
                continue
            pj = pred_ids.index(int(p_id))
            p_mask = pred_masks[int(p_id)]
            inter = np.logical_and(t_mask, p_mask).sum()
            union = np.logical_or(t_mask, p_mask).sum()
            pair_inter[ti, pj] = inter
            pair_union[ti, pj] = union

    pair_iou = pair_inter / (pair_union + 1.0e-6)
    paired_pred = np.argmax(pair_iou, axis=1)
    pair_iou_max = pair_iou[np.arange(len(true_ids)), paired_pred]
    paired_true = np.nonzero(pair_iou_max > 0)[0]
    paired_pred = paired_pred[paired_true]

    overall_inter = pair_inter[paired_true, paired_pred].sum()
    overall_union = pair_union[paired_true, paired_pred].sum()

    unpaired_true = set(range(len(true_ids))) - set(paired_true.tolist())
    unpaired_pred = set(range(len(pred_ids))) - set(paired_pred.tolist())
    for ti in unpaired_true:
        overall_union += true_masks[true_ids[ti]].sum()
    for pi in unpaired_pred:
        overall_union += pred_masks[pred_ids[pi]].sum()
    return float(overall_inter / (overall_union + 1.0e-6))


def get_fast_pq(true: np.ndarray, pred: np.ndarray, match_iou: float = 0.5) -> tuple[float, float, float]:
    """Return DQ, SQ, PQ. Matching follows HoVer-Net-style PQ at IoU threshold 0.5."""
    true = remap_label(true)
    pred = remap_label(pred)
    true_ids = [i for i in np.unique(true) if i != 0]
    pred_ids = [i for i in np.unique(pred) if i != 0]
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0, 1.0, 1.0
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0, 0.0, 0.0

    iou = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)
    for ti, t_id in enumerate(true_ids):
        t_mask = true == t_id
        for p_id in np.unique(pred[t_mask]):
            if p_id == 0:
                continue
            pj = pred_ids.index(int(p_id))
            p_mask = pred == p_id
            inter = np.logical_and(t_mask, p_mask).sum()
            union = np.logical_or(t_mask, p_mask).sum()
            iou[ti, pj] = inter / (union + 1.0e-6)

    if match_iou >= 0.5:
        paired_true, paired_pred = np.nonzero(iou > match_iou)
        paired_iou = iou[paired_true, paired_pred]
    else:
        true_ind, pred_ind = linear_sum_assignment(-iou)
        valid = iou[true_ind, pred_ind] > match_iou
        paired_true, paired_pred = true_ind[valid], pred_ind[valid]
        paired_iou = iou[paired_true, paired_pred]

    tp = len(paired_true)
    fp = len(pred_ids) - tp
    fn = len(true_ids) - tp
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)
    sq = float(paired_iou.mean()) if tp > 0 else 0.0
    pq = dq * sq
    return float(dq), float(sq), float(pq)


def dice_binary(true: np.ndarray, pred: np.ndarray) -> float:
    t = true > 0
    p = pred > 0
    inter = np.logical_and(t, p).sum()
    return float((2.0 * inter + 1.0e-6) / (t.sum() + p.sum() + 1.0e-6))


@dataclass
class MetricRow:
    name: str
    dice: float
    aji: float
    dq: float
    sq: float
    pq: float


def compute_instance_metrics(true: np.ndarray, pred: np.ndarray, match_iou: float = 0.5) -> dict[str, float]:
    true = remap_label(true)
    pred = remap_label(pred)
    dq, sq, pq = get_fast_pq(true, pred, match_iou=match_iou)
    return {
        "dice": dice_binary(true, pred),
        "aji": fast_aji(true, pred),
        "dq": dq,
        "sq": sq,
        "pq": pq,
    }


def find_matching_label(gt_dir: Path, stem: str, suffixes: Iterable[str]) -> Path | None:
    for suffix in suffixes:
        p = gt_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    # fallback for exact stem with any image extension
    for ext in IMG_EXTS:
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def evaluate_prediction_dir(
    pred_dir: str | Path,
    gt_dir: str | Path,
    pred_suffix: str = "_pred_inst.png",
    gt_suffixes: tuple[str, ...] = ("_label.png", "_inst.png", ".png"),
    pred_binary: bool = False,
    min_area: int = 0,
    match_iou: float = 0.5,
    save_csv: str | Path | None = None,
) -> dict[str, float]:
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    pred_paths = sorted([p for p in pred_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    rows: list[dict] = []
    for pred_path in pred_paths:
        stem = pred_path.name[:-len(pred_suffix)] if pred_path.name.endswith(pred_suffix) else pred_path.stem.replace("_pred_inst", "").replace("_pred", "")
        gt_path = find_matching_label(gt_dir, stem, gt_suffixes)
        if gt_path is None:
            continue
        pred = read_label(pred_path)
        if pred_binary or pred.max() <= 1 or set(np.unique(pred).tolist()).issubset({0, 255}):
            pred = binary_to_instances(pred > 0, min_area=min_area)
        true = read_label(gt_path)
        m = compute_instance_metrics(true, pred, match_iou=match_iou)
        rows.append({"name": stem, **m})

    if not rows:
        raise FileNotFoundError(f"No matched prediction/GT pairs found under {pred_dir} and {gt_dir}")
    df = pd.DataFrame(rows)
    summary = {k: float(df[k].mean()) for k in ["dice", "aji", "dq", "sq", "pq"]}
    summary["num_images"] = int(len(df))
    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        pd.DataFrame([{**{"name": "mean"}, **summary}]).to_csv(save_csv.with_name(save_csv.stem + "_summary.csv"), index=False)
    return summary
