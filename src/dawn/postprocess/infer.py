from __future__ import annotations

from pathlib import Path
import argparse
import json
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io, segmentation, color
from tqdm import tqdm

from dawn.models import DAWN
from dawn.data.datasets import read_rgb, read_gray, IMG_EXTS
from dawn.evaluation.metrics import binary_to_instances, evaluate_prediction_dir
from dawn.utils.checkpoint import load_checkpoint
from dawn.utils.masks import combined_pseudo_label, remove_bad_regions


def image_to_tensor(arr: np.ndarray, mean, std) -> torch.Tensor:
    x = arr.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)) / (
        np.asarray(std, dtype=np.float32).reshape(1, 1, 3) + 1e-6
    )
    return torch.from_numpy(x.transpose(2, 0, 1))[None].float()


def _save_overlay(image: np.ndarray, pred_inst: np.ndarray, out_path: Path, gt_inst: np.ndarray | None = None, point: np.ndarray | None = None):
    img = image.astype(np.float32) / 255.0
    overlay = segmentation.mark_boundaries(img, pred_inst > 0, color=(1, 0, 0), mode="thick")
    if gt_inst is not None:
        overlay = segmentation.mark_boundaries(overlay, gt_inst > 0, color=(0, 1, 0), mode="thick")
    if point is not None and point.max() > 0:
        ys, xs = np.nonzero(point > 0)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 2), min(overlay.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(overlay.shape[1], x + 3)
            overlay[y0:y1, x0:x1] = np.array([0.0, 0.3, 1.0])
    io.imsave(out_path, (overlay * 255).clip(0, 255).astype(np.uint8), check_contrast=False)


def predict_folder(cfg: dict) -> dict | None:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = cfg.get("model", {})
    model = DAWN(
        detector_backbone=model_cfg.get("detector_backbone", "resnet34"),
        detector_pretrained=False,
        hover_mode=model_cfg.get("hover_mode", "fast"),
    ).to(device)
    load_checkpoint(model, cfg["checkpoint"], strict=False)
    model.eval()

    data_cfg = cfg["data"]
    pp = cfg.get("postprocess", {}) | cfg.get("cpl", {})
    img_dir = Path(data_cfg["image_dir"])
    out_dir = Path(cfg["output_dir"])
    point_dir = Path(data_cfg["point_dir"]) if data_cfg.get("point_dir") else None
    gt_dir = Path(data_cfg["gt_dir"]) if data_cfg.get("gt_dir") else None

    for sub in ["seg_prob", "det_prob", "pred_mask", "pred_inst"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    make_pseudo = point_dir is not None
    if make_pseudo:
        (out_dir / "pseudo").mkdir(exist_ok=True)
    if cfg.get("visualize", {}).get("enabled", True):
        (out_dir / "vis").mkdir(exist_ok=True)

    image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    for p in tqdm(image_paths, desc="inference"):
        arr = read_rgb(p)
        x = image_to_tensor(arr, data_cfg.get("mean", [0.5] * 3), data_cfg.get("std", [0.5] * 3)).to(device)
        with torch.no_grad():
            out = model(x)
            seg_prob = torch.softmax(out["np"], dim=1)[:, 1:2]
            seg_prob = F.interpolate(seg_prob, size=arr.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            det_prob = F.interpolate(out["det_prob"], size=arr.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        mask = remove_bad_regions(
            seg_prob > pp.get("seg_threshold", 0.5),
            pp.get("min_area", 12),
            pp.get("max_area", 2000),
        )
        inst = binary_to_instances(mask, min_area=pp.get("min_area", 12))
        io.imsave(out_dir / "seg_prob" / f"{p.stem}_seg_prob.png", (seg_prob * 255).clip(0, 255).astype(np.uint8), check_contrast=False)
        io.imsave(out_dir / "det_prob" / f"{p.stem}_det_prob.png", (det_prob * 255).clip(0, 255).astype(np.uint8), check_contrast=False)
        io.imsave(out_dir / "pred_mask" / f"{p.stem}_pred.png", (mask * 255).astype(np.uint8), check_contrast=False)
        io.imsave(out_dir / "pred_inst" / f"{p.stem}_pred_inst.png", inst.astype(np.uint16), check_contrast=False)

        point = None
        if make_pseudo:
            point_path = point_dir / f"{p.stem}{data_cfg.get('point_suffix', '_label_point.png')}"
            if not point_path.exists():
                point_path = point_dir / f"{p.stem}.png"
            if point_path.exists():
                point = read_gray(point_path) > 0
                pseudo = combined_pseudo_label(
                    seg_prob,
                    det_prob,
                    point,
                    pp.get("theta", pp.get("det_threshold", 0.2)),
                    pp.get("distance", 25),
                    pp.get("min_area", 12),
                )
                io.imsave(out_dir / "pseudo" / f"{p.stem}_pseudo.png", (pseudo * 255).astype(np.uint8), check_contrast=False)

        if cfg.get("visualize", {}).get("enabled", True):
            gt = None
            if gt_dir is not None:
                for suffix in data_cfg.get("gt_suffixes", ["_label.png", "_inst.png", ".png"]):
                    gt_path = gt_dir / f"{p.stem}{suffix}"
                    if gt_path.exists():
                        gt = read_gray(gt_path)
                        break
            _save_overlay(arr, inst, out_dir / "vis" / f"{p.stem}_overlay.png", gt_inst=gt, point=point)

    summary = None
    if gt_dir is not None and gt_dir.exists():
        eval_cfg = cfg.get("evaluation", {})
        summary = evaluate_prediction_dir(
            out_dir / "pred_inst",
            gt_dir,
            pred_suffix="_pred_inst.png",
            gt_suffixes=tuple(data_cfg.get("gt_suffixes", ["_label.png", "_inst.png", ".png"])),
            pred_binary=False,
            min_area=pp.get("min_area", 12),
            match_iou=eval_cfg.get("match_iou", 0.5),
            save_csv=out_dir / "metrics_per_image.csv",
        )
        (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
        print("Evaluation summary:", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    predict_folder(cfg)


if __name__ == "__main__":
    main()
