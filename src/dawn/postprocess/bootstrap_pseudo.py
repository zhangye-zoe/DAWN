from __future__ import annotations

from pathlib import Path
import argparse
import json
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io, measure, morphology
from tqdm import tqdm

from dawn.data.datasets import IMG_EXTS, read_gray, read_rgb
from dawn.models import DAWN
from dawn.utils.checkpoint import load_checkpoint
from dawn.utils.masks import filter_mask_by_points, remove_bad_regions


def image_to_tensor(arr: np.ndarray, mean, std) -> torch.Tensor:
    x = arr.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)) / (
        np.asarray(std, dtype=np.float32).reshape(1, 1, 3) + 1e-6
    )
    return torch.from_numpy(x.transpose(2, 0, 1))[None].float()


@torch.no_grad()
def generate_bootstrap_pseudo(cfg: dict) -> dict:
    """Generate initial pseudo labels on the target training set using a
    source-domain HoVerNet/segmentor checkpoint.

    This is the intended first-step pseudo label for DAWN target adaptation:
    source segmentor prediction -> threshold/morphology -> point-supported
    filtering. The output can be used as `data.pseudo_dir` in the first target
    training round, so round0 already uses segmentation masks rather than sparse
    point maps as segmentation supervision.
    """
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = cfg["data"]
    pp = cfg.get("postprocess", {}) | cfg.get("cpl", {}) | cfg.get("bootstrap", {})

    image_dir = Path(data_cfg["image_dir"])
    point_dir = Path(data_cfg["point_dir"]) if data_cfg.get("point_dir") else None
    out_dir = Path(cfg["output_dir"])

    for sub in ["seg_prob", "source_mask", "pseudo"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    model_cfg = cfg.get("model", {})
    model = DAWN(
        detector_backbone=model_cfg.get("detector_backbone", "resnet34"),
        detector_pretrained=False,
        hover_mode=model_cfg.get("hover_mode", "fast"),
    ).to(device)

    checkpoint = cfg.get("checkpoint") or cfg.get("source_checkpoint") or cfg.get("segmentor_checkpoint")
    if checkpoint is None:
        raise ValueError("Bootstrap pseudo generation needs `checkpoint`, `source_checkpoint`, or `segmentor_checkpoint`.")

    # Prefer loading the checkpoint into the segmentor only, because source
    # pre-training checkpoints usually do not contain detector weights.
    try:
        model.load_segmentor(checkpoint, strict=False)
        load_mode = "segmentor_only"
    except Exception:
        load_checkpoint(model, checkpoint, strict=False)
        load_mode = "full_model"
    print(f"[bootstrap] loaded {checkpoint} ({load_mode})")

    model.eval()
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    rows = []
    for p in tqdm(image_paths, desc="bootstrap pseudo from source segmentor"):
        arr = read_rgb(p)
        x = image_to_tensor(arr, data_cfg.get("mean", [0.5] * 3), data_cfg.get("std", [0.5] * 3)).to(device)
        out = model(x)
        seg_prob = torch.softmax(out["np"], dim=1)[:, 1:2]
        seg_prob = F.interpolate(seg_prob, size=arr.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        source_mask = remove_bad_regions(
            seg_prob > pp.get("seg_threshold", 0.5),
            pp.get("min_area", 12),
            pp.get("max_area", 2000),
        ).astype(np.uint8)

        pseudo = source_mask
        point_count = None
        if point_dir is not None:
            point_path = point_dir / f"{p.stem}{data_cfg.get('point_suffix', '_label_point.png')}"
            if not point_path.exists():
                point_path = point_dir / f"{p.stem}.png"
            if point_path.exists():
                point = read_gray(point_path) > 0
                point_count = int(point.sum())
                pseudo = filter_mask_by_points(
                    source_mask,
                    point,
                    distance=pp.get("distance", 25),
                    min_area=pp.get("min_area", 12),
                    keep_if_no_point=pp.get("keep_if_no_point", False),
                )

        io.imsave(out_dir / "seg_prob" / f"{p.stem}_seg_prob.png", (seg_prob * 255).clip(0, 255).astype(np.uint8), check_contrast=False)
        io.imsave(out_dir / "source_mask" / f"{p.stem}_source_mask.png", (source_mask * 255).astype(np.uint8), check_contrast=False)
        io.imsave(out_dir / "pseudo" / f"{p.stem}_pseudo.png", (pseudo * 255).astype(np.uint8), check_contrast=False)
        rows.append({
            "name": p.name,
            "source_area": int(source_mask.sum()),
            "pseudo_area": int(pseudo.sum()),
            "point_count": point_count,
        })

    summary = {
        "output_dir": str(out_dir),
        "pseudo_dir": str(out_dir / "pseudo"),
        "num_images": len(image_paths),
        "checkpoint": str(checkpoint),
        "load_mode": load_mode,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_dir / "pseudo_stats.csv", index=False)
    except Exception:
        pass
    print("[bootstrap] summary:", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    generate_bootstrap_pseudo(cfg)


if __name__ == "__main__":
    main()
