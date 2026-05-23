from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dawn.data import TargetPointDataset, RandomTransform
from dawn.data.datasets import IMG_EXTS, read_rgb, read_gray
from dawn.evaluation.metrics import binary_to_instances, compute_instance_metrics
from dawn.models import DAWN
from dawn.training.losses import masked_mse, cfc_loss, dynamic_loss
from dawn.utils.checkpoint import save_checkpoint, load_checkpoint
from dawn.utils.masks import remove_bad_regions
from dawn.utils.plotting import plot_history_csv


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_to_tensor(arr: np.ndarray, mean, std) -> torch.Tensor:
    x = arr.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)) / (
        np.asarray(std, dtype=np.float32).reshape(1, 1, 3) + 1e-6
    )
    return torch.from_numpy(x.transpose(2, 0, 1))[None].float()


def _find_gt(gt_dir: Path, stem: str, suffixes: list[str]) -> Path | None:
    for suffix in suffixes:
        p = gt_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    for ext in IMG_EXTS:
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


@torch.no_grad()
def evaluate_model_on_folder(model: DAWN, cfg: dict, device: torch.device, split: str = "val") -> dict[str, float] | None:
    data_cfg = cfg["data"]
    val_cfg = cfg.get("validation", {})
    root = Path(data_cfg["target_root"])
    image_dir = Path(val_cfg.get("image_dir", root / data_cfg.get("image_dir", "images") / split))
    gt_dir_cfg = val_cfg.get("gt_dir") or data_cfg.get("val_gt_dir")
    if gt_dir_cfg is None:
        inst_dir = data_cfg.get("instance_dir", "labels_instance")
        candidate = root / inst_dir / split
        gt_dir = candidate if candidate.exists() else None
    else:
        gt_dir = Path(gt_dir_cfg)
    if gt_dir is None or not gt_dir.exists() or not image_dir.exists():
        return None

    suffixes = val_cfg.get("gt_suffixes", data_cfg.get("gt_suffixes", ["_label.png", "_inst.png", ".png"]))
    pp = cfg.get("postprocess", {}) | cfg.get("cpl", {}) | cfg.get("dawn", {})
    model.eval()
    rows = []
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    max_images = val_cfg.get("max_images")
    if max_images:
        image_paths = image_paths[: int(max_images)]
    for p in tqdm(image_paths, desc=f"validate {split}", leave=False):
        gt_path = _find_gt(gt_dir, p.stem, suffixes)
        if gt_path is None:
            continue
        arr = read_rgb(p)
        x = image_to_tensor(arr, data_cfg.get("mean", [0.5] * 3), data_cfg.get("std", [0.5] * 3)).to(device)
        out = model(x)
        seg_prob = torch.softmax(out["np"], dim=1)[:, 1:2]
        seg_prob = F.interpolate(seg_prob, size=arr.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        mask = remove_bad_regions(seg_prob > pp.get("seg_threshold", 0.5), pp.get("min_area", 12), pp.get("max_area", 2000))
        pred_inst = binary_to_instances(mask, min_area=pp.get("min_area", 12))
        true_inst = read_gray(gt_path)
        rows.append(compute_instance_metrics(true_inst, pred_inst, match_iou=val_cfg.get("match_iou", 0.5)))
    if not rows:
        return None
    return {k: float(np.mean([r[k] for r in rows])) for k in ["dice", "aji", "dq", "sq", "pq"]} | {"num_images": len(rows)}


def train_target(cfg: dict) -> dict:
    seed_everything(cfg.get("seed", 2024))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data_cfg, train_cfg, dawn_cfg = cfg["data"], cfg["train"], cfg.get("dawn", {})

    tfm = RandomTransform(
        train_cfg.get("patch_size", 256),
        data_cfg.get("mean", [0.5] * 3),
        data_cfg.get("std", [0.5] * 3),
        train=True,
    )
    ds = TargetPointDataset(
        data_cfg["target_root"],
        split=data_cfg.get("split", "train"),
        image_dir=data_cfg.get("image_dir", "images"),
        point_dir=data_cfg.get("point_dir", "labels_point"),
        point_suffix=data_cfg.get("point_suffix", "_label_point.png"),
        pseudo_dir=data_cfg.get("pseudo_dir"),
        transform=tfm,
        r1=dawn_cfg.get("r1", 11),
        r2=dawn_cfg.get("r2", 22),
        sigma=dawn_cfg.get("sigma", 2.75),
    )
    loader = DataLoader(
        ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    model = DAWN(
        detector_backbone=cfg.get("model", {}).get("detector_backbone", "resnet34"),
        detector_pretrained=cfg.get("model", {}).get("detector_pretrained", False),
        hover_mode=cfg.get("model", {}).get("hover_mode", "fast"),
    ).to(device)
    if cfg.get("source_checkpoint"):
        print(f"Loading segmentor/source checkpoint: {cfg['source_checkpoint']}")
        model.load_segmentor(cfg["source_checkpoint"], strict=False)
    if cfg.get("resume"):
        load_checkpoint(model, cfg["resume"], strict=False)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", True) and device.type == "cuda")
    alpha = dawn_cfg.get("alpha", dawn_cfg.get("alpha_cfc", 0.1))
    beta = dawn_cfg.get("beta", dawn_cfg.get("beta_dyn", 0.15))
    monitor = train_cfg.get("monitor", "val_pq")
    best_score = None
    history = []
    val_interval = int(train_cfg.get("val_interval", 1))

    for epoch in range(1, train_cfg.get("epochs", 80) + 1):
        model.train()
        totals = {"loss": 0.0, "l_det": 0.0, "l_cfc": 0.0, "l_dyn": 0.0}
        for batch in tqdm(loader, desc=f"target epoch {epoch}"):
            img = batch["image"].to(device)
            gaussian = batch["gaussian"].to(device)
            valid = batch["valid"].to(device)
            pseudo = batch["pseudo"].to(device)
            has_pseudo = batch.get("has_pseudo")
            has_pseudo = has_pseudo.to(device) if has_pseudo is not None else torch.zeros_like(pseudo)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(img)
                det_prob = out["det_prob"]
                if det_prob.shape[-2:] != gaussian.shape[-2:]:
                    gaussian_i = F.interpolate(gaussian, size=det_prob.shape[-2:], mode="bilinear", align_corners=False)
                    valid_i = F.interpolate(valid, size=det_prob.shape[-2:], mode="nearest")
                else:
                    gaussian_i, valid_i = gaussian, valid
                weights = torch.ones_like(gaussian_i)
                weights[gaussian_i > 0] = dawn_cfg.get("foreground_weight", 10.0)
                l_det = masked_mse(det_prob, gaussian_i, mask=valid_i, weight=weights)
                l_cfc = cfc_loss(out["seg_encoding"], out["det_encoding"])
                l_dyn = dynamic_loss(out["np"], det_prob, pseudo, has_pseudo=has_pseudo)
                loss = l_det + alpha * l_cfc + beta * l_dyn
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            totals["loss"] += float(loss.item())
            totals["l_det"] += float(l_det.item())
            totals["l_cfc"] += float(l_cfc.item())
            totals["l_dyn"] += float(l_dyn.item())

        row = {"epoch": epoch}
        row.update({k: v / max(1, len(loader)) for k, v in totals.items()})
        val_metrics = None
        if val_interval > 0 and epoch % val_interval == 0:
            val_metrics = evaluate_model_on_folder(model, cfg, device, split=cfg.get("validation", {}).get("split", "val"))
            if val_metrics:
                row.update({f"val_{k}": v for k, v in val_metrics.items() if k != "num_images"})
                row["val_num_images"] = val_metrics.get("num_images", 0)
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        plot_history_csv(out_dir / "history.csv", out_dir, prefix="target")

        print("epoch summary:", row)
        save_checkpoint(out_dir / "checkpoint_last.pt", model, opt, epoch, {"history": history, **row})

        current = row.get(monitor)
        if current is None:
            current = row["loss"]
            mode_is_max = False
        else:
            mode_is_max = monitor.startswith("val_")
        if best_score is None:
            improved = True
        else:
            improved = current > best_score if mode_is_max else current < best_score
        if improved:
            best_score = current
            save_checkpoint(out_dir / "checkpoint_best.pt", model, opt, epoch, {"history": history, "best_score": best_score, "monitor": monitor, **row})

    result = {
        "output_dir": str(out_dir),
        "best_checkpoint": str(out_dir / "checkpoint_best.pt"),
        "last_checkpoint": str(out_dir / "checkpoint_last.pt"),
        "monitor": monitor,
        "best_score": float(best_score),
        "history_csv": str(out_dir / "history.csv"),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_target(cfg)


if __name__ == "__main__":
    main()
