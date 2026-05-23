from __future__ import annotations
from pathlib import Path
import argparse, yaml, random, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# from dawn.data import SourceSegDataset, SourceSegNpyDataset, RandomTransform
from dawn.data import SourceSegDataset, SourceSegNpyDataset, HoVerNetAugmentor
from dawn.models import HoVerNet
from dawn.training.losses import hover_supervised_loss
from dawn.utils.checkpoint import save_checkpoint, load_checkpoint
from dawn.utils.plotting import plot_history_csv


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_source(cfg: dict):
    seed_everything(cfg.get("seed", 2024))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    aug_cfg = data_cfg.get("augment", {})
    use_augment = aug_cfg.get("enable", True)

    augmentor = None
    if use_augment:
        augmentor = HoVerNetAugmentor(
            input_shape=tuple(aug_cfg.get("input_shape", [256, 256])),
            mask_shape=tuple(aug_cfg.get("mask_shape", aug_cfg.get("input_shape", [256, 256]))),
            mode=aug_cfg.get("mode", "train"),
            seed=int(cfg.get("seed", 2024)),
            normalize=True,
            mean=tuple(data_cfg.get("mean", [0.5, 0.5, 0.5])),
            std=tuple(data_cfg.get("std", [0.5, 0.5, 0.5])),
        )

    data_format = data_cfg.get("format", "folder")

    if data_format == "npy":
        ds = SourceSegNpyDataset(
            image_npy=data_cfg["image_npy"],
            label_npy=data_cfg["label_npy"],
            label_channel=data_cfg.get("label_channel", None),
            augmentor=augmentor,
            mmap_mode=data_cfg.get("mmap_mode", "r"),
        )
    elif data_format == "folder":
        ds = SourceSegDataset(
            root=data_cfg["source_root"],
            split=data_cfg.get("split", "train"),
            image_dir=data_cfg.get("image_dir", "images"),
            label_dir=data_cfg.get("instance_label_dir", "labels_instance"),
            label_suffix=data_cfg.get("instance_label_suffix", "_label.png"),
            augmentor=augmentor,
            label_channel=data_cfg.get("label_channel", None),
        )
    else:
        raise ValueError(f"Unsupported source data format: {data_format}")


    loader = DataLoader(ds, batch_size=train_cfg.get("batch_size", 4), shuffle=True, num_workers=train_cfg.get("num_workers", 4), pin_memory=True)
    model = HoVerNet(mode=cfg.get("model", {}).get("hover_mode", "fast")).to(device)
    if cfg.get("resume"):
        load_checkpoint(model, cfg["resume"], strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 1e-4), weight_decay=train_cfg.get("weight_decay", 1e-4))
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", True) and device.type == "cuda")
    best = 1e9
    history = []
    for epoch in range(1, train_cfg.get("epochs", 50)+1):
        model.train(); total = 0.0
        for batch in tqdm(loader, desc=f"source epoch {epoch}"):
            img = batch["image"].to(device)
            np_t = batch["np"].to(device)
            hv_t = batch["hv"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(img)
                loss = hover_supervised_loss(out, np_t, hv_t)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item()
        avg = total / max(1, len(loader))
        row = {"epoch": epoch, "loss": avg}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        plot_history_csv(out_dir / "history.csv", out_dir, prefix="source")
        print(f"epoch={epoch} loss={avg:.5f}")
        save_checkpoint(out_dir / "checkpoint_last.pt", model, opt, epoch, {"loss": avg, "history": history})
        if avg < best:
            best = avg; save_checkpoint(out_dir / "checkpoint_best.pt", model, opt, epoch, {"loss": avg, "history": history})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    train_source(cfg)

if __name__ == "__main__":
    main()
