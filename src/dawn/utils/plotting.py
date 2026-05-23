from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_history_csv(csv_path: str | Path, out_dir: str | Path | None = None, prefix: str = "training"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if "loss" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["loss"], marker="o")
        for col in ["l_det", "l_cfc", "l_dyn"]:
            if col in df.columns:
                plt.plot(df["epoch"], df[col], marker="o", label=col)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend() if any(c in df.columns for c in ["l_det", "l_cfc", "l_dyn"]) else None
        plt.tight_layout()
        path = out_dir / f"{prefix}_loss.png"
        plt.savefig(path, dpi=200)
        plt.close()
        saved.append(path)

    metric_cols = [c for c in ["val_dice", "val_aji", "val_dq", "val_sq", "val_pq"] if c in df.columns and df[c].notna().any()]
    if metric_cols:
        plt.figure()
        for col in metric_cols:
            plt.plot(df["epoch"], df[col], marker="o", label=col.replace("val_", ""))
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"{prefix}_metrics.png"
        plt.savefig(path, dpi=200)
        plt.close()
        saved.append(path)
    return saved


def plot_round_summary(csv_path: str | Path, out_dir: str | Path | None = None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
    saved = []
    for ycols, name, ylabel in [
        (["best_loss"], "round_loss", "Best validation/train loss"),
        (["best_val_dice", "best_val_aji", "best_val_dq", "best_val_sq", "best_val_pq"], "round_metrics", "Best validation metric"),
    ]:
        cols = [c for c in ycols if c in df.columns and df[c].notna().any()]
        if not cols:
            continue
        plt.figure()
        for col in cols:
            plt.plot(df["round"], df[col], marker="o", label=col.replace("best_val_", ""))
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        if len(cols) > 1:
            plt.legend()
        plt.tight_layout()
        path = out_dir / f"{name}.png"
        plt.savefig(path, dpi=200)
        plt.close()
        saved.append(path)
    return saved
