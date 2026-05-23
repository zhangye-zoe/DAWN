from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import argparse
import json
import yaml

import pandas as pd

from dawn.postprocess.infer import predict_folder
from dawn.postprocess.bootstrap_pseudo import generate_bootstrap_pseudo
from dawn.training.target_train import train_target
from dawn.utils.plotting import plot_round_summary


def run_rounds(cfg: dict) -> dict:
    pipe = cfg.get("pipeline", {})
    rounds = int(pipe.get("rounds", 3))
    root_output_dir = Path(pipe.get("root_output_dir", cfg.get("output_dir", "outputs/dawn_pipeline")))
    root_output_dir.mkdir(parents=True, exist_ok=True)

    target_base = deepcopy(cfg["target_train"])
    infer_base = deepcopy(cfg["pseudo_infer"])
    initial_checkpoint = pipe.get("initial_source_checkpoint", target_base.get("source_checkpoint"))
    if not initial_checkpoint:
        raise ValueError("Please set pipeline.initial_source_checkpoint or target_train.source_checkpoint")

    current_checkpoint = initial_checkpoint
    current_pseudo_dir = pipe.get("initial_pseudo_dir")

    # Optional but recommended: before round0 target training, run the source-domain
    # HoVerNet/segmentor on target training images and build initial pseudo masks.
    # This prevents round0 from using sparse point maps as segmentation targets.
    bootstrap_cfg = cfg.get("bootstrap_pseudo")
    bootstrap_info = None
    if bootstrap_cfg and bootstrap_cfg.get("enabled", True) and current_pseudo_dir is None:
        print("\n========== DAWN bootstrap pseudo labels from source HoVerNet ==========")
        bootstrap_cfg = deepcopy(bootstrap_cfg)
        bootstrap_cfg.setdefault("checkpoint", initial_checkpoint)
        bootstrap_cfg.setdefault("output_dir", str(root_output_dir / "bootstrap_pseudo"))
        bootstrap_path = root_output_dir / "bootstrap_pseudo.yaml"
        bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_path.write_text(yaml.safe_dump(bootstrap_cfg, sort_keys=False))
        bootstrap_info = generate_bootstrap_pseudo(bootstrap_cfg)
        current_pseudo_dir = bootstrap_info["pseudo_dir"]

    rows = []

    for r in range(rounds):
        print(f"\n========== DAWN target round {r}/{rounds - 1} ==========")
        round_dir = root_output_dir / f"round{r}"
        train_cfg = deepcopy(target_base)
        train_cfg["output_dir"] = str(round_dir / "train")
        if r == 0:
            # Round0 initializes only the segmentation network from source HoVerNet.
            train_cfg["source_checkpoint"] = str(initial_checkpoint)
            train_cfg.pop("resume", None)
        else:
            # Later rounds should continue both segmentation and detection networks.
            # Therefore we resume the full DAWN checkpoint from the previous round.
            train_cfg["source_checkpoint"] = None
            train_cfg["resume"] = str(current_checkpoint)
        train_cfg.setdefault("data", {})["pseudo_dir"] = str(current_pseudo_dir) if current_pseudo_dir else None
        train_cfg_path = round_dir / "target_train.yaml"
        train_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        train_cfg_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False))

        train_result = train_target(train_cfg)
        best_ckpt = train_result["best_checkpoint"]

        row = {"round": r, "train_dir": train_result["output_dir"], "best_checkpoint": best_ckpt, "best_loss": None}
        hist_path = Path(train_result["history_csv"])
        if hist_path.exists():
            hist = pd.read_csv(hist_path)
            if "loss" in hist.columns:
                row["best_loss"] = float(hist["loss"].min())
            for m in ["dice", "aji", "dq", "sq", "pq"]:
                col = f"val_{m}"
                if col in hist.columns and hist[col].notna().any():
                    row[f"best_val_{m}"] = float(hist[col].max())

        if r < rounds - 1:
            infer_cfg = deepcopy(infer_base)
            infer_cfg["checkpoint"] = best_ckpt
            infer_cfg["output_dir"] = str(round_dir / "pseudo")
            infer_cfg_path = round_dir / "generate_pseudo.yaml"
            infer_cfg_path.write_text(yaml.safe_dump(infer_cfg, sort_keys=False))
            predict_folder(infer_cfg)
            current_pseudo_dir = str(Path(infer_cfg["output_dir"]) / "pseudo")
            row["pseudo_dir"] = current_pseudo_dir
            current_checkpoint = best_ckpt
        else:
            current_checkpoint = best_ckpt

        rows.append(row)
        pd.DataFrame(rows).to_csv(root_output_dir / "round_summary.csv", index=False)
        plot_round_summary(root_output_dir / "round_summary.csv", root_output_dir)

    summary = {"rounds": rows, "final_checkpoint": current_checkpoint, "output_dir": str(root_output_dir), "bootstrap": bootstrap_info}
    (root_output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Pipeline YAML file, e.g. configs/pipeline_tnbc.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_rounds(cfg)


if __name__ == "__main__":
    main()
