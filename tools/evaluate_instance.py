from __future__ import annotations
import argparse
import json
from pathlib import Path
from dawn.evaluation.metrics import evaluate_prediction_dir


def main():
    p = argparse.ArgumentParser(description="Evaluate instance segmentation masks with DICE, AJI, DQ, SQ, and PQ.")
    p.add_argument("--pred-dir", required=True, help="Prediction instance directory, e.g. outputs/.../pred_inst")
    p.add_argument("--gt-dir", required=True, help="Ground-truth instance directory")
    p.add_argument("--pred-suffix", default="_pred_inst.png")
    p.add_argument("--gt-suffixes", default="_label.png,_inst.png,.png")
    p.add_argument("--pred-binary", action="store_true")
    p.add_argument("--min-area", type=int, default=12)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--out", default=None, help="Output CSV path")
    args = p.parse_args()
    out_csv = args.out or str(Path(args.pred_dir).parent / "metrics_per_image.csv")
    summary = evaluate_prediction_dir(
        args.pred_dir,
        args.gt_dir,
        pred_suffix=args.pred_suffix,
        gt_suffixes=tuple(args.gt_suffixes.split(",")),
        pred_binary=args.pred_binary,
        min_area=args.min_area,
        match_iou=args.match_iou,
        save_csv=out_csv,
    )
    summary_path = Path(out_csv).with_name(Path(out_csv).stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
