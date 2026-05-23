from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def link_or_copy(src: Path, dst: Path, copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split TNBC images, point labels, and optional true masks "
            "according to train_val_test.json."
        )
    )

    parser.add_argument(
        "--image-dir",
        required=True,
        help="Original image directory, e.g. datasets/target_data/TNBC/images",
    )
    parser.add_argument(
        "--point-dir",
        required=True,
        help="Original point label directory, e.g. datasets/target_data/TNBC/labels_point",
    )
    parser.add_argument(
        "--true-mask-dir",
        default=None,
        help="Optional true instance mask directory, e.g. datasets/target_data/TNBC/true_mask",
    )
    parser.add_argument(
        "--split-json",
        required=True,
        help="Path to train_val_test.json",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output prepared dataset root",
    )
    parser.add_argument(
        "--point-suffix",
        default="_label_point.png",
        help="Point label suffix. Example: 0.png -> 0_label_point.png",
    )
    parser.add_argument(
        "--true-mask-suffix",
        default="_true_mask.png",
        help="True mask suffix. Example: 0.png -> 0_true_mask.png",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    point_dir = Path(args.point_dir)
    true_mask_dir = Path(args.true_mask_dir) if args.true_mask_dir else None
    split_json = Path(args.split_json)
    out = Path(args.out)

    with split_json.open("r") as f:
        split_dict = json.load(f)

    total_images = 0
    total_points = 0
    total_true_masks = 0
    missing = []

    for split, names in split_dict.items():
        for image_name in names:
            image_name = Path(image_name).name
            stem = Path(image_name).stem

            image_path = image_dir / image_name
            point_path = point_dir / f"{stem}{args.point_suffix}"

            if not image_path.exists():
                missing.append(str(image_path))
                continue

            if not point_path.exists():
                missing.append(str(point_path))
                continue

            link_or_copy(
                image_path,
                out / "images" / split / image_name,
                copy=args.copy,
            )
            total_images += 1

            link_or_copy(
                point_path,
                out / "labels_point" / split / f"{stem}{args.point_suffix}",
                copy=args.copy,
            )
            total_points += 1

            if true_mask_dir is not None:
                true_mask_path = true_mask_dir / f"{stem}{args.true_mask_suffix}"

                if not true_mask_path.exists():
                    missing.append(str(true_mask_path))
                    continue

                # 这里统一保存成 stem.png，方便后续 pred_mask/GT 按同名匹配
                link_or_copy(
                    true_mask_path,
                    out / "labels_instance" / split / f"{stem}.png",
                    copy=args.copy,
                )
                total_true_masks += 1

    (out / "manifests").mkdir(parents=True, exist_ok=True)

    for split, names in split_dict.items():
        manifest = {
            "split": split,
            "num_images": len(names),
            "images": names,
            "point_suffix": args.point_suffix,
            "true_mask_suffix": args.true_mask_suffix if true_mask_dir else None,
        }

        with (out / "manifests" / f"{split}.json").open("w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Prepared dataset under: {out}")
    print(f"  images:      {total_images}")
    print(f"  point labels: {total_points}")
    if true_mask_dir is not None:
        print(f"  true masks:  {total_true_masks}")

    print("\nSplit summary:")
    for split, names in split_dict.items():
        image_count = len(list((out / "images" / split).glob("*")))
        point_count = len(list((out / "labels_point" / split).glob("*")))
        mask_count = (
            len(list((out / "labels_instance" / split).glob("*")))
            if (out / "labels_instance" / split).exists()
            else 0
        )

        print(
            f"  {split}: "
            f"images={image_count}, "
            f"points={point_count}, "
            f"true_masks={mask_count}"
        )

    if missing:
        print("\nMissing files:")
        for x in missing[:50]:
            print("  ", x)
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
        raise FileNotFoundError(f"{len(missing)} files are missing.")


if __name__ == "__main__":
    main()