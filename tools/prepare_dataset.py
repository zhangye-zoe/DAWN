from __future__ import annotations
from pathlib import Path
import argparse, shutil, json


def link_or_copy(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    if copy: shutil.copy2(src, dst)
    else: dst.symlink_to(src.resolve())


def main():
    p = argparse.ArgumentParser(description="Create DAWN standard data layout from existing folders.")
    p.add_argument("--images", required=True, help="Folder containing images")
    p.add_argument("--points", help="Folder containing point labels")
    p.add_argument("--instances", help="Folder containing instance masks for source pre-training")
    p.add_argument("--out", required=True, help="Output dataset root")
    p.add_argument("--split", default="train", choices=["train","val","test"])
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = p.parse_args()
    out = Path(args.out)
    imgs = sorted([x for x in Path(args.images).iterdir() if x.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}])
    for im in imgs:
        link_or_copy(im, out/"images"/args.split/im.name, args.copy)
    if args.points:
        for lab in Path(args.points).iterdir():
            if lab.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}:
                link_or_copy(lab, out/"labels_point"/args.split/lab.name, args.copy)
    if args.instances:
        for lab in Path(args.instances).iterdir():
            if lab.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}:
                link_or_copy(lab, out/"labels_instance"/args.split/lab.name, args.copy)
    manifest = {"split": args.split, "num_images": len(imgs), "images": [x.name for x in imgs]}
    (out/"manifests").mkdir(parents=True, exist_ok=True)
    (out/"manifests"/f"{args.split}.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(imgs)} images under {out}")

if __name__ == "__main__":
    main()
