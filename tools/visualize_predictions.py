from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from skimage import io, segmentation
from PIL import Image
from tqdm import tqdm

IMG_EXTS={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}

def read_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))

def read_gray(path):
    arr=np.asarray(Image.open(path))
    if arr.ndim==3: arr=arr[...,0]
    return arr

def find_file(d, stem, suffixes):
    if not d: return None
    d=Path(d)
    for s in suffixes:
        p=d/f'{stem}{s}'
        if p.exists(): return p
    return None

def main():
    ap=argparse.ArgumentParser(description='Create boundary overlays for predictions, GT, and point labels.')
    ap.add_argument('--image-dir', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--gt-dir')
    ap.add_argument('--point-dir')
    ap.add_argument('--pred-suffix', default='_pred_inst.png')
    ap.add_argument('--gt-suffixes', default='_label.png,_inst.png,.png')
    ap.add_argument('--point-suffix', default='_label_point.png')
    args=ap.parse_args()
    image_dir=Path(args.image_dir); pred_dir=Path(args.pred_dir); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gt_suffixes=args.gt_suffixes.split(',')
    for img_path in tqdm(sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])):
        stem=img_path.stem
        pred_path=pred_dir/f'{stem}{args.pred_suffix}'
        if not pred_path.exists():
            pred_path=pred_dir/f'{stem}_pred.png'
        if not pred_path.exists():
            continue
        img=read_rgb(img_path).astype(np.float32)/255.0
        pred=read_gray(pred_path)>0
        overlay=segmentation.mark_boundaries(img, pred, color=(1,0,0), mode='thick')
        gt_path=find_file(args.gt_dir, stem, gt_suffixes)
        if gt_path:
            overlay=segmentation.mark_boundaries(overlay, read_gray(gt_path)>0, color=(0,1,0), mode='thick')
        point_path=find_file(args.point_dir, stem, [args.point_suffix, '.png'])
        if point_path:
            point=read_gray(point_path)>0
            ys,xs=np.nonzero(point)
            for y,x in zip(ys,xs):
                overlay[max(0,y-2):y+3, max(0,x-2):x+3]=np.array([0,0.3,1])
        io.imsave(out_dir/f'{stem}_overlay.png', (overlay*255).clip(0,255).astype(np.uint8), check_contrast=False)

if __name__=='__main__':
    main()
