#!/usr/bin/env bash

ratio='1.0'
dataset='MO'
repeat=3
# detection
python main.py --random-seed 2025 --lr 0.00001 --batch-size 4 --epochs 10 \
  --gpus 6  --root-save-dir ../result/${dataset}/${ratio}_repeat=${repeat}

# python test.py --img-dir ../data_for_train/MO/images/test --label-dir ../data/MO/labels_point \
#  --model-path ../experiments/detection/MO/1.0_repeat=3/2/checkpoints/checkpoint_best.pth.tar \
#  --threshold 0.35 --save-dir ../experiments/detection/MO/1.0_repeat=3/iter=2_ob_thre=0.35



