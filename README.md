# DAWN: Domain-Adaptive Weakly Supervised Nuclei Segmentation

## The code still be orginization ... Please wait

This repository contains the official implementation of **DAWN**, a Domain-Adaptive Weakly Supervised Nuclei Segmentation via Cross-Task Interactions.



## 🔍 Overview

DAWN addresses the challenge of cell nuclei segmentation under domain shift and weak annotations. It leverages point-level supervision and introduces a cross-task interaction mechanism to enhance generalization across different staining styles and image sources.


## 📁 Project Structure

```bash
DAWN/
├── configs/ # Configuration files
├── data/ # Datasets or data loaders
├── models/ # Model architectures
├── trainers/ # Training logic
├── utils/ # Utility functions
├── main.py # Entry point for training/evaluation
└── README.md
```


## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/zhangye-zoe/DAWN.git
cd DAWN
```

### 2. Create and activate environment
```bash
conda create -n dawn python=3.8 -y
conda activate dawn
pip install -r requirements.txt
```

### 3. Prepare the dataset
Place your dataset under the data/ directory. The expected structure is:
```bash
data/
└── DatasetName/
    ├── images/
    └── annotations/
```
### 4. Training
```bash
python main.py --config configs/train_config.yaml
```
### 5. Evaluation
```bash
python main.py --config configs/eval_config.yaml
```


## 📄 Citation
If you use this code, please cite our paper:

```bash
@ARTICLE{10798459,
  author={Zhang, Ye and Wang, Yifeng and Fang, Zijie and Bian, Hao and Cai, Linghan and Wang, Ziyue and Zhang, Yongbing},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={DAWN: Domain-Adaptive Weakly Supervised Nuclei Segmentation via Cross-Task Interactions}, 
  year={2025},
  volume={35},
  number={5},
  pages={4753-4767},
  keywords={Image segmentation;Training;Annotations;Adaptation models;Accuracy;Optimization;Instance segmentation;Data models;Cams;Semantics;Nuclei instance segmentation;weakly supervised learning;domain adaptation},
  doi={10.1109/TCSVT.2024.3515467}}

```
