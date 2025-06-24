# DAWN: Domain-Adaptive Weakly Supervised Nuclei Segmentation

# The code still be orginization ... Please wait

This repository contains the official implementation of **DAWN**, a Domain-Adaptive Weakly Supervised Nuclei Segmentation via Cross-Task Interactions.



## 🔍 Overview

DAWN addresses the challenge of cell nuclei segmentation under domain shift and weak annotations. It leverages point-level supervision and introduces a cross-task interaction mechanism to enhance generalization across different staining styles and image sources.

<p align="center">
  <img src="assets/demo.png" alt="Demo" width="600"/>
</p>

## 📁 Project Structure

DAWN/
├── configs/ # Configuration files
├── data/ # Datasets or data loaders
├── models/ # Model architectures
├── trainers/ # Training logic
├── utils/ # Utility functions
├── main.py # Entry point for training/evaluation
└── README.md

bash
复制
编辑

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/zhangye-zoe/DAWN.git
cd DAWN
2. Create and activate environment
bash
复制
编辑
conda create -n dawn python=3.8 -y
conda activate dawn
pip install -r requirements.txt
3. Prepare the dataset
Place your dataset under the data/ directory. The expected structure is:

kotlin
复制
编辑
data/
└── DatasetName/
    ├── images/
    └── annotations/
4. Training
bash
复制
编辑
python main.py --config configs/train_config.yaml
5. Evaluation
bash
复制
编辑
python main.py --config configs/eval_config.yaml
📊 Results
Method	Dice	AJI	F1-score
DAWN	0.83	0.70	0.88

See our paper for full benchmark comparisons.

📄 Citation
If you use this code, please cite our paper:

bibtex
复制
编辑
@article{zhang2025dawn,
  title={DAWN: Domain-Adaptive Weakly Supervised Nuclei Segmentation via Cross-Task Interactions},
  author={Zhang, Ye and ...},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
🤝 Acknowledgements
This project was developed at Harbin Institute of Technology, Shenzhen. We thank our collaborators for helpful discussions.

📬 Contact
If you have any questions, feel free to contact:

Ye Zhang: zhangye-hit@xxx.com

GitHub