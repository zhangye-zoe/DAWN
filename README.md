# DAWN: Domain-Adaptive Weakly Supervised Nuclei Segmentation

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2%2Bcu121-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Official implementation of **DAWN**, a weakly supervised nuclei segmentation framework that transfers source-domain segmentation priors to target-domain point-annotated data through cross-task interactions.

- **Paper Link**: [Read the Paper](https://ieeexplore.ieee.org/abstract/document/10798459)
- **Publication**: IEEE Transactions on Circuits and Systems for Video Technology
- **Authors**: Ye Zhang, Yifeng Wang, Zijie Fang, Hao Bian, Linghan Cai, Ziyue Wang, and Yongbing Zhang

---

## ✨ Highlights

- Point-supervised target-domain nuclei segmentation.
- Source-domain HoVerNet pretraining with full instance masks.
- Bootstrap pseudo-label generation on target-domain images.
- Cross-task interaction via **CFC**, **CPL**, and **IST**.
- Automatic multi-round target training, inference, metrics, and visualization.
- Evaluation with **DICE, AJI, DQ, SQ, and PQ**.

---

## 🧠 Framework Overview

DAWN follows a source-to-target training pipeline:

```text
Source-domain segmentation pretraining
        ↓
Bootstrap pseudo-label generation
        ↓
Round0 target training with IST
        ↓
CPL pseudo-label update
        ↓
Round1 / Round2 target refinement
        ↓
Inference and evaluation
```

Sparse point labels are **not** directly used as segmentation masks. DAWN first generates target-domain bootstrap pseudo masks using the source-pretrained segmentation network, filters them with point annotations, and then progressively updates pseudo labels by fusing segmentation predictions, detection predictions, and point annotations.

---

## 🛠️ Installation

The current implementation was tested with:

```bash
# CUDA compiler
nvcc -V
# Cuda compilation tools, release 12.2

# PyTorch runtime
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# torch 2.1.2+cu121, CUDA runtime 12.1
```

Recommended installation:

```bash
conda create -n dawn python=3.10 -y
conda activate dawn

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -e .
```

`requirements.txt` does not include `torch`, `torchvision`, or `torchaudio`, so it will not overwrite the CUDA-enabled PyTorch installation.

If NumPy/SciPy compatibility issues occur:

```bash
pip uninstall -y numpy scipy
pip install numpy==1.26.4 scipy==1.11.4
```

---

## 📁 Data Preparation

Target-domain data should be organized as:

```text
datasets/TNBC_prepared/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels_point/
│   ├── train/
│   ├── val/
│   └── test/
└── labels_instance/          # optional; required for validation/test metrics
    ├── val/
    └── test/
```

Example naming:

```text
images/train/0.png
labels_point/train/0_label_point.png
labels_instance/test/0.png
```

The image/point-label naming mismatch is handled by:

```yaml
point_suffix: _label_point.png
```

Split TNBC with `train_val_test.json`:

```bash
python tools/split_tnbc_by_json.py \
  --image-dir /path/to/TNBC/images \
  --point-dir /path/to/TNBC/labels_point \
  --true-mask-dir /path/to/TNBC/true_mask \
  --split-json /path/to/train_val_test.json \
  --out datasets/TNBC_prepared \
  --point-suffix _label_point.png \
  --true-mask-suffix _true_mask.png
```

---

## 🚀 Source-Domain Pretraining

Edit:

```text
configs/source_pannuke.yaml
```

Run:

```bash
bash scripts/train_source.sh configs/source_pannuke.yaml
```

Expected outputs:

```text
outputs/source_pannuke/checkpoint_best.pt
outputs/source_pannuke/checkpoint_last.pt
```

---

## 🧩 Bootstrap Pseudo Labels

Generate initial target-domain pseudo masks using the source-pretrained HoVerNet:

```bash
bash scripts/generate_bootstrap_pseudo.sh configs/bootstrap_pseudo_tnbc.yaml
```

Expected outputs:

```text
outputs/dawn_tnbc_bootstrap_pseudo/
├── seg_prob/
├── source_mask/
├── pseudo/
├── pseudo_stats.csv
└── summary.json
```

The `pseudo/` masks supervise both segmentation and detection networks in the first target-domain training round.

---

## 🔁 Target-Domain Training

Run the full target-domain pipeline:

```bash
bash scripts/run_target_rounds.sh configs/pipeline_tnbc.yaml
```

This performs:

```text
bootstrap pseudo-label generation
→ round0 target training
→ round0 CPL pseudo-label update
→ round1 target training
→ round1 CPL pseudo-label update
→ round2 target training
```

Expected outputs:

```text
outputs/dawn_tnbc_pipeline/
├── bootstrap_pseudo/
├── round0/
├── round1/
├── round2/
├── round_summary.csv
├── round_loss.png
├── round_metrics.png
└── pipeline_summary.json
```

---

## 🔍 Inference and Evaluation

Edit `configs/infer_tnbc_test.yaml` and set the final checkpoint, for example:

```yaml
checkpoint: outputs/dawn_tnbc_pipeline/round2/train/checkpoint_best.pt
```

Run inference:

```bash
bash scripts/infer.sh configs/infer_tnbc_test.yaml
```

Expected outputs:

```text
outputs/dawn_tnbc_test/
├── seg_prob/
├── det_prob/
├── pred_mask/
├── pred_inst/
└── vis/
```

Evaluate test-set predictions:

```bash
python tools/evaluate_instance.py \
  --pred-dir outputs/dawn_tnbc_test/pred_inst \
  --gt-dir datasets/TNBC_prepared/labels_instance/test \
  --pred-suffix .png \
  --gt-suffixes .png \
  --out outputs/dawn_tnbc_test/test_metrics.csv
```

---

## 📜 Main Scripts

```text
scripts/train_source.sh                  Source-domain segmentation pretraining
scripts/generate_bootstrap_pseudo.sh     Bootstrap target pseudo-label generation
scripts/train_target.sh                  Single target-domain training round
scripts/run_target_rounds.sh             Automatic multi-round target pipeline
scripts/infer.sh                         Inference and visualization
scripts/evaluate.sh                      Metric calculation
```

---

## 🙏 Acknowledgements

This implementation is built upon and inspired by the following open-source projects:

- [HoVer-Net](https://github.com/vqdang/hover_net): used as the segmentation backbone and source-domain fully supervised nuclei segmentation framework. We follow the HoVerNet-style target generation, including nuclei foreground and horizontal/vertical distance maps.
- [WeaklySegPointAnno](https://github.com/huiqu18/WeaklySegPointAnno): used as an important reference for weakly supervised nuclei segmentation with point annotations, pseudo-label generation, and staged training.

We thank the authors for making their implementations publicly available.

---

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@article{zhang2025dawn,
  title   = {DAWN: Domain-Adaptive Weakly Supervised Nuclei Segmentation via Cross-Task Interactions},
  author  = {Zhang, Ye and Wang, Yifeng and Fang, Zijie and Bian, Hao and Cai, Linghan and Wang, Ziyue and Zhang, Yongbing},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2025},
  doi     = {10.1109/TCSVT.2024.3515467}
}
```
