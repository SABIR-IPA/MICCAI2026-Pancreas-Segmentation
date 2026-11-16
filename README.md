# Reducing False Positives in Pancreatic Tumor Segmentation via Multi-Scale Texture Enhancement and Curriculum Learning

**Accepted at MICCAI 2026, Machine Learning in Medical Imaging (MLMI) Workshop**

This repository contains the core implementation of our proposed method for pancreatic tumor segmentation in 3D CT images.

The method focuses on improving tumor segmentation by combining multi-scale texture enhancement with curriculum learning to reduce false-positive predictions.

## Method Overview

The proposed framework uses a SwinUNETR-based architecture and a two-phase training strategy:

1. **Phase 1: Recall-oriented adaptation**
   - Fine-tunes a pretrained baseline model.
   - Prioritizes tumor detection.
   - Disables the false-positive penalty.

2. **Phase 2: Curriculum-based precision optimization**
   - Introduces false-positive control during training.
   - Uses hard-case training to improve segmentation reliability.
   - Aims to reduce false positives while preserving tumor detection.

## Repository Structure

```text
MICCAI2026-Pancreas-Segmentation/
├── main_train.py       # Training and curriculum learning
├── preprocessing.py    # CT preprocessing and data loading
├── utilities.py        # Model, loss functions, and utilities
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## Requirements

- PyTorch 2.5.1
- MONAI 1.3.0
- NumPy 1.24.3
- NiBabel 5.1.0
- SciPy 1.11.1

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The dataset should contain:

```text
dataset/
├── imagesTr/
└── labelsTr/
```

A JSON file defining the training, validation, test, and hard-case splits is required.

A pretrained baseline SwinUNETR checkpoint is also required.

## Training

```bash
python main_train.py \
    --data_dir /path/to/dataset \
    --baseline_ckpt /path/to/baseline_checkpoint.pth \
    --data_splits /path/to/splits.json \
    --out_root ./results \
    --device cuda:0
```

## Publication

**Reducing False Positives in Pancreatic Tumor Segmentation via Multi-Scale Texture Enhancement and Curriculum Learning**

MICCAI 2026, Machine Learning in Medical Imaging (MLMI) Workshop.

## Citation

**Muhammad Sabir, Yuichiro Hayashi, Masahiro Oda, and Kensaku Mori.**

*Reducing False Positives in Pancreatic Tumor Segmentation via
Multi-Scale Texture Enhancement and Curriculum Learning.*

Accepted at the MICCAI 2026 Machine Learning in Medical Imaging
(MLMI) Workshop.

Official BibTeX citation and publication link will be added
when verified.

## Contact

Email: sabir.ipa@gmail.com
