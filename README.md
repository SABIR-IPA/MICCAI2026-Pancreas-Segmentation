# Reducing False Positives in Pancreatic Tumor Segmentation via Multi-Scale Texture Enhancement and Curriculum Learning

**Accepted at MICCAI 2026, Machine Learning in Medical Imaging (MLMI) Workshop**

This repository contains the core implementation of our proposed method for pancreatic tumor segmentation in 3D CT images.

The method focuses on improving tumor segmentation by combining multi-scale texture enhancement with curriculum learning to reduce false-positive predictions.

## Method Overview
This project demonstrates strong understanding of medical AI, 3D computer vision, and deep learning engineering.
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

## Reproducibility and Evaluation Protocol

### Dataset and splits

The implementation expects paired NIfTI CT images and segmentation masks in the `imagesTr/` and `labelsTr/` directories. Image and label filenames must share the same case identifier. macOS AppleDouble metadata files beginning with `._` are excluded.

The predefined MSD Task07 split contains:

| Group | Cases |
|---|---:|
| Training (`train_clean`) | 195 |
| Validation (`val_clean`) | 22 |
| Independent test (`test_clean`) | 55 |
| Hard cases (`hard_cases`) | 9 |

All 281 cases have matching image-label filenames. The split groups contain no duplicate identifiers or overlaps.

The training script requires a JSON split manifest containing the four keys listed above.

### Pretrained checkpoint

Training requires a compatible pretrained SwinUNETR checkpoint supplied through `--baseline_ckpt`. The checkpoint is not included in this repository.

The checkpoint loader validates backbone parameter coverage and segmentation-head compatibility. Exact reproduction requires the original pretrained checkpoint.

### Evaluation protocol and limitations

**Hard-case exposure:** The nine hard cases are incorporated into Phase 2 training. Results on these cases are training-exposed diagnostics and must not be interpreted as independent test performance.

**Postprocessing selection:** Postprocessing parameters are selected using the validation group, separately from the independent test group.

### Reproduction requirements

Exact reproduction requires the MSD Task07 dataset, the predefined split manifest, the original baseline checkpoint, and a compatible software environment. These data and checkpoint artifacts are not included in this repository.
