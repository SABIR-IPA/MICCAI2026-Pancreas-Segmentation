import os
import json
from glob import glob

import numpy as np
import nibabel as nib

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstD, Spacingd, Orientationd,
    CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandRotated, RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianNoised, ToTensord, MapTransform, DeleteItemsd, RandZoomd,
)
from monai.data import CacheDataset, Dataset, DataLoader
from monai.utils import set_determinism

# ============================================================
# Constants (used throughout the pipeline)
# ============================================================
PIXDIM = (1.0, 1.0, 1.0)
PATCH_SIZE = (96, 96, 96)
WINDOWS = [(-100.0, 240.0), (-400.0, 240.0)]
SEED = 42

TUMOR_LABEL = 2
ORGAN_LABEL = 1
TUMOR_CHANNEL = 2
NUM_CLASSES = 3

# Oversampling thresholds and factors
TINY_TUMOR_THRESHOLD = 1500
SMALL_TUMOR_THRESHOLD = 3000
LOW_RATIO_THRESHOLD = 0.015
OVERSAMPLE_TINY = 4
OVERSAMPLE_SMALL = 3
OVERSAMPLE_MEDIUM = 2
OVERSAMPLE_NORMAL = 1
HARD_REPEAT_PHASE2 = 1

# Training hyperparameters
MAX_EPOCHS_PHASE1 = 80
PATIENCE_PHASE1 = 20
MAX_EPOCHS_PHASE2 = 40
PATIENCE_PHASE2 = 10

WEIGHT_DECAY = 2e-4
OVERLAP = 0.5

# Learning rates for three parameter groups
LR_BACKBONE_P1 = 5e-6
LR_REST_P1 = 1e-4
LR_NEW_P1 = 3e-4

LR_BACKBONE_P2 = 0.0
LR_REST_P2 = 1e-5
LR_NEW_P2 = 2e-4

# Post‑processing grid for final tuning (larger set)
THR_LIST = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_VOX_LIST = [10, 30, 50, 100, 150, 200, 300, 400]

# Smaller grid used during validation for checkpoint selection
VAL_THR_LIST = [0.35, 0.45, 0.50, 0.55, 0.60, 0.70]
VAL_MIN_VOX_LIST = [10, 30, 50, 100, 200, 400]
VAL_PAN_DILATE = 2
VAL_MAX_LESIONS = 2


# ============================================================
# Data analysis and indexing
# ============================================================
def analyze_label(label_path, tumor_label=2):
    seg = nib.load(label_path).get_fdata()
    tvox = int((seg == tumor_label).sum())
    ovox = int(((seg == 1) | (seg == 2)).sum())
    ratio = float(tvox / ovox) if ovox > 0 else 0.0
    return tvox, ovox, ratio, (tvox > 0)


def build_case_index(data_dir):
    images = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    assert len(images) == len(labels) > 0
    idx = {}
    for img, lbl in zip(images, labels):
        case = os.path.basename(lbl).replace(".nii.gz", "")
        tvox, ovox, ratio, has_tum = analyze_label(lbl, TUMOR_LABEL)
        idx[case] = {
            "case": case,
            "vol": img,
            "seg": lbl,
            "tumor_size": tvox,
            "organ_voxels": ovox,
            "ratio": ratio,
            "has_tumor": has_tum,
        }
    return idx


def load_splits(splits_path):
    with open(splits_path) as f:
        splits = json.load(f)
    for k in ("train_clean", "val_clean", "test_clean", "hard_cases"):
        if k not in splits:
            raise KeyError(f"splits.json missing required key: {k}")
    print("\n" + "=" * 80)
    print("Loaded splits from paper (verbatim, no regeneration)")
    print("=" * 80)
    print(f"  train_clean : {len(splits['train_clean'])}")
    print(f"  val_clean   : {len(splits['val_clean'])}")
    print(f"  test_clean  : {len(splits['test_clean'])}")
    print(f"  hard_cases  : {len(splits['hard_cases'])}")
    print("=" * 80)
    return splits


def files_from_case_list(case_index, names):
    out, missing = [], []
    for n in names:
        if n in case_index:
            out.append(dict(case_index[n]))
        else:
            missing.append(n)
    if missing:
        print(f"WARNING: {len(missing)} cases missing on disk: {missing[:5]}...")
    return out


def oversample_train(files, hard_set=None, hard_repeat=1):
    hard_set = set(hard_set) if hard_set is not None else set()
    out = []
    for f in files:
        if not f["has_tumor"]:
            r = 1
        elif f["tumor_size"] < TINY_TUMOR_THRESHOLD:
            r = OVERSAMPLE_TINY
        elif f["tumor_size"] < SMALL_TUMOR_THRESHOLD or f["ratio"] < LOW_RATIO_THRESHOLD:
            r = OVERSAMPLE_SMALL
        elif f["tumor_size"] < 10000:
            r = OVERSAMPLE_MEDIUM
        else:
            r = OVERSAMPLE_NORMAL
        if f["case"] in hard_set:
            r *= int(hard_repeat)
        for _ in range(r):
            out.append(dict(f))
    return out


# ============================================================
# Custom MONAI transforms
# ============================================================
class MultiWindowIntensityd(MapTransform):
    def __init__(self, keys, windows):
        super().__init__(keys)
        self.windows = windows

    @staticmethod
    def _scale(ct, a_min, a_max):
        return ((np.clip(ct, a_min, a_max) - a_min) / (a_max - a_min + 1e-8)).astype(np.float32)

    def __call__(self, data):
        d = dict(data)
        vol = d["vol"]
        if vol.ndim != 4 or vol.shape[0] != 1:
            raise ValueError(f"Expected vol (1,D,H,W), got {vol.shape}")
        ct = vol[0].astype(np.float32)
        d["vol"] = np.stack([self._scale(ct, w[0], w[1]) for w in self.windows], axis=0)
        return d


class CreateTumorMaskd(MapTransform):
    def __init__(self, keys, seg_key="seg", out_key="tumor_mask", tumor_label=2):
        super().__init__(keys)
        self.seg_key, self.out_key, self.tumor_label = seg_key, out_key, tumor_label

    def __call__(self, data):
        d = dict(data)
        d[self.out_key] = (d[self.seg_key] == self.tumor_label).astype(d[self.seg_key].dtype)
        return d


class CreatePancreasHardNegIndicesd(MapTransform):
    def __init__(self, seg_key="seg", fg_key="fg_idx", bg_key="bg_idx",
                 organ_label=1, tumor_label=2):
        super().__init__([seg_key])
        self.seg_key = seg_key
        self.fg_key, self.bg_key = fg_key, bg_key
        self.organ_label, self.tumor_label = organ_label, tumor_label

    def __call__(self, data):
        d = dict(data)
        seg = d[self.seg_key]
        seg3 = seg[0] if (seg.ndim == 4 and seg.shape[0] == 1) else seg
        tumor = (seg3 == self.tumor_label)
        organ = (seg3 == self.organ_label)
        bg = (seg3 == 0)
        fg_mask = tumor if tumor.any() else organ
        bg_mask = organ if organ.any() else bg
        d[self.fg_key] = np.where(fg_mask.reshape(-1))[0].astype(np.int64)
        d[self.bg_key] = np.where(bg_mask.reshape(-1))[0].astype(np.int64)
        return d


# ============================================================
# Transformation pipelines and DataLoader factories
# ============================================================
def build_transforms_train():
    return Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstD(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=PIXDIM, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        MultiWindowIntensityd(keys=["vol"], windows=WINDOWS),

        CropForegroundd(keys=["vol", "seg"], source_key="seg", allow_smaller=True, margin=10),
        SpatialPadd(keys=["vol", "seg"], spatial_size=PATCH_SIZE, mode="constant"),

        CreateTumorMaskd(keys=["seg"], seg_key="seg", out_key="tumor_mask", tumor_label=TUMOR_LABEL),
        CreatePancreasHardNegIndicesd(seg_key="seg", fg_key="fg_idx", bg_key="bg_idx",
                                      organ_label=ORGAN_LABEL, tumor_label=TUMOR_LABEL),

        RandCropByPosNegLabeld(
            keys=["vol", "seg", "tumor_mask"],
            label_key="tumor_mask",
            spatial_size=PATCH_SIZE,
            pos=3, neg=1, num_samples=4,
            image_key="vol", image_threshold=0.0, allow_smaller=True,
            fg_indices_key="fg_idx", bg_indices_key="bg_idx",
        ),
        DeleteItemsd(keys=["fg_idx", "bg_idx"]),

        RandFlipd(keys=["vol", "seg", "tumor_mask"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotated(keys=["vol", "seg", "tumor_mask"],
                    range_x=0.3, range_y=0.3, range_z=0.3, prob=0.3,
                    mode=("bilinear", "nearest", "nearest")),
        RandZoomd(keys=["vol", "seg", "tumor_mask"],
                  min_zoom=0.8, max_zoom=1.2, prob=0.2,
                  mode=("trilinear", "nearest", "nearest")),
        RandScaleIntensityd(keys=["vol"], factors=0.10, prob=0.20),
        RandShiftIntensityd(keys=["vol"], offsets=0.10, prob=0.20),
        RandGaussianNoised(keys=["vol"], std=0.02, prob=0.10),

        ToTensord(keys=["vol", "seg", "tumor_mask"]),
    ])


def build_transforms_eval():
    return Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstD(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=PIXDIM, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        MultiWindowIntensityd(keys=["vol"], windows=WINDOWS),

        CropForegroundd(keys=["vol", "seg"], source_key="seg", allow_smaller=True, margin=10),
        SpatialPadd(keys=["vol", "seg"], spatial_size=PATCH_SIZE, mode="constant"),

        ToTensord(keys=["vol", "seg"]),
    ])


def make_loader(files, train_mode, cache=True, num_workers=4):
    tx = build_transforms_train() if train_mode else build_transforms_eval()
    ds = CacheDataset(files, tx, cache_rate=1.0, num_workers=num_workers) if cache else Dataset(files, tx)
    return DataLoader(ds, batch_size=1, shuffle=train_mode, num_workers=num_workers, pin_memory=True)