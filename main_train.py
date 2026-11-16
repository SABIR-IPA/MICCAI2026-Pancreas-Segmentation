import os
import sys
import json
import shutil
import argparse
import datetime

import torch
import numpy as np
from monai.utils import set_determinism

from preprocessing import (
    build_case_index, load_splits, files_from_case_list,
    oversample_train, make_loader, SEED,
    MAX_EPOCHS_PHASE1, PATIENCE_PHASE1,
    MAX_EPOCHS_PHASE2, PATIENCE_PHASE2,
    HARD_REPEAT_PHASE2,
    LR_BACKBONE_P1, LR_REST_P1, LR_NEW_P1,
    LR_BACKBONE_P2, LR_REST_P2, LR_NEW_P2,
    VAL_THR_LIST, VAL_MIN_VOX_LIST,
    VAL_PAN_DILATE, VAL_MAX_LESIONS,
)
from utilities import (
    build_enhanced_model, StablePrecisionLoss,
    load_checkpoint_clean_multiwindow, load_state_strictish,
    build_optimizer, freeze_base_network,
    train_model, precompute_logits,
    eval_set_from_logits, pick_best_postprocess,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Path to the dataset (must contain imagesTr/ and labelsTr/)")
    parser.add_argument("--baseline_ckpt", required=True,
                        help="Path to the pre‑trained baseline SwinUNETR checkpoint ")
    parser.add_argument("--data_splits", required=True,
                        help="JSON file containing train/val/test/hard splits")
    parser.add_argument("--out_root", default="./results",
                        help="Root directory where all outputs will be saved")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g., 'cuda:0' or 'cpu'); if None, auto‑detect")
    args = parser.parse_args()

    # Set device and seeds
    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    set_determinism(seed=SEED)

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(args.out_root, f"task07_valppsearch_{timestamp}")
    os.makedirs(root_dir, exist_ok=True)

    # Load splits and copy to output
    splits = load_splits(args.data_splits)
    shutil.copy(args.data_splits, os.path.join(root_dir, "splits.json"))

    # Build case index and prepare file lists
    case_index = build_case_index(args.data_dir)
    train_clean = files_from_case_list(case_index, splits["train_clean"])
    val_clean = files_from_case_list(case_index, splits["val_clean"])
    test_clean = files_from_case_list(case_index, splits["test_clean"])
    hard = files_from_case_list(case_index, splits["hard_cases"])
    hard_set = set(splits["hard_cases"])

    # -------------------------
    # PHASE 1: recall/adaptation (no FP penalty)
    # -------------------------
    phase1_dir = os.path.join(root_dir, "phase1")
    os.makedirs(phase1_dir, exist_ok=True)

    train_p1 = oversample_train(train_clean, hard_set=None, hard_repeat=1)
    train_loader1 = make_loader(train_p1, train_mode=True, cache=True, num_workers=4)
    val_loader = make_loader(val_clean, train_mode=False, cache=True, num_workers=4)

    model1 = build_enhanced_model(device)
    model1 = load_checkpoint_clean_multiwindow(model1, args.baseline_ckpt, device, in_ch=2)

    loss_fn1 = StablePrecisionLoss(
        warmup_epochs=10,
        ramp_epochs=25,
        fp_start_epoch=10**9,   # never enabled
        fp_weight_max=0.0,
        fp_ramp_epochs=1,
        alpha_start=0.30, beta_start=0.70,
        alpha_end=0.30, beta_end=0.70,
        tumor_weight_start=2.0,
        tumor_weight_end=2.0,
    ).to(device)

    optim1 = build_optimizer(model1, LR_BACKBONE_P1, LR_REST_P1, LR_NEW_P1)

    print("\n" + "=" * 80)
    print("PHASE-1: TRAIN_CLEAN (score = best ValPP tumor Dice via VAL‑only PP search)")
    print("=" * 80)

    res1 = train_model(
        model1, train_loader1, val_loader, loss_fn1, optim1,
        MAX_EPOCHS_PHASE1, phase1_dir, PATIENCE_PHASE1, device,
        val_thr_list=VAL_THR_LIST, val_minvox_list=VAL_MIN_VOX_LIST
    )

    phase1_best = os.path.join(phase1_dir, "best_model.pth")
    assert os.path.exists(phase1_best)

    # -------------------------
    # PHASE 2: precision stage (FP penalty), base frozen
    # -------------------------
    phase2_dir = os.path.join(root_dir, "phase2")
    os.makedirs(phase2_dir, exist_ok=True)

    train_p2_base = train_clean + hard
    train_p2 = oversample_train(train_p2_base, hard_set=hard_set, hard_repeat=HARD_REPEAT_PHASE2)
    train_loader2 = make_loader(train_p2, train_mode=True, cache=True, num_workers=4)

    model2 = build_enhanced_model(device)
    model2 = load_state_strictish(model2, phase1_best, device)
    freeze_base_network(model2)

    loss_fn2 = StablePrecisionLoss(
        warmup_epochs=0,
        ramp_epochs=20,
        fp_start_epoch=2,
        fp_weight_max=1.2,
        fp_ramp_epochs=25,
        alpha_start=0.50, beta_start=0.50,
        alpha_end=0.85, beta_end=0.15,
        tumor_weight_start=1.5,
        tumor_weight_end=1.0,
    ).to(device)

    optim2 = build_optimizer(model2, LR_BACKBONE_P2, LR_REST_P2, LR_NEW_P2)

    print("\n" + "=" * 80)
    print("PHASE-2: TRAIN_CLEAN + HARD (base frozen; score = best ValPP tumor Dice)")
    print("=" * 80)

    res2 = train_model(
        model2, train_loader2, val_loader, loss_fn2, optim2,
        MAX_EPOCHS_PHASE2, phase2_dir, PATIENCE_PHASE2, device,
        val_thr_list=VAL_THR_LIST, val_minvox_list=VAL_MIN_VOX_LIST
    )

    phase2_best = os.path.join(phase2_dir, "best_model.pth")
    assert os.path.exists(phase2_best)

    # -------------------------
    # FINAL: tune PP on VAL only, report TEST + HARD
    # -------------------------
    print("\n" + "=" * 80)
    print("TUNING postprocess ONLY on VAL_CLEAN (gated PP)")
    print("=" * 80)

    val_eval_loader = make_loader(val_clean, train_mode=False, cache=True, num_workers=4)
    test_eval_loader = make_loader(test_clean, train_mode=False, cache=True, num_workers=4)
    hard_eval_loader = make_loader(hard, train_mode=False, cache=True, num_workers=4)

    final_model = build_enhanced_model(device)
    final_model = load_state_strictish(final_model, phase2_best, device)
    final_model.eval()

    val_logits = precompute_logits(final_model, val_eval_loader, device)
    best_pp = pick_best_postprocess(val_logits, target_precision=0.70,
                                    pan_dilate=VAL_PAN_DILATE,
                                    max_lesions=VAL_MAX_LESIONS,
                                    temperature=2.0)
    print(f"\nBest postprocess (VAL only): {best_pp}")

    print("\n" + "=" * 80)
    print("FINAL REPORTING — test_clean once, hard transparent")
    print("=" * 80)

    test_logits = precompute_logits(final_model, test_eval_loader, device)
    hard_logits = precompute_logits(final_model, hard_eval_loader, device)

    test_metrics = eval_set_from_logits(
        test_logits,
        thr=best_pp["thr"], min_vox=best_pp["min_vox"],
        pan_dilate=best_pp["pan_dilate"], max_lesions=best_pp["max_lesions"],
        temperature=best_pp["temperature"],
    )
    hard_metrics = eval_set_from_logits(
        hard_logits,
        thr=best_pp["thr"], min_vox=best_pp["min_vox"],
        pan_dilate=best_pp["pan_dilate"], max_lesions=best_pp["max_lesions"],
        temperature=best_pp["temperature"],
    )

    print("\nTEST_CLEAN:")
    print(test_metrics)
    print("\nHARD (seen in phase‑2 training):")
    print(hard_metrics)

    # Save summary
    summary = {
        "phase1": {"dir": phase1_dir, **res1},
        "phase2": {"dir": phase2_dir, **res2},
        "best_pp_val_only": best_pp,
        "test_clean_metrics": test_metrics,
        "hard_metrics": hard_metrics,
    }
    with open(os.path.join(root_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to: {root_dir}/summary.json")


if __name__ == "__main__":
    main()