"""Read-only paired historical vs corrected postprocessing evaluation.

Run from the public repository.
"""
import argparse
import csv
import gc
import json
import math
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from preprocessing import (
    SEED, PATCH_SIZE, OVERLAP, build_case_index, load_splits,
    validate_splits, files_from_case_list, make_loader,
)
from utilities import (
    build_enhanced_model, load_state_strictish, postprocess_ranked_gated,
    keep_largest_cc, remove_small_cc, dice_for_label, lesion_stats,
    nanmean, ORGAN_LABEL, TUMOR_LABEL, TUMOR_CHANNEL,
)


def historical_postprocess(pred_raw, tum_prob, thr, min_vox, max_lesions=2, pan_dilate=2):
    """Exact historical branch logic from before Commit 2B."""
    pred = pred_raw.copy()
    pan = (pred_raw == ORGAN_LABEL) | (pred_raw == TUMOR_LABEL)
    if pan_dilate and pan_dilate > 0:
        pan = ndi.binary_dilation(pan.astype(bool), iterations=int(pan_dilate))
    pred[pred == TUMOR_LABEL] = ORGAN_LABEL
    pred[(tum_prob > thr) & pan] = TUMOR_LABEL
    union = (pred == ORGAN_LABEL) | (pred == TUMOR_LABEL)
    pred[~keep_largest_cc(union)] = 0
    tumor_mask = remove_small_cc((pred == TUMOR_LABEL), min_vox=min_vox)
    lbl, n = ndi.label(tumor_mask.astype(np.uint8))
    if n == 0:
        return pred.astype(np.int16)  # Original defect, deliberately preserved.
    lesion_scores = []
    for i in range(1, n + 1):
        mask = lbl == i
        mean_prob = float(tum_prob[mask].mean())
        size = int(mask.sum())
        lesion_scores.append((i, mean_prob * np.log(size + 1)))
    lesion_scores.sort(key=lambda x: -x[1])
    keep_ids = [x[0] for x in lesion_scores[:max_lesions]]
    final_mask = np.isin(lbl, keep_ids)
    pred[pred == TUMOR_LABEL] = ORGAN_LABEL
    pred[final_mask] = TUMOR_LABEL
    return pred.astype(np.int16)


def metric_row(case, pred, gt):
    organ_dice = float(dice_for_label(pred, gt, ORGAN_LABEL))
    tumor_dice = float(dice_for_label(pred, gt, TUMOR_LABEL))
    tp, fp, fn = lesion_stats(pred == TUMOR_LABEL, gt == TUMOR_LABEL, 400)
    blind = int(np.any(gt == TUMOR_LABEL) and (math.isnan(tumor_dice) or tumor_dice <= 1e-6))
    return dict(case=case, organ_dice=organ_dice, tumor_dice=tumor_dice,
                tp=int(tp), fp=int(fp), fn=int(fn), blind=blind,
                tumor_voxels=int(np.count_nonzero(pred == TUMOR_LABEL)))


def aggregate(rows):
    tp = sum(r['tp'] for r in rows)
    fp = sum(r['fp'] for r in rows)
    fn = sum(r['fn'] for r in rows)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return dict(organ_mean=float(nanmean([r['organ_dice'] for r in rows])),
                tumor_mean=float(nanmean([r['tumor_dice'] for r in rows])),
                tp=tp, fp=fp, fn=fn, prec=precision, rec=recall,
                f1=2 * precision * recall / (precision + recall + 1e-8),
                blind=sum(r['blind'] for r in rows),
                blind_cases=[r['case'] for r in rows if r['blind']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--splits', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--original-summary', required=True)
    ap.add_argument('--output-dir', default='diagnostics/postprocessing_2b')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-cases', type=int, default=None,
                    help='Smoke test only; do not compare aggregate with historical summary')
    args = ap.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        ap.error('--max-cases must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        ap.error('CUDA requested but not available')
    output = Path(args.output_dir).resolve()
    experiment = Path(args.original_summary).resolve().parent
    if output == experiment or experiment in output.parents:
        ap.error('Output directory must not be inside the original experiment')
    output.mkdir(parents=True, exist_ok=True)
    with open(args.original_summary) as f:
        historical_summary = json.load(f)
    pp = historical_summary['best_pp_val_only']
    settings = dict(thr=float(pp['thr']), min_vox=int(pp['min_vox']),
                    pan_dilate=int(pp['pan_dilate']), max_lesions=int(pp['max_lesions']))
    temperature = float(pp['temperature'])
    if settings['min_vox'] != 400:
        ap.error('This diagnostic expects min_vox=400; adjust metric_row if changed')
    splits = load_splits(args.splits)
    index = build_case_index(args.data_dir)
    validate_splits(splits, index)
    files = files_from_case_list(index, splits['test_clean'])
    if args.max_cases is not None:
        files = files[:args.max_cases]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    set_determinism(seed=SEED)
    device = torch.device(args.device)
    model = build_enhanced_model(device)
    model = load_state_strictish(model, args.checkpoint, device)
    model.eval()
    # No CacheDataset and no worker prefetch; process one case at a time.
    loader = make_loader(files, train_mode=False, cache=False, num_workers=0)
    rows = []
    old_rows, new_rows = [], []
    for i, batch in enumerate(loader, 1):
        case = batch.get('case', ['unknown'])
        case = str(case[0] if isinstance(case, (list, tuple)) else case)
        vol = batch['vol'].to(device)
        seg_np = batch['seg'][0, 0].cpu().numpy().astype(np.int16)
        with torch.no_grad():
            logits = sliding_window_inference(
                vol, roi_size=PATCH_SIZE, sw_batch_size=1,
                predictor=model, overlap=OVERLAP, mode='gaussian')
            probs = torch.softmax(logits / temperature, dim=1)[0]
            tum_prob = probs[TUMOR_CHANNEL].float().cpu().numpy().astype(np.float32)
            pred_raw = torch.argmax(probs, dim=0).cpu().numpy().astype(np.int16)
        old_pred = historical_postprocess(pred_raw, tum_prob, **settings)
        new_pred = postprocess_ranked_gated(pred_raw, tum_prob, **settings)
        old = metric_row(case, old_pred, seg_np)
        new = metric_row(case, new_pred, seg_np)
        old_rows.append(old)
        new_rows.append(new)
        changed = int(np.count_nonzero(old_pred != new_pred))
        rows.append(dict(case=case, changed_voxels=changed,
                         historical_tumor_dice=old['tumor_dice'],
                         corrected_tumor_dice=new['tumor_dice'],
                         historical_organ_dice=old['organ_dice'],
                         corrected_organ_dice=new['organ_dice'],
                         historical_tp=old['tp'], corrected_tp=new['tp'],
                         historical_fp=old['fp'], corrected_fp=new['fp'],
                         historical_fn=old['fn'], corrected_fn=new['fn'],
                         historical_tumor_voxels=old['tumor_voxels'],
                         corrected_tumor_voxels=new['tumor_voxels']))
        print(f'[{i}/{len(files)}] {case}: changed_voxels={changed}, '
              f'tumor_dice={old["tumor_dice"]:.5f} -> {new["tumor_dice"]:.5f}', flush=True)
        del batch, vol, logits, probs, tum_prob, pred_raw, old_pred, new_pred, seg_np
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    old_agg = aggregate(old_rows)
    new_agg = aggregate(new_rows)
    expected = historical_summary['test_clean_metrics']
    check_fields = ('organ_mean', 'tumor_mean', 'tp', 'fp', 'fn', 'blind')
    differences = {k: float(old_agg[k]) - float(expected[k]) for k in check_fields}
    reproduced = (args.max_cases is None and len(files) == len(splits['test_clean'])
                  and all(abs(differences[k]) < (1e-4 if k.endswith('mean') else 0.5)
                          for k in check_fields))
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                  original_summary=str(Path(args.original_summary).resolve()),
                  split_manifest=str(Path(args.splits).resolve()),
                  cases_evaluated=len(rows), parameters={**settings, 'temperature': temperature},
                  historical=old_agg, corrected=new_agg,
                  published_summary=expected,
                  historical_minus_published=differences,
                  historical_reproduction_matches=reproduced,
                  affected_cases=[r['case'] for r in rows if r['changed_voxels'] > 0],
                  note=('Historical reproduction must match before attributing aggregate '
                        'differences to the postprocessing correction. Original GT-dependent '
                        'cropping is retained for this paired comparison.'))
    with open(output / 'per_case_comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / 'comparison_summary.json', 'w') as f:
        json.dump(report, f, indent=2, allow_nan=True)
    print('\nHistorical:', old_agg)
    print('Corrected: ', new_agg)
    print('Historical reproduction matches summary:', reproduced)
    print('Changed cases:', report['affected_cases'])
    print('Reports:', output)
    if not reproduced and args.max_cases is None:
        print('WARNING: Historical metrics differ; investigate before interpreting correction.')


if __name__ == '__main__':
    main()
