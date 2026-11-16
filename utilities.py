import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import scipy.ndimage as ndi

from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetOutBlock, Convolution
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# Import constants from preprocessing
from preprocessing import (
    PATCH_SIZE, OVERLAP, NUM_CLASSES, TUMOR_CHANNEL, TUMOR_LABEL,
    ORGAN_LABEL, VAL_PAN_DILATE, VAL_MAX_LESIONS, WEIGHT_DECAY,
    THR_LIST, MIN_VOX_LIST
)


# ============================================================
# Model components
# ============================================================
class TextureFeatureExtractor(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        c = out_ch // 4
        self.c3 = Convolution(3, in_ch, c, kernel_size=3, padding=1, norm="instance", act="relu")
        self.c5 = Convolution(3, in_ch, c, kernel_size=5, padding=2, norm="instance", act="relu")
        self.c7 = Convolution(3, in_ch, c, kernel_size=7, padding=3, norm="instance", act="relu")
        self.cd = Convolution(3, in_ch, c, kernel_size=3, padding=2, dilation=2, norm="instance", act="relu")
        self.fuse = Convolution(3, out_ch, out_ch, kernel_size=1, norm="instance", act="relu")

    def forward(self, x):
        return self.fuse(torch.cat([self.c3(x), self.c5(x), self.c7(x), self.cd(x)], dim=1))


class ContrastNormalizationModule(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))

    def forward(self, x):
        m = x.mean(dim=(2, 3, 4), keepdim=True)
        s = x.std(dim=(2, 3, 4), keepdim=True) + 1e-5
        return self.gamma * (x - m) / s + self.beta


class EnhancedSwinUNETR(nn.Module):
    def __init__(self, base, feature_size=48, in_channels=2, num_classes=3):
        super().__init__()
        self.base = base
        self.base.out = nn.Identity()

        self.contrast_norm = ContrastNormalizationModule(in_channels)
        self.texture = TextureFeatureExtractor(feature_size * 16, feature_size * 16)

        self.tex_head = nn.Sequential(
            Convolution(3, feature_size, feature_size, kernel_size=3, padding=1,
                        norm="instance", act="relu", dropout=0.2),
            Convolution(3, feature_size, feature_size, kernel_size=3, padding=1,
                        norm="instance", act="relu", dropout=0.2),
        )

        self.base_out = UnetOutBlock(3, in_channels=feature_size, out_channels=num_classes)
        self.delta_out = UnetOutBlock(3, in_channels=feature_size * 2, out_channels=num_classes)

        self.cn_gate = nn.Parameter(torch.tensor(-3.0))
        self.tex_gate = nn.Parameter(torch.tensor(-3.0))
        self.delta_gate = nn.Parameter(torch.tensor(-3.0))

        # Small delta initialisation (not zero)
        with torch.no_grad():
            self.delta_out.conv.conv.weight.mul_(0.01)
            if self.delta_out.conv.conv.bias is not None:
                self.delta_out.conv.conv.bias.zero_()

    def gate_values(self):
        return {
            "g_cn": float(torch.sigmoid(self.cn_gate).detach().cpu()),
            "g_tex": float(torch.sigmoid(self.tex_gate).detach().cpu()),
            "g_delta": float(torch.sigmoid(self.delta_gate).detach().cpu()),
        }

    def forward(self, x):
        gcn = torch.sigmoid(self.cn_gate)
        x = x + gcn * (self.contrast_norm(x) - x)

        hs = self.base.swinViT(x, self.base.normalize)
        e0 = self.base.encoder1(x)
        e1 = self.base.encoder2(hs[0])
        e2 = self.base.encoder3(hs[1])
        e3 = self.base.encoder4(hs[2])

        bot0 = self.base.encoder10(hs[4])
        gtex = torch.sigmoid(self.tex_gate)
        bot = bot0 + gtex * self.texture(bot0)

        d3 = self.base.decoder5(bot, hs[3])
        d2 = self.base.decoder4(d3, e3)
        d1 = self.base.decoder3(d2, e2)
        d0 = self.base.decoder2(d1, e1)
        feat = self.base.decoder1(d0, e0)

        logits_base = self.base_out(feat)
        logits_delta = self.delta_out(torch.cat([feat, self.tex_head(feat)], dim=1))
        gd = torch.sigmoid(self.delta_gate)

        return logits_base + gd * logits_delta


def build_enhanced_model(device):
    base = SwinUNETR(
        in_channels=2, out_channels=3, img_size=PATCH_SIZE,
        depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        feature_size=48, norm_name="instance", spatial_dims=3,
        use_checkpoint=True, downsample="mergingv2", use_v2=True,
        drop_rate=0.1, attn_drop_rate=0.1, dropout_path_rate=0.2,
    )
    return EnhancedSwinUNETR(base, feature_size=48, in_channels=2, num_classes=3).to(device)


# ============================================================
# Loss functions
# ============================================================
def dilate3d(mask_bool, r=2):
    if r <= 0:
        return mask_bool
    x = mask_bool.float().unsqueeze(1)  # (B,1,D,H,W)
    k = 2 * r + 1
    x = F.max_pool3d(x, kernel_size=k, stride=1, padding=r)
    return (x[:, 0] > 0.0)


class WeightedFocalCELoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target, weight=None):
        ce = F.cross_entropy(logits, target.long(), weight=weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


class StablePrecisionLoss(nn.Module):
    def __init__(
        self,
        warmup_epochs=10,
        ramp_epochs=25,
        fp_start_epoch=15,
        alpha_start=0.30, beta_start=0.70,
        alpha_end=0.80, beta_end=0.20,
        tumor_weight_start=2.0,
        tumor_weight_end=1.0,
        w_tversky=0.6,
        w_focal=0.4,
        fp_weight_max=0.6,
        fp_ramp_epochs=25,
        hard_ratio_start=0.10,
        hard_ratio_end=0.30,
        fp_region="pancreas_ring",
        ring_dilate=2,
        fp_prob_floor=0.50,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.fp_start_epoch = fp_start_epoch

        self.alpha_start, self.beta_start = alpha_start, beta_start
        self.alpha_end, self.beta_end = alpha_end, beta_end
        self.tumor_w0, self.tumor_w1 = tumor_weight_start, tumor_weight_end

        self.w_tversky, self.w_focal = w_tversky, w_focal
        self.fp_weight_max = fp_weight_max
        self.fp_ramp_epochs = fp_ramp_epochs
        self.hard0, self.hard1 = hard_ratio_start, hard_ratio_end

        self.fp_region = fp_region
        self.ring_dilate = ring_dilate
        self.fp_prob_floor = fp_prob_floor

        self.current_epoch = 0
        self.focal = WeightedFocalCELoss(gamma=2.0)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    @staticmethod
    def _lin(epoch, e0, e1, v0, v1):
        if epoch <= e0:
            return float(v0)
        if epoch >= e1:
            return float(v1)
        t = (epoch - e0) / max(1.0, (e1 - e0))
        return float(v0 + t * (v1 - v0))

    @staticmethod
    def _smooth_ramp(epoch, start, duration, maxv):
        if epoch <= start:
            return 0.0
        t = (epoch - start) / max(1.0, duration)
        return float(maxv * (1.0 / (1.0 + math.exp(-12.0 * (t - 0.5)))))

    def _sched(self):
        e0 = self.warmup_epochs
        e1 = self.warmup_epochs + self.ramp_epochs
        a = self._lin(self.current_epoch, e0, e1, self.alpha_start, self.alpha_end)
        b = self._lin(self.current_epoch, e0, e1, self.beta_start, self.beta_end)
        tw = self._lin(self.current_epoch, e0, e1, self.tumor_w0, self.tumor_w1)
        hr = self._lin(self.current_epoch, e0, e1, self.hard0, self.hard1)
        fpw = self._smooth_ramp(self.current_epoch, self.fp_start_epoch,
                                self.fp_ramp_epochs, self.fp_weight_max)
        return a, b, tw, fpw, hr

    def false_positive_penalty(self, logits, tgt_class, hard_ratio):
        probs = F.softmax(logits, dim=1)
        tumor_prob = probs[:, TUMOR_CHANNEL]

        pancreas = (tgt_class == ORGAN_LABEL)
        tumor_gt = (tgt_class == TUMOR_LABEL)

        if self.fp_region == "pancreas_ring":
            ring = dilate3d(tumor_gt, r=self.ring_dilate) & pancreas & (~tumor_gt)
            neg_mask = ring
        else:
            neg_mask = pancreas & (~tumor_gt)

        neg_mask = neg_mask & (tumor_prob > self.fp_prob_floor)
        if not neg_mask.any():
            return logits.new_zeros(())

        fp_vals = tumor_prob[neg_mask]
        if fp_vals.numel() == 0:
            return logits.new_zeros(())

        k = max(1, int(hard_ratio * fp_vals.numel()))
        topk, _ = torch.topk(fp_vals, k)
        return (topk ** 2).mean()

    def forward(self, logits, seg):
        tgt = seg.squeeze(1).long()
        alpha, beta, tumor_w, fpw, hard_ratio = self._sched()

        loss_tv = TverskyLoss(
            to_onehot_y=True, softmax=True, include_background=False,
            alpha=alpha, beta=beta
        )(logits, seg)

        w = torch.tensor([1.0, 1.0, float(tumor_w)], device=logits.device, dtype=torch.float32)
        loss_fc = self.focal(logits, tgt, weight=w)

        total = self.w_tversky * loss_tv + self.w_focal * loss_fc
        if fpw > 0:
            total = total + fpw * self.false_positive_penalty(logits, tgt, hard_ratio=hard_ratio)
        return total


# ============================================================
# Checkpoint loading utilities
# ============================================================
def _checkpoint_state_dict(ckpt_path, device):
    """Read a trusted checkpoint and extract its model state."""
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)

    if not isinstance(raw, dict):
        raise TypeError("Checkpoint must contain a state dictionary.")

    for key in ("state_dict", "model_state_dict", "net", "model"):
        if key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break

    state = {}

    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue

        normalized_key = key.removeprefix("module.")

        if normalized_key in state:
            raise ValueError(
                f"Duplicate checkpoint parameter after prefix removal: "
                f"{normalized_key}"
            )

        state[normalized_key] = value

    if not state:
        raise ValueError("Checkpoint contains no model tensors.")

    return state


def load_checkpoint_clean_multiwindow(model, ckpt_path, device, in_ch=2):
    """Initialize the enhanced model from a compatible baseline checkpoint."""
    raw = _checkpoint_state_dict(ckpt_path, device)
    target = model.state_dict()
    prefix = "base." if hasattr(model, "base") else ""

    new_state = {}
    adapted = []
    in_ch_adapted = []

    for key, value in raw.items():
        for target_key in (key, f"{prefix}{key}"):
            if target_key not in target:
                continue

            expected = target[target_key]

            if expected.shape == value.shape:
                new_state[target_key] = value
                adapted.append(target_key)
                break

            if (
                value.ndim == 5
                and expected.ndim == 5
                and value.shape[0] == expected.shape[0]
                and value.shape[2:] == expected.shape[2:]
                and value.shape[1] == 1
                and expected.shape[1] == in_ch
            ):
                new_state[target_key] = (
                    value.repeat(1, in_ch, 1, 1, 1) / in_ch
                )
                in_ch_adapted.append(target_key)
                break

    expected_backbone = {
        key for key in target if key.startswith(prefix)
    } if prefix else set(target)

    missing_backbone = expected_backbone - set(new_state)

    if missing_backbone:
        sample = sorted(missing_backbone)[:10]
        raise ValueError(
            f"Baseline checkpoint is missing {len(missing_backbone)} "
            f"backbone tensors: {sample}"
        )

    head_weight = raw.get("out.conv.conv.weight")
    head_bias = raw.get("out.conv.conv.bias")

    if head_weight is None or head_bias is None:
        raise ValueError("Baseline checkpoint is missing segmentation head tensors.")

    target_weight = model.base_out.conv.conv.weight
    target_bias = model.base_out.conv.conv.bias

    if (
        head_weight.shape != target_weight.shape
        or target_bias is None
        or head_bias.shape != target_bias.shape
    ):
        raise ValueError("Baseline segmentation head has incompatible shapes.")

    model.load_state_dict(new_state, strict=False)

    with torch.no_grad():
        target_weight.copy_(head_weight)
        target_bias.copy_(head_bias)
        model.cn_gate.fill_(-3.0)
        model.tex_gate.fill_(-3.0)
        model.delta_gate.fill_(-3.0)

    print(f"Loaded baseline: {ckpt_path}")
    print(
        f"  matched: {len(adapted)}, "
        f"in-channel adapted: {len(in_ch_adapted)}"
    )
    return model


def load_state_strictish(model, ckpt_path, device):
    """Restore a complete enhanced-model checkpoint."""
    raw = _checkpoint_state_dict(ckpt_path, device)
    model.load_state_dict(raw, strict=True)
    print(f"Loaded enhanced-model checkpoint: {ckpt_path}")
    return model


# ============================================================
# Optimiser and freezing helpers
# ============================================================
def build_optimizer(model, lr_backbone, lr_rest, lr_new):
    new_keys = ("contrast_norm", "texture", "tex_head", "delta_out",
                "cn_gate", "tex_gate", "delta_gate")
    backbone, rest, new = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "base.swinViT" in n:
            backbone.append(p)
        elif any(k in n for k in new_keys):
            new.append(p)
        else:
            rest.append(p)

    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": lr_backbone},
            {"params": rest, "lr": lr_rest},
            {"params": new, "lr": lr_new},
        ],
        weight_decay=WEIGHT_DECAY
    )


def freeze_base_network(model):
    for p in model.base.parameters():
        p.requires_grad = False


# ============================================================
# Post‑processing and metrics
# ============================================================
def dice_for_label(pred, gt, lab):
    p = (pred == lab).astype(np.uint8)
    g = (gt == lab).astype(np.uint8)
    if g.sum() == 0:
        return float("nan")
    return float(2.0 * int((p & g).sum()) / (int(p.sum() + g.sum()) + 1e-8))


def nanmean(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")


def keep_largest_cc(mask):
    if not np.any(mask):
        return mask
    s = np.ones((3, 3, 3), dtype=np.uint8)
    lbl, n = ndi.label(mask.astype(np.uint8), structure=s)
    if n <= 1:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return (lbl == int(np.argmax(sizes)))


def remove_small_cc(mask, min_vox):
    if min_vox is None or min_vox <= 1 or not np.any(mask):
        return mask
    s = np.ones((3, 3, 3), dtype=np.uint8)
    lbl, n = ndi.label(mask.astype(np.uint8), structure=s)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = np.zeros_like(mask, dtype=bool)
    for i in range(1, n + 1):
        if sizes[i] >= min_vox:
            keep |= (lbl == i)
    return keep


def postprocess_ranked_gated(pred_raw, tum_prob, thr, min_vox, max_lesions=2, pan_dilate=2):
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
        pred[pred == TUMOR_LABEL] = ORGAN_LABEL
        return pred.astype(np.int16)

    lesion_scores = []
    for i in range(1, n + 1):
        mask = (lbl == i)
        mean_prob = float(tum_prob[mask].mean())
        size = int(mask.sum())
        score = mean_prob * np.log(size + 1)
        lesion_scores.append((i, score))

    lesion_scores.sort(key=lambda x: -x[1])
    keep_ids = [x[0] for x in lesion_scores[:max_lesions]]

    final_mask = np.isin(lbl, keep_ids)
    pred[pred == TUMOR_LABEL] = ORGAN_LABEL
    pred[final_mask] = TUMOR_LABEL
    return pred.astype(np.int16)


def lesion_stats(pr, gt, min_vox):
    s = np.ones((3, 3, 3), dtype=np.uint8)
    p_lbl, p_n = ndi.label(pr.astype(np.uint8), structure=s)
    g_lbl, g_n = ndi.label(gt.astype(np.uint8), structure=s)

    if min_vox and min_vox > 1:
        for arr, n in [(p_lbl, p_n), (g_lbl, g_n)]:
            if n > 0:
                sizes = np.bincount(arr.ravel())
                sizes[0] = 0
                for i in range(1, n + 1):
                    if sizes[i] < min_vox:
                        arr[arr == i] = 0
        p_lbl, p_n = ndi.label((p_lbl > 0).astype(np.uint8), structure=s)
        g_lbl, g_n = ndi.label((g_lbl > 0).astype(np.uint8), structure=s)

    tp = fn = 0
    for gi in range(1, g_n + 1):
        ov = np.unique(p_lbl[g_lbl == gi])
        ov = ov[ov != 0]
        tp += int(ov.size > 0)
        fn += int(ov.size == 0)

    fp = 0
    for pi in range(1, p_n + 1):
        ov = np.unique(g_lbl[p_lbl == pi])
        ov = ov[ov != 0]
        fp += int(ov.size == 0)

    return tp, fp, fn


# ============================================================
# Validation PP search (used inside training loop)
# ============================================================
def search_valpp_best(val_cache, thr_list, min_vox_list, pan_dilate=2, max_lesions=2):
    best = None
    for thr in thr_list:
        for mv in min_vox_list:
            od_list, td_list = [], []
            for it in val_cache:
                seg_np = it["seg_np"]
                pred_pp = postprocess_ranked_gated(
                    it["pred_raw"], it["tum_prob"],
                    thr=thr, min_vox=mv,
                    max_lesions=max_lesions, pan_dilate=pan_dilate
                )
                od_list.append(dice_for_label(pred_pp, seg_np, ORGAN_LABEL))
                td_list.append(dice_for_label(pred_pp, seg_np, TUMOR_LABEL))
            cand = {"thr": thr, "min_vox": mv,
                    "organ_mean": nanmean(od_list), "tumor_mean": nanmean(td_list)}
            if best is None or cand["tumor_mean"] > best["tumor_mean"]:
                best = cand
    return best


# ============================================================
# Training loop
# ============================================================
def train_model(model, train_loader, val_loader, loss_fn, optimizer,
                max_epochs, model_dir, patience, device,
                val_thr_list, val_minvox_list):
    os.makedirs(model_dir, exist_ok=True)
    best_score = -1.0
    best_epoch = -1
    scaler = GradScaler(enabled=True)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", ignore_empty=True)

    for epoch in range(1, max_epochs + 1):
        loss_fn.set_epoch(epoch)

        if epoch == loss_fn.fp_start_epoch:
            for g in optimizer.param_groups:
                g["lr"] *= 0.5

        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            x = batch["vol"].to(device)
            y = batch["seg"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += float(loss.item())
        train_loss = ep_loss / max(1, len(train_loader))

        # Validation inference + PP search
        model.eval()
        val_cache = []
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                x = batch["vol"].to(device)
                y = batch["seg"].to(device)

                logits = sliding_window_inference(x, PATCH_SIZE, 1, model, overlap=OVERLAP)
                val_loss += float(loss_fn(logits, y).item())

                probs = torch.softmax(logits, dim=1)[0]
                tum_prob = probs[TUMOR_CHANNEL].detach().cpu().numpy().astype(np.float32)
                pred_raw = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.int16)
                seg_np = y[0, 0].detach().cpu().numpy().astype(np.int16)

                preds = torch.argmax(logits, dim=1, keepdim=True)
                preds_oh = F.one_hot(preds.squeeze(1).long(), num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
                labels_oh = F.one_hot(y.squeeze(1).long(), num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
                dice_metric(preds_oh, labels_oh)

                val_cache.append({"seg_np": seg_np, "pred_raw": pred_raw, "tum_prob": tum_prob})

            val_loss = val_loss / max(1, len(val_loader))
            dvals = dice_metric.aggregate()
            pan_arg = dvals[0].item()
            tum_arg = dvals[1].item()
            dice_metric.reset()

        best_pp = search_valpp_best(
            val_cache,
            thr_list=val_thr_list,
            min_vox_list=val_minvox_list,
            pan_dilate=VAL_PAN_DILATE,
            max_lesions=VAL_MAX_LESIONS,
        )

        lr_str = " | ".join([f"g{i}:{g['lr']:.2e}" for i, g in enumerate(optimizer.param_groups)])
        a, b, tw, fpw, hr = loss_fn._sched()

        print(f"Epoch {epoch}/{max_epochs} | {lr_str}")
        print(f"  Train | Loss: {train_loss:.4f}")
        print(f"  Val   | Loss: {val_loss:.4f} | Argmax Pan/Tum: {pan_arg:.4f}/{tum_arg:.4f}")
        print(f"  ValPP | Pan/Tum: {best_pp['organ_mean']:.4f}/{best_pp['tumor_mean']:.4f} (thr={best_pp['thr']}, mv={best_pp['min_vox']})")
        print(f"  Sched | alpha={a:.2f} beta={b:.2f} tumor_w={tw:.2f} fpw={fpw:.3f} hard={hr:.2f}")
        print(f"  Gates | {model.gate_values()}")

        score = best_pp["tumor_mean"]

        fp_disabled = (getattr(loss_fn, "fp_weight_max", 0.0) <= 0.0) or (loss_fn.fp_start_epoch > max_epochs)
        patience_active = (epoch >= loss_fn.fp_start_epoch) or fp_disabled

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print(f"  *** NEW BEST (ValPP tumor): {best_score:.4f} @epoch {epoch}")
        else:
            if patience_active:
                print(f"  No improvement ({epoch - best_epoch}/{patience})")
            else:
                print("  No improvement (Warm‑up phase, patience delayed)")

        if patience_active and (epoch - best_epoch) >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    return {"best_epoch": best_epoch, "best_score": best_score, "total_epochs": epoch}


# ============================================================
# Final evaluation helpers
# ============================================================
@torch.no_grad()
def precompute_logits(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        vol = batch["vol"].to(device)
        seg = batch["seg"].to(device).long()
        case = batch.get("case", ["unknown"])
        case = case[0] if isinstance(case, (list, tuple)) else case

        logits = sliding_window_inference(
            vol, roi_size=PATCH_SIZE, sw_batch_size=1,
            predictor=model, overlap=OVERLAP, mode="gaussian",
        )
        seg_np = seg[0, 0].detach().cpu().numpy().astype(np.int16)
        out.append({"case": str(case), "seg_np": seg_np, "logits_cpu": logits.detach().cpu()})
    return out


def eval_set_from_logits(cases_logits, thr, min_vox, pan_dilate=2, max_lesions=2, temperature=2.0):
    organ_d, tumor_d = [], []
    tp_s = fp_s = fn_s = 0
    blind = 0
    blind_cases = []

    for item in cases_logits:
        seg_np = item["seg_np"]

        logits = item["logits_cpu"]
        probs = torch.softmax(logits / float(temperature), dim=1)[0]

        tum_prob = probs[TUMOR_CHANNEL].numpy().astype(np.float32)
        pred_raw = torch.argmax(probs, dim=0).numpy().astype(np.int16)

        pred_post = postprocess_ranked_gated(
            pred_raw, tum_prob,
            thr=thr, min_vox=min_vox,
            max_lesions=max_lesions, pan_dilate=pan_dilate,
        )

        od = dice_for_label(pred_post, seg_np, ORGAN_LABEL)
        td = dice_for_label(pred_post, seg_np, TUMOR_LABEL)
        organ_d.append(od)
        tumor_d.append(td)

        gt_t = (seg_np == TUMOR_LABEL)
        pr_t = (pred_post == TUMOR_LABEL)
        tp, fp, fn = lesion_stats(pr_t, gt_t, min_vox)
        tp_s += tp
        fp_s += fp
        fn_s += fn

        if int(gt_t.sum()) > 0 and (0.0 if math.isnan(td) else td) <= 1e-6:
            blind += 1
            blind_cases.append(item["case"])

    prec = tp_s / (tp_s + fp_s + 1e-8)
    rec = tp_s / (tp_s + fn_s + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    return {
        "organ_mean": nanmean(organ_d),
        "tumor_mean": nanmean(tumor_d),
        "tp": tp_s, "fp": fp_s, "fn": fn_s,
        "prec": prec, "rec": rec, "f1": f1,
        "blind": blind,
        "blind_cases": blind_cases,
    }


def pick_best_postprocess(val_logits, target_precision=0.70, pan_dilate=2, max_lesions=2, temperature=2.0):
    best = None
    all_configs = []

    for thr in THR_LIST:
        for mv in MIN_VOX_LIST:
            m = eval_set_from_logits(
                val_logits, thr=thr, min_vox=mv,
                pan_dilate=pan_dilate, max_lesions=max_lesions,
                temperature=temperature,
            )
            cand = {"thr": thr, "min_vox": mv, "pan_dilate": pan_dilate,
                    "max_lesions": max_lesions, "temperature": temperature, **m}
            all_configs.append(cand)

            if m["prec"] >= target_precision:
                if best is None or m["tumor_mean"] > best["tumor_mean"]:
                    best = cand

    if best is None:
        print("No config reached target precision; fallback to best tumor_mean Dice.")
        best = max(all_configs, key=lambda x: x["tumor_mean"])

    return best
