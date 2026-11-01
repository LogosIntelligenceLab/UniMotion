"""Stateless objectives for motion generation and trajectory prediction.

NTP retains the geometric token target and label smoothing. Prediction
retains best-mode Smooth-L1 plus mode classification, not single-mode LFR.
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.unimotion.metrics.utils import get_euclidean_targets, get_prob_targets


class PretrainingLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, use_gt_raw=True,
                 ntp_weight=1.0, regression_weight=1.0, classification_weight=1.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.use_gt_raw = use_gt_raw
        self.weights = (ntp_weight, regression_weight, classification_weight)

    def forward(self, agent, output):
        losses = {}
        if "sim" in output:
            sim = output["sim"]
            suffix = "_raw" if self.use_gt_raw else ""
            target, valid = get_euclidean_targets(
                sim["pred_pos"], sim["pred_head"], sim["pred_valid"],
                sim["gt_pos" + suffix], sim["gt_head" + suffix], sim["gt_valid" + suffix],
            )
            target_prob = get_prob_targets(target, agent["token_agent_shape"], agent["token_traj"])
            ce = F.cross_entropy(sim["next_token_logits"].transpose(1, 2),
                                 target_prob.transpose(1, 2), reduction="none",
                                 label_smoothing=self.label_smoothing)
            valid = valid & sim["next_token_valid"]
            if self.training:
                valid = valid & agent["train_mask"][:, None]
            losses["ntp_loss"] = ce.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)
        if "pred" in output:
            pred = output["pred"]
            trajectory, logits = pred["pred_mm_token_logits"], pred["pred_mm_prob"]
            gt, mask = pred["pred_gt_pos"], pred["pred_mask"]
            # Regress all valid future targets; token validity gates classification.
            # Ignore empty futures instead of assigning an arbitrary mode label.
            cls_mask = mask.any(-1) & pred["token_mask"]
            distances = (trajectory - gt[:, None]).norm(dim=-1)
            best = distances.masked_fill(~mask[:, None], 0).sum(-1).argmin(-1).detach()
            selected = trajectory[torch.arange(trajectory.size(0), device=trajectory.device), best]
            reg = F.smooth_l1_loss(selected, gt, reduction="none").sum(-1)
            # Normalize each future timestep separately, then average over time.
            losses["prediction_reg_loss"] = (
                reg.masked_fill(~mask, 0).sum(0) / mask.sum(0).clamp_min(1)
            ).mean()
            ce = F.cross_entropy(logits, best, reduction="none")
            losses["prediction_cls_loss"] = ce.masked_fill(~cls_mask, 0).sum() / cls_mask.sum().clamp_min(1)
        keys = ("ntp_loss", "prediction_reg_loss", "prediction_cls_loss")
        losses["loss"] = sum(self.weights[i] * losses[k] for i, k in enumerate(keys) if k in losses)
        return losses
