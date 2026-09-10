"""Joint enhancement and crack-segmentation losses."""
import torch
import torch.nn.functional as F


def balanced_bce(logits, target, max_pos_weight=100.0):
    positive = target.sum().detach()
    negative = target.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(max=max_pos_weight)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def dice_loss(logits, target, eps=1e-6):
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def segmentation_loss(logits, target):
    return balanced_bce(logits, target) + dice_loss(logits, target)


def edge_loss(prediction, target):
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    true_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    true_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, true_dx) + F.l1_loss(pred_dy, true_dy)


def joint_loss(result, reference, label, weights):
    final_seg = segmentation_loss(result["logits"], label)
    side_seg = torch.stack([
        segmentation_loss(side, label) for side in result["side_outputs"]
    ]).mean()
    enhancement = F.l1_loss(result["enhanced"], reference)
    edges = edge_loss(result["enhanced"], reference)
    total = (
        weights["segmentation"] * final_seg
        + weights["side"] * side_seg
        + weights["enhancement"] * enhancement
        + weights["edge"] * edges
    )
    parts = {"seg": final_seg, "side": side_seg,
             "enh": enhancement, "edge": edges}
    return total, parts
