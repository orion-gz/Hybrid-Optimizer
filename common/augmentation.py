import torch
import numpy as np
import random


def rand_bbox(size, lam):
    """Return a random bounding box for CutMix, scaled by the mix ratio lam."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_cutmix_data(inputs, targets, use_mixup, use_cutmix, mixup_alpha):
    """Apply Mixup or CutMix to a batch with 50% probability.

    Returns the mixed inputs, a label tuple, and a bool indicating whether
    mixing was applied. The label tuple contains (targets_a, targets_b, lam)
    when mixed, and (targets,) otherwise.
    """
    apply_mix = (use_mixup or use_cutmix) and random.random() < 0.5
    if apply_mix:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        rand_index = torch.randperm(inputs.size(0))
        targets_a, targets_b = targets, targets[rand_index]

        if use_cutmix:
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            mixed_inputs = inputs.clone()
            mixed_inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # recalculate lam based on the actual box area
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
        else:
            mixed_inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]

        return mixed_inputs, (targets_a, targets_b, lam), True
    else:
        return inputs, (targets,), False


def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the mixed loss as a weighted sum of two cross-entropy terms."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)