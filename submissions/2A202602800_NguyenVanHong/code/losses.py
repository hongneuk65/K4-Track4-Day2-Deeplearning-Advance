import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothingCE(nn.CrossEntropyLoss):
    def __init__(self, smoothing=0.1):
        super().__init__(label_smoothing=smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma

        self.register_buffer(
            "alpha",
            None if alpha is None
            else torch.tensor(alpha, dtype=torch.float32),
        )

    def forward(self, logits, target):
        log_prob = F.log_softmax(logits, dim=1)

        log_pt = log_prob.gather(
            1, target.unsqueeze(1)
        ).squeeze(1)

        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            loss = loss * self.alpha[target]

        return loss.mean()


def build_criterion(kind="ce", **kwargs):
    if kind == "ce":
        return nn.CrossEntropyLoss()

    if kind == "ls":
        return LabelSmoothingCE(
            kwargs.get("smoothing", 0.1)
        )

    if kind == "focal":
        return FocalLoss(
            gamma=kwargs.get("gamma", 2.0),
            alpha=kwargs.get("alpha"),
        )

    if kind == "ce_weighted":
        return nn.CrossEntropyLoss(
            weight=kwargs["weight"]
        )

    raise ValueError(f"Loss không hỗ trợ: {kind}")


def class_weights(counts, beta=0.0):
    counts = torch.tensor(
        np.array(counts, copy=True),
        dtype=torch.float32,
    )

    if (counts <= 0).any():
        raise ValueError("Mỗi lớp train phải có ít nhất một ảnh")

    if not 0 <= beta < 1:
        raise ValueError("beta phải thuộc [0, 1)")

    if beta == 0:
        weights = 1.0 / counts
    else:
        weights = (1 - beta) / (1 - beta ** counts)

    return weights / weights.mean()


def mix_batch(x, y, alpha=1.0, mode="cutmix"):
    if alpha <= 0:
        raise ValueError("alpha phải > 0")

    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(
        x.size(0), device=x.device
    )

    if mode == "mixup":
        mixed = lam * x + (1 - lam) * x[permutation]

    elif mode == "cutmix":
        height, width = x.shape[-2:]
        ratio = np.sqrt(1 - lam)

        box_height = int(height * ratio)
        box_width = int(width * ratio)

        center_y = np.random.randint(height)
        center_x = np.random.randint(width)

        y1 = max(0, center_y - box_height // 2)
        y2 = min(height, center_y + (box_height + 1) // 2)
        x1 = max(0, center_x - box_width // 2)
        x2 = min(width, center_x + (box_width + 1) // 2)

        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = (
            x[permutation, :, y1:y2, x1:x2]
        )

        lam = 1 - (
            (y2 - y1) * (x2 - x1)
            / (height * width)
        )

    else:
        raise ValueError(f"Mix không hỗ trợ: {mode}")

    return mixed, (y, y[permutation], lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets

    return (
        lam * criterion(logits, y_a)
        + (1 - lam) * criterion(logits, y_b)
    )
