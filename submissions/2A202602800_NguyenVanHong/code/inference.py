import numpy as np
import copy
import torch
from torch.nn import functional as F


def softmax_numpy(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)

    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(
        axis=1, keepdims=True
    )


def view_identity(x):
    return x


def view_hflip(x):
    return x.flip(-1)


def views_multicrop(x, crop=224):
    height, width = x.shape[-2:]

    if crop > min(height, width):
        raise ValueError("Crop lớn hơn ảnh đầu vào")

    positions = [
        (0, 0),
        (0, width - crop),
        (height - crop, 0),
        (height - crop, width - crop),
        ((height - crop) // 2, (width - crop) // 2),
    ]

    return [
        x[:, :, top:top + crop, left:left + crop]
        for top, left in positions
    ]


def views_multiscale(x, sizes):
    """Caller must verify that its backbone supports the requested resolutions."""
    sizes = list(sizes)
    if not sizes or any(not isinstance(s, int) or s <= 0 for s in sizes):
        raise ValueError("sizes must contain positive integers")
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            for s in sizes]


def fuse_conv_bn(model):
    """Fuse proven forward pairs only; leave unknown architectures untouched.

    Supports adjacent Sequential Conv2d/BatchNorm2d and timm ResNet pairs.
    Named-child adjacency alone does not prove forward adjacency.
    """
    from torch import nn
    from timm.models.resnet import ResNet, BasicBlock, Bottleneck
    fused = copy.deepcopy(model).eval()

    def fuse_pair(parent, conv_name, bn_name):
        conv, bn = getattr(parent, conv_name), getattr(parent, bn_name)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            setattr(parent, conv_name, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
            setattr(parent, bn_name, nn.Identity())

    for module in list(fused.modules()):
        if isinstance(module, nn.Sequential):
            names = list(module._modules)
            for a, b in zip(names, names[1:]):
                fuse_pair(module, a, b)
        if isinstance(module, (BasicBlock, Bottleneck)):
            for index in (1, 2, 3):
                a, b = f"conv{index}", f"bn{index}"
                if hasattr(module, a) and hasattr(module, b):
                    fuse_pair(module, a, b)
        if isinstance(module, ResNet):
            fuse_pair(module, "conv1", "bn1")
    return fused


def aggregate_views(logits_per_view, space="prob"):
    if space == "prob":
        return np.mean(
            [softmax_numpy(z) for z in logits_per_view],
            axis=0,
        )

    if space == "logit":
        return softmax_numpy(
            np.mean(logits_per_view, axis=0)
        )

    raise ValueError(space)


def ensemble_probs(list_of_probs):
    probabilities = np.mean(list_of_probs, axis=0)
    return probabilities / probabilities.sum(
        axis=1, keepdims=True
    )


def fit_temperature(val_logits, val_labels):
    from scipy.optimize import minimize_scalar

    logits = torch.as_tensor(
        val_logits, dtype=torch.float64
    )
    labels = torch.as_tensor(
        val_labels, dtype=torch.long
    )

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        return F.cross_entropy(
            logits / temperature, labels
        ).item()

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError(result.message)

    return float(np.exp(result.x))


def apply_temperature(logits, T=None, *, temperature=None):
    """Keep starter's T keyword and accept the previous temperature alias."""
    if T is not None and temperature is not None:
        raise ValueError("Specify only T or temperature")
    temperature = T if T is not None else temperature
    if temperature is None:
        raise ValueError("Temperature is required")
    if temperature <= 0:
        raise ValueError("Temperature phải > 0")

    return softmax_numpy(
        np.asarray(logits) / temperature
    )


@torch.inference_mode()
def predict_logits(model, loader, device, view=None):
    model.eval()

    names = []
    labels = []
    logits = []

    for images, targets, filenames in loader:
        images = images.to(device)

        if view is not None:
            images = view(images)

        output = model(images).float()

        names.extend(filenames)
        labels.append(targets.numpy())
        logits.append(output.cpu().numpy())

    return (
        names,
        np.concatenate(labels),
        np.concatenate(logits),
    )


@torch.inference_mode()
def forward_method(model, images, method, temperature=1.0):
    if method == "I01":
        views = [images, view_hflip(images)]
    elif method == "I02":
        views = views_multicrop(images, crop=224)
    elif method in ["I00", "I03", "I04"]:
        views = [images]
    else:
        raise ValueError(method)

    with torch.autocast(
        device_type=images.device.type,
        enabled=method == "I04" and images.is_cuda,
    ):
        logits = [model(view).float() for view in views]

    if method == "I03":
        return (logits[0] / temperature).softmax(dim=1)

    # TTA: trung bình xác suất của các view.
    return torch.stack([
        output.softmax(dim=1)
        for output in logits
    ]).mean(dim=0)


@torch.inference_mode()
def predict_method(
    model, loader, device, method, temperature=1.0
):
    model.eval()

    names = []
    labels = []
    probabilities = []

    for images, targets, filenames in loader:
        output = forward_method(
            model,
            images.to(device),
            method,
            temperature,
        )

        names.extend(filenames)
        labels.append(targets.numpy())
        probabilities.append(output.cpu().numpy())

    return (
        names,
        np.concatenate(labels),
        np.concatenate(probabilities),
    )
