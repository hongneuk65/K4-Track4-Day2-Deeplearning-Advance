import copy

import torch
import timm


def build_model(
    name,
    pretrained=True,
    num_classes=9,
    drop_rate=0.0,
    init="finetune",
):
    if init not in ["scratch", "frozen", "finetune"]:
        raise ValueError(f"Init không hỗ trợ: {init}")

    model = timm.create_model(
        name,
        pretrained=pretrained and init != "scratch",
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model):
    # Đóng băng toàn bộ, sau đó mở lại head.
    model.requires_grad_(False)
    model.get_classifier().requires_grad_(True)
    model.eval()


def param_groups(
    model,
    lr_backbone,
    lr_head,
    weight_decay,
):
    head_ids = {
        id(parameter)
        for parameter in model.get_classifier().parameters()
    }

    groups = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        learning_rate = (
            lr_head if id(parameter) in head_ids else lr_backbone
        )

        # Bias và tham số norm không nhận weight decay.
        decay = (
            0.0
            if parameter.ndim <= 1 or name.endswith(".bias")
            else weight_decay
        )

        key = (learning_rate, decay)
        groups.setdefault(key, []).append(parameter)

    return [
        {
            "params": parameters,
            "lr": learning_rate,
            "weight_decay": decay,
        }
        for (learning_rate, decay), parameters in groups.items()
    ]


def count_params(model):
    # Đếm cả tham số bị đóng băng, đơn vị triệu.
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size=224):
    from fvcore.nn import FlopCountAnalysis

    cpu_model = copy.deepcopy(model).cpu().float().eval()
    sample = torch.zeros(1, 3, img_size, img_size)

    analysis = FlopCountAnalysis(cpu_model, sample)
    gmac = analysis.total() / 1e9

    unsupported = {
        str(operation): int(count)
        for operation, count in analysis.unsupported_ops().items()
    }

    if unsupported:
        print("Lưu ý GMAC: toán tử chưa được fvcore tính:", unsupported)

    return float(gmac)
