from dataclasses import dataclass, asdict
from pathlib import Path

import json
import random
import time
import platform
import copy

import numpy as np
import pandas as pd
import torch
import timm
import matplotlib.pyplot as plt

import dataset
import model as model_utils
import losses

from eval import compute_metrics, save_predictions


class EMA:
    """Optional helper; not used in the submitted experiments.

    Average parameters, copy buffers (including BN statistics) from the source.
    Evaluate self.model rather than the live training model.
    """
    def __init__(self, model, decay):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = dict(model.named_parameters())
        for name, parameter in self.model.named_parameters():
            parameter.lerp_(source[name].detach(), 1 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, buffer in self.model.named_buffers():
            buffer.copy_(source_buffers[name])

    def copy_to(self, model):
        model.load_state_dict(self.model.state_dict())


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0

    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0

    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0

    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None

    epochs: int = 12
    batch_size: int = 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0

    amp: bool = True
    num_workers: int = 2

    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"

    save_test_predictions: bool = False
    resume: bool = True


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg, split):
    return (
        Path(cfg.pred_dir)
        / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def softmax_numpy(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)

    probabilities = np.exp(logits)

    return probabilities / probabilities.sum(
        axis=1, keepdims=True
    )


def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model_utils.param_groups(
            model,
            cfg.lr_backbone,
            cfg.lr_head,
            cfg.weight_decay,
        )
    )


def build_scheduler(optimizer, cfg, steps_per_epoch):
    warmup_steps = round(
        cfg.warmup_epochs * steps_per_epoch
    )
    total_steps = cfg.epochs * steps_per_epoch

    def lr_multiplier(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        progress = (
            (step - warmup_steps)
            / max(1, total_steps - warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))

        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
    )


def train_one_epoch(
    model, loader, criterion, optimizer,
    scheduler, scaler, cfg, device,
):
    model.train()

    if cfg.init == "frozen":
        # Backbone/BN ở eval; chỉ head ở train.
        model.eval()
        model.get_classifier().train()

    total_loss = 0.0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        targets = labels

        if cfg.mix is not None:
            images, targets = losses.mix_batch(
                images,
                labels,
                alpha=cfg.mix_alpha,
                mode=cfg.mix,
            )

        with torch.autocast(
            device_type=device.type,
            enabled=cfg.amp and device.type == "cuda",
        ):
            logits = model(images)

            if cfg.mix is None:
                loss = criterion(logits, labels)
            else:
                loss = losses.mixed_loss(
                    criterion, logits, targets
                )

        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss không hữu hạn")

        scaler.scale(loss).backward()

        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # Không tiến scheduler nếu AMP bỏ qua optimizer step.
        if scaler.get_scale() >= old_scale:
            scheduler.step()

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

    if total_samples == 0:
        raise ValueError("Không có batch train")

    return {
        "train_loss": total_loss / total_samples,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()

    filenames = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    total_samples = 0

    for images, labels, names in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images).float()
        loss = criterion(logits, labels)

        filenames.extend(names)
        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

    return (
        filenames,
        np.concatenate(all_labels),
        np.concatenate(all_logits),
        total_loss / total_samples,
    )


def plot_curves(history, path, title):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(
        df["epoch"], df["train_loss"],
        label="Train objective",
    )
    axes[0].plot(
        df["epoch"], df["val_loss"],
        label="Val CE",
    )
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        df["epoch"], df["macro_f1"],
        label="Val macro-F1",
    )
    axes[1].plot(
        df["epoch"], df["top1"],
        label="Val top-1",
    )
    axes[1].set_ylabel("Metric")
    axes[1].legend()

    axes[2].plot(df["epoch"], df["lr"])
    axes[2].set_ylabel("LR — first parameter group")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")

    torch.save(state, temporary)
    temporary.replace(path)


def run(cfg):
    if cfg.save_test_predictions:
        raise ValueError(
            "Bước 1–2 chỉ dùng val. "
            "Phần chạy test sẽ thực hiện riêng ở Bước 4."
        )

    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    directory = run_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(cfg)
    config_path = directory / "config.json"

    if config_path.exists():
        previous = json.loads(config_path.read_text())

        if previous != config_dict:
            raise ValueError(
                "exp_id/seed này đã có cấu hình khác. "
                "Dùng exp_id mới nếu thay đổi cấu hình."
            )

    config_path.write_text(
        json.dumps(config_dict, indent=2),
        encoding="utf-8",
    )

    train_df, val_df, test_df = dataset.load_split(
        cfg.labels_dir, cfg.fold
    )

    dataset.check_split(
        train_df, val_df, test_df, cfg.images_dir
    )

    model = model_utils.build_model(
        cfg.backbone,
        init=cfg.init,
        drop_rate=cfg.drop_rate,
    ).to(device)

    pretrained_cfg = model.pretrained_cfg

    mean = pretrained_cfg.get(
        "mean", dataset.IMAGENET_MEAN
    )
    std = pretrained_cfg.get(
        "std", dataset.IMAGENET_STD
    )

    train_loader = dataset.make_loader(
        train_df,
        cfg.images_dir,
        dataset.build_transforms(
            True, cfg.img_size, cfg.aug, mean, std
        ),
        cfg.batch_size,
        True,
        sampler=cfg.sampler,
        num_workers=cfg.num_workers,
    )

    val_loader = dataset.make_loader(
        val_df,
        cfg.images_dir,
        dataset.build_transforms(
            False, cfg.img_size, mean=mean, std=std
        ),
        cfg.batch_size,
        False,
        num_workers=cfg.num_workers,
    )

    counts = (
        train_df["Label"].value_counts()
        .reindex(range(9), fill_value=0)
        .to_numpy()
    )

    weights = losses.class_weights(
        counts,
        beta=cfg.class_weight_beta or 0.0,
    )

    criterion = losses.build_criterion(
        cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=weights,
    ).to(device)

    # Val loss luôn dùng CE không trọng số.
    val_criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(
        optimizer, cfg, len(train_loader)
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.amp and device.type == "cuda",
    )

    history = []
    best_f1 = -1.0
    best_epoch = 0
    start_epoch = 0

    last_path = directory / "last.pt"

    if cfg.resume and last_path.exists():
        state = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])

        history = state["history"]
        best_f1 = state["best_f1"]
        best_epoch = state["best_epoch"]
        start_epoch = state["epoch"]

        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())

        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(
                [s.cpu() for s in state["cuda_rng"]]
            )

        print(f"Resume từ epoch {start_epoch}")

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda" else "CPU"
        ),
        "pretrained_cfg": pretrained_cfg,
        "preprocessing": {
            "train": "RandomResizedCrop + HorizontalFlip",
            "val": "Resize(size*256/224) + CenterCrop",
            "interpolation": "torchvision default bilinear",
            "mean": mean,
            "std": std,
        },
    }

    (directory / "environment.json").write_text(
        json.dumps(environment, indent=2, default=str),
        encoding="utf-8",
    )

    for epoch in range(start_epoch, cfg.epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        row = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scaler, cfg, device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        train_seconds = time.perf_counter() - start_time

        names, y_true, logits, val_loss = evaluate(
            model, val_loader, val_criterion, device
        )

        probabilities = softmax_numpy(logits)
        metrics = compute_metrics(
            y_true,
            probabilities.argmax(axis=1),
            probabilities,
        )

        row.update({
            "epoch": epoch + 1,
            "val_loss": float(val_loss),
            "macro_f1": float(metrics["macro_f1"]),
            "top1": float(metrics["top1"]),
            "train_seconds": float(train_seconds),
        })

        history.append(row)

        # Chỉ dùng > để hòa thì giữ epoch sớm hơn.
        if row["macro_f1"] > best_f1:
            best_f1 = row["macro_f1"]
            best_epoch = epoch + 1

            atomic_save(
                {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "macro_f1": best_f1,
                },
                directory / "best.pt",
            )

        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all()
                    if device.type == "cuda" else None
                ),
            },
            last_path,
        )

        pd.DataFrame(history).to_csv(
            directory / "history.csv",
            index=False,
        )

        plot_curves(
            history,
            Path(cfg.curves_dir)
            / f"{cfg.exp_id}_seed{cfg.seed}.png",
            f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}",
        )

        print(
            f"{cfg.exp_id} | epoch {epoch + 1}/{cfg.epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"macro-F1={row['macro_f1']:.4f} | "
            f"top1={row['top1']:.4f} | "
            f"train={train_seconds:.1f}s",
            flush=True,
        )

    best_state = torch.load(
        directory / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_state["model"])

    names, y_true, logits, _ = evaluate(
        model, val_loader, val_criterion, device
    )

    probabilities = softmax_numpy(logits)
    metrics = compute_metrics(
        y_true, probabilities.argmax(axis=1), probabilities
    )

    np.savez_compressed(
        directory / "val_logits.npz",
        filenames=np.asarray(names),
        y_true=y_true,
        logits=logits,
    )

    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)

    save_predictions(
        pred_path(cfg, "val"),
        names,
        y_true,
        probabilities,
    )

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "tag": pretrained_cfg.get("tag", ""),
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "img_size": cfg.img_size,
        "best_epoch": best_epoch,
        "macro_f1_val": float(metrics["macro_f1"]),
        "top1_val": float(metrics["top1"]),
        "params_M": model_utils.count_params(model),
        "gmac": model_utils.count_gmacs(
            model, cfg.img_size
        ),
        "train_seconds_per_epoch": float(
            np.mean([r["train_seconds"] for r in history])
        ),
        "gmac_tool": "fvcore; 1 multiply-add = 1",
    }

    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def parse_overrides(pairs):
    """Parse CLI overrides without silently accepting invalid fields/types."""
    defaults = asdict(Config())
    optional_numeric = {"class_weight_beta"}
    optional_strings = {"sampler", "mix"}
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE: {pair}")
        key, value = pair.split("=", 1)
        if key not in defaults:
            raise ValueError(f"Unknown Config field: {key}")
        default = defaults[key]
        if key in optional_numeric | optional_strings:
            parsed = None if value.lower() == "none" else (
                float(value) if key in optional_numeric else value
            )
        elif isinstance(default, bool):
            if value.lower() not in {"true", "false"}:
                raise ValueError(f"{key}: expected true or false")
            parsed = value.lower() == "true"
        else:
            parsed = type(default)(value)
        result[key] = parsed
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train one validation-selected experiment")
    parser.add_argument("--set", nargs="*", default=[])
    cfg = Config(**parse_overrides(parser.parse_args().set))
    print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()
