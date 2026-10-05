from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T


NUM_CLASSES = 9

CLASS_NAMES = [
    "Chinee Apple",
    "Lantana",
    "Parkinsonia",
    "Parthenium",
    "Prickly Acacia",
    "Rubber Vine",
    "Siam Weed",
    "Snake Weed",
    "Negatives",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir, fold=0):
    """Đọc nguyên bản ba CSV, không lọc hay chia lại dữ liệu."""
    labels_dir = Path(labels_dir)

    frames = []
    for split in ["train", "val", "test"]:
        path = labels_dir / f"{split}_subset{fold}.csv"
        df = pd.read_csv(path)

        required = {"Filename", "Label"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path}: thiếu cột {required - set(df.columns)}")

        frames.append(df)

    return tuple(frames)


def check_split(train_df, val_df, test_df, images_dir):
    """Kiểm tra split chính thức và trả số liệu phục vụ báo cáo."""
    images_dir = Path(images_dir)

    frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    filename_sets = {}

    for name, df in frames.items():
        if df["Filename"].isna().any() or df["Label"].isna().any():
            raise ValueError(f"{name}: có tên ảnh hoặc nhãn bị thiếu")

        if df["Filename"].duplicated().any():
            raise ValueError(f"{name}: có tên ảnh trùng trong CSV")

        if not df["Label"].isin(range(NUM_CLASSES)).all():
            raise ValueError(f"{name}: nhãn phải thuộc 0..8")

        missing = [
            filename
            for filename in df["Filename"]
            if not (images_dir / filename).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{name}: thiếu {len(missing)} ảnh. Ví dụ: {missing[:5]}"
            )

        filename_sets[name] = set(df["Filename"])

    overlap = {
        "train_val": len(filename_sets["train"] & filename_sets["val"]),
        "train_test": len(filename_sets["train"] & filename_sets["test"]),
        "val_test": len(filename_sets["val"] & filename_sets["test"]),
    }

    if any(overlap.values()):
        raise ValueError(f"Có ảnh xuất hiện ở nhiều tập: {overlap}")

    union = set.union(*filename_sets.values())

    if len(union) != 17509:
        raise ValueError(f"Hợp ba tập có {len(union)} ảnh, kỳ vọng 17509")

    for name, expected_ratio in [
        ("train", 0.6),
        ("val", 0.2),
        ("test", 0.2),
    ]:
        actual_ratio = len(frames[name]) / len(union)

        if abs(actual_ratio - expected_ratio) > 0.01:
            raise ValueError(
                f"{name}: tỷ lệ {actual_ratio:.2%} lệch khỏi "
                f"{expected_ratio:.0%} quá 1 điểm phần trăm. "
                "Kiểm tra CSV và báo giảng viên trước khi train."
            )

    result = {
        "n": {name: len(df) for name, df in frames.items()},
        "per_class": {
            name: {
                int(label): int(count)
                for label, count in (
                    df["Label"]
                    .value_counts()
                    .reindex(range(NUM_CLASSES), fill_value=0)
                    .items()
                )
            }
            for name, df in frames.items()
        },
        "overlap": overlap,
        "union_count": len(union),
    }

    print("Số ảnh mỗi tập:", result["n"])
    print("Giao giữa các tập:", result["overlap"])
    print("Hợp ba tập:", result["union_count"])
    print("Mọi ảnh trong CSV đều tồn tại.")

    return result


def build_transforms(
    train,
    img_size=224,
    aug="basic",
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    """Train có augmentation; val/test dùng tiền xử lý cố định."""
    if train:
        operations = [
            T.RandomResizedCrop(
                img_size,
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(),
        ]

        if aug == "color":
            operations.append(T.ColorJitter(0.2, 0.2, 0.2, 0.1))
        elif aug == "trivial":
            operations.append(T.TrivialAugmentWide())
        elif aug == "randaug":
            operations.append(T.RandAugment())
        elif aug != "basic":
            raise ValueError(f"Augmentation không hỗ trợ: {aug}")
    else:
        operations = [
            T.Resize(
                round(img_size * 256 / 224),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.CenterCrop(img_size),
        ]

    operations.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(operations)


class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = str(row["Filename"])
        label = int(row["Label"])

        with Image.open(self.images_dir / filename) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, filename


def seed_worker(worker_id):
    """Seed cho augmentation trong mỗi DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    df,
    images_dir,
    transform,
    batch_size,
    train,
    sampler=None,
    num_workers=2,
):
    dataset = DeepWeedsDataset(df, images_dir, transform)

    if sampler not in [None, "balanced"]:
        raise ValueError(f"Sampler không hỗ trợ: {sampler}")

    if not train and sampler is not None:
        raise ValueError("Val/test không dùng sampler cân bằng")

    weighted_sampler = None

    if train and sampler == "balanced":
        counts = df["Label"].value_counts()
        weights = [
            1.0 / counts[label]
            for label in df["Label"]
        ]

        weighted_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(df),
            replacement=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and weighted_sampler is None,
        sampler=weighted_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        worker_init_fn=seed_worker,
    )
