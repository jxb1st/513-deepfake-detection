"""Dataset loading for real vs AI-generated image detection.

Strategy: try several HuggingFace datasets in priority order; fall back to
a smaller one if downloads fail. We resize to 128x128 for speed on
Apple Silicon MPS.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

IMG_SIZE = 96  # CIFAKE is 32x32; upsample to 96 for transfer-learning models
NUM_WORKERS = 2

# Dataset candidates (HuggingFace IDs, no auth required)
DATASET_CANDIDATES = [
    "dragonintelligence/CIFAKE-image-dataset",  # 60K real + 60K SD-1.4-generated
]


def set_seed(seed: int = 513) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_transforms(train: bool) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class HFImageDataset(Dataset):
    """Wraps a HuggingFace image dataset for binary real/fake classification.

    Standardizes column names by checking common variants and exposes a
    PyTorch-friendly (tensor, label) interface.
    """

    LABEL_KEYS = ["label", "labels", "class", "target"]
    IMAGE_KEYS = ["image", "img", "picture", "data"]

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

        cols = self.ds.column_names
        self.label_key = next(k for k in self.LABEL_KEYS if k in cols)
        self.image_key = next(k for k in self.IMAGE_KEYS if k in cols)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        img = row[self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = self.transform(img)
        y = int(row[self.label_key])
        return x, y


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    name: str
    n_train: int
    n_val: int
    n_test: int
    label_names: tuple[str, str]   # (label 0 name, label 1 name)


def _subsample(ds, max_n: int, seed: int = 513):
    """Stratified subsample that ALWAYS shuffles (CIFAKE rows are class-sorted)."""
    labels = np.array(ds["label"]) if "label" in ds.column_names \
        else np.array(ds["labels"])
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = (max_n // len(classes)) if max_n < len(ds) else None
    keep = []
    for c in classes:
        idx_c = np.where(labels == c)[0].tolist()
        rng.shuffle(idx_c)
        if per_class is not None:
            idx_c = idx_c[:per_class]
        keep.extend(idx_c)
    rng.shuffle(keep)
    return ds.select(keep)


def load_data(
    n_train: int = 6000,
    n_val: int = 1000,
    n_test: int = 2000,
    batch_size: int = 64,
    cache_dir: Optional[str] = None,
) -> DataBundle:
    """Load and return train/val/test loaders, trying datasets in priority order.

    The total downloaded sample is bounded by (n_train+n_val+n_test) so the
    runtime budget is predictable even on tiny laptops.
    """
    from datasets import load_dataset, DatasetDict

    last_err = None
    for name in DATASET_CANDIDATES:
        try:
            print(f"[data] trying {name}...")
            raw = load_dataset(name, cache_dir=cache_dir)
            print(f"[data] loaded {name}, splits = {list(raw.keys())}")

            if isinstance(raw, DatasetDict):
                if "train" in raw and "test" in raw:
                    train_full = raw["train"]
                    test_full = raw["test"]
                elif "train" in raw:
                    split = raw["train"].train_test_split(test_size=0.2, seed=513)
                    train_full, test_full = split["train"], split["test"]
                else:
                    only = list(raw.values())[0]
                    split = only.train_test_split(test_size=0.2, seed=513)
                    train_full, test_full = split["train"], split["test"]
            else:
                split = raw.train_test_split(test_size=0.2, seed=513)
                train_full, test_full = split["train"], split["test"]

            # CRITICAL: shuffle before splitting (CIFAKE is class-sorted).
            train_full = train_full.shuffle(seed=513)
            split2 = train_full.train_test_split(test_size=0.15, seed=513)
            train_ds = split2["train"]
            val_ds = split2["test"]

            train_ds = _subsample(train_ds, n_train)
            val_ds = _subsample(val_ds, n_val)
            test_ds = _subsample(test_full, n_test)

            train_t = HFImageDataset(train_ds, _build_transforms(train=True))
            val_t = HFImageDataset(val_ds, _build_transforms(train=False))
            test_t = HFImageDataset(test_ds, _build_transforms(train=False))

            label_names = ("real", "fake")
            try:
                feat = train_ds.features.get("label") or train_ds.features.get("labels")
                if feat is not None and hasattr(feat, "names"):
                    label_names = tuple(feat.names)[:2]
            except Exception:
                pass

            return DataBundle(
                train_loader=DataLoader(
                    train_t, batch_size=batch_size, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=False, drop_last=True),
                val_loader=DataLoader(
                    val_t, batch_size=batch_size, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=False),
                test_loader=DataLoader(
                    test_t, batch_size=batch_size, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=False),
                name=name, n_train=len(train_t), n_val=len(val_t), n_test=len(test_t),
                label_names=label_names,
            )
        except Exception as e:
            print(f"[data] {name} failed: {type(e).__name__}: {e}")
            last_err = e
            continue

    raise RuntimeError(
        f"All dataset candidates failed; last error: {last_err}")


if __name__ == "__main__":
    set_seed()
    bundle = load_data(n_train=200, n_val=50, n_test=100, batch_size=16)
    print(f"Loaded {bundle.name}: train={bundle.n_train} "
          f"val={bundle.n_val} test={bundle.n_test}")
    xb, yb = next(iter(bundle.train_loader))
    print(f"Batch shapes: x={tuple(xb.shape)} y={tuple(yb.shape)}, "
          f"dtype={xb.dtype} device={xb.device}")
