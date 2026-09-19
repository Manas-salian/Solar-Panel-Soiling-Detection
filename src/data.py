"""
Data pipeline: torchvision ImageFolder datasets + augmentation + DataLoaders.

Folder layout expected (produced by scripts/prepare_dataset.py):
    data/{train,val,test}/<class_name>/*.jpg
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    """Train: geometric + mild photometric augmentation for a small, web-scraped
    dataset. Eval: plain squash-resize (no centre crop, so nothing at the panel
    edges is thrown away)."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([transforms.RandomRotation(12)], p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), value="random"),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


def build_dataloaders(data_dir, img_size: int, batch_size: int, num_workers: int):
    """Return ({'train','val','test'} -> DataLoader, train_class_counts)."""
    data_dir = Path(data_dir)
    sets = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"{split_dir} not found. Run scripts/prepare_dataset.py first.")
        sets[split] = datasets.ImageFolder(split_dir, transform=build_transforms(img_size, split == "train"))
        if sets[split].classes != CLASS_NAMES:
            raise ValueError(
                f"Class folders in {split_dir} are {sets[split].classes}, expected {CLASS_NAMES}")

    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    loaders = {
        "train": DataLoader(sets["train"], batch_size=batch_size, shuffle=True, drop_last=False, **common),
        "val": DataLoader(sets["val"], batch_size=batch_size, shuffle=False, **common),
        "test": DataLoader(sets["test"], batch_size=batch_size, shuffle=False, **common),
    }
    train_counts = np.bincount(np.asarray(sets["train"].targets), minlength=NUM_CLASSES)
    return loaders, train_counts


def class_weights_from_counts(counts) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1, for nn.CrossEntropyLoss."""
    counts = np.asarray(counts, dtype=np.float64)
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)
