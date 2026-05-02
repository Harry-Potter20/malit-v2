from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = ["Uninfected", "Parasitized"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def build_transforms(
    image_size: int = 224,
    is_train: bool = True,
) -> Callable:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class MalariaDataset(Dataset):
    """
    NIH Malaria Cell Images dataset.

    Expected directory layout:
        root/
            Uninfected/  *.png
            Parasitized/ *.png
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: Callable | None = None,
    ):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        transform: Callable | None = None,
    ) -> "MalariaDataset":
        root = Path(root)
        paths, labels = [], []
        for class_name, idx in CLASS_TO_IDX.items():
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in cls.EXTENSIONS:
                    paths.append(p)
                    labels.append(idx)
        return cls(paths, labels, transform)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    @property
    def class_names(self) -> list[str]:
        return CLASS_NAMES

    def subset(self, indices: list[int], transform: Callable | None = None) -> "MalariaDataset":
        paths = [self.image_paths[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        return MalariaDataset(paths, labels, transform or self.transform)
