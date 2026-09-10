"""Dataset utilities for aligned degraded/reference/crack-label triplets."""
import csv
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def source_id(path):
    return re.sub(r"_u\d+$", "", Path(path).stem)


def read_manifest(data_root, manifest="train/manifest.csv"):
    data_root = Path(data_root)
    manifest_path = data_root / manifest
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"degraded", "reference", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid or empty manifest: {manifest_path}")
    for row in rows:
        for key in required:
            path = data_root / row[key]
            if not path.is_file():
                raise FileNotFoundError(path)
    return rows


def grouped_train_val_split(rows, val_ratio=0.2, seed=42):
    groups = {}
    for row in rows:
        groups.setdefault(source_id(row["degraded"]), []).append(row)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    val_count = max(1, round(len(names) * val_ratio))
    val_names = set(names[:val_count])
    train_rows = [row for name in names if name not in val_names for row in groups[name]]
    val_rows = [row for name in names if name in val_names for row in groups[name]]
    if not train_rows:
        raise ValueError("Validation ratio leaves no training samples")
    return train_rows, val_rows


class FusionTripletDataset(Dataset):
    def __init__(self, data_root, rows, size=(256, 256), augment=False):
        self.data_root = Path(data_root)
        self.rows = rows
        self.height, self.width = size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        degraded = Image.open(self.data_root / row["degraded"]).convert("RGB")
        reference = Image.open(self.data_root / row["reference"]).convert("RGB")
        label = Image.open(self.data_root / row["label"]).convert("L")
        size = (self.width, self.height)
        degraded = degraded.resize(size, Image.Resampling.BICUBIC)
        reference = reference.resize(size, Image.Resampling.BICUBIC)
        label = label.resize(size, Image.Resampling.NEAREST)
        if self.augment and random.random() < 0.5:
            degraded = degraded.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            reference = reference.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        degraded = self._rgb_tensor(degraded)
        reference = self._rgb_tensor(reference)
        label_array = (np.asarray(label, dtype=np.uint8) > 127).astype(np.float32).copy()
        label = torch.from_numpy(label_array).unsqueeze(0)
        return {"degraded": degraded, "reference": reference,
                "label": label, "name": Path(row["degraded"]).stem}

    @staticmethod
    def _rgb_tensor(image):
        array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1)
