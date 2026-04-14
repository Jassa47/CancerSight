"""HAM10000 Dataset class and dataloader factory."""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dirs: list, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        label = CLASS_TO_IDX[row["dx"]]

        img_path = None
        for d in self.img_dirs:
            candidate = os.path.join(d, img_id + ".jpg")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_id}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(train_df, val_df, test_df, img_dirs, train_transform, val_transform, batch_size=32, num_workers=2):
    train_dataset = HAM10000Dataset(train_df, img_dirs, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, img_dirs, transform=val_transform)
    test_dataset = HAM10000Dataset(test_df, img_dirs, transform=val_transform)

    # Weighted sampler for class imbalance
    labels = [CLASS_TO_IDX[dx] for dx in train_df["dx"]]
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_counts.float()
    sample_weights = weights[torch.tensor(labels)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
