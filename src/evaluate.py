"""Evaluation and TTA for CancerSight."""

import logging
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import wandb

from src.data.dataset import CLASSES

logger = logging.getLogger(__name__)


def evaluate(model, test_loader, device, tta=False, tta_transforms=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if tta and tta_transforms:
                # Average predictions across TTA transforms
                probs = torch.zeros(images.size(0), len(CLASSES)).to(device)
                for t in tta_transforms:
                    aug_images = torch.stack([t(img) for img in images.cpu()]).to(device)
                    outputs = model(aug_images)
                    probs += F.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
            else:
                outputs = model(images)
                preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(all_labels, all_preds, target_names=CLASSES)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"\n{report}")

    wandb.log({"test_acc": acc})

    return acc, report, cm
