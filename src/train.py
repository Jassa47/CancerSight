"""Training loop for CancerSight."""

import logging
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)


def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=20, experiment=1):
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_model_path = f"experiments/experiment-{experiment:02d}.pth"

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation ---
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(val_loss)

        logger.info(f"Epoch {epoch} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with val_acc={best_val_acc:.4f}")

    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}")
    return model


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total
