from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from dataset import get_dataloaders


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "metadata"
    / "splits"
    / "train.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_CLASSES = 38

BATCH_SIZE = 32

# Quick sanity run
EPOCHS = 1

LEARNING_RATE = 1e-3

# Number of images for this first test
TRAIN_LIMIT = 2000
VAL_LIMIT = 500


# --------------------------------------------------
# Device
# --------------------------------------------------

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    device = get_device()

    print(f"Using device: {device}", flush=True)


    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------

    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    # Use only a small subset for the first test
    train_loader.dataset.data = (
        train_loader.dataset.data
        .iloc[:TRAIN_LIMIT]
        .reset_index(drop=True)
    )

    val_loader.dataset.data = (
        val_loader.dataset.data
        .iloc[:VAL_LIMIT]
        .reset_index(drop=True)
    )

    print(
        f"Training images: {len(train_loader.dataset)}",
        flush=True
    )

    print(
        f"Validation images: {len(val_loader.dataset)}",
        flush=True
    )


    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("Loading ResNet18...", flush=True)

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Replace ImageNet's 1000 outputs
    # with our 38 PlantVillage classes
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = model.to(device)


    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    # For this quick sanity test we use
    # normal CrossEntropyLoss.
    #
    # Class-weighted loss will be used
    # in the real training run.

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        print(
            f"\nStarting Epoch {epoch + 1}/{EPOCHS}",
            flush=True
        )

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0


        for batch_idx, (images, labels) in enumerate(
            train_loader
        ):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update model
            optimizer.step()


            # Statistics
            train_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)


            # Progress
            if (batch_idx + 1) % 10 == 0:

                print(
                    f"Batch {batch_idx + 1} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True
                )


        train_loss /= train_total

        train_accuracy = (
            train_correct / train_total
        )


        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        print(
            "\nRunning validation...",
            flush=True
        )

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += (
                    loss.item() * images.size(0)
                )

                predictions = outputs.argmax(
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)


        val_loss /= val_total

        val_accuracy = (
            val_correct / val_total
        )


        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}",
            flush=True
        )

        print(
            f"Train Loss: {train_loss:.4f}",
            flush=True
        )

        print(
            f"Train Accuracy: {train_accuracy:.4f}",
            flush=True
        )

        print(
            f"Validation Loss: {val_loss:.4f}",
            flush=True
        )

        print(
            f"Validation Accuracy: {val_accuracy:.4f}",
            flush=True
        )


    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_path = MODEL_DIR / "resnet18_sanity.pth"

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        f"\nModel saved to: {model_path}",
        flush=True
    )


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()