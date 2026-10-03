from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = PROJECT_ROOT / "ml" / "data" / "metadata" / "splits" / "train.csv"


def main():
    df = pd.read_csv(TRAIN_CSV)

    counts = df["class_name"].value_counts().sort_index()

    total_images = len(df)
    num_classes = len(counts)

    weights = total_images / (num_classes * counts)

    weights_tensor = torch.tensor(
        weights.values,
        dtype=torch.float32
    )

    print("Number of training images:", total_images)
    print("Number of classes:", num_classes)

    print("\nClass weights:\n")

    for class_name, weight in zip(counts.index, weights):
        print(f"{class_name}: {weight:.4f}")

    print("\nWeight tensor:")
    print(weights_tensor)


if __name__ == "__main__":
    main()