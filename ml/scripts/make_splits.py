from pathlib import Path
import hashlib
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    Path.home()
    / ".cache/kagglehub/datasets/mohitsingh1804/plantvillage/versions/1/PlantVillage"
)

OUTPUT_DIR = PROJECT_ROOT / "ml/data/metadata/splits"

SEED = 42


def get_images(split_dir):
    rows = []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append(
                    {
                        "path": str(image_path),
                        "class_name": class_dir.name,
                    }
                )

    return rows


def sha256_file(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def main():
    rows = (
        get_images(DATASET_DIR / "train")
        + get_images(DATASET_DIR / "val")
    )

    df = pd.DataFrame(rows)

    print(f"Total images: {len(df)}")
    print("\nComputing SHA-256 hashes...")

    df["sha256"] = df["path"].apply(sha256_file)

    # Images with identical SHA-256 hashes belong to the same group.
    hash_to_group = {}
    next_group_id = 0
    group_ids = []

    for file_hash in df["sha256"]:
        if file_hash not in hash_to_group:
            hash_to_group[file_hash] = f"group_{next_group_id:05d}"
            next_group_id += 1

        group_ids.append(hash_to_group[file_hash])

    df["group_id"] = group_ids

    duplicate_groups = (
        df.groupby("group_id")
        .size()
        .loc[lambda x: x > 1]
    )

    print(f"Exact duplicate groups: {len(duplicate_groups)}")
    print(f"Images involved in duplicate groups: {duplicate_groups.sum()}")

    # Deterministic class IDs.
    classes = sorted(df["class_name"].unique())
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    df["class_id"] = df["class_name"].map(class_to_id)

    rng = random.Random(SEED)

    train_parts = []
    val_parts = []
    test_parts = []

    # Split independently within each class.
    for class_name, class_df in df.groupby("class_name", sort=True):

        # Work with groups instead of individual images.
        groups = list(class_df.groupby("group_id"))

        rng.shuffle(groups)

        total_images = len(class_df)
        target_train = total_images * 0.80
        target_val = total_images * 0.10

        train_images = 0
        val_images = 0

        train_groups = []
        val_groups = []
        test_groups = []

        for group_id, group_df in groups:

            group_size = len(group_df)

            if train_images + group_size <= target_train:
                train_groups.append(group_df)
                train_images += group_size

            elif val_images + group_size <= target_val:
                val_groups.append(group_df)
                val_images += group_size

            else:
                test_groups.append(group_df)

        train_parts.extend(train_groups)
        val_parts.extend(val_groups)
        test_parts.extend(test_groups)

    train_df = pd.concat(train_parts).copy()
    val_df = pd.concat(val_parts).copy()
    test_df = pd.concat(test_parts).copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nFinal split:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)}")

    print(f"\nClasses: {len(classes)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
