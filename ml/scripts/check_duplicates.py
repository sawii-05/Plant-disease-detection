from pathlib import Path
from collections import defaultdict

import hashlib
import imagehash
import kagglehub
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_dataset_path() -> Path:
    dataset_path = kagglehub.dataset_download(
        "mohitsingh1804/plantvillage"
    )
    return Path(dataset_path) / "PlantVillage"


def get_image_files(folder: Path):
    return [
        file
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def exact_hash(image_path: Path):
    hasher = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def perceptual_hash(image_path: Path):
    with Image.open(image_path) as image:
        return imagehash.phash(image)


def main():
    dataset_path = get_dataset_path()

    train_path = dataset_path / "train"
    val_path = dataset_path / "val"

    train_images = get_image_files(train_path)
    val_images = get_image_files(val_path)

    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

    # --------------------------------------------------
    # 1. Exact duplicate detection
    # --------------------------------------------------

    print("\nChecking exact duplicates...")

    train_hashes = defaultdict(list)

    for image_path in train_images:
        train_hashes[exact_hash(image_path)].append(image_path)

    exact_cross_split = []

    for image_path in val_images:
        file_hash = exact_hash(image_path)

        if file_hash in train_hashes:
            for train_image in train_hashes[file_hash]:
                exact_cross_split.append(
                    (train_image, image_path)
                )

    print(
        "Exact train-validation duplicate pairs:",
        len(exact_cross_split),
    )

    # --------------------------------------------------
    # 2. Perceptual hash candidates
    # --------------------------------------------------

    print("\nComputing perceptual hashes...")

    train_phashes = []

    for index, image_path in enumerate(train_images, start=1):
        train_phashes.append(
            (image_path, perceptual_hash(image_path))
        )

        if index % 5000 == 0:
            print(f"  Train: {index}/{len(train_images)}")

    val_phashes = []

    for index, image_path in enumerate(val_images, start=1):
        val_phashes.append(
            (image_path, perceptual_hash(image_path))
        )

        if index % 2000 == 0:
            print(f"  Validation: {index}/{len(val_images)}")

    # --------------------------------------------------
    # 3. Candidate near-duplicate search
    # --------------------------------------------------

    print("\nSearching for near-duplicate candidates...")

    threshold = 2
    candidates = []

    # Compare only within the same class.
    # Different classes cannot be legitimate duplicates
    # for this dataset structure.
    train_by_class = defaultdict(list)

    for image_path, image_hash in train_phashes:
        class_name = image_path.parent.name
        train_by_class[class_name].append(
            (image_path, image_hash)
        )

    for val_path, val_hash in val_phashes:
        class_name = val_path.parent.name

        for train_path, train_hash in train_by_class[class_name]:
            distance = val_hash - train_hash

            if distance <= threshold:
                candidates.append(
                    (distance, train_path, val_path)
                )

    candidates.sort(key=lambda x: x[0])

    print()
    print("Near-duplicate candidates:", len(candidates))

    print("\nClosest candidates:")

    for distance, train_path, val_path in candidates[:20]:
        print(f"\nDistance: {distance}")
        print(f"Train: {train_path}")
        print(f"Val:   {val_path}")


if __name__ == "__main__":
    main()
