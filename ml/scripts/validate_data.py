from pathlib import Path

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


def check_images(folder: Path):
    image_files = get_image_files(folder)

    corrupted = []
    non_rgb = []

    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                if image.mode not in {"RGB", "RGBA"}:
                    non_rgb.append((image_path, image.mode))

        except Exception as error:
            corrupted.append((image_path, str(error)))

    return image_files, corrupted, non_rgb


def count_classes(folder: Path):
    class_counts = {}

    for class_folder in sorted(folder.iterdir()):
        if not class_folder.is_dir():
            continue

        count = sum(
            1
            for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        class_counts[class_folder.name] = count

    return class_counts


def main():
    dataset_path = get_dataset_path()

    train_path = dataset_path / "train"
    val_path = dataset_path / "val"

    train_counts = count_classes(train_path)
    val_counts = count_classes(val_path)

    print(f"Dataset: {dataset_path}")
    print()

    train_images, train_corrupted, train_non_rgb = check_images(train_path)
    val_images, val_corrupted, val_non_rgb = check_images(val_path)

    print("Train images:", len(train_images))
    print("Validation images:", len(val_images))

    print()
    print("Train corrupted:", len(train_corrupted))
    print("Validation corrupted:", len(val_corrupted))

    print()
    print("Train non-RGB/RGBA:", len(train_non_rgb))
    print("Validation non-RGB/RGBA:", len(val_non_rgb))

    print()
    print("Train classes:", len(train_counts))
    print("Validation classes:", len(val_counts))
    print("Classes identical:", set(train_counts) == set(val_counts))

    print()
    print("Smallest training classes:")
    for class_name, count in sorted(
        train_counts.items(), key=lambda x: x[1]
    )[:5]:
        print(f"  {class_name}: {count}")

    print()
    print("Largest training classes:")
    for class_name, count in sorted(
        train_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
