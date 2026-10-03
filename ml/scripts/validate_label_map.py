from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    Path.home()
    / ".cache/kagglehub/datasets/mohitsingh1804/plantvillage/versions/1/PlantVillage"
)

TRAIN_DIR = DATASET_DIR / "train"
LABEL_MAP_FILE = PROJECT_ROOT / "ml/data/metadata/label_map.yaml"


def main():
    with open(LABEL_MAP_FILE, "r") as file:
        label_map = yaml.safe_load(file)

    dataset_classes = sorted(
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    )

    mapped_classes = [
        label_map[class_id]["class_name"]
        for class_id in sorted(label_map, key=int)
    ]

    print("Dataset classes:", len(dataset_classes))
    print("Mapped classes:", len(mapped_classes))

    print("IDs:", sorted(label_map, key=int))

    print("Classes match:", dataset_classes == mapped_classes)

    required_fields = {"class_name", "plant", "condition"}

    valid_fields = all(
        required_fields.issubset(entry.keys())
        for entry in label_map.values()
    )

    print("Required fields present:", valid_fields)

    if dataset_classes != mapped_classes:
        raise ValueError("Dataset classes and label map do not match.")

    if len(label_map) != 38:
        raise ValueError("Expected exactly 38 classes.")

    if not valid_fields:
        raise ValueError("One or more label entries are missing fields.")

    print("\nLabel map validation PASSED.")


if __name__ == "__main__":
    main()