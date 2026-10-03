from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = Path.home() / ".cache/kagglehub/datasets/mohitsingh1804/plantvillage/versions/1/PlantVillage"
TRAIN_DIR = DATASET_DIR / "train"

OUTPUT_FILE = PROJECT_ROOT / "ml/data/metadata/label_map.yaml"


def main():
    classes = sorted(
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    )

    label_map = {}

    for class_id, class_name in enumerate(classes):
        plant, condition = class_name.split("___", maxsplit=1)

        label_map[class_id] = {
            "class_name": class_name,
            "plant": plant,
            "condition": condition,
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        yaml.safe_dump(label_map, f, sort_keys=False)

    print(f"Created label map with {len(classes)} classes.")
    print(f"Saved to: {OUTPUT_FILE}")

    for class_id, info in label_map.items():
        print(f"{class_id}: {info['class_name']}")


if __name__ == "__main__":
    main()
