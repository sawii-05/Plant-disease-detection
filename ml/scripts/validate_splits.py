from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "ml" / "data" / "metadata" / "splits"

FILES = {
    "train": SPLIT_DIR / "train.csv",
    "val": SPLIT_DIR / "val.csv",
    "test": SPLIT_DIR / "test.csv",
}


def main():
    data = {
        name: pd.read_csv(path)
        for name, path in FILES.items()
    }

    # 1. Total images
    total = sum(len(df) for df in data.values())
    print(f"Total images: {total}")

    # 2. Number of classes
    classes = set()
    for df in data.values():
        classes.update(df["class_name"])

    print(f"Classes: {len(classes)}")

    # 3. Check group leakage
    group_sets = {
        name: set(df["group_id"])
        for name, df in data.items()
    }

    train_val = group_sets["train"] & group_sets["val"]
    train_test = group_sets["train"] & group_sets["test"]
    val_test = group_sets["val"] & group_sets["test"]

    print(f"Train-Val shared groups: {len(train_val)}")
    print(f"Train-Test shared groups: {len(train_test)}")
    print(f"Val-Test shared groups: {len(val_test)}")

    # 4. Check every class exists in every split
    for name, df in data.items():
        missing = classes - set(df["class_name"])

        if missing:
            print(f"{name}: MISSING classes -> {sorted(missing)}")
        else:
            print(f"{name}: all 38 classes present")

    # 5. Smallest class in each split
    print("\nSmallest class counts:")

    for name, df in data.items():
        counts = df["class_name"].value_counts()
        print(f"{name}: {counts.min()}")

    # 6. Overall result
    passed = (
        total == 54305
        and len(classes) == 38
        and len(train_val) == 0
        and len(train_test) == 0
        and len(val_test) == 0
        and all(
            set(df["class_name"]) == classes
            for df in data.values()
        )
    )

    print("\nValidation:", "PASSED" if passed else "FAILED")


if __name__ == "__main__":
    main()