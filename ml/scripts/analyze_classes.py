from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "ml" / "data" / "metadata" / "splits"


def main():
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    val_df = pd.read_csv(SPLIT_DIR / "val.csv")
    test_df = pd.read_csv(SPLIT_DIR / "test.csv")

    train_counts = train_df["class_name"].value_counts().sort_index()
    val_counts = val_df["class_name"].value_counts().sort_index()
    test_counts = test_df["class_name"].value_counts().sort_index()

    report = pd.DataFrame({
        "train": train_counts,
        "validation": val_counts,
        "test": test_counts,
    })

    report["total"] = (
        report["train"]
        + report["validation"]
        + report["test"]
    )

    report["train_percentage"] = (
        report["train"] / len(train_df) * 100
    )

    print("\nClass distribution:\n")
    print(report.to_string())

    print("\nMost frequent training classes:")
    print(train_counts.nlargest(5))

    print("\nLeast frequent training classes:")
    print(train_counts.nsmallest(5))

    print("\nImbalance ratio:")
    print(
        f"Max / Min = "
        f"{train_counts.max() / train_counts.min():.2f}"
    )

    output_path = PROJECT_ROOT / "ml" / "data" / "metadata" / "class_distribution.csv"
    report.to_csv(output_path)

    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()