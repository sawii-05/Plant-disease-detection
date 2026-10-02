from pathlib import Path
import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = kagglehub.dataset_download(
        "mohitsingh1804/plantvillage"
    )

    print(f"Kaggle dataset location: {dataset_path}")
    print(f"Project raw-data directory: {DATA_DIR}")


if __name__ == "__main__":
    main()