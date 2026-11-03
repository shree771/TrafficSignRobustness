
from pathlib import Path
from torchvision.datasets import GTSRB

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading the real GTSRB training dataset...")

train_dataset = GTSRB(
    root=str(DATA_DIR),
    split="train",
    download=True
)

print(f"Training images downloaded: {len(train_dataset)}")

print("\nDownloading the official GTSRB test dataset...")

test_dataset = GTSRB(
    root=str(DATA_DIR),
    split="test",
    download=True
)

print(f"Test images downloaded: {len(test_dataset)}")
print("\nDataset download and loading completed successfully.")
