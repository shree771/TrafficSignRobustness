
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
from torchvision.datasets import GTSRB

# Locate the project and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

print("Loading GTSRB dataset...")

train_dataset = GTSRB(
    root=str(DATA_DIR),
    split="train",
    download=False
)

test_dataset = GTSRB(
    root=str(DATA_DIR),
    split="test",
    download=False
)

print("\n========== DATASET VERIFICATION ==========")
print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))
train_labels = [label for _, label in train_dataset._samples]

print("Number of classes:", len(set(train_labels)))
# Check image dimensions and label range
image, label = train_dataset[0]
print("Sample image size:", image.size)
print("Sample image mode:", image.mode)
print("Sample label:", label)

train_labels = [target for _, target in train_dataset._samples]
print("Minimum label:", min(train_labels))
print("Maximum label:", max(train_labels))
print("Unique training labels:", len(set(train_labels)))

# Display 12 real training images
fig, axes = plt.subplots(3, 4, figsize=(12, 8))

for index, ax in enumerate(axes.flat):
    image, label = train_dataset[index]
    ax.imshow(image)
    ax.set_title(f"Class: {label}")
    ax.axis("off")

plt.tight_layout()

# Save the sample image grid
output_dir = PROJECT_ROOT / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "gtsrb_sample_images.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print("\nSample image grid saved to:", output_path)

plt.show()

print("\nDataset verification completed.")

from collections import Counter

class_counts = Counter(train_labels)

print("\n========== CLASS DISTRIBUTION ==========")

for class_id in sorted(class_counts):
    print(f"Class {class_id:02d}: {class_counts[class_id]} images")

print("\nTotal images counted:", sum(class_counts.values()))
print("Classes found:", len(class_counts))
print("Total training images:", len(train_dataset))
print("Total test images:", len(test_dataset))

