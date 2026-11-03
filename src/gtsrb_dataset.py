
from pathlib import Path
import csv

from PIL import Image
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for class_id in range(43):
            class_folder = (
                self.root / f"{class_id:05d}"
            )
            annotation_file = (
                class_folder / f"GT-{class_id:05d}.csv"
            )

            if not annotation_file.exists():
                raise FileNotFoundError(
                    f"Missing annotation: {annotation_file}"
                )

            with annotation_file.open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                reader = csv.DictReader(file, delimiter=";")

                for row in reader:
                    image_path = class_folder / row["Filename"]
                    label = int(row["ClassId"])

                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"Missing image: {image_path}"
                        )

                    self.samples.append((image_path, label))

        if len(self.samples) != 39209:
            raise ValueError(
                f"Expected 39209 training images, "
                f"but found {len(self.samples)}"
            )

        print(
            f"Loaded {len(self.samples)} real GTSRB training images "
            f"across 43 classes."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    image_root = (
        project_root
        / "data"
        / "gtsrb"
        / "GTSRB"
        / "Final_Training"
        / "Images"
    )

    dataset = GTSRBDataset(image_root)

    print("Dataset size:", len(dataset))
    print("First sample:", dataset.samples[0])
    print("Last sample:", dataset.samples[-1])

    labels = [label for _, label in dataset.samples]
    print("Unique classes:", len(set(labels)))
