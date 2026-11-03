
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

image_root = (
    PROJECT_ROOT
    / "data"
    / "gtsrb"
    / "GTSRB"
    / "Final_Training"
    / "Images"
)

annotation_file = image_root / "GT-final_train.csv"

print("Training image folder exists:", image_root.exists())
print("Training annotation file exists:", annotation_file.exists())

image_count = sum(1 for _ in image_root.rglob("*.ppm"))
class_folders = sorted(
    folder.name
    for folder in image_root.iterdir()
    if folder.is_dir()
)

print("Training images:", image_count)
print("Class folders:", len(class_folders))

if annotation_file.exists():
    with annotation_file.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.reader(file, delimiter=";")
        header = next(reader, None)
        print("Annotation columns:", header)
        print("First annotation:", next(reader, None))
else:
    print("Check the extracted folder for GT-final_train.csv.")

if image_count == 39209 and len(class_folders) == 43:
    print("\nDataset image verification PASSED.")
else:
    print("\nDataset image verification FAILED.")
