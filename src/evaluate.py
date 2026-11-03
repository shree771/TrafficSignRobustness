from pathlib import Path
import csv
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import existing project models and corruptions.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import LightweightCNN, create_resnet18
from corruptions import apply_corruption


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = (
    ROOT / "data" / "gtsrb" / "GTSRB"
    / "Final_Test" / "Images"
)

CSV_PATH = ROOT / "data" / "gtsrb" / "GT-final_test.csv"

CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

IMAGE_SIZE = 48
BATCH_SIZE = 32
DEVICE = torch.device("cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

MODEL_EXPERIMENTS = [
    ("lightweight_cnn", "none"),
    ("lightweight_cnn", "standard"),
    ("lightweight_cnn", "corruption"),
    ("resnet18", "none"),
    ("resnet18", "standard"),
    ("resnet18", "corruption"),
]

CORRUPTIONS = [
    "blur",
    "brightness_dark",
    "brightness_bright",
    "noise",
    "occlusion",
]


# ============================================================
# OFFICIAL TEST DATASET
# ============================================================

class OfficialTestDataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_csv,
        corruption=None,
        severity=0
    ):
        self.image_dir = image_dir
        self.corruption = corruption
        self.severity = severity
        self.samples = []

        with annotation_csv.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:

            reader = csv.DictReader(file, delimiter=";")

            for row in reader:
                filename = row["Filename"]
                label = int(row["ClassId"])
                image_path = image_dir / filename

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Missing test image: {image_path}"
                    )

                self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(
                f"No test images found in {annotation_csv}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        with Image.open(image_path) as original:
            image = original.convert("RGB")

        if self.corruption is not None:
            image = apply_corruption(
                image,
                self.corruption,
                self.severity
            )

        image = TRANSFORM(image)

        return image, label


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(probabilities, labels):

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels

    accuracy = float(correct.mean())

    # Expected Calibration Error with 15 bins.
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, 16)

    for i in range(15):

        if i == 14:
            mask = (
                (confidence >= bin_edges[i])
                & (confidence <= bin_edges[i + 1])
            )
        else:
            mask = (
                (confidence >= bin_edges[i])
                & (confidence < bin_edges[i + 1])
            )

        if mask.any():

            bin_accuracy = correct[mask].mean()
            bin_confidence = confidence[mask].mean()

            ece += (
                mask.mean()
                * abs(bin_accuracy - bin_confidence)
            )

    incorrect = ~correct

    if incorrect.any():
        avg_wrong_confidence = float(
            confidence[incorrect].mean()
        )
    else:
        avg_wrong_confidence = 0.0

    return {
        "accuracy": accuracy,
        "ece": float(ece),
        "avg_confidence_incorrect": avg_wrong_confidence,
        "predictions": predictions,
        "confidence": confidence,
        "correct": correct,
    }


# ============================================================
# LOAD SAVED MODELS
# ============================================================

def load_model(model_name, strategy):

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{model_name}_{strategy}"
        / "best_model.pth"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if model_name == "lightweight_cnn":

        model = LightweightCNN(num_classes=43)

    elif model_name == "resnet18":

        # Do not download weights; load saved checkpoint instead.
        model = create_resnet18(
            num_classes=43,
            pretrained=False
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Missing model_state_dict: {checkpoint_path}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Loaded model: {model_name} / {strategy}")

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_probabilities = []
    all_labels = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(DEVICE)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(labels.numpy())

    probabilities = np.concatenate(
        all_probabilities, axis=0
    )

    labels = np.concatenate(
        all_labels, axis=0
    )

    metrics = calculate_metrics(probabilities, labels)

    return metrics, labels


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    labels,
    predictions,
    model_name,
    strategy
):

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(43))
    )

    fig, ax = plt.subplots(figsize=(14, 14))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(range(43))
    )

    display.plot(
        ax=ax,
        include_values=False,
        xticks_rotation="vertical",
        colorbar=False
    )

    ax.set_title(
        f"{model_name} - {strategy} - Clean Test"
    )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_DIR
        / f"{model_name}_{strategy}_clean.png",
        dpi=160
    )

    plt.close(fig)


# ============================================================
# RELIABILITY DIAGRAM
# ============================================================

def save_reliability_diagram(
    metrics,
    model_name,
    strategy
):

    confidence = metrics["confidence"]
    correct = metrics["correct"]

    edges = np.linspace(0.0, 1.0, 16)

    bin_confidences = []
    bin_accuracies = []

    for i in range(15):

        if i == 14:
            mask = (
                (confidence >= edges[i])
                & (confidence <= edges[i + 1])
            )
        else:
            mask = (
                (confidence >= edges[i])
                & (confidence < edges[i + 1])
            )

        if mask.any():
            bin_confidences.append(
                float(confidence[mask].mean())
            )
            bin_accuracies.append(
                float(correct[mask].mean())
            )

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    ax.plot(
        bin_confidences,
        bin_accuracies,
        marker="o",
        label=f"{model_name} - {strategy}"
    )

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction Correct")

    ax.set_title(
        f"Reliability: {model_name} / {strategy}"
    )

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / f"reliability_{model_name}_{strategy}.png",
        dpi=160
    )

    plt.close(fig)


# ============================================================
# GRAPH GENERATION
# ============================================================

def save_severity_graphs(results):

    for corruption in CORRUPTIONS:

        subset = results[
            (results["corruption"] == corruption)
            & (results["severity"] > 0)
        ]

        for metric, ylabel in [
            ("accuracy", "Accuracy"),
            ("ece", "Expected Calibration Error")
        ]:

            fig, ax = plt.subplots(figsize=(10, 6))

            for model_name, strategy in MODEL_EXPERIMENTS:

                line = subset[
                    (subset["model"] == model_name)
                    & (subset["strategy"] == strategy)
                ].sort_values("severity")

                ax.plot(
                    line["severity"],
                    line[metric],
                    marker="o",
                    label=f"{model_name} - {strategy}"
                )

            ax.set_xlabel("Corruption Severity")
            ax.set_ylabel(ylabel)

            ax.set_title(
                f"{ylabel} vs Severity: {corruption}"
            )

            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

            fig.tight_layout()

            fig.savefig(
                FIGURES_DIR
                / f"{metric}_vs_severity_{corruption}.png",
                dpi=160
            )

            plt.close(fig)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test image directory missing: {TEST_DIR}"
        )

    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Test annotation CSV missing: {CSV_PATH}"
        )

    test_dataset = OfficialTestDataset(
        TEST_DIR,
        CSV_PATH
    )

    print(f"Official test images: {len(test_dataset)}")
    print(f"Device: {DEVICE}")

    # 1 clean + 5 corruption types * 5 severity levels = 26.
    total_conditions = (
        len(MODEL_EXPERIMENTS)
        * (1 + len(CORRUPTIONS) * 5)
    )

    current_condition = 0
    rows = []

    for model_name, strategy in MODEL_EXPERIMENTS:

        model = load_model(model_name, strategy)

        # ----------------------------------------------------
        # Clean test evaluation
        # ----------------------------------------------------

        current_condition += 1

        print(
            f"\n[{current_condition}/{total_conditions}] "
            f"{model_name} / {strategy} / clean"
        )

        clean_metrics, labels = evaluate_model(
            model,
            test_dataset
        )

        rows.append({
            "model": model_name,
            "strategy": strategy,
            "corruption": "clean",
            "severity": 0,
            "accuracy": clean_metrics["accuracy"],
            "ece": clean_metrics["ece"],
            "avg_confidence_incorrect":
                clean_metrics["avg_confidence_incorrect"]
        })

        save_confusion_matrix(
            labels,
            clean_metrics["predictions"],
            model_name,
            strategy
        )

        save_reliability_diagram(
            clean_metrics,
            model_name,
            strategy
        )

        # ----------------------------------------------------
        # Corrupted test evaluation
        # ----------------------------------------------------

        for corruption in CORRUPTIONS:

            for severity in range(1, 6):

                current_condition += 1

                print(
                    f"\n[{current_condition}/{total_conditions}] "
                    f"{model_name} / {strategy} / "
                    f"{corruption} / severity {severity}"
                )

                corrupted_dataset = OfficialTestDataset(
                    TEST_DIR,
                    CSV_PATH,
                    corruption=corruption,
                    severity=severity
                )

                metrics, _ = evaluate_model(
                    model,
                    corrupted_dataset
                )

                rows.append({
                    "model": model_name,
                    "strategy": strategy,
                    "corruption": corruption,
                    "severity": severity,
                    "accuracy": metrics["accuracy"],
                    "ece": metrics["ece"],
                    "avg_confidence_incorrect":
                        metrics["avg_confidence_incorrect"]
                })

        del model

    # --------------------------------------------------------
    # Save final metrics
    # --------------------------------------------------------

    results = pd.DataFrame(rows)

    results.to_csv(
        RESULTS_DIR / "evaluation_metrics.csv",
        index=False
    )

    summary = (
        results.groupby(
            ["model", "strategy", "corruption"],
            as_index=False
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_ece=("ece", "mean"),
            mean_wrong_confidence=(
                "avg_confidence_incorrect", "mean"
            )
        )
    )

    summary.to_csv(
        RESULTS_DIR / "evaluation_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save graphs
    # --------------------------------------------------------

    save_severity_graphs(results)

    print("\n" + "=" * 60)
    print("EVALUATION FINISHED")
    print("=" * 60)

    print(
        f"Conditions completed: "
        f"{current_condition}/{total_conditions}"
    )

    print(
        f"Metrics: "
        f"{RESULTS_DIR / 'evaluation_metrics.csv'}"
    )

    print(
        f"Summary: "
        f"{RESULTS_DIR / 'evaluation_summary.csv'}"
    )

    print(f"Graphs: {FIGURES_DIR}")
    print(f"Confusion matrices: {CONFUSION_DIR}")


if __name__ == "__main__":
    main()