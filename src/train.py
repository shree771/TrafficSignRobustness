import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from gtsrb_dataset import GTSRBDataset
from models import LightweightCNN, create_resnet18
from augmentations import get_clean_transform, get_training_transform


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# DATASET VIEW
# ============================================================

class TransformedSubset(Dataset):

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_index = self.indices[position]

        image_path, label = self.base_dataset.samples[
            original_index
        ]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# MODEL FACTORY
# ============================================================

def make_model(model_name, pretrained):

    if model_name == "lightweight_cnn":
        return LightweightCNN(num_classes=43)

    if model_name == "resnet18":
        return create_resnet18(
            num_classes=43,
            pretrained=pretrained
        )

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, loader, criterion, device):

    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            count = labels.size(0)

            loss_sum += loss.item() * count

            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total += count

    if total == 0:
        return 0.0, 0.0

    average_loss = loss_sum / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ============================================================
# CSV UTILITIES
# ============================================================

def load_csv_rows(path):

    if not path.exists():
        return []

    with path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        return list(csv.DictReader(file))


def write_csv(path, fieldnames, rows, append=False):

    mode = "a" if append else "w"

    needs_header = (
        not append
        or not path.exists()
        or path.stat().st_size == 0
    )

    with path.open(
        mode,
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if needs_header:
            writer.writeheader()

        writer.writerows(rows)


# ============================================================
# CHECKPOINT CONFIGURATION VALIDATION
# ============================================================

def config_matches(saved_config, config):

    keys = (
        "seed",
        "image_size",
        "num_classes",
        "batch_size",
        "validation_fraction",
        "learning_rate",
        "weight_decay",
        "epochs",
        "early_stopping_patience",
        "pretrained_resnet18",
    )

    return all(
        saved_config.get(key) == config.get(key)
        for key in keys
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    seed = config["seed"]
    set_seed(seed)

    device = torch.device("cpu")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    torch.backends.mkldnn.enabled = True

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    num_workers = config.get("num_workers", 0)
    validation_fraction = config["validation_fraction"]

    print("=" * 65)
    print("TRAFFIC SIGN RECOGNITION - RESUMABLE TRAINING")
    print("=" * 65)

    print("Device:", device)
    print("Batch size:", batch_size)
    print("Maximum epochs:", epochs)

    print(
        "Total experiments:",
        len(config["models"])
        * len(config["augmentation_strategies"])
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    image_root = (
        PROJECT_ROOT
        / "data"
        / "gtsrb"
        / "GTSRB"
        / "Final_Training"
        / "Images"
    )

    if not image_root.exists():

        raise FileNotFoundError(
            f"Training directory not found: {image_root}"
        )

    print("\nLoading dataset index...")

    base_dataset = GTSRBDataset(image_root)

    total_samples = len(base_dataset)

    if total_samples != 39209:

        raise ValueError(
            f"Expected 39209 training samples; "
            f"got {total_samples}."
        )

    labels = [
        label
        for _, label in base_dataset.samples
    ]

    if len(set(labels)) != 43:

        raise ValueError(
            "The dataset does not contain all 43 classes."
        )

    print("Verified training images:", total_samples)
    print("Verified classes:", len(set(labels)))

    # --------------------------------------------------------
    # DETERMINISTIC TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    generator = torch.Generator().manual_seed(seed)

    permutation = torch.randperm(
        total_samples,
        generator=generator
    ).tolist()

    val_size = int(
        total_samples * validation_fraction
    )

    train_indices = permutation[:-val_size]
    val_indices = permutation[-val_size:]

    print("Training samples:", len(train_indices))
    print("Validation samples:", len(val_indices))

    # --------------------------------------------------------
    # VALIDATION DATASET
    # --------------------------------------------------------

    validation_dataset = TransformedSubset(
        base_dataset,
        val_indices,
        get_clean_transform()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history_path = (
        RESULTS_DIR / "training_history.csv"
    )

    history_fields = [
        "model",
        "strategy",
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "epoch_seconds",
    ]

    # Preserve existing history instead of deleting it.
    completed_count = 0

    # --------------------------------------------------------
    # EXPERIMENT LOOP
    # --------------------------------------------------------

    for model_name in config["models"]:

        for strategy in config["augmentation_strategies"]:

            experiment_name = (
                f"{model_name}_{strategy}"
            )

            experiment_dir = (
                CHECKPOINT_DIR / experiment_name
            )

            experiment_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            best_checkpoint_path = (
                experiment_dir / "best_model.pth"
            )

            last_checkpoint_path = (
                experiment_dir / "last_checkpoint.pth"
            )

            status_path = (
                experiment_dir / "status.json"
            )

            # ------------------------------------------------
            # SKIP COMPLETED EXPERIMENTS
            # ------------------------------------------------

            if status_path.exists():

                try:

                    status = json.loads(
                        status_path.read_text(
                            encoding="utf-8"
                        )
                    )

                    if (
                        status.get("completed") is True
                        and status.get("epochs_requested") == epochs
                        and status.get("config")
                        and config_matches(
                            status["config"],
                            config
                        )
                    ):

                        print(
                            "\nSKIPPING COMPLETED EXPERIMENT:",
                            experiment_name
                        )

                        print(
                            "Best validation accuracy:",
                            f'{status.get("best_validation_accuracy", 0):.2f}%'
                        )

                        completed_count += 1
                        continue

                except (
                    json.JSONDecodeError,
                    OSError,
                    TypeError
                ):

                    print(
                        "Could not read experiment status. "
                        "Checking its checkpoint."
                    )

            # ------------------------------------------------
            # START EXPERIMENT
            # ------------------------------------------------

            print("\n" + "=" * 65)
            print("MODEL:", model_name)
            print("AUGMENTATION:", strategy)
            print("=" * 65)

            set_seed(seed)

            training_dataset = TransformedSubset(
                base_dataset,
                train_indices,
                get_training_transform(strategy)
            )

            training_loader = DataLoader(
                training_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

            use_pretrained = (
                model_name == "resnet18"
                and config.get(
                    "pretrained_resnet18",
                    True
                )
            )

            model = make_model(
                model_name,
                use_pretrained
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"]
            )

            criterion = nn.CrossEntropyLoss()

            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=1
                )
            )

            # ------------------------------------------------
            # INITIAL TRAINING STATE
            # ------------------------------------------------

            start_epoch = 1
            best_accuracy = -1.0
            best_epoch = 0
            patience_count = 0
            resumed = False

            # ------------------------------------------------
            # RESUME INTERRUPTED EXPERIMENT
            # ------------------------------------------------

            if last_checkpoint_path.exists():

                try:

                    saved = torch.load(
                        last_checkpoint_path,
                        map_location=device,
                        weights_only=False
                    )

                    if (
                        saved.get("experiment_name")
                        == experiment_name
                        and saved.get("config")
                        and config_matches(
                            saved["config"],
                            config
                        )
                    ):

                        model.load_state_dict(
                            saved["model_state_dict"]
                        )

                        optimizer.load_state_dict(
                            saved["optimizer_state_dict"]
                        )

                        scheduler.load_state_dict(
                            saved["scheduler_state_dict"]
                        )

                        start_epoch = (
                            int(saved["epoch"]) + 1
                        )

                        best_accuracy = float(
                            saved["best_validation_accuracy"]
                        )

                        best_epoch = int(
                            saved["best_epoch"]
                        )

                        patience_count = int(
                            saved["patience_count"]
                        )

                        resumed = True

                        print(
                            f"Resuming from epoch "
                            f"{start_epoch}."
                        )

                except Exception as error:

                    print(
                        "Resume checkpoint could not be loaded:",
                        error
                    )

                    print(
                        "Starting this experiment from epoch 1."
                    )

            if not resumed:

                print(
                    "Starting this experiment from epoch 1."
                )

            experiment_start = time.time()
            stopped_early = False
            last_epoch_completed = start_epoch - 1

            # ------------------------------------------------
            # EPOCH LOOP
            # ------------------------------------------------

            for epoch in range(
                start_epoch,
                epochs + 1
            ):

                epoch_start = time.time()

                model.train()

                total = 0
                correct = 0
                loss_sum = 0.0

                for batch_index, (
                    images,
                    labels_batch
                ) in enumerate(
                    training_loader,
                    start=1
                ):

                    images = images.to(device)
                    labels_batch = labels_batch.to(device)

                    optimizer.zero_grad()

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels_batch
                    )

                    loss.backward()
                    optimizer.step()

                    count = labels_batch.size(0)

                    loss_sum += (
                        loss.item() * count
                    )

                    correct += (
                        outputs.argmax(dim=1)
                        == labels_batch
                    ).sum().item()

                    total += count

                    # Display progress during CPU training.
                    if batch_index % 100 == 0:

                        print(
                            f"\rEpoch {epoch}/{epochs} | "
                            f"Batch {batch_index}/"
                            f"{len(training_loader)} | "
                            f"Processed {total}/"
                            f"{len(training_dataset)}",
                            end="",
                            flush=True
                        )

                train_loss = loss_sum / max(total, 1)

                train_accuracy = (
                    100.0 * correct / max(total, 1)
                )

                validation_loss, validation_accuracy = evaluate(
                    model,
                    validation_loader,
                    criterion,
                    device
                )

                scheduler.step(validation_accuracy)

                epoch_seconds = (
                    time.time() - epoch_start
                )

                last_epoch_completed = epoch

                # --------------------------------------------
                # SAVE TRAINING HISTORY
                # --------------------------------------------

                record = {
                    "model": model_name,
                    "strategy": strategy,
                    "epoch": epoch,
                    "train_loss": round(
                        train_loss, 6
                    ),
                    "train_accuracy": round(
                        train_accuracy, 4
                    ),
                    "validation_loss": round(
                        validation_loss, 6
                    ),
                    "validation_accuracy": round(
                        validation_accuracy, 4
                    ),
                    "epoch_seconds": round(
                        epoch_seconds, 2
                    ),
                }

                write_csv(
                    history_path,
                    history_fields,
                    [record],
                    append=True
                )

                print()

                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Train accuracy: {train_accuracy:.2f}% | "
                    f"Validation loss: {validation_loss:.4f} | "
                    f"Validation accuracy: "
                    f"{validation_accuracy:.2f}% | "
                    f"Time: {epoch_seconds / 60:.1f} min"
                )

                # --------------------------------------------
                # SAVE BEST MODEL
                # --------------------------------------------

                if validation_accuracy > best_accuracy:

                    best_accuracy = validation_accuracy
                    best_epoch = epoch
                    patience_count = 0

                    torch.save(
                        {
                            "model_state_dict": (
                                model.state_dict()
                            ),
                            "model_name": model_name,
                            "strategy": strategy,
                            "epoch": epoch,
                            "best_validation_accuracy": (
                                best_accuracy
                            ),
                            "config": config,
                        },
                        best_checkpoint_path
                    )

                    print(
                        "Best-model checkpoint saved:",
                        best_checkpoint_path
                    )

                else:

                    patience_count += 1

                # --------------------------------------------
                # SAVE RESUMABLE CHECKPOINT
                # --------------------------------------------

                torch.save(
                    {
                        "experiment_name": (
                            experiment_name
                        ),
                        "model_state_dict": (
                            model.state_dict()
                        ),
                        "optimizer_state_dict": (
                            optimizer.state_dict()
                        ),
                        "scheduler_state_dict": (
                            scheduler.state_dict()
                        ),
                        "epoch": epoch,
                        "best_validation_accuracy": (
                            best_accuracy
                        ),
                        "best_epoch": best_epoch,
                        "patience_count": patience_count,
                        "config": config,
                    },
                    last_checkpoint_path
                )

                # Mark incomplete until training finishes.
                status_path.write_text(
                    json.dumps(
                        {
                            "completed": False,
                            "epochs_requested": epochs,
                            "last_epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_validation_accuracy": (
                                best_accuracy
                            ),
                            "config": config,
                        },
                        indent=4
                    ),
                    encoding="utf-8"
                )

                # --------------------------------------------
                # EARLY STOPPING
                # --------------------------------------------

                if (
                    patience_count
                    >= config["early_stopping_patience"]
                ):

                    print(
                        "Early stopping activated."
                    )

                    stopped_early = True
                    break

            # ------------------------------------------------
            # MARK EXPERIMENT COMPLETED
            # ------------------------------------------------

            status_path.write_text(
                json.dumps(
                    {
                        "completed": True,
                        "epochs_requested": epochs,
                        "last_epoch": last_epoch_completed,
                        "best_epoch": best_epoch,
                        "best_validation_accuracy": (
                            best_accuracy
                        ),
                        "stopped_early": stopped_early,
                        "config": config,
                    },
                    indent=4
                ),
                encoding="utf-8"
            )

            elapsed_minutes = (
                time.time() - experiment_start
            ) / 60

            print(
                "\nExperiment completed:",
                experiment_name
            )

            print(
                f"Best validation accuracy: "
                f"{best_accuracy:.2f}%"
            )

            print("Best epoch:", best_epoch)

            print(
                f"Elapsed time this session: "
                f"{elapsed_minutes:.1f} minutes"
            )

            completed_count += 1

            del model
            del optimizer
            del scheduler
            del training_loader
            del training_dataset

    # ========================================================
    # FINAL TRAINING SUMMARY
    # ========================================================

    history_rows = load_csv_rows(history_path)

    best_by_experiment = {}

    for row in history_rows:

        try:

            row["epoch"] = int(row["epoch"])

            row["validation_accuracy"] = float(
                row["validation_accuracy"]
            )

            row["train_accuracy"] = float(
                row["train_accuracy"]
            )

            row["validation_loss"] = float(
                row["validation_loss"]
            )

        except (ValueError, KeyError):

            continue

        key = (
            row["model"],
            row["strategy"]
        )

        if (
            key not in best_by_experiment
            or row["validation_accuracy"]
            > best_by_experiment[key][
                "validation_accuracy"
            ]
        ):

            best_by_experiment[key] = row

    summary_fields = [
        "model",
        "strategy",
        "best_epoch",
        "train_accuracy",
        "validation_accuracy",
        "validation_loss",
    ]

    summary_rows = []

    for (
        model_name,
        strategy
    ), row in sorted(
        best_by_experiment.items()
    ):

        summary_rows.append(
            {
                "model": model_name,
                "strategy": strategy,
                "best_epoch": row["epoch"],
                "train_accuracy": (
                    row["train_accuracy"]
                ),
                "validation_accuracy": (
                    row["validation_accuracy"]
                ),
                "validation_loss": (
                    row["validation_loss"]
                ),
            }
        )

    write_csv(
        RESULTS_DIR / "training_summary.csv",
        summary_fields,
        summary_rows
    )

    print("\n" + "=" * 65)
    print("TRAINING PIPELINE FINISHED")
    print("=" * 65)

    print(
        "Completed or skipped experiments:",
        completed_count
    )

    print("Training history:", history_path)

    print(
        "Training summary:",
        RESULTS_DIR / "training_summary.csv"
    )

    print("Checkpoints:", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()