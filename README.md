# Traffic Sign Robustness

## Project Overview

Traffic Sign Robustness is a deep learning project that evaluates how accurately traffic sign recognition models perform under different image conditions and visual corruptions.

The project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset to train and evaluate two deep learning architectures: Lightweight CNN and ResNet18.

It investigates the impact of image augmentation and corruption strategies on model accuracy and reliability.

## Objectives

- Develop a traffic sign recognition system using deep learning.
- Compare Lightweight CNN and ResNet18 architectures.
- Evaluate model performance under different image corruption conditions.
- Analyze the effect of image augmentation on classification accuracy.
- Study model reliability and calibration using Expected Calibration Error (ECE).

## Key Features

- Traffic sign classification across 43 classes.
- Lightweight CNN and ResNet18 model implementation.
- Multiple image augmentation strategies.
- Evaluation under different corruption conditions.
- Accuracy and reliability analysis.
- Confusion matrix visualization.
- Reliability diagrams and experimental result plots.

## Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**

- Number of classes: 43
- Input image size: 48 × 48 pixels
- Validation split: 20%
- Random seed: 42

Note: The dataset is not included in this GitHub repository because of its large file size. Use the dataset download and verification scripts in the `src` folder to prepare the data.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Deep Learning
- Computer Vision

## Models Implemented

### 1. Lightweight CNN
A compact convolutional neural network designed for traffic sign classification with relatively low computational requirements.

### 2. ResNet18
A residual neural network architecture used to evaluate classification performance and compare robustness against the Lightweight CNN.

## Augmentation Strategies

The project evaluates three strategies:

1. **None:** Training without additional augmentation.
2. **Standard:** Training with standard image augmentation techniques.
3. **Corruption:** Training with corruption-based image augmentation.

The models are evaluated under different image conditions, including:

- Blur
- Brightness variation
- Noise
- Occlusion

## Experimental Configuration

The main configuration is stored in `configs/config.json`.

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Image size | 48 × 48 |
| Number of classes | 43 |
| Batch size | 16 |
| Validation fraction | 0.2 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Epochs | 10 |
| Early stopping patience | 3 |
| Device | CPU |
| Pretrained ResNet18 | Enabled |

## Project Structure

```text
TrafficSignRobustness/
│
├── configs/
│   └── config.json
│
├── results/
│   ├── confusion_matrices/
│   ├── figures/
│   └── training_history.csv
│
├── src/
│   ├── augmentations.py
│   ├── check_training_dataset.py
│   ├── corruptions.py
│   ├── download_dataset.py
│   ├── evaluate.py
│   ├── gtsrb_dataset.py
│   ├── models.py
│   ├── train.py
│   └── verify_dataset.py
│
├── .gitignore
└── README.md
```

## Results and Evaluation

The project generates experimental outputs to compare the performance and reliability of the implemented models.

The results folder contains:

- Confusion matrices for both models.
- Accuracy versus corruption severity graphs.
- Expected Calibration Error (ECE) graphs.
- Reliability diagrams.
- Training history in CSV format.
- Sample traffic sign
