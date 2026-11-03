
import torch
from torchvision import transforms


IMAGE_SIZE = 48

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_clean_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


def get_training_transform(strategy="none"):
    if strategy == "none":
        return get_clean_transform()

    if strategy == "standard":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    if strategy == "corruption":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.3),
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    raise ValueError(
        "Strategy must be 'none', 'standard', or 'corruption'."
    )


def add_gaussian_noise(image, severity=1):
    if severity not in range(1, 6):
        raise ValueError("Severity must be between 1 and 5.")

    noise_std = 0.03 * severity
    noise = torch.randn_like(image) * noise_std

    return torch.clamp(image + noise, 0.0, 1.0)


if __name__ == "__main__":
    from PIL import Image

    sample = Image.new("RGB", (48, 48), color="red")

    for strategy in ("none", "standard", "corruption"):
        transform = get_training_transform(strategy)
        result = transform(sample)

        print(
            f"{strategy}: shape={tuple(result.shape)}, "
            f"finite={bool(torch.isfinite(result).all())}"
        )

    print("Augmentation smoke test completed.")
