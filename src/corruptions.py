from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def apply_corruption(image, corruption, severity):
    """
    Apply a corruption to a PIL image.

    Supported corruptions:
        blur
        brightness_dark
        brightness_bright
        noise
        occlusion

    Severity: integer from 1 to 5.
    """

    if severity not in (1, 2, 3, 4, 5):
        raise ValueError("Severity must be between 1 and 5.")

    # 1. Increasing Gaussian blur
    if corruption == "blur":
        radii = [0.5, 1.0, 1.5, 2.0, 3.0]
        return image.filter(
            ImageFilter.GaussianBlur(
                radius=radii[severity - 1]
            )
        )

    # 2. Increasing darkness
    elif corruption == "brightness_dark":
        factors = [0.9, 0.75, 0.6, 0.45, 0.3]
        return ImageEnhance.Brightness(image).enhance(
            factors[severity - 1]
        )

    # 3. Increasing brightness
    elif corruption == "brightness_bright":
        factors = [1.1, 1.25, 1.4, 1.6, 1.8]
        return ImageEnhance.Brightness(image).enhance(
            factors[severity - 1]
        )

    # 4. Increasing Gaussian noise
    elif corruption == "noise":
        noise_levels = [5, 10, 20, 30, 40]

        array = np.asarray(image).astype(np.float32)

        # Deterministic noise makes experiments reproducible.
        rng = np.random.default_rng(seed=severity * 100)

        noise = rng.normal(
            loc=0,
            scale=noise_levels[severity - 1],
            size=array.shape
        )

        noisy = np.clip(
            array + noise, 0, 255
        ).astype(np.uint8)

        return Image.fromarray(noisy)

    # 5. Increasing partial occlusion
    elif corruption == "occlusion":
        array = np.asarray(image).copy()

        height, width = array.shape[:2]

        fractions = [0.05, 0.10, 0.15, 0.20, 0.30]
        fraction = fractions[severity - 1]

        box_width = max(1, int(width * fraction))
        box_height = max(1, int(height * fraction))

        # Put a black square in the center.
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2

        array[
            y1:y1 + box_height,
            x1:x1 + box_width
        ] = 0

        return Image.fromarray(array)

    else:
        raise ValueError(
            f"Unknown corruption type: {corruption}"
        )