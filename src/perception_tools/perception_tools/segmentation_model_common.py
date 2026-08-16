from pathlib import Path

import cv2
import numpy as np


IGNORE_INDEX = 255

DEFAULT_COLORS_RGB = {
    "grass": (0, 255, 0),
    "sand": (255, 255, 0),
    "mud": (160, 80, 40),
    "concrete": (120, 120, 120),
}


#Read class names from label_map.txt and return them ordered by class id.
def read_label_map(label_map_path: Path):
    class_names_by_id = {}

    for raw_line in label_map_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid label map line: {raw_line}")

        class_id = int(parts[0])
        class_name = parts[1].strip()
        if class_id == IGNORE_INDEX:
            continue

        class_names_by_id[class_id] = class_name

    if 0 not in class_names_by_id:
        class_names_by_id[0] = "background"

    max_class_id = max(class_names_by_id)
    labels = []

    for class_id in range(max_class_id + 1):
        labels.append(class_names_by_id.get(class_id, f"class_{class_id}"))

    return labels

#Return paths to matching PNG images and masks using the {image_stem}_mask.png naming convention.
def find_image_mask_pairs(images_dir: Path, masks_dir: Path):
    pairs = []
    for image_path in sorted(images_dir.glob("*.png")):
        mask_path = masks_dir / f"{image_path.stem}_mask.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            print(f"Warning: no mask found for {image_path.name}")

    if not pairs:
        raise ValueError(f"No image/mask pairs found in {images_dir} and {masks_dir}")

    return pairs

#Convert an RGB uint8 image to float and apply ImageNet mean/std normalization.
def normalize_image_rgb(image_rgb: np.ndarray):
    image = image_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean) / std


#Select CUDA when available, unless the user explicitly requested CPU.
def select_device(torch, requested_device):
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    return requested_device


#Extract the main prediction tensor from models that may return auxiliary outputs.
def extract_logits(model_output):
    if isinstance(model_output, (list, tuple)):
        return model_output[0]
    return model_output


#Convert class-id mask values into a colored BGR image for OpenCV output.
def colorize_mask_bgr(mask: np.ndarray, labels: list[str]):
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, label in enumerate(labels):
        if class_id == 0:
            continue

        color_rgb = DEFAULT_COLORS_RGB.get(label, (255, 255, 255))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        colored[mask == class_id] = color_bgr

    return colored


#Blend predicted class colors over the original BGR image for visualization.
def make_prediction_overlay(image_bgr: np.ndarray, mask: np.ndarray, labels: list[str], alpha: float):
    colored = colorize_mask_bgr(mask, labels)
    overlay = image_bgr.copy()
    has_label = (mask > 0) & (mask != IGNORE_INDEX)

    if np.any(has_label):
        overlay[has_label] = cv2.addWeighted(
            image_bgr[has_label],
            1.0 - alpha,
            colored[has_label],
            alpha,
            0.0,
        )

    return overlay


#Print the percentage of pixels assigned to each predicted class.
def print_mask_coverage(mask: np.ndarray, labels: list[str]):
    values, counts = np.unique(mask, return_counts=True)
    total = mask.size

    print("\nPrediction class coverage:")
    for value, count in zip(values, counts):
        class_id = int(value)
        if class_id == IGNORE_INDEX:
            label = "__ignore__"
        elif class_id < len(labels):
            label = labels[class_id]
        else:
            label = f"class_{class_id}"

        percent = 100.0 * float(count) / float(total)
        print(f"  {class_id:3d} {label:12s}: {percent:6.2f}%")
