from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


IGNORE_LABELS = {"__ignore__", "_background_", "background"}
DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "grass": (0, 255, 0),
    "sand": (255, 255, 0),
    "mud": (255, 0, 255),
    "concrete": (0, 180, 255),
}


def read_labels(labels_path: Path) -> List[str]:
    labels: List[str] = []
    for raw_line in labels_path.read_text().splitlines():
        label = raw_line.strip()
        if not label or label.startswith("#") or label in IGNORE_LABELS:
            continue
        labels.append(label)
    if not labels:
        raise ValueError(f"No semantic labels found in {labels_path}")
    return labels


def polygon_points(points: List[List[float]], width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return np.round(pts).astype(np.int32)


def rectangle_points(points: List[List[float]], width: int, height: int) -> np.ndarray:
    if len(points) != 2:
        raise ValueError("Labelme rectangle must have exactly two points")
    (x1, y1), (x2, y2) = points
    rect = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return polygon_points(rect, width, height)


def shape_to_points(shape: Dict, width: int, height: int) -> np.ndarray:
    shape_type = shape.get("shape_type", "polygon")
    points = shape.get("points", [])
    if shape_type == "polygon":
        return polygon_points(points, width, height)
    if shape_type == "rectangle":
        return rectangle_points(points, width, height)
    raise ValueError(f"Unsupported shape_type: {shape_type}")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def make_preview(image: np.ndarray, mask: np.ndarray, labels: List[str], alpha: float) -> np.ndarray:
    overlay = image.copy()
    color_layer = np.zeros_like(image)

    for class_id, label in enumerate(labels, start=1):
        color_rgb = DEFAULT_COLORS.get(label, (255, 255, 255))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        color_layer[mask == class_id] = color_bgr

    has_label = mask > 0
    overlay[has_label] = cv2.addWeighted(
        image[has_label],
        1.0 - alpha,
        color_layer[has_label],
        alpha,
        0.0,
    )
    return overlay


def convert_one(
    json_path: Path,
    labels: List[str],
    out_dirs: Dict[str, Path],
    alpha: float,
    output_prefix: str = "",
) -> Dict:
    label_to_id = {label: idx for idx, label in enumerate(labels, start=1)}
    data = json.loads(json_path.read_text())

    width = int(data["imageWidth"])
    height = int(data["imageHeight"])
    image_path = json_path.parent / data["imagePath"]
    if not image_path.exists():
        raise FileNotFoundError(f"Image referenced by {json_path.name} not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if image.shape[:2] != (height, width):
        raise ValueError(
            f"Image size mismatch for {image_path.name}: "
            f"json={width}x{height}, image={image.shape[1]}x{image.shape[0]}"
        )

    mask = np.zeros((height, width), dtype=np.uint8)
    shape_counts = {label: 0 for label in labels}
    skipped = []

    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if label in IGNORE_LABELS:
            continue
        if label not in label_to_id:
            skipped.append(label)
            continue

        try:
            points = shape_to_points(shape, width, height)
        except ValueError as exc:
            skipped.append(f"{label} ({exc})")
            continue

        if len(points) < 3:
            skipped.append(f"{label} (<3 points)")
            continue

        cv2.fillPoly(mask, [points], int(label_to_id[label]))
        shape_counts[label] += 1

    output_stem = f"{output_prefix}{image_path.stem}"
    image_out = out_dirs["images"] / f"{output_stem}{image_path.suffix}"
    mask_out = out_dirs["masks"] / f"{output_stem}_mask.png"
    preview_out = out_dirs["previews"] / f"{output_stem}_preview.png"

    shutil.copy2(image_path, image_out)
    cv2.imwrite(str(mask_out), mask)
    cv2.imwrite(str(preview_out), make_preview(image, mask, labels, alpha))

    coverage = {}
    total_pixels = mask.size
    for class_id, label in enumerate(labels, start=1):
        pixels = int(np.count_nonzero(mask == class_id))
        coverage[label] = {
            "pixels": pixels,
            "percent": round(100.0 * pixels / total_pixels, 2),
            "shapes": shape_counts[label],
        }

    return {
        "json": str(json_path),
        "source_image": str(image_path),
        "image": str(image_out),
        "mask": str(mask_out),
        "preview": str(preview_out),
        "coverage": coverage,
        "skipped": skipped,
    }


def write_label_map(output_dir: Path, labels: List[str]) -> None:
    lines = ["0 background"]
    lines.extend(f"{idx} {label}" for idx, label in enumerate(labels, start=1))
    (output_dir / "label_map.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        nargs="+",
        default="dataset/raw_images/terrain_rgb_01",
        help="One or more directories containing Labelme JSON files and source images",
    )
    parser.add_argument(
        "--labels",
        default="dataset/terrain_labels.txt",
        help="Label list file. __ignore__ and _background_ are ignored.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/labelme_segmentation",
        help="Output dataset root",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Preview overlay opacity",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear existing accepted images, masks, and previews before writing output.",
    )
    args = parser.parse_args()

    input_dirs = [Path(path) for path in args.input_dir]
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)

    labels = read_labels(labels_path)
    json_groups = []
    for input_dir in input_dirs:
        json_paths = sorted(input_dir.glob("*.json"))
        if not json_paths:
            raise RuntimeError(f"No Labelme JSON files found in {input_dir}")
        json_groups.append((input_dir, json_paths))

    out_dirs = {
        "images": output_dir / "labeling" / "accepted" / "images",
        "masks": output_dir / "labeling" / "accepted" / "masks",
        "previews": output_dir / "labeling" / "accepted" / "previews",
    }
    if args.clear_output:
        for path in out_dirs.values():
            if path.exists():
                shutil.rmtree(path)

    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    write_label_map(output_dir, labels)

    use_prefixes = len(input_dirs) > 1
    results = []
    for input_dir, json_paths in json_groups:
        prefix = f"{safe_name(input_dir.name)}_" if use_prefixes else ""
        results.extend(convert_one(path, labels, out_dirs, args.alpha, prefix) for path in json_paths)

    summary_path = output_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps({"labels": labels, "results": results}, indent=2))

    print(f"Converted {len(results)} Labelme files")
    print(f"Images:   {out_dirs['images']}")
    print(f"Masks:    {out_dirs['masks']}")
    print(f"Previews: {out_dirs['previews']}")
    print(f"Summary:  {summary_path}")
    print("\nClass IDs:")
    print("  0 background")
    for idx, label in enumerate(labels, start=1):
        print(f"  {idx} {label}")

    skipped_total = sum(len(result["skipped"]) for result in results)
    if skipped_total:
        print(f"\nWarnings: skipped {skipped_total} shapes with unknown/unsupported labels.")


if __name__ == "__main__":
    main()
