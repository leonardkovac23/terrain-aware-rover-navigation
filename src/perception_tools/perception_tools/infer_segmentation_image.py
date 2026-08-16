import argparse
from pathlib import Path

import cv2
import numpy as np

from perception_tools.segmentation_model_common import (
    extract_logits,
    make_prediction_overlay,
    normalize_image_rgb,
    print_mask_coverage,
    read_label_map,
    select_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference on one RGB image.")
    parser.add_argument("image", help="Input RGB image.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pth path.")
    parser.add_argument(
        "--checkpoint-dir",
        default="dataset/models/checkpoints",
        help="Directory used when --checkpoint is omitted.",
    )
    parser.add_argument("--model", default="ddrnet_39", help="Model name if checkpoint has no metadata.")
    parser.add_argument(
        "--label-map",
        default="dataset/segmentation_dataset/label_map.txt",
        help="Fallback label map if checkpoint has no labels.",
    )
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output", default=None, help="Overlay output PNG path.")
    parser.add_argument("--mask-output", default=None)
    parser.add_argument("--alpha", type=float, default=0.35, help="Overlay opacity.")
    return parser.parse_args()


def import_inference_dependencies():
    try:
        import torch
        from super_gradients.training import models
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependencies. Activate your training environment or install "
            "src/perception_tools/requirements-training.txt."
        ) from exc

    return torch, models


def find_latest_checkpoint(checkpoint_dir: Path):
    best_models = list(checkpoint_dir.glob("**/best_model.pth"))
    if best_models:
        return max(best_models, key=lambda path: path.stat().st_mtime)

    checkpoints = list(checkpoint_dir.glob("**/*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def load_checkpoint(torch, checkpoint_path: Path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def main():
    args = parse_args()
    torch, models = import_inference_dependencies()

    device = select_device(torch, args.device)
    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else find_latest_checkpoint(
        Path(args.checkpoint_dir).expanduser()
    )
    checkpoint = load_checkpoint(torch, checkpoint_path, device)

    labels = checkpoint.get("labels")
    if labels is None:
        labels = read_label_map(Path(args.label_map).expanduser())

    model_name = checkpoint.get("model_name", args.model)
    num_classes = int(checkpoint.get("num_classes", len(labels)))
    input_size = args.input_size or int(checkpoint.get("input_size", 256))

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_h, original_w = image_bgr.shape[:2]
    model_bgr = cv2.resize(image_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    model_rgb = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)
    model_rgb = normalize_image_rgb(model_rgb)

    input_tensor = torch.from_numpy(model_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)

    model = models.get(model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = extract_logits(model(input_tensor))
        prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    if prediction.shape[:2] != (original_h, original_w):
        prediction = cv2.resize(prediction, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    print(f"Image:      {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model:      {model_name}")
    print_mask_coverage(prediction, labels)

    overlay = make_prediction_overlay(image_bgr, prediction, labels, args.alpha)

    output_path = Path(args.output).expanduser() if args.output else Path("dataset/infer") / f"{image_path.stem}_preview.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    print(f"\nSaved overlay: {output_path}")

    if args.mask_output:
        mask_output_path = Path(args.mask_output).expanduser()
        mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_output_path), prediction)
        print(f"Saved mask:    {mask_output_path}")


if __name__ == "__main__":
    main()
