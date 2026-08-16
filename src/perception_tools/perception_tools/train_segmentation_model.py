import argparse
import random
from pathlib import Path

import numpy as np

from perception_tools.segmentation_model_common import IGNORE_INDEX, read_label_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a semantic segmentation model from image/mask pairs."
    )
    parser.add_argument(
        "--data-dir",
        default="dataset/segmentation_dataset",
        help="Dataset root containing images/, masks/, and label_map.txt.",
    )
    parser.add_argument(
        "--model",
        default="ddrnet_39",
        help="SuperGradients model name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--train-split", type=float, default=0.85)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        default="dataset/models/checkpoints",
        help="Root directory for saved checkpoints.",
    )
    return parser.parse_args()

# Import ML dependencies lazily so --help works outside the training env.
def import_training_dependencies():
    try:
        import torch
        from torch.nn import CrossEntropyLoss
        from torch.optim import SGD
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader, Subset
        from super_gradients.training import models
        from perception_tools.segmentation_dataset import SegmentationDataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Activate your training environment or install "
            "src/perception_tools/requirements-training.txt."
        ) from exc

    return torch, CrossEntropyLoss, SGD, CosineAnnealingLR, DataLoader, Subset, models, SegmentationDataset


def select_device(torch, requested_device):
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    return requested_device


def split_indices(num_samples, train_split, val_split, seed):
    split_sum = train_split + val_split
    
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError("train_split + val_split must equal 1.0.")

    if num_samples < 2:
        raise ValueError("Need at least two image/mask pairs for train/validation split.")

    indices = list(range(num_samples))
    random.Random(seed).shuffle(indices)

    train_count = max(1, int(round(num_samples * train_split)))
    val_count = max(1, int(round(num_samples * val_split))) if val_split > 0.0 else 0

    if train_count + val_count > num_samples:
        val_count = max(1, num_samples - train_count)

    if train_count + val_count > num_samples:
        train_count = num_samples - val_count

    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]

    if not train_indices or not val_indices:
        raise ValueError("Train and validation splits must both contain at least one sample.")

    return train_indices, val_indices


def update_metrics(logits, targets, num_classes, intersections, unions, correct_total):
    predictions = logits.argmax(dim=1)
    valid_pixels = targets != IGNORE_INDEX

    correct_total[0] += ((predictions == targets) & valid_pixels).sum().item()
    correct_total[1] += valid_pixels.sum().item()

    for class_id in range(num_classes):
        pred_mask = (predictions == class_id) & valid_pixels
        target_mask = (targets == class_id) & valid_pixels
        intersections[class_id] += (pred_mask & target_mask).sum().item()
        unions[class_id] += (pred_mask | target_mask).sum().item()


def summarize_metrics(intersections, unions, correct_total):
    pixel_accuracy = correct_total[0] / correct_total[1] if correct_total[1] else 0.0
    valid_ious = []

    for intersection, union in zip(intersections, unions):
        if union > 0:
            valid_ious.append(intersection / union)

    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return pixel_accuracy, mean_iou


def extract_logits(model_output):
    if isinstance(model_output, (list, tuple)):
        return model_output[0]
    return model_output


def run_epoch(torch, model, loader, criterion, optimizer, device, num_classes):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0

    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)

    correct_total = [0, 0]

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = extract_logits(model(images))
            loss = criterion(logits, masks)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        update_metrics(logits.detach(), masks, num_classes, intersections, unions, correct_total)

    pixel_accuracy, mean_iou = summarize_metrics(intersections, unions, correct_total)

    return total_loss / max(total_samples, 1), pixel_accuracy, mean_iou


def save_checkpoint(path, torch, model, optimizer, epoch, args, labels, val_loss, val_miou):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "labels": labels,
            "num_classes": len(labels),
            "input_size": args.input_size,
            "val_loss": val_loss,
            "val_miou": val_miou,
        },
        path,
    )


def main():
    args = parse_args()
    (
        torch,
        CrossEntropyLoss,
        SGD,
        CosineAnnealingLR,
        DataLoader,
        Subset,
        models,
        SegmentationDataset,
    ) = import_training_dependencies()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser()
    label_map_path = data_dir / "label_map.txt"
    labels = read_label_map(label_map_path)
    num_classes = len(labels)

    if args.model.startswith("ddrnet") and args.batch_size < 2:
        raise ValueError("DDRNet needs --batch-size >= 2 during training.")

    dataset = SegmentationDataset(
        images_dir=data_dir / "images",
        masks_dir=data_dir / "masks",
        labels=labels,
        input_size=args.input_size,
        use_augmentation=args.use_augmentation,
    )

    train_indices, val_indices = split_indices(
        len(dataset),
        args.train_split,
        args.val_split,
        args.seed,
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    if args.model.startswith("ddrnet") and len(train_dataset) < args.batch_size:
        raise ValueError("DDRNet needs at least batch-size samples in the training split.")

    device = select_device(torch, args.device)
    print(f"Dataset: {data_dir}")
    print(f"Classes: {num_classes} ({', '.join(labels)})")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Device:        {device}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    model = models.get(args.model, num_classes=num_classes)
    model.to(device)

    criterion = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = Path(args.checkpoint_dir).expanduser() / args.model
    best_model_path = checkpoint_dir / "best_model.pth"
    latest_model_path = checkpoint_dir / "latest_model.pth"
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_miou = run_epoch(
            torch, model, train_loader, criterion, optimizer, device, num_classes
        )

        with torch.no_grad():
            val_loss, val_acc, val_miou = run_epoch(
                torch, model, val_loader, criterion, None, device, num_classes
            )

        scheduler.step()

        save_checkpoint(latest_model_path, torch, model, optimizer, epoch, args, labels, val_loss, val_miou)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_model_path, torch, model, optimizer, epoch, args, labels, val_loss, val_miou)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_miou={train_miou:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_miou={val_miou:.3f}"
        )

    print(f"\nSaved latest checkpoint: {latest_model_path}")
    print(f"Saved best checkpoint:   {best_model_path}")


if __name__ == "__main__":
    main()
