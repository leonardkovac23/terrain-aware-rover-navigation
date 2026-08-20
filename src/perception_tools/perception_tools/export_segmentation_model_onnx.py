import argparse
from pathlib import Path

import torch
from super_gradients.training import models

try:
    from perception_tools.segmentation_model_common import select_device
except ModuleNotFoundError:
    from segmentation_model_common import select_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a trained semantic segmentation checkpoint to ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        default="dataset/models/checkpoints/ddrnet_39/best_model.pth",
        help="Path to trained .pth checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="dataset/models_onnx/terrain_segmentation/ddrnet_39.onnx",
        help="Output ONNX model path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used during export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version.",
    )

    return parser.parse_args()

#Wrap the model so ONNX export receives only the main logits tensor
class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        output = self.model(image)

        if isinstance(output, (list, tuple)):
            return output[0]

        return output


def load_checkpoint(checkpoint_path: Path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def validate_checkpoint(checkpoint, checkpoint_path: Path):
    required_keys = [
        "model_name",
        "model_state_dict",
        "num_classes",
        "input_size",
    ]

    missing_keys = [key for key in required_keys if key not in checkpoint]

    if missing_keys:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing required keys: "
            f"{', '.join(missing_keys)}"
        )


def export_onnx(checkpoint, output_path: Path, device, opset: int):
    model_name = checkpoint["model_name"]
    num_classes = int(checkpoint["num_classes"])
    input_size = int(checkpoint["input_size"])

    model = models.get(model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    wrapped_model = ExportWrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    return model_name, num_classes, input_size


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    output_path = Path(args.output).expanduser()
    device = select_device(torch, args.device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    validate_checkpoint(checkpoint, checkpoint_path)

    model_name, num_classes, input_size = export_onnx(
        checkpoint,
        output_path,
        device,
        args.opset,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model:      {model_name}")
    print(f"Classes:    {num_classes}")
    print(f"Input size: {input_size}")
    print(f"ONNX:       {output_path}")


if __name__ == "__main__":
    main()
