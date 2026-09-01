import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.cfg import py2cfg
from train import SegmentationModule, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Run GWSegNet inference.")
    parser.add_argument("-c", "--config", type=Path, required=True, help="Path to a Python config.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory for PNG masks.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path; defaults to the config checkpoint.")
    parser.add_argument("--tta", choices=("none", "flips", "d4"), default="none")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--raw-labels",
        action="store_true",
        help="Write class IDs instead of 0/255 binary masks.",
    )
    return parser.parse_args()


def build_tta(model, mode):
    if mode == "none":
        return model
    try:
        import ttach as tta
    except ImportError as error:
        raise ImportError(
            "TTA requires ttach. Install it with `pip install ttach` or use --tta none."
        ) from error
    transforms = [tta.HorizontalFlip(), tta.VerticalFlip()]
    if mode == "d4":
        transforms.append(tta.Rotate90(angles=[0, 90, 180, 270]))
    return tta.SegmentationTTAWrapper(model, tta.Compose(transforms))


def main():
    args = parse_args()
    config = py2cfg(args.config)
    seed_everything(getattr(config, "seed", 42))

    checkpoint = args.checkpoint or (
        Path(config.weights_path) / f"{config.test_weights_name}.ckpt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegmentationModule.load_from_checkpoint(checkpoint, config=config)
    model.to(device).eval()
    model = build_tta(model, args.tta)

    dataset = config.test_dataset
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or getattr(config, "test_batch_size", config.val_batch_size),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
        drop_last=False,
    )
    args.output.mkdir(exist_ok=True, parents=True)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            logits = model(batch["img"].to(device, non_blocking=True))
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            predictions = nn.functional.softmax(logits, dim=1).argmax(dim=1)

            for image_id, prediction in zip(batch["img_id"], predictions):
                mask = prediction.cpu().numpy().astype(np.uint8)
                if config.num_classes == 2 and not args.raw_labels:
                    mask = mask * 255
                output_path = args.output / f"{image_id}.png"
                if not cv2.imwrite(str(output_path), mask):
                    raise OSError(f"Failed to write prediction: {output_path}")


if __name__ == "__main__":
    main()
