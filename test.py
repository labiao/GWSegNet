import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.cfg import py2cfg
from tools.metric import Evaluator
from train import SegmentationModule, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GWSegNet on a labeled Great Wall test set."
    )
    parser.add_argument(
        "-c", "--config", type=Path, required=True,
        help="Path to the GWSegNet Python config.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Directory for binary prediction masks.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Checkpoint path; defaults to the checkpoint defined by the config.",
    )
    parser.add_argument(
        "--tta", choices=("none", "flips", "d4"), default="none",
        help="Test-time augmentation mode.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-size", type=int, default=512,
        help="Square evaluation/output size; use 0 to keep the transformed size.",
    )
    parser.add_argument(
        "--save-error-maps", action="store_true",
        help="Save TP/FP/FN visualizations to <output>_errors.",
    )
    parser.add_argument(
        "--no-save-masks", action="store_true",
        help="Only report metrics without writing prediction masks.",
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


def resize_mask(mask, output_size):
    if not output_size:
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST,
    )


def save_binary_mask(mask, output_path):
    output = (mask > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(output_path), output):
        raise OSError(f"Failed to write prediction: {output_path}")


def save_error_map(prediction, target, output_path):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    visualization = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    visualization[prediction & target] = (0, 255, 0)       # TP: green
    visualization[prediction & ~target] = (0, 0, 255)      # FP: red
    visualization[~prediction & target] = (255, 0, 0)      # FN: blue
    if not cv2.imwrite(str(output_path), visualization):
        raise OSError(f"Failed to write error map: {output_path}")


def print_metrics(config, evaluator):
    iou = evaluator.Intersection_over_Union()
    f1 = evaluator.F1()
    recall = evaluator.Recall()
    precision = evaluator.Precision()
    overall_accuracy = evaluator.OA()

    print("\nPer-class metrics")
    print("class\tprecision\trecall\tF1\tIoU")
    for class_name, class_precision, class_recall, class_f1, class_iou in zip(
        config.classes, precision, recall, f1, iou
    ):
        print(
            f"{class_name}\t{class_precision:.6f}\t{class_recall:.6f}"
            f"\t{class_f1:.6f}\t{class_iou:.6f}"
        )

    foreground_indices = list(getattr(config, "eval_class_indices", [1]))
    foreground_f1 = float(np.nanmean(f1[foreground_indices]))
    foreground_iou = float(np.nanmean(iou[foreground_indices]))
    mean_f1 = float(np.nanmean(f1))
    mean_iou = float(np.nanmean(iou))
    print("\nSummary")
    print(f"Great Wall F1:  {foreground_f1:.6f}")
    print(f"Great Wall IoU: {foreground_iou:.6f}")
    print(f"Mean F1:        {mean_f1:.6f}")
    print(f"Mean IoU:       {mean_iou:.6f}")
    print(f"OA:             {overall_accuracy:.6f}")


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

    dataset = getattr(config, "evaluation_dataset", config.val_dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or getattr(config, "test_batch_size", 2),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
        drop_last=False,
    )

    if not args.no_save_masks:
        args.output.mkdir(exist_ok=True, parents=True)
    error_output = Path(f"{args.output}_errors")
    if args.save_error_maps:
        error_output.mkdir(exist_ok=True, parents=True)

    evaluator = Evaluator(num_class=config.num_classes)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Testing"):
            logits = model(batch["img"].to(device, non_blocking=True))
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            predictions = nn.functional.softmax(logits, dim=1).argmax(dim=1)

            for image_id, prediction, target in zip(
                batch["img_id"], predictions, batch["gt_semantic_seg"]
            ):
                prediction = resize_mask(prediction.cpu().numpy(), args.output_size)
                target = resize_mask(target.cpu().numpy(), args.output_size)
                evaluator.add_batch(gt_image=target, pre_image=prediction)

                if not args.no_save_masks:
                    save_binary_mask(prediction, args.output / f"{image_id}.png")
                if args.save_error_maps:
                    save_error_map(
                        prediction,
                        target,
                        error_output / f"{image_id}.png",
                    )

    print_metrics(config, evaluator)


if __name__ == "__main__":
    main()
