from pathlib import Path
import random

import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transform import Compose, RandomScale, Resize, SmartCropV1


CLASSES = ("background", "great_wall")
PALETTE = ((0, 0, 0), (255, 255, 255))

ORIGIN_IMG_SIZE = (512, 512)
MODEL_INPUT_SIZE = (504, 504)
SUPPORTED_IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def get_training_transform():
    return albu.Compose([
        albu.RandomRotate90(p=0.5),
        albu.Normalize(),
    ])


def train_aug(img, mask):
    crop_aug = Compose([
        RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode="value"),
        SmartCropV1(
            crop_size=MODEL_INPUT_SIZE[0],
            max_ratio=0.75,
            ignore_index=len(CLASSES),
            nopad=False,
        ),
    ])
    img, mask = crop_aug(img, mask)
    augmented = get_training_transform()(
        image=np.asarray(img).copy(),
        mask=np.asarray(mask).copy(),
    )
    return augmented["image"], augmented["mask"]


def get_val_transform():
    return albu.Compose([albu.Normalize()])


def val_aug(img, mask):
    img, mask = Compose([Resize(MODEL_INPUT_SIZE)])(img, mask)
    img = np.asarray(img)
    if mask is None:
        return get_val_transform()(image=img.copy())["image"]
    augmented = get_val_transform()(
        image=img.copy(),
        mask=np.asarray(mask).copy(),
    )
    return augmented["image"], augmented["mask"]


class GreatWallDataset(Dataset):
    """Binary Great Wall segmentation dataset.

    Expected layout::

        <data_root>/images/*.{png,tif,tiff,jpg,jpeg}
        <data_root>/masks/*.{png,tif,tiff,jpg,jpeg}

    Mask values greater than zero are converted to the Great Wall class (1).
    Image and mask extensions may differ. A mask named ``mask_001.tif`` is
    paired with either ``001.*`` or ``mask_001.*`` in the images directory.
    """

    def __init__(
        self,
        data_root="data/greatwall/test",
        mode="val",
        img_dir="images",
        mask_dir="masks",
        transform=val_aug,
        mosaic_ratio=0.0,
        img_size=ORIGIN_IMG_SIZE,
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")

        self.data_root = Path(data_root)
        self.image_dir = self.data_root / img_dir
        self.mask_dir = self.data_root / mask_dir
        self.mode = mode
        self.transform = transform
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.samples = self._build_samples()

    @staticmethod
    def _list_images(directory):
        if not directory.is_dir():
            return []
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )

    def _build_samples(self):
        image_paths = self._list_images(self.image_dir)
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in {self.image_dir}")

        image_by_stem = {path.stem: path for path in image_paths}
        mask_paths = self._list_images(self.mask_dir)
        if not mask_paths:
            if self.mode == "test":
                return [(path.stem, path, None) for path in image_paths]
            raise FileNotFoundError(f"No supported masks found in {self.mask_dir}")

        samples = []
        missing = []
        for mask_path in mask_paths:
            candidates = [mask_path.stem]
            if mask_path.stem.startswith("mask_"):
                candidates.insert(0, mask_path.stem[len("mask_"):])
            image_path = next(
                (image_by_stem[stem] for stem in candidates if stem in image_by_stem),
                None,
            )
            if image_path is None:
                missing.append(mask_path.name)
                continue
            samples.append((mask_path.stem, image_path, mask_path))

        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Could not pair {len(missing)} masks with images in {self.image_dir}: {preview}"
            )
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_mask(mask_path):
        mask = np.asarray(Image.open(mask_path).convert("L"))
        return Image.fromarray((mask > 0).astype(np.uint8))

    def load_img_and_mask(self, index):
        _, image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = self._load_mask(mask_path) if mask_path is not None else None
        return image, mask

    def load_mosaic_img_and_mask(self, index):
        indices = [index] + [random.randrange(len(self.samples)) for _ in range(3)]
        tiles = [self.load_img_and_mask(sample_index) for sample_index in indices]
        if any(mask is None for _, mask in tiles):
            raise RuntimeError("Mosaic augmentation requires masks.")

        height, width = self.img_size
        offset_x = random.randint(width // 4, width - width // 4)
        offset_y = random.randint(height // 4, height - height // 4)
        crop_sizes = [
            (offset_x, offset_y),
            (width - offset_x, offset_y),
            (offset_x, height - offset_y),
            (width - offset_x, height - offset_y),
        ]

        cropped = []
        for (image, mask), (crop_width, crop_height) in zip(tiles, crop_sizes):
            crop = albu.RandomCrop(width=crop_width, height=crop_height)(
                image=np.asarray(image).copy(),
                mask=np.asarray(mask).copy(),
            )
            cropped.append((crop["image"], crop["mask"]))

        top_image = np.concatenate((cropped[0][0], cropped[1][0]), axis=1)
        bottom_image = np.concatenate((cropped[2][0], cropped[3][0]), axis=1)
        top_mask = np.concatenate((cropped[0][1], cropped[1][1]), axis=1)
        bottom_mask = np.concatenate((cropped[2][1], cropped[3][1]), axis=1)
        image = Image.fromarray(np.ascontiguousarray(np.concatenate((top_image, bottom_image))))
        mask = Image.fromarray(np.ascontiguousarray(np.concatenate((top_mask, bottom_mask))))
        return image, mask

    def __getitem__(self, index):
        image_id, _, mask_path = self.samples[index]
        use_mosaic = (
            self.mode == "train"
            and mask_path is not None
            and random.random() <= self.mosaic_ratio
        )
        if use_mosaic:
            image, mask = self.load_mosaic_img_and_mask(index)
        else:
            image, mask = self.load_img_and_mask(index)

        if self.transform:
            if mask is None:
                image = self.transform(image, None)
            else:
                image, mask = self.transform(image, mask)
        else:
            image = np.asarray(image)
            mask = np.asarray(mask) if mask is not None else None

        result = {
            "img_id": image_id,
            "img": torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float(),
        }
        if mask is not None and self.mode != "test":
            result["gt_semantic_seg"] = torch.from_numpy(
                np.ascontiguousarray(mask)
            ).long()
        return result


__all__ = [
    "CLASSES",
    "PALETTE",
    "GreatWallDataset",
    "train_aug",
    "val_aug",
]
