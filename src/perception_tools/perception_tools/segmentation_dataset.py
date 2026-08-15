import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from perception_tools.segmentation_model_common import (
    IGNORE_INDEX,
    find_image_mask_pairs,
    normalize_image_rgb,
)


class SegmentationDataset(Dataset):

    def __init__(self, images_dir: Path, masks_dir: Path, labels: list[str], input_size: int, use_augmentation: bool,):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.labels = labels
        self.num_classes = len(labels)
        self.input_size = int(input_size)
        self.use_augmentation = use_augmentation
        self.samples = find_image_mask_pairs(self.images_dir, self.masks_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.input_size > 0:
            size = (self.input_size, self.input_size)
            image_rgb = cv2.resize(image_rgb, size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        valid_pixels = mask != IGNORE_INDEX
        mask[valid_pixels] = np.clip(mask[valid_pixels], 0, self.num_classes - 1)

        if self.use_augmentation:
            image_rgb, mask = self.augment(image_rgb, mask)

        image_rgb = normalize_image_rgb(image_rgb)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask.copy()).long()

        return image_tensor, mask_tensor

    def augment(self, image_rgb, mask):
        if random.random() < 0.5:
            image_rgb = np.ascontiguousarray(np.fliplr(image_rgb))
            mask = np.ascontiguousarray(np.fliplr(mask))

        return image_rgb, mask
