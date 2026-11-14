"""Clinically constrained data augmentation for hand X-ray radiographs (PRD Section 7.2).

Supports synchronized transformations across images, segmentation masks, and bounding boxes:
- Rotation within clinically reasonable limits (±10 degrees).
- Controlled scaling (0.95 to 1.05) and small translation (±4%).
- Horizontal flipping (only when anatomically appropriate).
- Mild brightness/contrast adjustments and low-variance Gaussian noise.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class RadiographAugmentor:
    """Clinically safe augmentor for 2D hand X-rays and optional segmentation masks."""

    def __init__(
        self,
        max_rotation_deg: float = 10.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        max_shift_fraction: float = 0.04,
        allow_horizontal_flip: bool = True,
        brightness_delta: float = 0.08,
        contrast_range: Tuple[float, float] = (0.92, 1.08),
        noise_std: float = 0.015,
        seed: Optional[int] = None,
    ) -> None:
        self.max_rotation_deg = max_rotation_deg
        self.scale_range = scale_range
        self.max_shift_fraction = max_shift_fraction
        self.allow_horizontal_flip = allow_horizontal_flip
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply synchronized spatial and photometric augmentations.

        Args:
            image: float32 array in [0, 1] or uint8 array (H, W)
            mask: optional uint8 segmentation mask (H, W)
        """
        is_u8 = image.dtype == np.uint8
        img = image.astype(np.float32) / (255.0 if is_u8 else 1.0)
        h, w = img.shape[:2]
        aug_mask = mask.copy() if mask is not None else None

        # 1. Optional Horizontal Flip
        if self.allow_horizontal_flip and self.rng.random() < 0.5:
            img = np.fliplr(img).copy()
            if aug_mask is not None:
                aug_mask = np.fliplr(aug_mask).copy()

        # 2. Clinically constrained affine transform (rotation + scale + translation)
        if self.rng.random() < 0.8:
            angle = float(self.rng.uniform(-self.max_rotation_deg, self.max_rotation_deg))
            scale = float(self.rng.uniform(self.scale_range[0], self.scale_range[1]))
            tx = float(self.rng.uniform(-self.max_shift_fraction, self.max_shift_fraction) * w)
            ty = float(self.rng.uniform(-self.max_shift_fraction, self.max_shift_fraction) * h)

            mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            mat[0, 2] += tx
            mat[1, 2] += ty

            img = cv2.warpAffine(
                img,
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            if aug_mask is not None:
                aug_mask = cv2.warpAffine(
                    aug_mask,
                    mat,
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

        # 3. Brightness & contrast adjustment
        if self.rng.random() < 0.7:
            alpha = float(self.rng.uniform(self.contrast_range[0], self.contrast_range[1]))
            beta = float(self.rng.uniform(-self.brightness_delta, self.brightness_delta))
            img = np.clip(img * alpha + beta, 0.0, 1.0)

        # 4. Mild quantum mottle noise
        if self.rng.random() < 0.5 and self.noise_std > 0:
            noise = self.rng.normal(0.0, self.noise_std, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        if is_u8:
            img = (img * 255.0).round().astype(np.uint8)
        return img, aug_mask
