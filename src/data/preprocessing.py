"""Reusable medical X-ray preprocessing pipeline (PRD Section 7.1).

Ensures identical preprocessing between model training and FastAPI production inference:
1. Image loading & format validation (BMP, PNG, JPG, DICOM).
2. Grayscale conversion.
3. Aspect-ratio-preserving letterbox resize.
4. Contrast Limited Adaptive Histogram Equalization (CLAHE).
5. Gaussian / bilateral denoising.
6. Intensity normalization ([0, 1] and standardized tensor).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch


def load_grayscale_xray(image_path: str | Path) -> np.ndarray:
    """Load an X-ray image (BMP/PNG/JPG/DICOM) as a 2D uint8 grayscale array."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"X-ray image not found: {path}")

    ext = path.suffix.lower()
    if ext in {".dcm", ".dicom"}:
        import pydicom

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
        img = (arr * 255.0).astype(np.uint8)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to decode radiograph image: {path}")
    return img


def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    pad_value: int = 0,
    is_mask: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resize image while preserving anatomical aspect ratio using symmetric padding."""
    th, tw = target_size
    h, w = image.shape[:2]
    scale = min(tw / max(w, 1), th / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_top = (th - new_h) // 2
    pad_bottom = th - new_h - pad_top
    pad_left = (tw - new_w) // 2
    pad_right = tw - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    meta = {
        "orig_height": int(h),
        "orig_width": int(w),
        "scale": float(scale),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
        "target_height": int(th),
        "target_width": int(tw),
    }
    return padded, meta


def apply_clahe(
    image_u8: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local bone trabecular and cortical contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_u8)


def denoise_radiograph(image_u8: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """Apply edge-preserving or mild Gaussian noise reduction."""
    if method == "bilateral":
        return cv2.bilateralFilter(image_u8, d=5, sigmaColor=25, sigmaSpace=25)
    if method == "gaussian":
        return cv2.GaussianBlur(image_u8, (3, 3), 0.6)
    return image_u8


def normalize_intensity(image_u8: np.ndarray) -> np.ndarray:
    """Percentile clip (0.5% to 99.5%) and normalize to float32 in [0.0, 1.0]."""
    img_f = image_u8.astype(np.float32)
    p_low, p_high = np.percentile(img_f, [0.5, 99.5])
    if p_high > p_low:
        img_f = np.clip(img_f, p_low, p_high)
        img_f = (img_f - p_low) / (p_high - p_low)
    else:
        img_f = img_f / 255.0
    return img_f.astype(np.float32)


def preprocess_xray(
    image_or_path: str | Path | np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    use_clahe: bool = True,
    denoise_method: str = "gaussian",
    three_channel: bool = True,
) -> Dict[str, Any]:
    """Complete deterministic preprocessing pipeline for training and inference.

    Returns dictionary containing:
        - `tensor`: PyTorch float32 tensor (C, H, W)
        - `normalized`: float32 numpy array (H, W) in [0, 1]
        - `enhanced_u8`: uint8 numpy array (H, W) after resize + CLAHE
        - `meta`: resize and original dimension metadata
    """
    if isinstance(image_or_path, (str, Path)):
        raw_u8 = load_grayscale_xray(image_or_path)
    else:
        raw_u8 = image_or_path
        if raw_u8.ndim == 3:
            raw_u8 = cv2.cvtColor(raw_u8, cv2.COLOR_BGR2GRAY)
        raw_u8 = raw_u8.astype(np.uint8)

    resized_u8, meta = letterbox_resize(raw_u8, target_size=target_size)
    denoised_u8 = denoise_radiograph(resized_u8, method=denoise_method)
    enhanced_u8 = apply_clahe(denoised_u8) if use_clahe else denoised_u8
    norm_2d = normalize_intensity(enhanced_u8)

    if three_channel:
        stacked = np.stack([norm_2d, norm_2d, norm_2d], axis=0)
        # ImageNet-style normalization for pretrained backbones
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        tensor = torch.from_numpy((stacked - mean) / std)
    else:
        tensor = torch.from_numpy(norm_2d[None, ...])

    return {
        "tensor": tensor,
        "normalized": norm_2d,
        "enhanced_u8": enhanced_u8,
        "meta": meta,
    }
