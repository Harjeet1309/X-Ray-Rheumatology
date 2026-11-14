"""Multi-layer radiographic overlay renderer (PRD Section 11).

Generates the 5 interactive views required by the platform:
1. Original X-ray (`original.png`)
2. Classification Grad-CAM heatmap (`gradcam.png`)
3. Detection bounding-box overlay (`detection.png`)
4. Segmentation anatomical & erosion mask overlay (`segmentation.png`)
5. Combined multi-modal visualization (`combined.png`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def render_gradcam_overlay(
    gray_u8: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.42,
) -> np.ndarray:
    """Blend a [0, 1] Grad-CAM heatmap onto a grayscale radiograph (returns BGR uint8)."""
    h, w = gray_u8.shape[:2]
    base_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_u8 = np.clip(hm_resized * 255.0, 0, 255).astype(np.uint8)
    color_hm = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    # Suppress background glow outside the hand silhouette
    hand_fg = (gray_u8 > 22).astype(np.float32)[..., None]
    blended = (
        base_bgr.astype(np.float32) * (1.0 - alpha * hand_fg)
        + color_hm.astype(np.float32) * (alpha * hand_fg)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_detection_overlay(
    gray_u8: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw localized joint bounding boxes and abnormality indicators onto the radiograph."""
    h, w = gray_u8.shape[:2]
    canvas = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    for det in detections:
        norm_box = det.get("bbox_normalized")
        if norm_box and len(norm_box) == 4:
            x0 = int(norm_box[0] * w)
            y0 = int(norm_box[1] * h)
            bw = int(norm_box[2] * w)
            bh = int(norm_box[3] * h)
        else:
            x0, y0, bw, bh = [int(v) for v in det.get("bbox", [0, 0, 32, 32])]

        flagged = bool(det.get("abnormality_flagged", False))
        # BGR colors: Coral/Red for flagged erosion/JSN, Clinical Teal for normal localized joint
        color = (56, 68, 239) if flagged else (180, 165, 14)
        thickness = 2 if flagged else 1

        cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), color, thickness)
        label_txt = f"{det.get('joint_name', 'J')} {int(det.get('confidence', 0.9)*100)}%"
        cv2.putText(
            canvas,
            label_txt,
            (max(2, x0), max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def render_segmentation_overlay(
    gray_u8: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    """Overlay anatomical bone (cyan), carpal (blue), and erosion (red) masks."""
    h, w = gray_u8.shape[:2]
    base_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    color_layer = np.zeros_like(base_bgr, dtype=np.uint8)
    # Class 1: Cortical/trabecular bone -> Teal/Cyan (BGR: 190, 180, 30)
    color_layer[mask_resized == 1] = (190, 180, 30)
    # Class 2: Carpal/joint structures -> Royal Blue (BGR: 235, 130, 50)
    color_layer[mask_resized == 2] = (235, 130, 50)
    # Class 3: Bone Erosion (BE) -> High-contrast Crimson Red (BGR: 60, 50, 245)
    color_layer[mask_resized == 3] = (60, 50, 245)

    active = (mask_resized > 0)[..., None].astype(np.float32)
    erosion_boost = (mask_resized == 3)[..., None].astype(np.float32) * 0.35
    w_map = np.clip(active * alpha + erosion_boost, 0.0, 0.80)

    blended = base_bgr.astype(np.float32) * (1.0 - w_map) + color_layer.astype(np.float32) * w_map
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_all_visualizations(
    gray_u8: np.ndarray,
    heatmap: np.ndarray,
    detections: List[Dict[str, Any]],
    seg_mask: np.ndarray,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Render and save all 5 visualization layers to `output_dir`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    disp_u8 = cv2.resize(gray_u8, (512, 512), interpolation=cv2.INTER_AREA)
    orig_bgr = cv2.cvtColor(disp_u8, cv2.COLOR_GRAY2BGR)
    gradcam_bgr = render_gradcam_overlay(disp_u8, heatmap)
    det_bgr = render_detection_overlay(disp_u8, detections)
    seg_bgr = render_segmentation_overlay(disp_u8, seg_mask)

    # Combined multi-modal overlay: segmentation + subtle Grad-CAM + detection bounding boxes
    combined_base = cv2.addWeighted(seg_bgr, 0.65, gradcam_bgr, 0.35, 0)
    combined_bgr = render_detection_overlay(
        cv2.cvtColor(combined_base, cv2.COLOR_BGR2GRAY), detections
    )
    combined_bgr = cv2.addWeighted(combined_base, 0.55, combined_bgr, 0.45, 0)

    paths = {
        "original": str(out / "original.png"),
        "gradcam": str(out / "gradcam.png"),
        "detection": str(out / "detection.png"),
        "segmentation": str(out / "segmentation.png"),
        "combined": str(out / "combined.png"),
    }
    cv2.imwrite(paths["original"], orig_bgr)
    cv2.imwrite(paths["gradcam"], gradcam_bgr)
    cv2.imwrite(paths["detection"], det_bgr)
    cv2.imwrite(paths["segmentation"], seg_bgr)
    cv2.imwrite(paths["combined"], combined_bgr)
    return paths
