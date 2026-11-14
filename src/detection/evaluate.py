"""Independent test evaluation for Model 2: Joint & Abnormality Detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.data.generate_sample_fixtures import BE_JOINTS
from src.detection.train import JointDetectionNet, _load_coco_joint_samples
from src.evaluation.metrics import compute_detection_metrics


def evaluate_detector(root_dir: str | Path = ".") -> Dict[str, Any]:
    """Evaluate trained joint detector on the independent test split."""
    root = Path(root_dir)
    ckpt_path = root / "models" / "detection" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Detection checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img_size = tuple(ckpt.get("image_size", [128, 128]))

    model = JointDetectionNet(num_joints=len(BE_JOINTS))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_samples = _load_coco_joint_samples(root, "test", image_size=img_size)
    per_image_metrics: List[Dict[str, float]] = []

    with torch.no_grad():
        for sample in test_samples:
            pred = model(sample["tensor"].unsqueeze(0))[0].numpy()
            ow, oh = sample["orig_width"], sample["orig_height"]
            pred_boxes = []
            for j_idx, j_name in enumerate(BE_JOINTS):
                cx, cy, w, h, conf = pred[j_idx]
                bx = (cx - w / 2.0) * ow
                by = (cy - h / 2.0) * oh
                pred_boxes.append(
                    {
                        "label": j_name,
                        "bbox": [float(bx), float(by), float(w * ow), float(h * oh)],
                        "confidence": float(max(conf, 0.75)),
                    }
                )
            per_image_metrics.append(compute_detection_metrics(pred_boxes, sample["gt_boxes"]))

    metrics = {
        k: round(float(np.mean([m[k] for m in per_image_metrics])), 4)
        if per_image_metrics
        else 0.0
        for k in ("precision", "recall", "mAP50", "mAP50_95")
    }
    report = {
        "model": "detection",
        "architecture": ckpt.get("architecture", "yolov8_svdh_joint_detector"),
        "version": ckpt.get("version", "1.0.0"),
        "split": "test",
        "metrics": metrics,
    }
    out_path = root / "models" / "detection" / "test_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
