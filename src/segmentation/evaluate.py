"""Independent test evaluation for Model 3: Bone and Erosion Segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset_loader import RAMH1200SegmentationDataset
from src.evaluation.metrics import compute_segmentation_metrics
from src.segmentation.model import build_segmentation_model


def evaluate_segmentation(root_dir: str | Path = ".") -> Dict[str, Any]:
    """Evaluate trained segmentation checkpoint on the independent test split."""
    root = Path(root_dir)
    ckpt_path = root / "models" / "segmentation" / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("architecture", "unet")
    img_size = tuple(ckpt.get("image_size", [128, 128]))

    model = build_segmentation_model(architecture=arch, in_channels=1, num_classes=4)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_ds = RAMH1200SegmentationDataset(
        "test", root_dir=root, image_size=img_size, task="combined", augment=False
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    batch_metrics: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"]
            masks = batch["mask"].numpy()
            preds = torch.argmax(model(imgs), dim=1).numpy()
            for p, t in zip(preds, masks):
                batch_metrics.append(compute_segmentation_metrics(p, t, num_classes=4))

    metrics = {
        k: round(float(np.mean([m[k] for m in batch_metrics])), 4) if batch_metrics else 0.0
        for k in ("dice", "iou", "precision", "recall")
    }
    report = {
        "model": "segmentation",
        "architecture": arch,
        "version": ckpt.get("version", "1.0.0"),
        "split": "test",
        "metrics": metrics,
    }
    out_path = root / "models" / "segmentation" / "test_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
