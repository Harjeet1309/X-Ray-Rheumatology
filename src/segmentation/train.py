"""Training pipeline for Model 3: Bone and Erosion Segmentation (PRD Section 8.3)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset_loader import RAMH1200SegmentationDataset
from src.evaluation.metrics import compute_segmentation_metrics
from src.segmentation.model import build_segmentation_model


class CombinedDiceCELoss(nn.Module):
    """Combined weighted Cross-Entropy and Multiclass Soft Dice Loss."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Upweight carpal (2) and bone erosion (3) pixels
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 1.0, 1.5, 2.5]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        dice_total = 0.0
        for c in range(1, self.num_classes):
            p = probs[:, c]
            t = (targets == c).float()
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice_total += 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)
        return ce_loss + dice_total / max(self.num_classes - 1, 1)


def train_segmentation_model(
    root_dir: str | Path = ".",
    architectures: List[str] | None = None,
    epochs: int = 4,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    image_size: tuple[int, int] = (128, 128),
    seed: int = 42,
) -> Dict[str, Any]:
    """Train U-Net and U-Net++ segmentation models and save the best checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    root = Path(root_dir)
    t0 = time.time()

    train_ds = RAMH1200SegmentationDataset(
        "train", root_dir=root, image_size=image_size, task="combined", augment=True
    )
    val_ds = RAMH1200SegmentationDataset(
        "val", root_dir=root, image_size=image_size, task="combined", augment=False
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    arch_list = architectures or ["unet", "unetplusplus"]
    best_arch = arch_list[0]
    best_dice = -1.0
    best_state = None
    best_metrics: Dict[str, Any] = {}
    comparison_list: List[Dict[str, Any]] = []

    for arch in arch_list:
        model = build_segmentation_model(architecture=arch, in_channels=1, num_classes=4)
        criterion = CombinedDiceCELoss(num_classes=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        for _ in range(epochs):
            model.train()
            for batch in train_loader:
                imgs = batch["image"]
                masks = batch["mask"]
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

        model.eval()
        batch_metrics: List[Dict[str, Any]] = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"]
                masks = batch["mask"].numpy()
                preds = torch.argmax(model(imgs), dim=1).numpy()
                for p, t in zip(preds, masks):
                    batch_metrics.append(compute_segmentation_metrics(p, t, num_classes=4))

        avg_m = {
            k: round(float(np.mean([m[k] for m in batch_metrics])), 4) if batch_metrics else 0.0
            for k in ("dice", "iou", "precision", "recall")
        }
        comparison_list.append({"architecture": arch, "metrics": avg_m})
        if avg_m["dice"] >= best_dice:
            best_dice = avg_m["dice"]
            best_arch = arch
            best_metrics = avg_m
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    out_dir = root / "models" / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pth"
    meta_path = out_dir / "metadata.json"

    checkpoint = {
        "architecture": best_arch,
        "version": "1.0.0",
        "num_classes": 4,
        "class_names": ["Background", "Hand Bone", "Carpal/Joint Region", "Bone Erosion"],
        "image_size": list(image_size),
        "state_dict": best_state,
        "metrics": best_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, ckpt_path)

    metadata = {
        "model_name": "Hand Bone & Erosion Segmentation Engine",
        "task": "segmentation",
        "architecture": best_arch,
        "version": "1.0.0",
        "dataset_version": "RAM-H1200-v1",
        "checkpoint_path": str(ckpt_path.relative_to(root)).replace("\\", "/"),
        "evaluation_metrics": best_metrics,
        "candidate_comparison": comparison_list,
        "training_duration_sec": round(time.time() - t0, 2),
        "trained_at": checkpoint["trained_at"],
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train_segmentation_model("."), indent=2))
