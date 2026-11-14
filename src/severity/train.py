"""Training pipeline for Model 4: SvdH Radiographic Severity Assessment (PRD Section 8.4)."""

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

from src.data.dataset_loader import RAMH1200SeverityDataset
from src.data.generate_sample_fixtures import BE_JOINTS, JSN_JOINTS
from src.evaluation.metrics import compute_severity_metrics
from src.severity.model import SvdHSeverityNet


def train_severity_model(
    root_dir: str | Path = ".",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    image_size: tuple[int, int] = (128, 128),
    seed: int = 42,
) -> Dict[str, Any]:
    """Train the multi-head SvdH BE + JSN severity model and save the best checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    root = Path(root_dir)
    t0 = time.time()

    train_ds = RAMH1200SeverityDataset("train", root_dir=root, image_size=image_size, augment=True)
    val_ds = RAMH1200SeverityDataset("val", root_dir=root, image_size=image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SvdHSeverityNet(
        num_be_joints=len(BE_JOINTS),
        num_jsn_joints=len(JSN_JOINTS),
        pretrained=False,
    )
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            imgs = batch["image"]
            true_be = batch["be_scores"]
            true_jsn = batch["jsn_scores"]

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out["be_scores"], true_be) + criterion(out["jsn_scores"], true_jsn)
            loss.backward()
            optimizer.step()

    model.eval()
    be_true_list: List[float] = []
    be_pred_list: List[float] = []
    jsn_true_list: List[float] = []
    jsn_pred_list: List[float] = []

    with torch.no_grad():
        for batch in val_loader:
            out = model(batch["image"])
            be_true_list.extend(batch["be_scores"].numpy().flatten().tolist())
            be_pred_list.extend(out["be_scores"].numpy().flatten().tolist())
            jsn_true_list.extend(batch["jsn_scores"].numpy().flatten().tolist())
            jsn_pred_list.extend(out["jsn_scores"].numpy().flatten().tolist())

    be_metrics = compute_severity_metrics(be_true_list, be_pred_list, max_score=5)
    jsn_metrics = compute_severity_metrics(jsn_true_list, jsn_pred_list, max_score=4)
    combined_metrics = {
        "mae": round((be_metrics["mae"] + jsn_metrics["mae"]) / 2.0, 4),
        "rmse": round((be_metrics["rmse"] + jsn_metrics["rmse"]) / 2.0, 4),
        "quadratic_weighted_kappa": round(
            (be_metrics["quadratic_weighted_kappa"] + jsn_metrics["quadratic_weighted_kappa"]) / 2.0,
            4,
        ),
        "ordinal_accuracy": round(
            (be_metrics["ordinal_accuracy"] + jsn_metrics["ordinal_accuracy"]) / 2.0, 4
        ),
        "be_joint_metrics": be_metrics,
        "jsn_joint_metrics": jsn_metrics,
    }

    out_dir = root / "models" / "severity"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pth"
    meta_path = out_dir / "metadata.json"

    checkpoint = {
        "architecture": model.architecture,
        "version": "1.0.0",
        "image_size": list(image_size),
        "be_joints": BE_JOINTS,
        "jsn_joints": JSN_JOINTS,
        "state_dict": model.state_dict(),
        "metrics": combined_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, ckpt_path)

    metadata = {
        "model_name": "SvdH Radiographic Severity Scorer (BE + JSN)",
        "task": "severity",
        "architecture": model.architecture,
        "version": "1.0.0",
        "dataset_version": "RAM-H1200-v1",
        "checkpoint_path": str(ckpt_path.relative_to(root)).replace("\\", "/"),
        "evaluation_metrics": combined_metrics,
        "training_duration_sec": round(time.time() - t0, 2),
        "trained_at": checkpoint["trained_at"],
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train_severity_model("."), indent=2))
