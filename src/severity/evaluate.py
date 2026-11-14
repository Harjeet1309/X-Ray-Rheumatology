"""Independent test evaluation for Model 4: SvdH Severity Assessment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.dataset_loader import RAMH1200SeverityDataset
from src.data.generate_sample_fixtures import BE_JOINTS, JSN_JOINTS
from src.evaluation.metrics import compute_severity_metrics
from src.severity.model import SvdHSeverityNet


def evaluate_severity(root_dir: str | Path = ".") -> Dict[str, Any]:
    """Evaluate trained SvdH severity model on the independent test split."""
    root = Path(root_dir)
    ckpt_path = root / "models" / "severity" / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Severity checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img_size = tuple(ckpt.get("image_size", [128, 128]))

    model = SvdHSeverityNet(
        num_be_joints=len(BE_JOINTS),
        num_jsn_joints=len(JSN_JOINTS),
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_ds = RAMH1200SeverityDataset("test", root_dir=root, image_size=img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    be_true, be_pred, jsn_true, jsn_pred = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["image"])
            be_true.extend(batch["be_scores"].numpy().flatten().tolist())
            be_pred.extend(out["be_scores"].numpy().flatten().tolist())
            jsn_true.extend(batch["jsn_scores"].numpy().flatten().tolist())
            jsn_pred.extend(out["jsn_scores"].numpy().flatten().tolist())

    be_m = compute_severity_metrics(be_true, be_pred, max_score=5)
    jsn_m = compute_severity_metrics(jsn_true, jsn_pred, max_score=4)
    report = {
        "model": "severity",
        "architecture": ckpt.get("architecture", "svdh_multitask_resnet18"),
        "version": ckpt.get("version", "1.0.0"),
        "split": "test",
        "metrics": {
            "mae": round((be_m["mae"] + jsn_m["mae"]) / 2.0, 4),
            "rmse": round((be_m["rmse"] + jsn_m["rmse"]) / 2.0, 4),
            "quadratic_weighted_kappa": round(
                (be_m["quadratic_weighted_kappa"] + jsn_m["quadratic_weighted_kappa"]) / 2.0, 4
            ),
            "ordinal_accuracy": round((be_m["ordinal_accuracy"] + jsn_m["ordinal_accuracy"]) / 2.0, 4),
            "be_joint_metrics": be_m,
            "jsn_joint_metrics": jsn_m,
        },
    }
    out_path = root / "models" / "severity" / "test_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
