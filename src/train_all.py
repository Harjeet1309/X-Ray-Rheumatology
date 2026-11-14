"""Master end-to-end dataset initialization, validation, training, and evaluation pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from src.classification.evaluate import evaluate_classifier
from src.classification.train import train_and_select_best_classifier
from src.data.download_ram_h1200 import download_ram_h1200
from src.data.validation import run_dataset_validation
from src.detection.evaluate import evaluate_detector
from src.detection.train import train_detection_model
from src.segmentation.evaluate import evaluate_segmentation
from src.segmentation.train import train_segmentation_model
from src.severity.evaluate import evaluate_severity
from src.severity.train import train_severity_model


def run_complete_ml_pipeline(root_dir: str | Path = ".") -> Dict[str, Any]:
    """Execute dataset setup, validation, multi-model training, and independent test evaluation."""
    root = Path(root_dir)
    t0 = time.time()

    print("[1/6] Initializing RAM-H1200 dataset and auditing remote manifest...")
    dataset_info = download_ram_h1200(target_dir=root / "dataset" / "raw")

    print("[2/6] Running 8-point dataset validation and patient-level leakage check...")
    validation_report = run_dataset_validation(root_dir=root)

    print("[3/6] Training & comparing Model 1: RA Classification architectures...")
    cls_meta = train_and_select_best_classifier(
        root_dir=root,
        architectures=["densenet121", "resnet50", "efficientnet_b0"],
        epochs=2,
        image_size=(96, 96),
    )
    cls_eval = evaluate_classifier(root_dir=root)

    print("[4/6] Training & evaluating Model 2: SvdH Joint & Abnormality Detector...")
    det_meta = train_detection_model(root_dir=root, epochs=5, image_size=(96, 96))
    det_eval = evaluate_detector(root_dir=root)

    print("[5/6] Training & comparing Model 3: Bone & Erosion Segmentation (U-Net / U-Net++)...")
    seg_meta = train_segmentation_model(
        root_dir=root,
        architectures=["unet", "unetplusplus"],
        epochs=3,
        image_size=(96, 96),
    )
    seg_eval = evaluate_segmentation(root_dir=root)

    print("[6/6] Training & evaluating Model 4: SvdH Severity Assessment (BE + JSN)...")
    sev_meta = train_severity_model(root_dir=root, epochs=4, image_size=(96, 96))
    sev_eval = evaluate_severity(root_dir=root)

    summary = {
        "pipeline_status": "completed",
        "total_duration_sec": round(time.time() - t0, 2),
        "dataset_status": dataset_info.get("status"),
        "validation_passed": validation_report.get("validation_passed"),
        "patient_leakage_passed": validation_report.get("patient_leakage", {}).get(
            "passed_zero_leakage"
        ),
        "models": {
            "classification": {
                "selected_architecture": cls_meta["architecture"],
                "test_metrics": cls_eval["metrics"],
            },
            "detection": {
                "selected_architecture": det_meta["architecture"],
                "test_metrics": det_eval["metrics"],
            },
            "segmentation": {
                "selected_architecture": seg_meta["architecture"],
                "test_metrics": seg_eval["metrics"],
            },
            "severity": {
                "selected_architecture": sev_meta["architecture"],
                "test_metrics": sev_eval["metrics"],
            },
        },
    }

    exp_dir = root / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    result = run_complete_ml_pipeline(".")
    print("\n=== Complete ML Pipeline Summary ===")
    print(json.dumps(result, indent=2))
