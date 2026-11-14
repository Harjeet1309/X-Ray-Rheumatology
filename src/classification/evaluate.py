"""Independent test-set evaluation for Model 1: RA Classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from src.classification.model import build_classifier
from src.data.dataset_loader import create_dataloaders
from src.evaluation.metrics import compute_classification_metrics


def evaluate_classifier(root_dir: str | Path = ".") -> Dict[str, Any]:
    """Evaluate saved classification checkpoint on the independent test split."""
    root = Path(root_dir)
    ckpt_path = root / "models" / "classification" / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Classification checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("architecture", "densenet121")
    img_size = tuple(ckpt.get("image_size", [128, 128]))

    model = build_classifier(architecture=arch, num_classes=2, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loaders = create_dataloaders(root_dir=root, batch_size=4, image_size=img_size)
    y_true, y_prob = [], []
    predictions = []

    with torch.no_grad():
        for batch in loaders["test"]:
            imgs = batch["image"]
            labels = batch["label"]
            stems = batch["image_stem"]
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            for stem, t, p in zip(stems, labels.numpy(), probs):
                y_true.append(int(t))
                y_prob.append(float(p))
                predictions.append(
                    {
                        "image_stem": stem,
                        "true_label": int(t),
                        "predicted_probability": round(float(p), 4),
                        "predicted_label": int(p >= 0.5),
                    }
                )

    metrics = compute_classification_metrics(y_true, y_prob)
    report = {
        "model": "classification",
        "architecture": arch,
        "version": ckpt.get("version", "1.0.0"),
        "split": "test",
        "metrics": metrics,
        "predictions": predictions,
    }
    out_path = root / "models" / "classification" / "test_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
