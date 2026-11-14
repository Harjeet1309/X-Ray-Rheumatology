"""Reproducible training pipeline for Model 1: RA Classification (PRD Sections 8.1 & 9)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.classification.model import build_classifier
from src.data.dataset_loader import create_dataloaders
from src.evaluation.comparison import compare_models
from src.evaluation.metrics import compute_classification_metrics


def train_single_architecture(
    architecture: str = "densenet121",
    root_dir: str | Path = ".",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    image_size: tuple[int, int] = (128, 128),
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and validate a single classification architecture."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = create_dataloaders(root_dir=root_dir, batch_size=batch_size, image_size=image_size)
    model = build_classifier(architecture=architecture, num_classes=2, pretrained=False).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.15], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: List[Dict[str, float]] = []
    best_val_score = -1.0
    best_state = None
    best_val_metrics: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for batch in loaders["train"]:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * imgs.size(0)
            n_samples += imgs.size(0)
        scheduler.step()

        train_loss = running_loss / max(n_samples, 1)

        # Validation pass
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                imgs = batch["image"].to(device)
                labels = batch["label"]
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_true.extend(labels.numpy().tolist())
                y_prob.extend(probs.tolist())

        val_metrics = compute_classification_metrics(y_true, y_prob)
        composite = 0.5 * val_metrics["sensitivity"] + 0.5 * val_metrics["specificity"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_accuracy": val_metrics["accuracy"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )
        if composite >= best_val_score:
            best_val_score = composite
            best_val_metrics = val_metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return {
        "architecture": architecture,
        "best_state": best_state,
        "metrics": best_val_metrics,
        "history": history,
        "device": str(device),
    }


def train_and_select_best_classifier(
    root_dir: str | Path = ".",
    architectures: List[str] | None = None,
    epochs: int = 3,
    image_size: tuple[int, int] = (128, 128),
) -> Dict[str, Any]:
    """Train candidate architectures, compare them, and save the best checkpoint."""
    root = Path(root_dir)
    arch_list = architectures or ["densenet121", "resnet50", "efficientnet_b0"]
    candidates = []
    states = {}

    t0 = time.time()
    for arch in arch_list:
        res = train_single_architecture(
            architecture=arch,
            root_dir=root,
            epochs=epochs,
            image_size=image_size,
        )
        states[arch] = res["best_state"]
        candidates.append(
            {
                "architecture": arch,
                "metrics": res["metrics"],
                "history": res["history"],
            }
        )

    comparison = compare_models(
        candidates, output_path=root / "experiments" / "classification_comparison.json"
    )
    selected_arch = comparison["selected_model"] or arch_list[0]
    best_candidate = next(c for c in candidates if c["architecture"] == selected_arch)

    out_dir = root / "models" / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pth"
    meta_path = out_dir / "metadata.json"

    checkpoint = {
        "architecture": selected_arch,
        "version": "1.0.0",
        "num_classes": 2,
        "image_size": list(image_size),
        "state_dict": states[selected_arch],
        "metrics": best_candidate["metrics"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, ckpt_path)

    metadata = {
        "model_name": "RA Radiographic Pattern Classifier",
        "task": "classification",
        "architecture": selected_arch,
        "version": "1.0.0",
        "dataset_version": "RAM-H1200-v1",
        "checkpoint_path": str(ckpt_path.relative_to(root)).replace("\\", "/"),
        "evaluation_metrics": best_candidate["metrics"],
        "candidate_comparison": comparison,
        "training_duration_sec": round(time.time() - t0, 2),
        "trained_at": checkpoint["trained_at"],
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    meta = train_and_select_best_classifier(".")
    print(json.dumps(meta, indent=2))
