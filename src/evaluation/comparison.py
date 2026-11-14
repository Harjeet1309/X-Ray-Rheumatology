"""Multi-architecture model comparison and selection protocol (PRD Section 10.5).

Ranks candidate models using a composite clinical utility score that prioritizes
Sensitivity, Specificity, ROC-AUC, and Calibration rather than raw accuracy alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def compute_clinical_composite_score(metrics: Dict[str, Any]) -> float:
    """Compute weighted clinical composite score for classification models."""
    sens = float(metrics.get("sensitivity", 0.0))
    spec = float(metrics.get("specificity", 0.0))
    roc_auc = float(metrics.get("roc_auc", 0.0))
    f1 = float(metrics.get("f1_score", 0.0))
    # Medical screening prioritizes sensitivity + specificity balance (Youden's J) and ROC-AUC
    score = 0.35 * sens + 0.30 * spec + 0.20 * roc_auc + 0.15 * f1
    return round(score, 4)


def compare_models(
    candidates: List[Dict[str, Any]],
    output_path: str | Path = "experiments/model_comparison_report.json",
) -> Dict[str, Any]:
    """Compare multiple candidate architectures and select the clinically optimal model."""
    ranked: List[Dict[str, Any]] = []
    for entry in candidates:
        metrics = entry.get("metrics", {})
        composite = compute_clinical_composite_score(metrics)
        enriched = {
            **entry,
            "clinical_composite_score": composite,
        }
        ranked.append(enriched)

    ranked.sort(key=lambda x: x["clinical_composite_score"], reverse=True)
    best = ranked[0] if ranked else None

    report = {
        "selection_protocol": "Weighted Clinical Utility (0.35*Sensitivity + 0.30*Specificity + 0.20*ROC_AUC + 0.15*F1)",
        "selected_model": best["architecture"] if best else None,
        "selected_composite_score": best["clinical_composite_score"] if best else 0.0,
        "candidates": ranked,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
