"""Model registry, dataset audit, and ML monitoring endpoints (/api/models, /api/monitoring)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database.models import Analysis, ModelRegistry, ModelResult
from backend.app.database.session import get_db

router = APIRouter(prefix="/api", tags=["Models & Monitoring"])


@router.get("/models", response_model=Dict[str, Any])
def list_models(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve registered deep-learning models, versions, evaluation metrics, and dataset audit."""
    models = db.query(ModelRegistry).all()

    remote_audit = {}
    audit_path = Path("dataset/annotations/ram_h1200_remote_audit.json")
    if audit_path.exists():
        remote_audit = json.loads(audit_path.read_text(encoding="utf-8"))

    val_report = {}
    val_path = Path("dataset/splits/data_validation_report.json")
    if val_path.exists():
        val_report = json.loads(val_path.read_text(encoding="utf-8"))

    cls_comparison = {}
    comp_path = Path("experiments/classification_comparison.json")
    if comp_path.exists():
        cls_comparison = json.loads(comp_path.read_text(encoding="utf-8"))

    return {
        "models": [
            {
                "model_id": m.model_id,
                "model_name": m.model_name,
                "task": m.task,
                "architecture": m.architecture,
                "version": m.version,
                "dataset_version": m.dataset_version,
                "evaluation_metrics": m.evaluation_metrics,
                "deployment_status": m.deployment_status,
                "registered_at": m.registered_at.isoformat() if m.registered_at else None,
            }
            for m in models
        ],
        "dataset_audit": remote_audit,
        "validation_report": val_report,
        "architecture_comparison": cls_comparison,
    }


@router.get("/monitoring", response_model=Dict[str, Any])
def get_ml_monitoring_metrics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return production ML and application telemetry (PRD Section 17)."""
    analyses = db.query(Analysis).all()
    results = db.query(ModelResult).all()

    completed = [a for a in analyses if a.analysis_status == "completed"]
    failed = [a for a in analyses if a.analysis_status == "failed"]

    durations: List[float] = [
        float((a.summary_payload or {}).get("processing_time_ms", 0.0))
        for a in completed
        if (a.summary_payload or {}).get("processing_time_ms") is not None
    ]
    confidences: List[float] = [float(r.confidence) for r in results if r.confidence is not None]

    return {
        "application_monitoring": {
            "total_requests": len(analyses),
            "completed_analyses": len(completed),
            "failed_analyses": len(failed),
            "mean_processing_duration_ms": (
                round(sum(durations) / len(durations), 1) if durations else 0.0
            ),
        },
        "ml_monitoring": {
            "total_model_inferences": len(results),
            "mean_model_confidence": (
                round(sum(confidences) / len(confidences), 4) if confidences else 0.0
            ),
            "auto_retraining_enabled": False,
            "drift_status": "nominal",
        },
    }
