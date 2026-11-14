"""X-ray upload, AI inference execution, history, and PDF report routes (/api/analysis/*)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.database.models import Analysis, ModelResult, Report, User
from backend.app.database.session import get_db
from backend.app.middleware.auth import get_optional_user
from backend.app.schemas.api_schemas import (
    AnalysisDetailResponse,
    AnalysisRunResponse,
    AnalysisUploadResponse,
)
from backend.app.services.inference_service import UnifiedAnalysisOrchestrator
from backend.app.services.report_service import generate_analysis_pdf_report
from src.data.preprocessing import load_grayscale_xray
from src.data.validation import validate_single_image

router = APIRouter(prefix="/api/analysis", tags=["X-Ray Analysis"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".dcm", ".dicom"}
MAX_FILE_BYTES = 25 * 1024 * 1024

_orchestrator: Optional[UnifiedAnalysisOrchestrator] = None


def get_orchestrator() -> UnifiedAnalysisOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = UnifiedAnalysisOrchestrator(".")
    return _orchestrator


@router.post("/upload", response_model=AnalysisUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_xray_image(
    file: UploadFile = File(...),
    user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db),
) -> AnalysisUploadResponse:
    """Upload and validate a hand X-ray radiograph for AI analysis."""
    orig_name = file.filename or "hand_xray.png"
    ext = Path(orig_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported: JPG, JPEG, PNG, BMP, DICOM.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file exceeds the 25 MB limit.",
        )

    analysis_id = str(uuid.uuid4())
    uploads_dir = Path("storage") / "uploads"
    previews_dir = Path("storage") / "overlays" / analysis_id
    uploads_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    saved_path = uploads_dir / f"{analysis_id}{ext}"
    saved_path.write_bytes(content)

    validation = validate_single_image(saved_path)
    if not validation.get("valid", False):
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation.get("error", "Invalid radiograph image"),
        )

    # Render web-friendly PNG preview
    gray = load_grayscale_xray(saved_path)
    preview_path = previews_dir / "original.png"
    cv2.imwrite(str(preview_path), gray)
    preview_url = f"/static/overlays/{analysis_id}/original.png"

    record = Analysis(
        analysis_id=analysis_id,
        user_id=user.user_id if user else None,
        filename=orig_name,
        image_reference=str(saved_path).replace("\\", "/"),
        analysis_status="uploaded",
        quality_metadata=validation,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return AnalysisUploadResponse(
        analysis_id=record.analysis_id,
        filename=record.filename,
        analysis_status=record.analysis_status,
        upload_timestamp=record.upload_timestamp,
        quality_metadata=validation,
        preview_url=preview_url,
    )


@router.post("/sample", response_model=AnalysisUploadResponse, status_code=status.HTTP_201_CREATED)
def load_sample_ram_h1200_xray(
    case_type: str = Query("ra_positive", description="ra_positive or normal"),
    user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db),
) -> AnalysisUploadResponse:
    """Load a verified sample hand X-ray from the local RAM-H1200 test split for instant inspection."""
    test_dir = Path("dataset/raw/Segmentation/test")
    bmp_candidates = sorted(test_dir.glob("*.bmp"))
    if not bmp_candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No sample RAM-H1200 radiographs found in dataset/raw/Segmentation/test.",
        )

    chosen = bmp_candidates[0] if case_type == "ra_positive" else bmp_candidates[-1]
    analysis_id = str(uuid.uuid4())
    uploads_dir = Path("storage") / "uploads"
    previews_dir = Path("storage") / "overlays" / analysis_id
    uploads_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    saved_path = uploads_dir / f"{analysis_id}.bmp"
    shutil.copyfile(chosen, saved_path)

    validation = validate_single_image(saved_path)
    gray = load_grayscale_xray(saved_path)
    cv2.imwrite(str(previews_dir / "original.png"), gray)
    preview_url = f"/static/overlays/{analysis_id}/original.png"

    record = Analysis(
        analysis_id=analysis_id,
        user_id=user.user_id if user else None,
        filename=chosen.name,
        image_reference=str(saved_path).replace("\\", "/"),
        analysis_status="uploaded",
        quality_metadata=validation,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return AnalysisUploadResponse(
        analysis_id=record.analysis_id,
        filename=record.filename,
        analysis_status=record.analysis_status,
        upload_timestamp=record.upload_timestamp,
        quality_metadata=validation,
        preview_url=preview_url,
    )


@router.post("/{analysis_id}/run", response_model=AnalysisRunResponse)
def run_ai_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
) -> AnalysisRunResponse:
    """Execute the 4-model AI pipeline (Classification, Detection, Segmentation, SvdH Severity + Grad-CAM)."""
    record = db.query(Analysis).filter(Analysis.analysis_id == analysis_id).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis ID not found.")

    orchestrator = get_orchestrator()
    summary = orchestrator.run_analysis(analysis_id, db)
    db.refresh(record)

    return AnalysisRunResponse(
        analysis_id=record.analysis_id,
        analysis_status=record.analysis_status,
        completed_timestamp=record.completed_timestamp,
        summary=summary,
    )


@router.get("/history", response_model=Dict[str, Any])
def get_analysis_history(
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Return all recorded analyses and real database statistics (PRD Section 13.2 Page 3 & 7)."""
    records = db.query(Analysis).order_by(Analysis.upload_timestamp.desc()).all()
    total_reports = db.query(Report).count()

    completed = [r for r in records if r.analysis_status == "completed"]
    ra_detected_count = sum(
        1
        for r in completed
        if (r.summary_payload or {}).get("classification", {}).get("is_ra_positive", False)
    )

    items: List[Dict[str, Any]] = []
    for r in records:
        summary = r.summary_payload or {}
        cls_info = summary.get("classification", {})
        sev_info = summary.get("severity", {})
        items.append(
            {
                "analysis_id": r.analysis_id,
                "filename": r.filename,
                "upload_timestamp": r.upload_timestamp.isoformat() if r.upload_timestamp else None,
                "completed_timestamp": (
                    r.completed_timestamp.isoformat() if r.completed_timestamp else None
                ),
                "analysis_status": r.analysis_status,
                "predicted_class": cls_info.get("predicted_class"),
                "ra_probability": cls_info.get("ra_probability"),
                "total_svdh_score": sev_info.get("total_svdh_score"),
                "severity_category": sev_info.get("severity_category"),
                "preview_url": f"/static/overlays/{r.analysis_id}/original.png",
            }
        )

    return {
        "statistics": {
            "total_analyses": len(records),
            "completed_analyses": len(completed),
            "ra_pattern_detected_count": ra_detected_count,
            "generated_reports_count": total_reports,
        },
        "analyses": items,
    }


@router.get("/{analysis_id}", response_model=AnalysisDetailResponse)
def get_analysis_details(
    analysis_id: str,
    db: Session = Depends(get_db),
) -> AnalysisDetailResponse:
    """Retrieve complete multi-model results and visualization URLs for an analysis."""
    record = db.query(Analysis).filter(Analysis.analysis_id == analysis_id).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis ID not found.")

    m_results = (
        db.query(ModelResult).filter(ModelResult.analysis_id == analysis_id).all()
    )
    rep_exists = (
        db.query(Report).filter(Report.analysis_id == analysis_id).first() is not None
    )

    return AnalysisDetailResponse(
        analysis_id=record.analysis_id,
        filename=record.filename,
        upload_timestamp=record.upload_timestamp,
        completed_timestamp=record.completed_timestamp,
        analysis_status=record.analysis_status,
        quality_metadata=record.quality_metadata,
        summary=record.summary_payload,
        model_results=[
            {
                "result_id": mr.result_id,
                "model_name": mr.model_name,
                "model_version": mr.model_version,
                "confidence": mr.confidence,
                "output_reference": mr.output_reference,
                "prediction": mr.prediction,
            }
            for mr in m_results
        ],
        report_available=rep_exists or (record.analysis_status == "completed"),
    )


@router.get("/{analysis_id}/report")
def download_pdf_report(
    analysis_id: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Generate and download the PDF analysis report (PRD Section 13.2 Page 8)."""
    record = db.query(Analysis).filter(Analysis.analysis_id == analysis_id).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis ID not found.")
    if record.analysis_status != "completed":
        orchestrator = get_orchestrator()
        orchestrator.run_analysis(analysis_id, db)
        db.refresh(record)

    pdf_path = generate_analysis_pdf_report(record, db, root_dir=".")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"RA_XRay_Report_{analysis_id[:8]}.pdf",
    )


@router.delete("/{analysis_id}")
def delete_analysis_record(
    analysis_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Delete an uploaded image, overlays, generated PDF report, and database record (PRD Section 18)."""
    record = db.query(Analysis).filter(Analysis.analysis_id == analysis_id).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis ID not found.")

    if record.image_reference:
        Path(record.image_reference).unlink(missing_ok=True)
    overlay_dir = Path("storage") / "overlays" / analysis_id
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir, ignore_errors=True)

    for rep in record.reports:
        Path(rep.report_reference).unlink(missing_ok=True)

    db.delete(record)
    db.commit()
    return {"deleted": True, "analysis_id": analysis_id}
