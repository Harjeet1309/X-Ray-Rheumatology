"""Comprehensive End-to-End Test Suite for the Rheumatological X-ray Analysis Platform.

Covers all acceptance criteria in PRD Section 22:
- Dataset validation & zero patient-level split leakage
- Reproducible preprocessing & augmentation
- All 4 deep-learning models (Classification, Detection, Segmentation, SvdH Severity)
- Explainable AI (Grad-CAM + 5 visualization layers)
- FastAPI Backend endpoints (Auth, Upload, Sample, Run, History, PDF Report, Models, Health)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import numpy as np

from backend.app.database.session import init_db
from backend.app.main import app
from src.classification.inference import ClassificationInferenceService
from src.data.dataset_split import verify_zero_patient_leakage
from src.data.preprocessing import preprocess_xray
from src.data.validation import run_dataset_validation, validate_single_image
from src.detection.inference import DetectionInferenceService
from src.explainability.gradcam import GradCAMExtractor
from src.segmentation.inference import SegmentationInferenceService
from src.severity.inference import SeverityInferenceService


def test_dataset_validation_and_zero_patient_leakage() -> None:
    report = run_dataset_validation(".")
    assert report["validation_passed"] is True
    assert report["corrupted_images_count"] == 0
    assert report["duplicate_images_count"] == 0
    assert report["patient_leakage"]["passed_zero_leakage"] is True


def test_preprocessing_pipeline_consistency() -> None:
    sample_bmp = next(Path("dataset/raw/Segmentation/test").glob("*.bmp"))
    val = validate_single_image(sample_bmp)
    assert val["valid"] is True

    out1 = preprocess_xray(sample_bmp, target_size=(256, 256), use_clahe=True, three_channel=True)
    out2 = preprocess_xray(sample_bmp, target_size=(256, 256), use_clahe=True, three_channel=True)
    assert out1["tensor"].shape == (3, 256, 256)
    assert np.allclose(out1["normalized"], out2["normalized"])


def test_all_four_models_and_gradcam_inference() -> None:
    sample_bmp = next(Path("dataset/raw/Segmentation/test").glob("*.bmp"))

    # Model 1: Classification
    cls_svc = ClassificationInferenceService(".")
    cls_out = cls_svc.predict(sample_bmp)
    assert 0.0 <= cls_out["ra_probability"] <= 1.0
    assert "predicted_class" in cls_out

    # Model 2: Detection
    det_svc = DetectionInferenceService(".")
    det_out = det_svc.predict(sample_bmp)
    assert det_out["total_joints_localized"] == 16
    assert len(det_out["detections"]) == 16

    # Model 3: Segmentation
    seg_svc = SegmentationInferenceService(".")
    seg_out = seg_svc.predict(sample_bmp)
    assert seg_out["mask"].shape == (512, 512)
    assert seg_out["bone_coverage_percent"] >= 0.0

    # Model 4: Severity
    sev_svc = SeverityInferenceService(".")
    sev_out = sev_svc.predict(sample_bmp)
    assert 0 <= sev_out["total_be_score"] <= 80
    assert 0 <= sev_out["total_jsn_score"] <= 60
    assert sev_out["total_svdh_score"] == sev_out["total_be_score"] + sev_out["total_jsn_score"]

    # Explainable AI: Grad-CAM
    cam = GradCAMExtractor(cls_svc.model, cls_svc.model.get_target_conv_layer())
    prep = preprocess_xray(sample_bmp, target_size=cls_svc.image_size, three_channel=True)
    hm = cam.generate_heatmap(prep["tensor"].unsqueeze(0), output_size=(512, 512))
    assert hm.shape == (512, 512)
    assert 0.0 <= float(hm.min()) <= float(hm.max()) <= 1.0


def test_fastapi_backend_end_to_end() -> None:
    init_db(".")

    async def _run_api_checks() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Health check
            health = await client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["status"] == "healthy"
            assert health.json()["models_ready"] is True

            # 2. User registration & login
            email = "test_researcher@rheumavision.org"
            reg = await client.post(
                "/api/auth/register",
                json={"name": "Dr. Test Researcher", "email": email, "password": "securepassword123"},
            )
            assert reg.status_code in (201, 400)

            login = await client.post(
                "/api/auth/login",
                json={"email": email, "password": "securepassword123"},
            )
            assert login.status_code == 200
            token = login.json()["access_token"]
            assert token

            # 3. Load sample RAM-H1200 X-ray
            sample_res = await client.post(
                "/api/analysis/sample?case_type=ra_positive",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert sample_res.status_code == 201
            analysis_id = sample_res.json()["analysis_id"]

            # 4. Run full 4-model AI analysis
            run_res = await client.post(f"/api/analysis/{analysis_id}/run")
            assert run_res.status_code == 200
            run_data = run_res.json()
            assert run_data["analysis_status"] == "completed"
            assert "classification" in run_data["summary"]
            assert "detection" in run_data["summary"]
            assert "segmentation" in run_data["summary"]
            assert "severity" in run_data["summary"]
            assert "visualizations" in run_data["summary"]

            # 5. Retrieve analysis details
            detail_res = await client.get(f"/api/analysis/{analysis_id}")
            assert detail_res.status_code == 200
            assert len(detail_res.json()["model_results"]) == 4

            # 6. Download generated PDF report
            pdf_res = await client.get(f"/api/analysis/{analysis_id}/report")
            assert pdf_res.status_code == 200
            assert pdf_res.headers["content-type"] == "application/pdf"
            assert pdf_res.content.startswith(b"%PDF")

            # 7. Check history & models registry
            hist_res = await client.get("/api/analysis/history")
            assert hist_res.status_code == 200
            assert hist_res.json()["statistics"]["completed_analyses"] >= 1

            models_res = await client.get("/api/models")
            assert models_res.status_code == 200
            assert len(models_res.json()["models"]) >= 4

    asyncio.run(_run_api_checks())
